// KIVI packed-KV arena — see arena.hpp.
#include "engine/arena.hpp"

#include "kivi/usm.hpp"

namespace kivi {

namespace {
constexpr int64_t kMinGrowGroups = 4;
}

AllocFn usm_host_alloc() {
    return [](at::IntArrayRef sizes, at::ScalarType dt) { return alloc_usm_host(sizes, dt); };
}

AllocFn cpu_alloc() {
    return [](at::IntArrayRef sizes, at::ScalarType dt) {
        return at::empty(sizes, at::TensorOptions().dtype(dt).device(at::kCPU));
    };
}

int64_t grow_capacity(int64_t cap, int64_t needed, int64_t G) {
    const int64_t target = std::max(needed, cap + std::max(kMinGrowGroups * G, cap / 4));
    return round_up(target, G);
}

LayerArena::LayerArena(int64_t B_, int64_t H_, int64_t D_, int64_t G_, int64_t cap_,
                       at::ScalarType meta, AllocFn alloc, int64_t row_align_)
    : B(B_), H(H_), D(D_), G(G_), cap(cap_), meta_dtype(meta), row_align(row_align_),
      alloc_(std::move(alloc)) {
    TORCH_CHECK(cap > 0 && cap % G == 0, "arena capacity ", cap, " must be a positive multiple of G=", G);
    TORCH_CHECK(row_align >= 1, "row_align must be >= 1");
    // Keys: rows along the token axis (optionally padded).
    packed_k = rows({B, H, D}, cap / 4, at::kByte);
    scales_k = rows({B, H, D}, cap / G, meta_dtype);
    zeros_k = rows({B, H, D}, cap / G, meta_dtype);
    // Values: natural layout, tokens contiguous within each head.
    packed_v = plain({B, H, cap, D / 4}, at::kByte);
    scales_v = plain({B, H, cap, D / G}, meta_dtype);
    zeros_v = plain({B, H, cap, D / G}, meta_dtype);
}

at::Tensor LayerArena::rows(std::vector<int64_t> outer, int64_t row_elems, at::ScalarType dt) {
    const int64_t elt = static_cast<int64_t>(c10::elementSize(dt));
    const int64_t padded = round_up(row_elems * elt, row_align) / elt;
    outer.push_back(padded);
    at::Tensor base = alloc_(outer, dt);
    bases_.push_back(base);
    return base.narrow(-1, 0, row_elems);
}

at::Tensor LayerArena::plain(std::vector<int64_t> shape, at::ScalarType dt) {
    at::Tensor base = alloc_(shape, dt);
    bases_.push_back(base);
    return base;
}

std::array<at::Tensor, 3> LayerArena::key_slices(int64_t t0, int64_t t1) const {
    return {packed_k.slice(-1, t0 / 4, t1 / 4),
            scales_k.slice(-1, t0 / G, t1 / G),
            zeros_k.slice(-1, t0 / G, t1 / G)};
}

std::array<at::Tensor, 3> LayerArena::value_slices(int64_t t0, int64_t t1) const {
    return {packed_v.slice(2, t0, t1), scales_v.slice(2, t0, t1), zeros_v.slice(2, t0, t1)};
}

int64_t LayerArena::allocated_bytes() const {
    int64_t n = 0;
    for (const auto& b : bases_) n += b.numel() * static_cast<int64_t>(b.element_size());
    return n;
}

int64_t LayerArena::used_bytes() const {
    int64_t n = 0;
    for (const auto& views : {used_keys(), used_values()})
        for (const auto& t : views) n += t.numel() * static_cast<int64_t>(t.element_size());
    return n;
}

LayerArena LayerArena::grown(int64_t new_cap) const {
    TORCH_CHECK(new_cap >= cap && new_cap % G == 0, "grown: bad capacity ", new_cap);
    LayerArena next(B, H, D, G, new_cap, meta_dtype, alloc_, row_align);
    if (used) {
        auto dk = next.key_slices(0, used), sk = used_keys();
        auto dv = next.value_slices(0, used), sv = used_values();
        for (int i = 0; i < 3; ++i) {
            dk[i].copy_(sk[i]);
            dv[i].copy_(sv[i]);
        }
    }
    next.used = used;
    return next;
}

LayerArena ensure_capacity(std::optional<LayerArena> arena, int64_t needed,
                           int64_t B, int64_t H, int64_t D, int64_t G,
                           at::ScalarType meta_dtype, int64_t reserve,
                           const AllocFn& alloc) {
    const int64_t floor = std::max(needed, reserve);
    if (!arena) {
        // First allocation: exactly what this flush (or the reserve) needs.
        return LayerArena(B, H, D, G, round_up(floor, G), meta_dtype, alloc);
    }
    if (needed <= arena->cap) return std::move(*arena);
    const int64_t cap = reserve >= needed ? round_up(floor, G)
                                          : grow_capacity(arena->cap, needed, G);
    return arena->grown(cap);
}

}  // namespace kivi
