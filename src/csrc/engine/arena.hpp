#pragma once
// ==========================================================================
// KIVI packed-KV arena for one layer. Ported in Phase 5 (S1.3) from the
// former Python cache/arena.py, which was removed once the C++ engine
// became the only engine (S1.8); names and behaviour are unchanged.
//
// Layout:
//   keys,   channel-major: packed [B,H,D,cap/4] u8,  scales/zeros [B,H,D,cap/G]
//   values, token-major:   packed [B,H,cap,D/4] u8,  scales/zeros [B,H,cap,D/G]
// Key rows may be padded to `row_align` bytes (off by default — measured to
// cost more resident memory than it is worth, Phase 1 step 10). Capacity is in
// tokens and always a multiple of G; the first `used` tokens are valid.
// allocated_bytes() counts every base allocation, padding included.
// ==========================================================================

#include <torch/torch.h>

#include <array>
#include <functional>
#include <optional>
#include <vector>

namespace kivi {

// Allocates the arena's storage: USM host memory when the quantize kernels
// write it directly (zero-copy), ordinary CPU memory otherwise.
using AllocFn = std::function<at::Tensor(at::IntArrayRef, at::ScalarType)>;

AllocFn usm_host_alloc();   // kivi::alloc_usm_host, 64-byte aligned
AllocFn cpu_alloc();        // at::empty on the CPU

inline int64_t round_up(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

// Next capacity when `needed` tokens no longer fit in `cap`: max(4 groups,
// cap/4) growth — O(T) total copying, bounded spare tail.
int64_t grow_capacity(int64_t cap, int64_t needed, int64_t G);

class LayerArena {
public:
    LayerArena(int64_t B, int64_t H, int64_t D, int64_t G, int64_t cap,
               at::ScalarType meta_dtype, AllocFn alloc, int64_t row_align = 1);

    int64_t B, H, D, G, cap, used = 0;
    at::ScalarType meta_dtype;
    int64_t row_align;

    at::Tensor packed_k, scales_k, zeros_k;
    at::Tensor packed_v, scales_v, zeros_v;

    // (packed, scales, zeros) views covering tokens [t0, t1).
    std::array<at::Tensor, 3> key_slices(int64_t t0, int64_t t1) const;
    std::array<at::Tensor, 3> value_slices(int64_t t0, int64_t t1) const;
    std::array<at::Tensor, 3> used_keys() const { return key_slices(0, used); }
    std::array<at::Tensor, 3> used_values() const { return value_slices(0, used); }

    int64_t allocated_bytes() const;
    int64_t used_bytes() const;

    // A larger arena holding a copy of this one's used tokens. The caller
    // must ensure no kernel is still writing into this arena.
    LayerArena grown(int64_t new_cap) const;

private:
    at::Tensor rows(std::vector<int64_t> outer, int64_t row_elems, at::ScalarType dt);
    at::Tensor plain(std::vector<int64_t> shape, at::ScalarType dt);

    AllocFn alloc_;
    std::vector<at::Tensor> bases_;
};

// An arena with room for `needed` tokens, allocating or growing as required.
// `reserve` (a token count, 0 if unknown) sizes the first allocation and any
// growth up front, so the arena never grows and ends with no spare tail.
LayerArena ensure_capacity(std::optional<LayerArena> arena, int64_t needed,
                           int64_t B, int64_t H, int64_t D, int64_t G,
                           at::ScalarType meta_dtype, int64_t reserve,
                           const AllocFn& alloc);

}  // namespace kivi
