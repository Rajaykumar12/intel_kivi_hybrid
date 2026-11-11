// ==========================================================================
// Native tests for kivi::LayerArena / ensure_capacity (Phase 5 S1.3).
// Mirrors the behaviour pinned by the Python arena tests in
// tests/unit/test_manager_bookkeeping.py (growth policy, exact reserve,
// growth preserves the used region, byte accounting).
// ==========================================================================
#include <torch/torch.h>

#include <iostream>
#include <string>

#include "engine/arena.hpp"
#include "kivi/common.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  ok    " : "  FAIL  ") << what << "\n";
    if (!ok) ++failures;
}

void test_grow_capacity() {
    check(kivi::grow_capacity(0, 5, 4) == 16, "grow_capacity: empty -> 4 groups minimum");
    check(kivi::grow_capacity(256, 300, 32) == 384, "grow_capacity: +max(4G, cap/4), rounded to G");
    check(kivi::grow_capacity(1024, 1030, 32) == 1280, "grow_capacity: 25% step at large cap");
    check(kivi::grow_capacity(64, 500, 8) == 504, "grow_capacity: jumps straight to needed");
}

void test_first_allocation_and_reserve() {
    auto a = kivi::ensure_capacity(std::nullopt, 40, 1, 2, 8, 8, at::kHalf, 0, kivi::cpu_alloc());
    check(a.cap == 40 && a.used == 0, "first allocation is exact (no reserve)");
    auto r = kivi::ensure_capacity(std::nullopt, 8, 1, 2, 8, 4, at::kHalf, 100, kivi::cpu_alloc());
    check(r.cap == 100, "reserve sizes the first allocation");
    const void* p = r.packed_k.data_ptr();
    auto r2 = kivi::ensure_capacity(std::move(r), 96, 1, 2, 8, 4, at::kHalf, 100, kivi::cpu_alloc());
    check(r2.cap == 100 && r2.packed_k.data_ptr() == p, "fits -> same storage, no reallocation");
}

void test_shapes_and_slices() {
    const int64_t B = 2, H = 3, D = 8, G = 4, cap = 32;
    kivi::LayerArena a(B, H, D, G, cap, at::kHalf, kivi::cpu_alloc());
    check(a.packed_k.sizes() == at::IntArrayRef({B, H, D, cap / 4}), "packed_k [B,H,D,cap/4]");
    check(a.scales_k.sizes() == at::IntArrayRef({B, H, D, cap / G}), "scales_k [B,H,D,cap/G]");
    check(a.packed_v.sizes() == at::IntArrayRef({B, H, cap, D / 4}), "packed_v [B,H,cap,D/4]");
    check(a.scales_v.sizes() == at::IntArrayRef({B, H, cap, D / G}), "scales_v [B,H,cap,D/G]");
    auto ks = a.key_slices(8, 16);
    check(ks[0].size(-1) == 2 && ks[1].size(-1) == 2, "key_slices(8,16): 2 packed bytes, 2 groups");
    auto vs = a.value_slices(8, 16);
    check(vs[0].size(2) == 8 && vs[1].size(2) == 8, "value_slices(8,16): 8 token rows");
    const int64_t expect = B * H * D * (cap / 4) + 2 * B * H * D * (cap / G) * 2
                         + B * H * cap * (D / 4) + 2 * B * H * cap * (D / G) * 2;
    check(a.allocated_bytes() == expect, "allocated_bytes counts all six tensors");
    a.used = 16;
    check(a.used_bytes() * 2 == a.allocated_bytes(), "used_bytes = half at used=cap/2");
}

void test_growth_preserves_used_region() {
    const int64_t G = 4;
    kivi::LayerArena a(1, 2, 8, G, 16, at::kFloat, kivi::cpu_alloc());
    a.used = 12;
    for (auto& t : {a.packed_k, a.packed_v}) t.random_(0, 256);
    for (auto& t : {a.scales_k, a.zeros_k, a.scales_v, a.zeros_v}) t.normal_();
    auto before_k = a.used_keys(), before_v = a.used_values();
    auto b = a.grown(kivi::grow_capacity(a.cap, 20, G));
    check(b.cap == 32 && b.used == 12, "grown: capacity 16 -> 32, used kept");
    bool same = true;
    auto ak = b.used_keys(), av = b.used_values();
    for (int i = 0; i < 3; ++i) same = same && at::equal(ak[i], before_k[i]) && at::equal(av[i], before_v[i]);
    check(same, "grown: used region copied bit-exactly");
}

void test_row_padding() {
    kivi::LayerArena padded(1, 2, 8, 4, 12, at::kHalf, kivi::cpu_alloc(), 64);
    kivi::LayerArena plain(1, 2, 8, 4, 12, at::kHalf, kivi::cpu_alloc());
    check(padded.packed_k.stride(-2) == 64, "row_align=64: key rows on 64-byte boundaries");
    check(padded.scales_k.stride(-2) * 2 == 64, "row_align=64: fp16 scale rows padded too");
    check(padded.allocated_bytes() > plain.allocated_bytes(), "padding counted as resident");
}

void test_usm_arena() {
    kivi::LayerArena a(1, 2, 8, 4, 16, at::kHalf, kivi::usm_host_alloc());
    check(kivi::is_kernel_accessible(a.packed_k) && kivi::is_kernel_accessible(a.scales_v),
          "USM arena storage is kernel-accessible");
    check(kivi::is_kernel_accessible(a.key_slices(4, 12)[0]), "USM arena views stay accessible");
}

}  // namespace

int main() {
    std::cout << "kivi::LayerArena native tests\n";
    test_grow_capacity();
    test_first_allocation_and_reserve();
    test_shapes_and_slices();
    test_growth_preserves_used_region();
    test_row_padding();
    if (torch::xpu::is_available()) test_usm_arena();
    else std::cout << "  skip  USM arena (no XPU device)\n";
    std::cout << (failures ? "FAILED" : "PASSED") << "\n";
    return failures ? 1 : 0;
}
