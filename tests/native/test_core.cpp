// ==========================================================================
// Native (Python-free) tests for libkivi_core.
//
// 1. The process has no libpython mapped — the premise of Phase 5.
// 2. CPU fused decode attention vs an ATen dequantize-then-attend reference
//    (MHA and GQA, with and without residual).
// 3. If an XPU is present: quantize -> dequantize round trip for keys and
//    values, error bounded by half a quantization step.
//
// Exit code 0 on success; each failure is printed.
// ==========================================================================
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <string>

#include "cpu/fused_attention.hpp"
#include "ops/kv_cache_ops.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  ok    " : "  FAIL  ") << what << "\n";
    if (!ok) ++failures;
}

bool python_is_loaded() {
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line))
        if (line.find("libpython") != std::string::npos) return true;
    return false;
}

// Unpack LSB-first 2-bit codes along the last dim: [..., n/4] u8 -> [..., n] float.
torch::Tensor unpack(const torch::Tensor& packed) {
    auto shifts = torch::tensor({0, 2, 4, 6}, torch::kUInt8);
    auto q = torch::bitwise_and(torch::bitwise_right_shift(packed.unsqueeze(-1), shifts), 3);
    auto sizes = packed.sizes().vec();
    sizes.back() *= 4;
    return q.reshape(sizes).to(torch::kFloat32);
}

void test_fused_attention(int64_t Hq, int64_t Hkv, int64_t T, int64_t NR) {
    const int64_t B = 1, D = 64, G = 32;
    torch::manual_seed(0);
    auto q = torch::randn({B, Hq, 1, D});
    auto kr = torch::randn({B, Hkv, NR, D});
    auto vr = torch::randn({B, Hkv, NR, D});
    auto pk = torch::randint(0, 256, {B, Hkv, D, T / 4}, torch::kUInt8);
    auto sk = (torch::rand({B, Hkv, D, T / G}) * 0.5 + 0.01).to(torch::kHalf);
    auto zk = torch::randn({B, Hkv, D, T / G}).to(torch::kHalf);
    auto pv = torch::randint(0, 256, {B, Hkv, T, D / 4}, torch::kUInt8);
    auto sv = (torch::rand({B, Hkv, T, D / G}) * 0.5 + 0.01).to(torch::kHalf);
    auto zv = torch::randn({B, Hkv, T, D / G}).to(torch::kHalf);
    const double scaling = 1.0 / std::sqrt(static_cast<double>(D));

    auto out = kivi_fused_decode_attention(q, kr, vr, pk, sk, zk, pv, sv, zv, scaling, G);

    // Same op through the dispatcher by name: libkivi_core registered
    // kivi::fused_decode_attention on its own (no Python involved).
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("kivi::fused_decode_attention", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, const at::Tensor&, const at::Tensor&,
                          double, int64_t)>();
    auto via_op = op.call(q, kr, vr, pk, sk, zk, pv, sv, zv, scaling, G);
    check(torch::equal(out, via_op), "dispatcher op kivi::fused_decode_attention == direct call");

    // Reference: dequantize history, append residual, plain softmax attention.
    auto hk = (unpack(pk) * sk.to(torch::kFloat32).repeat_interleave(G, -1)
               + zk.to(torch::kFloat32).repeat_interleave(G, -1)).transpose(2, 3);
    auto hv = unpack(pv) * sv.to(torch::kFloat32).repeat_interleave(G, -1)
              + zv.to(torch::kFloat32).repeat_interleave(G, -1);
    auto k = torch::cat({hk, kr}, 2).repeat_interleave(Hq / Hkv, 1);
    auto v = torch::cat({hv, vr}, 2).repeat_interleave(Hq / Hkv, 1);
    auto p = torch::softmax(torch::matmul(q, k.transpose(2, 3)) * scaling, -1);
    auto ref = torch::matmul(p, v).transpose(1, 2);

    const double diff = (out - ref).abs().max().item<double>();
    check(out.sizes() == ref.sizes() && diff < 1e-4,
          "fused attention Hq=" + std::to_string(Hq) + " Hkv=" + std::to_string(Hkv) +
          " T=" + std::to_string(T) + " NR=" + std::to_string(NR) +
          " (max diff " + std::to_string(diff) + ")");
}

void test_xpu_roundtrip() {
    const int64_t B = 1, H = 2, D = 64, G = 32, L = 4 * G;
    torch::manual_seed(1);
    auto opts = torch::TensorOptions().device(torch::kXPU);

    auto xk = torch::randn({B, H, D, L}).to(torch::kXPU);
    auto pk = torch::empty({B, H, D, L / 4}, opts.dtype(torch::kUInt8));
    auto sk = torch::empty({B, H, D, L / G}, opts.dtype(torch::kHalf));
    auto zk = torch::empty_like(sk);
    kivi_quant_keys(xk, pk, sk, zk, G);
    auto ok = torch::empty_like(xk);
    kivi_dequant_keys(pk, sk, zk, ok, G);
    double err = (ok.cpu() - xk.cpu()).abs().max().item<double>();
    double half_step = sk.cpu().to(torch::kFloat32).max().item<double>() / 2;
    check(err <= half_step + 1e-2, "XPU key round trip (err " + std::to_string(err) + ")");

    auto xv = torch::randn({B, H, L, D}).to(torch::kXPU);
    auto pv = torch::empty({B, H, L, D / 4}, opts.dtype(torch::kUInt8));
    auto sv = torch::empty({B, H, L, D / G}, opts.dtype(torch::kHalf));
    auto zv = torch::empty_like(sv);
    kivi_quant_values(xv, pv, sv, zv, D, G);
    auto ov = torch::empty_like(xv);
    kivi_dequant_values(pv, sv, zv, ov, D, G);
    err = (ov.cpu() - xv.cpu()).abs().max().item<double>();
    half_step = sv.cpu().to(torch::kFloat32).max().item<double>() / 2;
    check(err <= half_step + 1e-2, "XPU value round trip (err " + std::to_string(err) + ")");
}

}  // namespace

int main() {
    std::cout << "libkivi_core native tests\n";
    check(!python_is_loaded(), "no libpython mapped in this process");

    test_fused_attention(4, 4, 64, 5);
    test_fused_attention(14, 2, 320, 17);
    test_fused_attention(8, 2, 640, 0);

    if (torch::xpu::is_available()) {
        test_xpu_roundtrip();
    } else {
        std::cout << "  skip  XPU round trip (no XPU device)\n";
    }

    std::cout << (failures ? "FAILED: " : "PASSED") << (failures ? std::to_string(failures) : "")
              << "\n";
    return failures ? 1 : 0;
}
