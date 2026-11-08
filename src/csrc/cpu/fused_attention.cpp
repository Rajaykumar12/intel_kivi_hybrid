// ==========================================================================
// KIVI fused decode attention on the CPU (roadmap Phase 2 §B.4).
//
// Computes one decode step of attention (query length 1) for every head
// directly from the packed 2-bit history plus the FP residual, without ever
// materialising the dequantized history: the history is read as 2-bit codes
// (1/16 of the fp32 bytes) plus per-group scales/zeros. On Iris Xe laptops
// the model and attention run on the CPU (attention on the iGPU measured
// 7-12x slower, docs/status_and_roadmap.md §2.4), so this is where the
// bandwidth saving pays.
//
// Parallelism: torch links GNU libgomp, and its at::parallel_for is a
// header-inline `#pragma omp parallel` that would need this file compiled
// with OpenMP — with icpx that means Intel's libiomp5 in the same process as
// libgomp (two runtimes, two spinning pools). Instead the two parallel
// phases run on torch's own inter-op pool via the exported at::launch.
// ==========================================================================

#include "cpu/fused_attention.hpp"

#include <ATen/Parallel.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <vector>

namespace {

#define KIVI_TARGET __attribute__((target("avx2,fma")))
namespace avx2 {
#include "cpu/fused_attention_impl.inc"
}
#undef KIVI_TARGET
#define KIVI_TARGET
namespace generic {
#include "cpu/fused_attention_impl.inc"
}
#undef KIVI_TARGET

const float* code_lut() {
    static const std::vector<float> lut = [] {
        std::vector<float> l(256 * 4);
        for (int b = 0; b < 256; ++b)
            for (int j = 0; j < 4; ++j) l[b * 4 + j] = static_cast<float>((b >> (2 * j)) & 3);
        return l;
    }();
    return lut.data();
}

// AVX2/FMA path when the CPU has it, unless KIVI_FUSED_GENERIC is set (read
// once) — that forces the portable build of the same loops, for testing.
bool cpu_has_avx2() {
    static const bool ok = !std::getenv("KIVI_FUSED_GENERIC") &&
                           __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return ok;
}

// Run f(i) for i in [0, n) on torch's inter-op pool (plus this thread).
template <typename F>
void parallel_tasks(int64_t n, const F& f) {
    const int64_t workers = std::min<int64_t>(
        n, std::max<int64_t>(1, static_cast<int64_t>(at::get_num_interop_threads())));
    if (workers <= 1) {
        for (int64_t i = 0; i < n; ++i) f(i);
        return;
    }
    std::atomic<int64_t> next{0};
    std::mutex m;
    std::condition_variable cv;
    int64_t done = 0;
    std::exception_ptr err;
    auto worker = [&] {
        try {
            for (int64_t i; (i = next.fetch_add(1)) < n;) f(i);
        } catch (...) {
            std::lock_guard<std::mutex> lk(m);
            if (!err) err = std::current_exception();
        }
        std::lock_guard<std::mutex> lk(m);
        if (++done == workers) cv.notify_one();
    };
    for (int64_t w = 1; w < workers; ++w) at::launch(worker);
    worker();
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [&] { return done == workers; });
    if (err) std::rethrow_exception(err);
}

}  // namespace

torch::Tensor kivi_fused_decode_attention(
    torch::Tensor query, torch::Tensor k_res, torch::Tensor v_res,
    torch::Tensor packed_k, torch::Tensor scales_k, torch::Tensor zeros_k,
    torch::Tensor packed_v, torch::Tensor scales_v, torch::Tensor zeros_v,
    double scaling, int64_t group_size)
{
    TORCH_CHECK(query.device().is_cpu(), "fused attention runs on the CPU");
    TORCH_CHECK(query.dim() == 4 && query.size(2) == 1,
                "query must be [B, Hq, 1, D] (decode step), got ", query.sizes());
    const int64_t B = query.size(0), Hq = query.size(1), D = query.size(3);
    TORCH_CHECK(k_res.dim() == 4 && v_res.sizes() == k_res.sizes(),
                "residual keys/values must be [B, Hkv, n_res, D] of equal shape");
    const int64_t Hkv = k_res.size(1), NR = k_res.size(2);
    TORCH_CHECK(k_res.size(0) == B && k_res.size(3) == D, "residual shape mismatch");
    TORCH_CHECK(Hkv > 0 && Hq % Hkv == 0, "query heads must be a multiple of KV heads");
    const int64_t n_rep = Hq / Hkv;
    const int64_t G = group_size;
    TORCH_CHECK(G > 0 && G % 4 == 0 && D % G == 0, "group_size must divide head_dim and be a multiple of 4");

    const int64_t T = packed_k.numel() ? packed_k.size(3) * 4 : 0;
    if (T) {
        TORCH_CHECK(packed_k.scalar_type() == torch::kUInt8 && packed_v.scalar_type() == torch::kUInt8,
                    "packed history must be uint8");
        TORCH_CHECK(packed_k.dim() == 4 && packed_k.size(0) == B && packed_k.size(1) == Hkv &&
                    packed_k.size(2) == D && packed_k.stride(3) == 1,
                    "packed_k must be [B, Hkv, D, T/4] with a contiguous last dim, got ",
                    packed_k.sizes(), " strides ", packed_k.strides());
        TORCH_CHECK(packed_v.dim() == 4 && packed_v.size(0) == B && packed_v.size(1) == Hkv &&
                    packed_v.size(2) == T && packed_v.size(3) == D / 4 &&
                    packed_v.stride(3) == 1 && packed_v.stride(2) == D / 4,
                    "packed_v must be [B, Hkv, T, D/4] with contiguous tokens, got ",
                    packed_v.sizes(), " strides ", packed_v.strides());
        TORCH_CHECK(T % G == 0, "history length must be a multiple of group_size");
        TORCH_CHECK(scales_k.sizes() == zeros_k.sizes() && scales_k.size(3) == T / G,
                    "scales_k/zeros_k must be [B, Hkv, D, T/G]");
        TORCH_CHECK(scales_v.sizes() == zeros_v.sizes() && scales_v.size(2) == T &&
                    scales_v.size(3) == D / G, "scales_v/zeros_v must be [B, Hkv, T, D/G]");
    }
    TORCH_CHECK(T + NR > 0, "nothing to attend to");

    // Small operands to contiguous fp32 (metadata is 1/G of the history).
    auto f32 = [](const torch::Tensor& t) { return t.to(torch::kFloat32).contiguous(); };
    torch::Tensor q = f32(query).view({B, Hq, D});
    torch::Tensor kr = f32(k_res), vr = f32(v_res);
    torch::Tensor sk, zk, sv, zv;
    if (T) { sk = f32(scales_k); zk = f32(zeros_k); sv = f32(scales_v); zv = f32(zeros_v); }

    const float scale = static_cast<float>(scaling);
    const int64_t BH = B * Hkv;
    const int64_t ld = T + NR;
    const int64_t chunk = std::max<int64_t>(G, (256 / G) * G);
    const int64_t n_chunks = T ? (T + chunk - 1) / chunk : 0;
    const float* lut = code_lut();
    const bool fast = cpu_has_avx2();

    std::vector<float> logits(static_cast<size_t>(BH * n_rep * ld));
    std::vector<float> partial(static_cast<size_t>(std::max<int64_t>(n_chunks, 1) * BH * n_rep * D), 0.0f);

    const float* qp = q.data_ptr<float>();
    const uint8_t* pkp = T ? packed_k.data_ptr<uint8_t>() : nullptr;
    const uint8_t* pvp = T ? packed_v.data_ptr<uint8_t>() : nullptr;

    // Phase A: history logits, parallel over (b*h, token chunk).
    if (T) {
        parallel_tasks(BH * n_chunks, [&](int64_t task) {
            const int64_t bh = task / n_chunks, c = task % n_chunks;
            const int64_t b = bh / Hkv, h = bh % Hkv;
            const int64_t t0 = c * chunk, t1 = std::min(T, t0 + chunk);
            std::vector<float> acc(static_cast<size_t>(n_rep * (t1 - t0)));
            std::vector<float> bias(static_cast<size_t>(n_rep * ((t1 - t0) / G)));
            const float* qh = qp + (b * Hq + h * n_rep) * D;
            const uint8_t* pkh = pkp + b * packed_k.stride(0) + h * packed_k.stride(1);
            const int64_t mrow = T / G;
            const float* skh = sk.data_ptr<float>() + bh * D * mrow;
            const float* zkh = zk.data_ptr<float>() + bh * D * mrow;
            float* lg = logits.data() + bh * n_rep * ld;
            auto fn = fast ? avx2::key_logits_chunk : generic::key_logits_chunk;
            fn(qh, static_cast<int>(n_rep), static_cast<int>(D), static_cast<int>(G),
               pkh, packed_k.stride(2), skh, zkh, mrow, t0, t1, scale, lg, ld, lut,
               acc.data(), bias.data());
        });
    }

    // Residual logits + softmax, per (b*h, query head). Cheap: ld floats each.
    const float* krp = kr.data_ptr<float>();
    const float* vrp = vr.data_ptr<float>();
    for (int64_t bh = 0; bh < BH; ++bh) {
        const int64_t b = bh / Hkv, h = bh % Hkv;
        for (int64_t r = 0; r < n_rep; ++r) {
            const float* qv = qp + (b * Hq + h * n_rep + r) * D;
            float* lg = logits.data() + (bh * n_rep + r) * ld;
            for (int64_t j = 0; j < NR; ++j) {
                const float* kj = krp + (bh * NR + j) * D;
                float s = 0.0f;
                for (int64_t d = 0; d < D; ++d) s += qv[d] * kj[d];
                lg[T + j] = scale * s;
            }
            float mx = -INFINITY;
            for (int64_t t = 0; t < ld; ++t) mx = std::max(mx, lg[t]);
            float sum = 0.0f;
            for (int64_t t = 0; t < ld; ++t) { lg[t] = std::exp(lg[t] - mx); sum += lg[t]; }
            const float inv = 1.0f / sum;
            for (int64_t t = 0; t < ld; ++t) lg[t] *= inv;
        }
    }

    // Phase B: history value accumulation, parallel over (b*h, token chunk).
    if (T) {
        parallel_tasks(BH * n_chunks, [&](int64_t task) {
            const int64_t bh = task / n_chunks, c = task % n_chunks;
            const int64_t b = bh / Hkv, h = bh % Hkv;
            const int64_t t0 = c * chunk, t1 = std::min(T, t0 + chunk);
            const uint8_t* pvh = pvp + b * packed_v.stride(0) + h * packed_v.stride(1);
            const int64_t mrow = D / G;
            const float* svh = sv.data_ptr<float>() + bh * T * mrow;
            const float* zvh = zv.data_ptr<float>() + bh * T * mrow;
            float* part = partial.data() + (c * BH + bh) * n_rep * D;
            auto fn = fast ? avx2::value_accum_chunk : generic::value_accum_chunk;
            fn(logits.data() + bh * n_rep * ld, ld, static_cast<int>(n_rep),
               static_cast<int>(D), static_cast<int>(G), pvh, packed_v.stride(2),
               svh, zvh, mrow, t0, t1, part, lut);
        });
    }

    // Reduce chunk partials, add the residual values, write [B, 1, Hq, D].
    torch::Tensor out = torch::empty({B, 1, Hq, D}, torch::kFloat32);
    float* op = out.data_ptr<float>();
    for (int64_t bh = 0; bh < BH; ++bh) {
        const int64_t b = bh / Hkv, h = bh % Hkv;
        for (int64_t r = 0; r < n_rep; ++r) {
            const float* p = logits.data() + (bh * n_rep + r) * ld;
            float* o = op + (b * Hq + h * n_rep + r) * D;
            for (int64_t d = 0; d < D; ++d) o[d] = 0.0f;
            for (int64_t c = 0; c < n_chunks; ++c) {
                const float* part = partial.data() + ((c * BH + bh) * n_rep + r) * D;
                for (int64_t d = 0; d < D; ++d) o[d] += part[d];
            }
            for (int64_t j = 0; j < NR; ++j) {
                const float pj = p[T + j];
                const float* vj = vrp + (bh * NR + j) * D;
                for (int64_t d = 0; d < D; ++d) o[d] += pj * vj[d];
            }
        }
    }
    return out.to(query.scalar_type());
}

bool kivi_fused_attention_uses_avx2() { return cpu_has_avx2(); }
