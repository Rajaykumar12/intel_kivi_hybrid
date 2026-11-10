#pragma once
// ==========================================================================
// kivi::Engine — the KIVI cache engine in C++ (Phase 5 S1.4).
//
// The KIVI cache state machine (formerly the Python KiviCache internals,
// replaced by this engine in Phase 5 S1.8):
// FP residual per layer, flush of the residual's excess into the packed
// arena (2-bit payload + per-group scales/zeros), asynchronous harvest,
// reserve(), per-layer group size, measured stats. The numerics are the
// existing kernels; this file is bookkeeping. It reproduced the former
// Python engine step for step before replacing it.
//
// Two update orders, both from the Python engine:
//   update(i, k, v)  HF-native path (KiviLayer.update): harvest layer i →
//                    flush the residual as it stood BEFORE this step →
//                    (sync mode: harvest) → append. Keeps the HF path
//                    bit-identical to the legacy loop.
//   append(i, k, v)  legacy add_tokens path: append → flush. The caller runs
//                    harvest_all() before the layer loop (and after it in
//                    sync mode), as KiviCache.add_tokens does.
//
// Devices:
//   "xpu"  SYCL kernels. zero_copy (default on) stages inputs in USM host
//          memory and writes straight into a USM arena; otherwise device
//          temporaries are copied into a CPU arena at harvest.
//   "cpu"  reference quantize in ATen (same math as the kernels; rounds half
//          to even like torch.round, the kernels round half away from zero —
//          only exact ties differ). Lets the engine run with no XPU at all.
// ==========================================================================

#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "engine/arena.hpp"
#include "kivi/event.hpp"

namespace kivi {

enum class ZeroCopy { Auto, On, Off };

struct EngineConfig {
    int64_t num_layers = 0;
    int64_t head_dim = 0;
    int64_t num_kv_heads = 0;                 // 0: take it from the first tensor
    int64_t residual_length = 128;
    std::vector<int64_t> group_size;          // one per layer
    std::optional<at::ScalarType> metadata_dtype;   // nullopt = "auto" (fp16)
    ZeroCopy zero_copy = ZeroCopy::Auto;
    bool async_flush = true;
    std::string device = "xpu";
    int64_t row_align = 1;
};

struct EngineStats {
    int64_t group_size, total_tokens, quantized_tokens, residual_tokens, pending_tokens;
    int64_t packed_bytes, packed_used_bytes, packed_capacity_tokens, residual_bytes;
    int64_t kivi_bytes, fp_baseline_bytes;
    double compression;
};

class Engine {
public:
    explicit Engine(EngineConfig cfg);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void update(int64_t layer, const at::Tensor& k_new, const at::Tensor& v_new);
    void append(int64_t layer, const at::Tensor& k_new, const at::Tensor& v_new);
    void harvest(int64_t layer);

    // -- attention (Phase 5 S1.5) -------------------------------------------
    // Attention for `layer` over its packed history + FP residual, for a
    // query whose keys/values were already added with update(). Returns
    // [B, T, Hq, D] — the layout transformers attention functions return.
    //   decode (T == 1, no mask hiding anything, history present, fused):
    //       fused kernel over the 2-bit codes; history never dequantized.
    //   otherwise: dequantize history, append residual, then exactly the
    //       SDPA call transformers' sdpa_attention_forward makes (GQA via
    //       enable_gqa, is_causal when T > 1 and no mask). If no mask is given
    //       and earlier tokens exist (kv_len > T > 1, e.g. chunked prefill), a
    //       bottom-right causal mask is built: new tokens see all earlier
    //       tokens and each other causally. (HF instead truncates keys to T
    //       in that case and relies on always passing a mask.)
    at::Tensor attend(int64_t layer, const at::Tensor& query,
                      std::optional<double> scaling = std::nullopt,
                      const std::optional<at::Tensor>& attn_mask = std::nullopt,
                      bool fused = true);
    // Full-precision [B, H, T, D] keys and values for a layer: dequantized
    // history followed by the residual (the legacy get_full_cache() view).
    std::optional<std::pair<at::Tensor, at::Tensor>> full_kv(int64_t layer) const;
    int64_t fused_calls() const { return fused_calls_; }
    int64_t reference_calls() const { return reference_calls_; }
    void harvest_all();
    void reserve(int64_t total_tokens);
    void reset();

    // -- introspection ------------------------------------------------------
    const EngineConfig& config() const { return cfg_; }
    bool zero_copy_enabled() const;
    std::optional<at::ScalarType> cache_dtype() const { return cache_dtype_; }
    std::optional<at::ScalarType> meta_dtype() const { return meta_dtype_; }
    std::optional<std::pair<at::Tensor, at::Tensor>> residual(int64_t layer) const;
    // (packed_k, scales_k, zeros_k, packed_v, scales_v, zeros_v) of the
    // quantized prefix; nullopt until something has been harvested.
    std::optional<std::array<at::Tensor, 6>> history(int64_t layer) const;
    int64_t quant_tokens(int64_t layer) const { return L(layer).quant_tokens; }
    int64_t flushes(int64_t layer) const { return L(layer).flushes; }
    int64_t pending_tokens(int64_t layer) const;
    int64_t arena_capacity(int64_t layer) const;
    int64_t seq_length(int64_t layer) const;
    int64_t total_flushes() const;
    EngineStats stats(int64_t layer) const;

private:
    struct Pending {
        int64_t n = 0;
        bool direct = false;
        std::optional<KiviEvent> ev_k, ev_v;
        at::Tensor in_k, in_v;                      // kept alive: kernels read them
        std::array<at::Tensor, 3> keys, values;     // staged outputs (not direct)
    };
    struct Layer {
        at::Tensor res_k, res_v;                    // [B,H,n,D], cache dtype
        std::optional<LayerArena> arena;
        std::optional<Pending> pending;
        at::Tensor stage_k, stage_v;                // reusable USM staging
        int64_t quant_tokens = 0, flushes = 0;
    };

    Layer& L(int64_t i);
    const Layer& L(int64_t i) const;
    void init_session(const at::Tensor& k, int64_t layer);
    void append_residual(int64_t i, const at::Tensor& k_new, const at::Tensor& v_new);
    void maybe_flush(int64_t i);
    void submit_flush(int64_t i, int64_t flush_len);
    int64_t layer_reserve(int64_t i) const;
    AllocFn arena_alloc() const;
    at::Tensor staging(at::Tensor& buf, at::IntArrayRef shape, at::ScalarType dt);

    EngineConfig cfg_;
    at::Device device_;
    std::vector<Layer> layers_;
    std::optional<int64_t> batch_size_;
    std::optional<at::ScalarType> cache_dtype_, meta_dtype_;
    int64_t reserve_tokens_ = 0;
    int64_t fused_calls_ = 0, reference_calls_ = 0;
};

// CPU reference of the quantize kernels' math, writing (packed, scales,
// zeros) for a contiguous input whose groups are contiguous runs of G
// elements (keys transposed to [B,H,D,L]; values [B,H,L,D]).
void reference_quantize(const at::Tensor& input, at::Tensor packed, at::Tensor scales,
                        at::Tensor zeros, int64_t group_size);

}  // namespace kivi
