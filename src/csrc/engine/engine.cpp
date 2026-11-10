// kivi::Engine — see engine.hpp. Comments reference the Python method each
// block ports (src/kivi_sycl/cache/manager.py).
#include "engine/engine.hpp"

#include "kivi/usm.hpp"
#include "ops/kv_cache_ops.hpp"

namespace kivi {

namespace {
constexpr double kHalfMax = 65504.0;
constexpr double kHalfScaleFloor = 6.103515625e-05;   // 2^-14 (quant_tile.hpp)
}

// ---------------------------------------------------------------------------
// reference quantize (CPU) — mirrors compute_scale<M> / saturate_zero<M> /
// round_through_storage<M> in kernels/internal/quant_tile.hpp
// ---------------------------------------------------------------------------

void reference_quantize(const at::Tensor& input, at::Tensor packed, at::Tensor scales,
                        at::Tensor zeros, int64_t G) {
    at::Tensor flat = input.reshape({-1, G}).to(at::kFloat);
    at::Tensor mn = std::get<0>(flat.min(1, /*keepdim=*/true));
    at::Tensor mx = std::get<0>(flat.max(1, /*keepdim=*/true));
    at::Tensor scale, zero;
    if (scales.scalar_type() == at::kFloat) {
        scale = (mx - mn) / 3.0 + 1e-10;
        zero = mn;
    } else {
        scale = at::clamp((mx - mn) / 3.0, kHalfScaleFloor, kHalfMax).to(at::kHalf).to(at::kFloat);
        zero = at::clamp(mn, -kHalfMax, kHalfMax).to(at::kHalf).to(at::kFloat);
    }
    at::Tensor q = at::clamp(at::round((flat - zero) / scale), 0.0, 3.0).to(at::kByte);
    at::Tensor qb = q.reshape({-1, G / 4, 4});
    at::Tensor byte = at::bitwise_or(
        at::bitwise_or(qb.select(2, 0), at::bitwise_left_shift(qb.select(2, 1), 2)),
        at::bitwise_or(at::bitwise_left_shift(qb.select(2, 2), 4),
                       at::bitwise_left_shift(qb.select(2, 3), 6)));
    packed.copy_(byte.reshape(packed.sizes()));
    scales.copy_(scale.reshape(scales.sizes()));
    zeros.copy_(zero.reshape(zeros.sizes()));
}

// ---------------------------------------------------------------------------
// construction / session
// ---------------------------------------------------------------------------

Engine::Engine(EngineConfig cfg) : cfg_(std::move(cfg)), device_(cfg_.device) {
    TORCH_CHECK(cfg_.num_layers > 0 && cfg_.head_dim > 0, "Engine: num_layers and head_dim must be > 0");
    if (cfg_.group_size.size() == 1)
        cfg_.group_size.assign(cfg_.num_layers, cfg_.group_size[0]);
    TORCH_CHECK(static_cast<int64_t>(cfg_.group_size.size()) == cfg_.num_layers,
                "group_size list has ", cfg_.group_size.size(), " entries for ",
                cfg_.num_layers, " layers");
    for (int64_t g : cfg_.group_size) {
        TORCH_CHECK(cfg_.head_dim % g == 0, "head_dim (", cfg_.head_dim,
                    ") must be divisible by group_size (", g, ")");
        TORCH_CHECK(g >= 4 && g % 4 == 0, "group_size (", g, ") must be >= 4 and divisible by 4");
        TORCH_CHECK(g <= cfg_.residual_length, "group_size (", g,
                    ") must not exceed residual_length (", cfg_.residual_length, ")");
    }
    TORCH_CHECK(device_.is_cpu() || device_.is_xpu(), "Engine: device must be cpu or xpu");
    if (cfg_.metadata_dtype)
        TORCH_CHECK(*cfg_.metadata_dtype == at::kHalf || *cfg_.metadata_dtype == at::kFloat,
                    "metadata_dtype must be float16 or float32");
    layers_.resize(cfg_.num_layers);
}

Engine::~Engine() {
    // Never free tensors a submitted kernel may still touch.
    for (auto& l : layers_)
        if (l.pending) {
            if (l.pending->ev_k) l.pending->ev_k->wait();
            if (l.pending->ev_v) l.pending->ev_v->wait();
        }
}

Engine::Layer& Engine::L(int64_t i) {
    TORCH_CHECK(i >= 0 && i < cfg_.num_layers, "layer index ", i, " out of range");
    return layers_[i];
}
const Engine::Layer& Engine::L(int64_t i) const {
    TORCH_CHECK(i >= 0 && i < cfg_.num_layers, "layer index ", i, " out of range");
    return layers_[i];
}

bool Engine::zero_copy_enabled() const {
    if (cfg_.zero_copy == ZeroCopy::Auto) return device_.is_xpu();
    return cfg_.zero_copy == ZeroCopy::On;
}

// _init_session
void Engine::init_session(const at::Tensor& k, int64_t layer) {
    if (!batch_size_) {
        batch_size_ = k.size(0);
        cfg_.num_kv_heads = k.size(1);
    } else {
        TORCH_CHECK(k.size(0) == *batch_size_, "KiviCache batch size changed mid-session (",
                    *batch_size_, " -> ", k.size(0), ") at layer ", layer,
                    ". Call reset() before starting a new request/batch.");
    }
    if (!cache_dtype_) {
        // bf16 has no kernel instantiation — upcast to fp32.
        const auto dt = k.scalar_type();
        cache_dtype_ = (dt == at::kFloat || dt == at::kHalf) ? dt : at::kFloat;
        // _resolve_meta_dtype
        if (!cfg_.metadata_dtype || *cfg_.metadata_dtype == at::kHalf) {
            meta_dtype_ = at::kHalf;
        } else {
            TORCH_CHECK_VALUE(*cache_dtype_ == at::kFloat,
                              "metadata_dtype=torch.float32 needs an fp32 model, got ",
                              *cache_dtype_, ": fp32 metadata on fp16 data has no kernel.");
            meta_dtype_ = at::kFloat;
        }
    }
}

// ---------------------------------------------------------------------------
// update orders
// ---------------------------------------------------------------------------

void Engine::update(int64_t i, const at::Tensor& k_new, const at::Tensor& v_new) {
    init_session(k_new, i);
    harvest(i);
    maybe_flush(i);
    if (!cfg_.async_flush) harvest(i);
    append_residual(i, k_new.to(*cache_dtype_), v_new.to(*cache_dtype_));
}

void Engine::append(int64_t i, const at::Tensor& k_new, const at::Tensor& v_new) {
    init_session(k_new, i);
    append_residual(i, k_new.to(*cache_dtype_), v_new.to(*cache_dtype_));
    maybe_flush(i);
}

// _append_residual
void Engine::append_residual(int64_t i, const at::Tensor& k_new, const at::Tensor& v_new) {
    Layer& l = L(i);
    if (!l.res_k.defined()) {
        l.res_k = k_new;
        l.res_v = v_new;
    } else {
        l.res_k = at::cat({l.res_k, k_new}, 2);
        l.res_v = at::cat({l.res_v, v_new}, 2);
    }
}

// _maybe_flush
void Engine::maybe_flush(int64_t i) {
    Layer& l = L(i);
    if (!l.res_k.defined() || l.pending) return;
    const int64_t res_len = l.res_k.size(2);
    if (res_len >= cfg_.residual_length) {
        const int64_t G = cfg_.group_size[i];
        submit_flush(i, ((res_len - (cfg_.residual_length - G)) / G) * G);
    }
}

// _layer_reserve
int64_t Engine::layer_reserve(int64_t i) const {
    if (!reserve_tokens_) return 0;
    const int64_t G = cfg_.group_size[i];
    return std::max<int64_t>(0, (reserve_tokens_ - (cfg_.residual_length - G)) / G * G);
}

void Engine::reserve(int64_t total_tokens) {
    TORCH_CHECK(total_tokens >= 0, "reserve: total_tokens must be >= 0");
    reserve_tokens_ = total_tokens;
}

AllocFn Engine::arena_alloc() const {
    return zero_copy_enabled() ? usm_host_alloc() : cpu_alloc();
}

// _staging: contiguous USM view from a reusable per-layer buffer.
at::Tensor Engine::staging(at::Tensor& buf, at::IntArrayRef shape, at::ScalarType dt) {
    const int64_t numel = c10::multiply_integers(shape);
    if (!buf.defined() || buf.numel() < numel || buf.scalar_type() != dt)
        buf = alloc_usm_host({numel}, dt);
    return buf.narrow(0, 0, numel).view(shape);
}

// _submit_flush
void Engine::submit_flush(int64_t i, int64_t flush_len) {
    Layer& l = L(i);
    const int64_t G = cfg_.group_size[i];
    TORCH_CHECK(!l.pending, "Layer ", i, " already has an in-flight flush");
    TORCH_CHECK(flush_len >= G && flush_len % G == 0, "flush_len (", flush_len,
                ") must be a positive multiple of G (", G, ")");
    TORCH_CHECK(l.res_k.size(2) >= flush_len, "Cannot flush ", flush_len,
                " tokens: residual only has ", l.res_k.size(2), " at layer ", i);

    at::Tensor k_chunk = l.res_k.slice(2, 0, flush_len);
    at::Tensor v_chunk = l.res_v.slice(2, 0, flush_len);
    const int64_t B = k_chunk.size(0), H = k_chunk.size(1), D = k_chunk.size(3);

    // Safe to grow (reallocate) here: no kernel is writing into the arena.
    l.arena = ensure_capacity(std::move(l.arena), l.quant_tokens + flush_len, B, H, D, G,
                              *meta_dtype_, layer_reserve(i), arena_alloc());
    const int64_t t0 = l.arena->used;
    Pending p;
    p.n = flush_len;

    auto make_outputs = [&](const at::TensorOptions& o) {
        p.keys = {at::empty({B, H, D, flush_len / 4}, o.dtype(at::kByte)),
                  at::empty({B, H, D, flush_len / G}, o.dtype(*meta_dtype_)),
                  at::empty({B, H, D, flush_len / G}, o.dtype(*meta_dtype_))};
        p.values = {at::empty({B, H, flush_len, D / 4}, o.dtype(at::kByte)),
                    at::empty({B, H, flush_len, D / G}, o.dtype(*meta_dtype_)),
                    at::empty({B, H, flush_len, D / G}, o.dtype(*meta_dtype_))};
    };

    if (device_.is_cpu()) {
        // Reference path: computed now, copied into the arena at harvest —
        // same timing as the staged XPU path, so serving semantics match.
        make_outputs(at::TensorOptions().device(at::kCPU));
        reference_quantize(k_chunk.transpose(2, 3).contiguous(), p.keys[0], p.keys[1], p.keys[2], G);
        reference_quantize(v_chunk.contiguous(), p.values[0], p.values[1], p.values[2], G);
    } else if (zero_copy_enabled()) {
        at::Tensor k_in = staging(l.stage_k, {B, H, D, flush_len}, k_chunk.scalar_type());
        k_in.copy_(k_chunk.transpose(2, 3));
        at::Tensor v_in = staging(l.stage_v, {B, H, flush_len, D}, v_chunk.scalar_type());
        v_in.copy_(v_chunk);
        auto ks = l.arena->key_slices(t0, t0 + flush_len);
        auto vs = l.arena->value_slices(t0, t0 + flush_len);
        p.ev_k = kivi_quant_keys_async(k_in, ks[0], ks[1], ks[2], static_cast<int>(G));
        p.ev_v = kivi_quant_values_async(v_in, vs[0], vs[1], vs[2], static_cast<int>(D),
                                         static_cast<int>(G));
        p.direct = true;
    } else {
        p.in_k = k_chunk.transpose(2, 3).contiguous().to(device_);
        p.in_v = v_chunk.contiguous().to(device_);
        make_outputs(at::TensorOptions().device(device_));
        p.ev_k = kivi_quant_keys_async(p.in_k, p.keys[0], p.keys[1], p.keys[2], static_cast<int>(G));
        p.ev_v = kivi_quant_values_async(p.in_v, p.values[0], p.values[1], p.values[2],
                                         static_cast<int>(D), static_cast<int>(G));
    }
    l.pending = std::move(p);
}

// _harvest_layer
void Engine::harvest(int64_t i) {
    Layer& l = L(i);
    if (!l.pending) return;
    Pending& p = *l.pending;
    if (p.ev_k) p.ev_k->wait();
    if (p.ev_v) p.ev_v->wait();
    const int64_t n = p.n, t0 = l.arena->used;
    if (!p.direct) {
        auto ks = l.arena->key_slices(t0, t0 + n);
        auto vs = l.arena->value_slices(t0, t0 + n);
        for (int j = 0; j < 3; ++j) {
            ks[j].copy_(p.keys[j]);
            vs[j].copy_(p.values[j]);
        }
    }
    l.arena->used = t0 + n;
    // Trim the flushed tokens from the residual only now — until this point
    // they were still served in full precision.
    l.res_k = l.res_k.slice(2, n).contiguous();
    l.res_v = l.res_v.slice(2, n).contiguous();
    l.quant_tokens += n;
    l.flushes += 1;
    l.pending.reset();
}

void Engine::harvest_all() {
    for (int64_t i = 0; i < cfg_.num_layers; ++i) harvest(i);
}

void Engine::reset() {
    for (auto& l : layers_) {
        if (l.pending) {
            if (l.pending->ev_k) l.pending->ev_k->wait();
            if (l.pending->ev_v) l.pending->ev_v->wait();
        }
        l = Layer{};
    }
    batch_size_.reset();
    cache_dtype_.reset();
    meta_dtype_.reset();
    reserve_tokens_ = 0;
}

// ---------------------------------------------------------------------------
// introspection / stats
// ---------------------------------------------------------------------------

std::optional<std::pair<at::Tensor, at::Tensor>> Engine::residual(int64_t i) const {
    const Layer& l = L(i);
    if (!l.res_k.defined()) return std::nullopt;
    return std::make_pair(l.res_k, l.res_v);
}

std::optional<std::array<at::Tensor, 6>> Engine::history(int64_t i) const {
    const Layer& l = L(i);
    if (!l.arena || l.arena->used == 0) return std::nullopt;
    auto k = l.arena->used_keys();
    auto v = l.arena->used_values();
    return std::array<at::Tensor, 6>{k[0], k[1], k[2], v[0], v[1], v[2]};
}

int64_t Engine::pending_tokens(int64_t i) const {
    const Layer& l = L(i);
    return l.pending ? l.pending->n : 0;
}

int64_t Engine::arena_capacity(int64_t i) const {
    const Layer& l = L(i);
    return l.arena ? l.arena->cap : 0;
}

int64_t Engine::seq_length(int64_t i) const {
    const Layer& l = L(i);
    return l.quant_tokens + (l.res_k.defined() ? l.res_k.size(2) : 0);
}

int64_t Engine::total_flushes() const {
    int64_t n = 0;
    for (const auto& l : layers_) n += l.flushes;
    return n;
}

// get_stats (serving_copy_bytes is a Python-side serving mode and is added there)
EngineStats Engine::stats(int64_t i) const {
    const Layer& l = L(i);
    const int64_t B = batch_size_.value_or(1);
    const int64_t H = cfg_.num_kv_heads ? cfg_.num_kv_heads : 1;
    const int64_t res_tokens = l.res_k.defined() ? l.res_k.size(2) : 0;
    const int64_t total = l.quant_tokens + res_tokens;
    auto nbytes = [](const at::Tensor& t) {
        return t.defined() ? t.numel() * static_cast<int64_t>(t.element_size()) : 0;
    };
    EngineStats s{};
    s.group_size = cfg_.group_size[i];
    s.total_tokens = total;
    s.quantized_tokens = l.quant_tokens;
    s.residual_tokens = res_tokens;
    s.pending_tokens = pending_tokens(i);
    s.packed_bytes = l.arena ? l.arena->allocated_bytes() : 0;
    s.packed_used_bytes = l.arena ? l.arena->used_bytes() : 0;
    s.packed_capacity_tokens = l.arena ? l.arena->cap : 0;
    s.residual_bytes = nbytes(l.res_k) + nbytes(l.res_v);
    s.kivi_bytes = s.packed_bytes + s.residual_bytes;
    const int64_t elt = static_cast<int64_t>(c10::elementSize(cache_dtype_.value_or(at::kFloat)));
    s.fp_baseline_bytes = 2 * B * H * total * cfg_.head_dim * elt;
    s.compression = s.kivi_bytes > 0 ? static_cast<double>(s.fp_baseline_bytes) / s.kivi_bytes : 0.0;
    return s;
}

}  // namespace kivi
