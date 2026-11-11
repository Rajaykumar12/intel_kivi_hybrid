// Next-token sampling (Phase 5 A.6).
//
// The order follows transformers' logits processors for generate():
//   repetition penalty -> temperature -> top-k -> top-p -> sample.
// temperature == 0 (or greedy) is argmax and ignores the rest except the
// repetition penalty, which changes the argmax too, as in HF.
// The RNG is a seeded mt19937_64, so a (seed, prompt, settings) triple
// always produces the same tokens.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include <ATen/Tensor.h>

namespace kivi::runtime {

struct SamplerConfig {
    double temperature = 0.0;        // 0: greedy
    int64_t top_k = 0;               // 0: off
    double top_p = 1.0;              // 1: off
    double repetition_penalty = 1.0; // 1: off
    uint64_t seed = 0;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig cfg);
    // `logits`: [V] (any float dtype, CPU). `history`: prompt + generated ids,
    // for the repetition penalty.
    int64_t next(const at::Tensor& logits, const std::vector<int64_t>& history);
    const SamplerConfig& config() const { return cfg_; }
    bool greedy() const { return cfg_.temperature <= 0.0; }

private:
    SamplerConfig cfg_;
    std::mt19937_64 rng_;
    std::vector<float> buf_;
    std::vector<int32_t> idx_;
};

}  // namespace kivi::runtime
