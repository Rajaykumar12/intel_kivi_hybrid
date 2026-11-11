#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include <ATen/ATen.h>

namespace kivi::runtime {

Sampler::Sampler(SamplerConfig cfg) : cfg_(cfg), rng_(cfg.seed) {
    if (cfg_.temperature < 0) throw std::invalid_argument("temperature must be >= 0");
    if (cfg_.top_k < 0) throw std::invalid_argument("top_k must be >= 0");
    if (!(cfg_.top_p > 0 && cfg_.top_p <= 1)) throw std::invalid_argument("top_p must be in (0, 1]");
    if (!(cfg_.repetition_penalty > 0)) throw std::invalid_argument("repetition_penalty must be > 0");
}

int64_t Sampler::next(const at::Tensor& logits, const std::vector<int64_t>& history) {
    at::Tensor l = logits.to(at::kFloat).contiguous();
    const int64_t V = l.numel();
    buf_.assign(l.data_ptr<float>(), l.data_ptr<float>() + V);
    float* x = buf_.data();

    // HF RepetitionPenaltyLogitsProcessor: each distinct seen token once.
    if (cfg_.repetition_penalty != 1.0) {
        const auto p = static_cast<float>(cfg_.repetition_penalty);
        std::unordered_set<int64_t> seen(history.begin(), history.end());
        for (int64_t t : seen)
            if (t >= 0 && t < V) x[t] = x[t] < 0 ? x[t] * p : x[t] / p;
    }
    if (greedy()) return std::max_element(x, x + V) - x;   // first max, like argmax

    const auto inv_t = static_cast<float>(1.0 / cfg_.temperature);
    for (int64_t i = 0; i < V; ++i) x[i] *= inv_t;

    // Candidates sorted by logit (descending); ties keep the lower id first.
    idx_.resize(V);
    std::iota(idx_.begin(), idx_.end(), 0);
    int64_t n = V;
    auto by_logit = [x](int32_t a, int32_t b) { return x[a] != x[b] ? x[a] > x[b] : a < b; };
    if (cfg_.top_k > 0 && cfg_.top_k < V) {
        n = cfg_.top_k;
        std::partial_sort(idx_.begin(), idx_.begin() + n, idx_.end(), by_logit);
    } else {
        std::sort(idx_.begin(), idx_.end(), by_logit);
    }

    // Softmax over the candidates.
    const float mx = x[idx_[0]];
    std::vector<double> prob(n);
    double sum = 0;
    for (int64_t i = 0; i < n; ++i) sum += prob[i] = std::exp(static_cast<double>(x[idx_[i]] - mx));
    for (auto& v : prob) v /= sum;

    // Top-p (nucleus): the smallest prefix whose mass reaches top_p, always >= 1 token.
    if (cfg_.top_p < 1.0) {
        double cum = 0;
        int64_t keep = 0;
        while (keep < n) {
            cum += prob[keep++];
            if (cum >= cfg_.top_p) break;
        }
        n = keep;
        double s = 0;
        for (int64_t i = 0; i < n; ++i) s += prob[i];
        for (int64_t i = 0; i < n; ++i) prob[i] /= s;
    }

    // Inverse-CDF draw: a fixed number of RNG calls per step keeps seeded
    // runs reproducible regardless of distribution shape.
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
    double cum = 0;
    for (int64_t i = 0; i < n; ++i) {
        cum += prob[i];
        if (u < cum) return idx_[i];
    }
    return idx_[n - 1];
}

}  // namespace kivi::runtime
