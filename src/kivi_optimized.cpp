#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include <iostream>

// Helper to get the correct device (Intel GPU) with error handling
sycl::queue& get_queue() {
    static sycl::queue* q_ptr = nullptr;
    
    if (!q_ptr) {
        try {
            q_ptr = new sycl::queue(sycl::gpu_selector_v, 
                [](sycl::exception_list exceptions) {
                    for (std::exception_ptr const& e : exceptions) {
                        try {
                            std::rethrow_exception(e);
                        } catch (sycl::exception const& ex) {
                            std::cerr << "SYCL exception: " << ex.what() << std::endl;
                        }
                    }
                });
        } catch (sycl::exception const& e) {
            std::cerr << "Failed to create SYCL queue: " << e.what() << std::endl;
            throw;
        }
    }
    return *q_ptr;
}

// ------------------------------------------------------------------
// KERNEL 1: Optimized Quantize (FP16 -> Packed INT2)
// ------------------------------------------------------------------
void quantize_kivi_kernel(const float* __restrict__ input, uint8_t* __restrict__ output, float* __restrict__ scales, int num_groups, int total_elements) {
    sycl::queue& q = get_queue();
    const int local_size = 256;
    const int num_work_groups = (num_groups + local_size - 1) / local_size;
    
    auto event = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_work_groups * local_size), sycl::range<1>(local_size)), [=](sycl::nd_item<1> item) {
            int g = item.get_global_id(0);
            if (g >= num_groups) return;
            
            int base_idx = g * 4;
            float v0 = input[base_idx + 0];
            float v1 = input[base_idx + 1];
            float v2 = input[base_idx + 2];
            float v3 = input[base_idx + 3];

            float max_val = sycl::fmax(sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::fmax(sycl::fabs(v2), sycl::fabs(v3)));
            float scale = sycl::fmax(max_val / 3.0f, 1e-8f);

            auto quantize_value = [scale](float val) -> uint8_t {
                return static_cast<uint8_t>(sycl::clamp(static_cast<int>(val / scale + 1.5f), 0, 3));
            };
            
            output[g] = (quantize_value(v3) << 6) | (quantize_value(v2) << 4) | (quantize_value(v1) << 2) | quantize_value(v0);
            scales[g] = scale;
        });
    });
    event.wait_and_throw();
}

// ------------------------------------------------------------------
// KERNEL 2: Batched Attention (Optimized Multi-Head)
// ------------------------------------------------------------------
void batched_dequant_attention_kernel(const uint8_t* __restrict__ packed_cache, const float* __restrict__ scales, const float* __restrict__ query, float* __restrict__ output_scores, int batch_size, int num_heads, int seq_len, int cache_len, int head_dim) {
    sycl::queue& q = get_queue();
    
    // Input validation
    if (head_dim % 4 != 0) {
        throw std::runtime_error("head_dim must be divisible by 4 for group quantization");
    }
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || cache_len <= 0) {
        throw std::runtime_error("Invalid dimensions: all must be positive");
    }
    
    // We need to compute attention for: batch × num_heads × seq_len × cache_len
    const int total_computations = batch_size * num_heads * seq_len * cache_len;
    const int local_size = 256;
    const int num_work_groups = (total_computations + local_size - 1) / local_size;
    const int groups_per_token = head_dim / 4;
    
    auto event = q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(num_work_groups * local_size), sycl::range<1>(local_size)), [=](sycl::nd_item<1> item) {
            int global_idx = item.get_global_id(0);
            if (global_idx >= total_computations) return;
            
            // Decode indices: we're computing attention between query token q_t and cache token c_t
            int b = global_idx / (num_heads * seq_len * cache_len);
            int remainder = global_idx % (num_heads * seq_len * cache_len);
            int h = remainder / (seq_len * cache_len);
            remainder = remainder % (seq_len * cache_len);
            int q_t = remainder / cache_len;  // Which query token (0 to seq_len-1)
            int c_t = remainder % cache_len;  // Which cache token (0 to cache_len-1)
            
            // Query offset: [batch][head][query_token][head_dim]
            int query_offset = ((b * num_heads + h) * seq_len + q_t) * head_dim;
            
            // Cache offset: [batch][head][cache_token][groups]
            int cache_offset = ((b * num_heads + h) * cache_len + c_t) * groups_per_token;

            // Compute dot product between query and dequantized cache
            float dot_product = 0.0f;
            for (int i = 0; i < groups_per_token; ++i) {
                uint8_t packed = packed_cache[cache_offset + i];
                float scale = scales[cache_offset + i];
                int q_idx = query_offset + i * 4;
                
                dot_product = sycl::fma(static_cast<float>((packed >> 0) & 0x3) * scale, query[q_idx + 0], dot_product);
                dot_product = sycl::fma(static_cast<float>((packed >> 2) & 0x3) * scale, query[q_idx + 1], dot_product);
                dot_product = sycl::fma(static_cast<float>((packed >> 4) & 0x3) * scale, query[q_idx + 2], dot_product);
                dot_product = sycl::fma(static_cast<float>((packed >> 6) & 0x3) * scale, query[q_idx + 3], dot_product);
            }
            
            output_scores[global_idx] = dot_product;
        });
    });
    event.wait_and_throw();
}

// ------------------------------------------------------------------
// KERNEL 3: Single Head Attention (Legacy Stub)
// ------------------------------------------------------------------
void dequant_attention_kernel(const uint8_t* packed_cache, const float* scales, const float* query, float* output_scores, int seq_len, int cache_len, int head_dim) {
    // Re-use the batched kernel logic for simplicity or keep basic stub
    batched_dequant_attention_kernel(packed_cache, scales, query, output_scores, 1, 1, seq_len, cache_len, head_dim);
}

// ------------------------------------------------------------------
// Python Bindings
// ------------------------------------------------------------------
void kivi_quantize_cuda(torch::Tensor input, torch::Tensor output, torch::Tensor scales) {
    quantize_kivi_kernel(input.data_ptr<float>(), output.data_ptr<uint8_t>(), scales.data_ptr<float>(), output.numel(), input.numel());
}
void kivi_attention_cuda(torch::Tensor cache, torch::Tensor scales, torch::Tensor query, torch::Tensor out, int cache_len) {
    int seq_len = out.size(0);
    int head_dim = query.size(0);
    dequant_attention_kernel(cache.data_ptr<uint8_t>(), scales.data_ptr<float>(), query.data_ptr<float>(), out.data_ptr<float>(), seq_len, cache_len, head_dim);
}
void kivi_batched_attention_cuda(torch::Tensor cache, torch::Tensor scales, torch::Tensor query, torch::Tensor out, int batch_size, int num_heads, int seq_len, int cache_len) {
    // query shape: [batch * num_heads * seq_len * head_dim] (flattened)
    // out shape: [batch * num_heads * seq_len * cache_len] (flattened)
    int total_query_elements = query.numel();
    int head_dim = total_query_elements / (batch_size * num_heads * seq_len);
    
    batched_dequant_attention_kernel(cache.data_ptr<uint8_t>(), scales.data_ptr<float>(), query.data_ptr<float>(), out.data_ptr<float>(), batch_size, num_heads, seq_len, cache_len, head_dim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize", &kivi_quantize_cuda, "KIVI 2-bit Quantize");
    m.def("attention", &kivi_attention_cuda, "KIVI 2-bit Single Attention");
    m.def("batched_attention", &kivi_batched_attention_cuda, "KIVI 2-bit Batched Attention");
}