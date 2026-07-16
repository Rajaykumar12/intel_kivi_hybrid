// ==========================================================================
// KIVI: Asymmetric 2-bit quantization SYCL kernels
// ==========================================================================
// Quantization: Asymmetric 2-bit
//   q = clamp(round((x - min) / scale), 0, 3)
//   scale = (max - min) / 3
//
// Packing: 4 x 2-bit values per uint8
//   byte = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
// ==========================================================================

#include "quantize_kernels.hpp"
#include "kivi/common.hpp"

void quantize_keys_per_channel(
    const float* __restrict__ input,
    uint8_t*     __restrict__ output,
    float*       __restrict__ scales,
    float*       __restrict__ zeros,
    int num_channels,
    int group_size)
{
    sycl::queue& q = kivi::get_queue();
    const int local_size  = 256;
    const int global_size = ((num_channels + local_size - 1) / local_size) * local_size;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int ch = item.get_global_id(0);
                if (ch >= num_channels) return;

                // --- Find min/max across the group ---
                float max_val = -1e30f;
                float min_val =  1e30f;
                int base = ch * group_size;

                for (int i = 0; i < group_size; ++i) {
                    float val = input[base + i];
                    max_val = sycl::fmax(max_val, val);
                    min_val = sycl::fmin(min_val, val);
                }

                // --- Asymmetric scale & zero ---
                float scale = (max_val - min_val) / 3.0f + 1e-10f;
                scales[ch] = scale;
                zeros[ch]  = min_val;

                // --- Pack 4 x 2-bit values per byte ---
                int out_base = ch * (group_size / 4);
                for (int i = 0; i < group_size / 4; ++i) {
                    uint8_t packed = 0;
                    for (int j = 0; j < 4; ++j) {
                        float val = input[base + i * 4 + j];
                        float qf  = (val - min_val) / scale;
                        int   qi  = static_cast<int>(sycl::round(qf));
                        qi = sycl::clamp(qi, 0, 3);
                        packed |= (static_cast<uint8_t>(qi) << (j * 2));
                    }
                    output[out_base + i] = packed;
                }
            });
    }).wait_and_throw();
}

void quantize_values_per_token(
    const float* __restrict__ input,
    uint8_t*     __restrict__ output,
    float*       __restrict__ scales,
    float*       __restrict__ zeros,
    int num_tokens,
    int head_dim,
    int group_size)
{
    sycl::queue& q = kivi::get_queue();
    int groups_per_token = head_dim / group_size;
    int total_groups     = num_tokens * groups_per_token;
    const int local_size  = 256;
    const int global_size = ((total_groups + local_size - 1) / local_size) * local_size;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int gid = item.get_global_id(0);
                if (gid >= total_groups) return;

                int token_idx    = gid / groups_per_token;
                int group_in_tok = gid % groups_per_token;
                int start_feat   = group_in_tok * group_size;
                int base_input   = token_idx * head_dim + start_feat;

                // --- Find min/max within the group ---
                float max_val = -1e30f;
                float min_val =  1e30f;
                for (int i = 0; i < group_size; ++i) {
                    float val = input[base_input + i];
                    max_val = sycl::fmax(max_val, val);
                    min_val = sycl::fmin(min_val, val);
                }

                float scale = (max_val - min_val) / 3.0f + 1e-10f;
                scales[gid] = scale;
                zeros[gid]  = min_val;

                // --- Pack ---
                // Output packed index: each token has head_dim/4 packed bytes
                int packed_per_token = head_dim / 4;
                int out_base = token_idx * packed_per_token + group_in_tok * (group_size / 4);

                for (int i = 0; i < group_size / 4; ++i) {
                    uint8_t packed = 0;
                    for (int j = 0; j < 4; ++j) {
                        float val = input[base_input + i * 4 + j];
                        float qf  = (val - min_val) / scale;
                        int   qi  = static_cast<int>(sycl::round(qf));
                        qi = sycl::clamp(qi, 0, 3);
                        packed |= (static_cast<uint8_t>(qi) << (j * 2));
                    }
                    output[out_base + i] = packed;
                }
            });
    }).wait_and_throw();
}
