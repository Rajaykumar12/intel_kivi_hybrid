#pragma once
// ==========================================================================
// CPU dequantize-on-read — C++ port of KiviCache._dequant_{keys,values}_into
// (src/kivi_sycl/cache/manager.py), Phase 5 S1.3. Same ATen op sequence, so
// the output is bit-identical to the Python version:
//   q = (packed.unsqueeze(-1) >> [0,2,4,6]) & 3      (LSB-first 2-bit codes)
//   q = q.reshape(..., groups, G).to(out.dtype)
//   q.mul_(scales.unsqueeze(-1)).add_(zeros.unsqueeze(-1))
// The two-step mul_ / add_ rounds exactly like `q * s + z` in torch — not
// like the SYCL kernel's contracted FMA (tests compare with tolerance there).
// G is derived from the shapes, so per-layer group sizes need no argument.
// ==========================================================================

#include <torch/torch.h>

namespace kivi {

// packed [B,H,D,T/4] u8, scales/zeros [B,H,D,T/G] -> out [B,H,T,D] (any strides).
void dequant_keys_into(at::Tensor out, const at::Tensor& packed,
                       const at::Tensor& scales, const at::Tensor& zeros);

// packed [B,H,T,D/4] u8, scales/zeros [B,H,T,D/G] -> out [B,H,T,D].
void dequant_values_into(at::Tensor out, const at::Tensor& packed,
                         const at::Tensor& scales, const at::Tensor& zeros);

// Allocating forms, [B,H,T,D] in `dtype`.
at::Tensor dequant_keys(const at::Tensor& packed, const at::Tensor& scales,
                        const at::Tensor& zeros, at::ScalarType dtype);
at::Tensor dequant_values(const at::Tensor& packed, const at::Tensor& scales,
                          const at::Tensor& zeros, at::ScalarType dtype);

}  // namespace kivi
