// CPU dequantize-on-read — see dequant_cpu.hpp.
#include "engine/dequant_cpu.hpp"

namespace kivi {

namespace {

const at::Tensor& unpack_shifts() {
    static const at::Tensor s = at::tensor({0, 2, 4, 6}, at::TensorOptions().dtype(at::kByte));
    return s;
}

// (packed.unsqueeze(-1) >> shifts) & 3, as in the Python version.
at::Tensor codes(const at::Tensor& packed) {
    return at::bitwise_and(at::bitwise_right_shift(packed.unsqueeze(-1), unpack_shifts()), 3);
}

}  // namespace

void dequant_keys_into(at::Tensor out, const at::Tensor& packed,
                       const at::Tensor& scales, const at::Tensor& zeros) {
    const int64_t B = packed.size(0), H = packed.size(1), D = packed.size(2);
    at::Tensor q = codes(packed);                                  // [B,H,D,T/4,4]
    const int64_t T = q.numel() / (B * H * D);
    const int64_t G = T / scales.size(-1);                         // per-layer G
    q = q.reshape({B, H, D, T / G, G}).to(out.scalar_type());
    q.mul_(scales.unsqueeze(-1)).add_(zeros.unsqueeze(-1));
    out.copy_(q.reshape({B, H, D, T}).transpose(2, 3));
}

void dequant_values_into(at::Tensor out, const at::Tensor& packed,
                         const at::Tensor& scales, const at::Tensor& zeros) {
    const int64_t B = packed.size(0), H = packed.size(1), T = packed.size(2);
    at::Tensor q = codes(packed);                                  // [B,H,T,D/4,4]
    const int64_t D = q.numel() / (B * H * T);
    const int64_t G = D / scales.size(-1);
    q = q.reshape({B, H, T, D / G, G}).to(out.scalar_type());
    q.mul_(scales.unsqueeze(-1)).add_(zeros.unsqueeze(-1));
    out.copy_(q.reshape({B, H, T, D}));
}

at::Tensor dequant_keys(const at::Tensor& packed, const at::Tensor& scales,
                        const at::Tensor& zeros, at::ScalarType dtype) {
    const int64_t B = packed.size(0), H = packed.size(1), D = packed.size(2);
    const int64_t T = packed.size(3) * 4;
    at::Tensor out = at::empty({B, H, T, D}, at::TensorOptions().dtype(dtype));
    dequant_keys_into(out, packed, scales, zeros);
    return out;
}

at::Tensor dequant_values(const at::Tensor& packed, const at::Tensor& scales,
                          const at::Tensor& zeros, at::ScalarType dtype) {
    const int64_t B = packed.size(0), H = packed.size(1), T = packed.size(2);
    const int64_t D = packed.size(3) * 4;
    at::Tensor out = at::empty({B, H, T, D}, at::TensorOptions().dtype(dtype));
    dequant_values_into(out, packed, scales, zeros);
    return out;
}

}  // namespace kivi
