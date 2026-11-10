#pragma once
// ==========================================================================
// KIVI: USM host allocations exposed as CPU torch tensors.
//
// On an integrated GPU (Iris Xe) "device" memory and host memory are the
// same LPDDR, so every .to("xpu") / .to("cpu") in the flush path is a
// DRAM-to-DRAM copy through the driver. A USM *host* allocation is ordinary
// host memory the CPU reads and writes at full speed (torch sees a plain
// CPU tensor) that SYCL kernels can also dereference directly — so the
// packed KV store can live in one place, written by the quantize kernels and
// read by the CPU dequant path, with no copies in between.
//
// malloc_host rather than malloc_shared: shared allocations exist to migrate
// pages between host and a device's own memory, and an iGPU has no own
// memory to migrate to. The CPU also reads this store on every decode step.
//
// Allocations are made in the context of kivi::get_queue(), the queue every
// kernel is submitted on, so kernels can always dereference them. That holds
// on both queue paths (torch's stream queue and the private fallback).
// ==========================================================================

#include "kivi/common.hpp"

namespace kivi {

inline torch::Tensor alloc_usm_host(at::IntArrayRef sizes, at::ScalarType dtype,
                                    int64_t align = 64)
{
    TORCH_CHECK(align > 0 && (align & (align - 1)) == 0,
                "[KIVI] alloc_usm_host: align must be a power of two, got ", align);
    for (int64_t s : sizes) {
        TORCH_CHECK(s >= 0, "[KIVI] alloc_usm_host: negative size in ", sizes);
    }
    sycl::queue& q = get_queue();
    const size_t bytes = static_cast<size_t>(c10::multiply_integers(sizes)) *
                         c10::elementSize(dtype);
    // A zero-byte request still gets a real (minimal) allocation so the
    // tensor has a valid, freeable data pointer.
    void* p = sycl::aligned_alloc_host(static_cast<size_t>(align),
                                       bytes ? bytes : static_cast<size_t>(align),
                                       q.get_context());
    TORCH_CHECK(p != nullptr, "[KIVI] USM host allocation of ", bytes, " bytes failed");
    sycl::context ctx = q.get_context();
    return torch::from_blob(
        p, sizes, [ctx](void* x) { sycl::free(x, ctx); },
        torch::TensorOptions().dtype(dtype).device(torch::kCPU));
}

// is_kernel_accessible / check_kernel_accessible live in kivi/common.hpp,
// where check_tensor uses them.

}  // namespace kivi
