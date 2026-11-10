// ==========================================================================
// KIVI: shared SYCL queue management and tensor-check helpers
// ==========================================================================
#pragma once

#include <sycl/sycl.hpp>
// LibTorch C++ API only — no pybind/Python, so this also builds into the
// Python-free libkivi_core.so (CMakeLists.txt). The pybind module adds
// <torch/extension.h> itself.
#include <torch/torch.h>

#ifdef KIVI_USE_TORCH_XPU_QUEUE
#include <c10/xpu/XPUStream.h>
#endif

namespace kivi {

#ifdef KIVI_USE_TORCH_XPU_QUEUE
// Use the SYCL queue backing PyTorch's current XPU stream instead of a
// private queue. A private queue has no ordering relationship with the
// queue torch/IPEX uses for `.to("xpu")` copies, and no guarantee of
// sharing their SYCL context — on stacks where the default context is not
// shared, dereferencing torch-allocated USM pointers from a foreign-context
// queue is undefined behavior (silent data corruption, not an error).
// Submitting on torch's own in-order stream queue removes both hazards.
// KIVI_USE_TORCH_XPU_QUEUE is defined by setup.py only when the installed
// torch ships libc10_xpu (torch >= 2.4 XPU builds); older stacks keep the
// private-queue fallback below.
inline sycl::queue& get_queue() {
    return c10::xpu::getCurrentXPUStream().queue();
}
#else
// Fallback: private process-wide queue. Function-local `static`
// initialization is thread-safe ("magic statics" since C++11) — no manual
// double-checked locking needed, unlike the raw `static sycl::queue*`
// pattern this replaces. Relies on torch/IPEX allocating USM in the DPC++
// default context (true on current Linux driver stacks) and on `.to()`
// copies being host-blocking (non_blocking=False, the only mode used here).
inline sycl::queue& get_queue() {
    static sycl::queue q = [] {
        auto async_handler = [](sycl::exception_list exceptions) {
            for (const std::exception_ptr& e : exceptions) {
                try {
                    std::rethrow_exception(e);
                } catch (const sycl::exception& ex) {
                    throw std::runtime_error(
                        std::string("[KIVI] Async SYCL error: ") + ex.what());
                }
            }
        };
        try {
            sycl::queue queue(sycl::gpu_selector_v, async_handler,
                               sycl::property::queue::in_order{});
            std::cout << "[KIVI] SYCL device: "
                      << queue.get_device().get_info<sycl::info::device::name>()
                      << std::endl;
            return queue;
        } catch (const sycl::exception& e) {
            throw std::runtime_error(
                std::string("[KIVI] Failed to create SYCL queue: ") + e.what());
        }
    }();
    return q;
}
#endif  // KIVI_USE_TORCH_XPU_QUEUE

// True if a kernel submitted on get_queue() may dereference `t`'s memory:
// an XPU tensor, or a CPU tensor whose storage is a USM allocation in the
// queue's context (see kivi/usm.hpp). An ordinary malloc-backed CPU tensor
// is not — passing one to a kernel is undefined behaviour (a page fault or
// silent garbage), which is why this is checked rather than assumed.
inline bool is_kernel_accessible(const torch::Tensor& t) {
    if (t.device().is_xpu()) return true;
    if (!t.device().is_cpu()) return false;
    // Empty tensors have a null data pointer and are never dereferenced.
    if (t.numel() == 0) return true;
    auto kind = sycl::get_pointer_type(t.data_ptr(), get_queue().get_context());
    return kind != sycl::usm::alloc::unknown;
}

inline void check_kernel_accessible(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(is_kernel_accessible(t),
                name, " is a ", torch::DeviceTypeName(t.device().type(), true),
                " tensor not backed by USM, so SYCL kernels cannot read or write "
                "it. Use an XPU tensor or allocate it with "
                "kivi_sycl._C.alloc_usm_host().");
}

// Contiguity, dtype, and kernel accessibility (XPU, or USM-backed CPU).
inline void check_tensor(const torch::Tensor& t, const char* name,
                          torch::ScalarType expected_dtype) {
    TORCH_CHECK(t.is_contiguous(),
                name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == expected_dtype,
                name, " must be ", torch::toString(expected_dtype),
                " but got ", torch::toString(t.scalar_type()));
    check_kernel_accessible(t, name);
}

}  // namespace kivi
