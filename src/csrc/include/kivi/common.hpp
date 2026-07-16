// ==========================================================================
// KIVI: shared SYCL queue management and tensor-check helpers
// ==========================================================================
#pragma once

#include <sycl/sycl.hpp>
#include <torch/extension.h>

namespace kivi {

// Function-local `static` initialization is thread-safe ("magic statics"
// since C++11) — no manual double-checked locking needed, unlike the raw
// `static sycl::queue*` pattern this replaces.
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

inline void check_tensor(const torch::Tensor& t, const char* name,
                          torch::ScalarType expected_dtype,
                          torch::DeviceType expected_device = torch::kXPU) {
    TORCH_CHECK(t.is_contiguous(),
                name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == expected_dtype,
                name, " must be ", torch::toString(expected_dtype),
                " but got ", torch::toString(t.scalar_type()));
    TORCH_CHECK(t.device().type() == expected_device,
                name, " must be on ", torch::DeviceTypeName(expected_device, true),
                " but got ", torch::DeviceTypeName(t.device().type(), true));
}

}  // namespace kivi
