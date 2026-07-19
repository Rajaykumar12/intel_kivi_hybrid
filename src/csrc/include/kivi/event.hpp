#pragma once
// ==========================================================================
// KIVI: pybind11-exposed handle wrapping a sycl::event.
//
// Returned by the *_async ops so Python can submit a kernel without
// blocking, do other CPU-side work, and synchronize later by calling
// `.wait()`. See docs/sycl_kernel_interface.md — §3 (Synchronization
// Behavior) for why this exists: the original design's `.wait_and_throw()`
// on every native call made every kivi_native.* call a blocking round-trip,
// which is what the CPU-residual flush path needs relief from for the
// async-flush integration (KiviCache).
// ==========================================================================

#include <sycl/sycl.hpp>

namespace kivi {

class KiviEvent {
public:
    explicit KiviEvent(sycl::event ev) : event_(std::move(ev)) {}

    // Blocks until the underlying kernel completes; rethrows any async
    // SYCL exception raised during its execution (mirrors what the
    // blocking kivi_native.* calls did implicitly via wait_and_throw()).
    void wait() { event_.wait_and_throw(); }

private:
    sycl::event event_;
};

}  // namespace kivi
