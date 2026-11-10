#pragma once
// ==========================================================================
// Process-wide engine registry: int64 id -> kivi::Engine (Phase 5 S1.7).
//
// The stateful op kivi::layer_attention takes an engine *id* — a plain int a
// traced/exported graph can carry as a constant — and resolves it here. The
// Python front end registers each native engine it creates; the C++ runtime
// registers its single engine as id 0 (the id baked into exported graphs).
// ==========================================================================

#include <memory>

#include "engine/engine.hpp"

namespace kivi {

// Register `engine`; returns its id. `id` >= 0 requests a specific id (e.g. 0
// for the runtime) and fails if taken; -1 picks the next free one (>= 1).
int64_t register_engine(std::shared_ptr<Engine> engine, int64_t id = -1);
void unregister_engine(int64_t id);
std::shared_ptr<Engine> get_engine(int64_t id);   // throws if unknown

}  // namespace kivi
