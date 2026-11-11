// Engine registry — see registry.hpp.
#include "engine/registry.hpp"

#include <mutex>
#include <unordered_map>

namespace kivi {

namespace {
std::mutex& mu() { static std::mutex m; return m; }
std::unordered_map<int64_t, std::shared_ptr<Engine>>& table() {
    static std::unordered_map<int64_t, std::shared_ptr<Engine>> t;
    return t;
}
int64_t& next_id() { static int64_t n = 1; return n; }
}  // namespace

int64_t register_engine(std::shared_ptr<Engine> engine, int64_t id) {
    TORCH_CHECK(engine, "register_engine: null engine");
    std::lock_guard<std::mutex> lk(mu());
    if (id < 0) {
        while (table().count(next_id())) ++next_id();
        id = next_id()++;
    } else {
        TORCH_CHECK(!table().count(id), "register_engine: id ", id, " already in use");
    }
    table()[id] = std::move(engine);
    return id;
}

void unregister_engine(int64_t id) {
    std::lock_guard<std::mutex> lk(mu());
    table().erase(id);
}

std::shared_ptr<Engine> get_engine(int64_t id) {
    std::lock_guard<std::mutex> lk(mu());
    auto it = table().find(id);
    TORCH_CHECK(it != table().end(), "kivi: no engine registered with id ", id);
    return it->second;
}

}  // namespace kivi
