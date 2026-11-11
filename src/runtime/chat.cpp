#include "chat.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace kivi::runtime {

namespace {

std::optional<nlohmann::json> read(const std::string& path, std::string* error) {
    std::ifstream in(path);
    if (!in) {
        if (error) *error = "no " + path + " (re-export the artifacts)";
        return std::nullopt;
    }
    return nlohmann::json::parse(in);
}

std::pair<std::string, std::string> pair_of(const nlohmann::json& j) {
    return {j.at(0).get<std::string>(), j.at(1).get<std::string>()};
}

}  // namespace

std::optional<ChatTemplate> ChatTemplate::load(const std::string& path, std::string* error) {
    auto j = read(path, error);
    if (!j) return std::nullopt;
    const auto& c = j->value("chat", nlohmann::json());
    if (c.is_null()) {
        if (error) {
            const auto& why = j->value("chat_error", nlohmann::json());
            *error = why.is_string() ? why.get<std::string>() : "the model has no chat template";
        }
        return std::nullopt;
    }
    ChatTemplate t;
    t.no_system_prefix_ = c.at("no_system_prefix").get<std::string>();
    t.generation_prompt_ = c.at("generation_prompt").get<std::string>();
    t.eos_token_ = c.value("eos_token", std::string());
    t.system_ = pair_of(c.at("system"));
    t.user_ = pair_of(c.at("user"));
    t.assistant_ = pair_of(c.at("assistant"));
    return t;
}

std::string ChatTemplate::render(const std::vector<ChatMessage>& messages,
                                 bool add_generation_prompt) const {
    std::string out;
    size_t i = 0;
    if (!messages.empty() && messages[0].role == "system") {
        out += system_.first + messages[0].content + system_.second;
        i = 1;
    } else {
        out += no_system_prefix_;
    }
    for (; i < messages.size(); ++i) {
        const auto& m = messages[i];
        const auto* p = m.role == "user" ? &user_ : m.role == "assistant" ? &assistant_ : nullptr;
        if (!p) throw std::runtime_error("chat: unsupported role '" + m.role + "' at message " +
                                         std::to_string(i));
        out += p->first + m.content + p->second;
    }
    if (add_generation_prompt) out += generation_prompt_;
    return out;
}

std::string ChatTemplate::next_turn(const std::string& user, bool ended_with_eos) const {
    std::string rest = assistant_.second;
    if (ended_with_eos && !eos_token_.empty() && rest.compare(0, eos_token_.size(), eos_token_) == 0)
        rest = rest.substr(eos_token_.size());
    // The reply was rendered after generation_prompt; the template's own
    // assistant prefix must be that same text for the delta to line up.
    if (generation_prompt_ != assistant_.first)
        throw std::runtime_error("chat: generation prompt differs from the assistant prefix; "
                                 "multi-turn needs them equal");
    return rest + user_.first + user + user_.second + generation_prompt_;
}

std::optional<SamplingDefaults> load_sampling_defaults(const std::string& path) {
    auto j = read(path, nullptr);
    if (!j) return std::nullopt;
    const auto& s = j->value("sampling", nlohmann::json());
    if (s.is_null()) return std::nullopt;
    SamplingDefaults d;
    d.temperature = s.value("temperature", 1.0);
    d.top_k = s.value("top_k", 0LL);
    d.top_p = s.value("top_p", 1.0);
    d.repetition_penalty = s.value("repetition_penalty", 1.0);
    return d;
}

}  // namespace kivi::runtime
