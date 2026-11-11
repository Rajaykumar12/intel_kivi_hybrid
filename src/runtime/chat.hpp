// Chat templates (Phase 5 A.8): the literal pieces the exporter derived from
// the HF Jinja template and verified against it (kivi_sycl/export/chat.py),
// read from generation.json. Rendering is concatenation:
//
//   (system.prefix + S + system.suffix | no_system_prefix)
//   + per message: role.prefix + content + role.suffix
//   + generation_prompt
//
// Multi-turn chat keeps the KV cache, so each later turn feeds only its delta
// (next_turn): the rest of the assistant suffix after the EOS the model
// generated, the new user block, and the generation prompt. The full
// conversation's text is then exactly what the template renders.
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace kivi::runtime {

struct ChatMessage {
    std::string role;      // "system", "user", "assistant"
    std::string content;
};

class ChatTemplate {
public:
    // nullopt if the artifact has no derivable chat template; *error says why.
    static std::optional<ChatTemplate> load(const std::string& generation_json,
                                            std::string* error = nullptr);

    std::string render(const std::vector<ChatMessage>& messages,
                       bool add_generation_prompt = true) const;
    // Text to feed after the model ended its reply. `ended_with_eos`: the
    // reply's last generated token was eos_token (its text is already in the
    // cache), otherwise the reply was cut off and the whole suffix is fed.
    std::string next_turn(const std::string& user, bool ended_with_eos) const;
    const std::string& eos_token() const { return eos_token_; }

private:
    std::string no_system_prefix_, generation_prompt_, eos_token_;
    std::pair<std::string, std::string> system_, user_, assistant_;
};

// Default sampling from the model's generation_config (if it samples).
struct SamplingDefaults {
    double temperature = 1.0, top_p = 1.0, repetition_penalty = 1.0;
    long long top_k = 0;
};
std::optional<SamplingDefaults> load_sampling_defaults(const std::string& generation_json);

}  // namespace kivi::runtime
