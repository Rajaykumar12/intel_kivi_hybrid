// Conformance test for the C++ tokenizer (Phase 5 A.5): every case of a
// tools/tokenizer_golden.py corpus must encode and decode exactly like the
// transformers tokenizer.
//
//   test_tokenizer <tokenizer.json> <golden.json> [generation.json]
//
// With generation.json, the golden's "chat" cases also check the chat
// template (A.8): rendering + tokenization equal HF apply_chat_template, and
// the incremental split kivi-run feeds turn by turn reproduces HF's ids for
// the whole conversation.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat.hpp"
#include "tokenizer/tokenizer.hpp"

namespace {

std::string show(const std::vector<int64_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size() && i < 24; ++i) s += (i ? " " : "") + std::to_string(v[i]);
    return s + (v.size() > 24 ? " ..." : "");
}

std::string snippet(const std::string& s) { return nlohmann::json(s.substr(0, 80)).dump(); }

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: test_tokenizer <tokenizer.json> <golden.json> [generation.json]\n");
        return 2;
    }
    auto tok = kivi::runtime::Tokenizer::from_file(argv[1]);
    std::ifstream in(argv[2]);
    const auto golden = nlohmann::json::parse(in);
    const auto& cases = golden.at("cases");

    size_t failed = 0, tokens = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& c : cases) {
        // text == null: a decode-only case (random ids).
        const std::string text = c.at("text").is_null() ? "<decode-only " + show(c["ids"].get<std::vector<int64_t>>()) + ">"
                                                        : c["text"].get<std::string>();
        const auto ids = c.at("ids").get<std::vector<int64_t>>();
        tokens += ids.size();
        std::vector<std::string> errs;
        if (!c["text"].is_null()) {
            const auto plain = c.at("ids_plain").get<std::vector<int64_t>>();
            auto got = tok.encode(text, true);
            if (got != ids) errs.push_back("encode: got " + show(got) + " want " + show(ids));
            auto got_plain = tok.encode(text, false);
            if (got_plain != plain)
                errs.push_back("encode(plain): got " + show(got_plain) + " want " + show(plain));
        }
        auto dec = tok.decode(ids, false);
        if (dec != c.at("decoded").get<std::string>())
            errs.push_back("decode: got " + snippet(dec) + " want " + snippet(c["decoded"].get<std::string>()));
        auto dec_skip = tok.decode(ids, true);
        if (dec_skip != c.at("decoded_skip").get<std::string>())
            errs.push_back("decode(skip): got " + snippet(dec_skip) + " want " +
                           snippet(c["decoded_skip"].get<std::string>()));
        if (!errs.empty()) {
            if (++failed <= 10) {
                std::fprintf(stderr, "FAIL %s\n", snippet(text).c_str());
                for (auto& e : errs) std::fprintf(stderr, "    %s\n", e.c_str());
            }
        }
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("tokenizer conformance: %zu/%zu cases identical (%zu tokens, %.2fs)\n",
                cases.size() - failed, cases.size(), tokens, s);
    if (argc == 4 && golden.contains("chat") && !golden["chat"].empty()) {
        std::string err;
        auto tmpl = kivi::runtime::ChatTemplate::load(argv[3], &err);
        if (!tmpl) {
            std::fprintf(stderr, "FAIL chat template: %s\n", err.c_str());
            return 1;
        }
        size_t chat_failed = 0, split_checked = 0;
        for (const auto& c : golden["chat"]) {
            std::vector<kivi::runtime::ChatMessage> msgs;
            for (const auto& m : c.at("messages"))
                msgs.push_back({m.at("role").get<std::string>(), m.at("content").get<std::string>()});
            const auto want_text = c.at("rendered").get<std::string>();
            const auto want_ids = c.at("ids").get<std::vector<int64_t>>();
            std::vector<std::string> errs;
            const auto text = tmpl->render(msgs);
            if (text != want_text) errs.push_back("render: got " + snippet(text) + " want " + snippet(want_text));
            const auto ids = tok.encode(text, false);
            if (ids != want_ids) errs.push_back("ids: got " + show(ids) + " want " + show(want_ids));
            if (c.contains("turn_prefix")) {
                ++split_checked;
                const auto prefix = c["turn_prefix"].get<std::string>();
                const auto delta = tmpl->next_turn(msgs.back().content, /*ended_with_eos=*/true);
                if (prefix + delta != want_text)
                    errs.push_back("next_turn: text " + snippet(prefix + delta) + " want " + snippet(want_text));
                auto inc = tok.encode(prefix, false);
                auto rest = tok.encode(delta, false, /*continuation=*/true);
                inc.insert(inc.end(), rest.begin(), rest.end());
                if (inc != want_ids) errs.push_back("incremental ids: got " + show(inc) + " want " + show(want_ids));
            }
            if (!errs.empty()) {
                ++chat_failed;
                std::fprintf(stderr, "FAIL chat %s\n", snippet(want_text).c_str());
                for (auto& e : errs) std::fprintf(stderr, "    %s\n", e.c_str());
            }
        }
        std::printf("chat conformance: %zu/%zu conversations identical (%zu incremental splits)\n",
                    golden["chat"].size() - chat_failed, golden["chat"].size(), split_checked);
        failed += chat_failed;
    }
    return failed == 0 ? 0 : 1;
}
