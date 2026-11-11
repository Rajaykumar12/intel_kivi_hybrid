// Conformance test for the C++ tokenizer (Phase 5 A.5): every case of a
// tools/tokenizer_golden.py corpus must encode and decode exactly like the
// transformers tokenizer.
//
//   test_tokenizer <tokenizer.json> <golden.json>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

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
    if (argc != 3) {
        std::fprintf(stderr, "usage: test_tokenizer <tokenizer.json> <golden.json>\n");
        return 2;
    }
    auto tok = kivi::runtime::Tokenizer::from_file(argv[1]);
    std::ifstream in(argv[2]);
    const auto golden = nlohmann::json::parse(in);
    const auto& cases = golden.at("cases");

    size_t failed = 0, tokens = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& c : cases) {
        const auto text = c.at("text").get<std::string>();
        const auto ids = c.at("ids").get<std::vector<int64_t>>();
        const auto plain = c.at("ids_plain").get<std::vector<int64_t>>();
        tokens += ids.size();
        std::vector<std::string> errs;
        auto got = tok.encode(text, true);
        if (got != ids) errs.push_back("encode: got " + show(got) + " want " + show(ids));
        auto got_plain = tok.encode(text, false);
        if (got_plain != plain) errs.push_back("encode(plain): got " + show(got_plain) + " want " + show(plain));
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
    return failed == 0 ? 0 : 1;
}
