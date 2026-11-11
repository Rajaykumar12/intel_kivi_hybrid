// kivi-run — Python-free KIVI inference (Phase 5 A.4: token ids in, token ids out).
//
//   kivi-run --model artifacts/tinyllama --prompt "Once upon a time" --max-new-tokens 64
//   kivi-run --model artifacts/tinyllama --ids 1,9038,2501,263,931 --max-new-tokens 64
//   kivi-run --model artifacts/tinyllama --golden golden.json     # exit 0 iff identical
//
// Greedy decoding: one prefill call over the prompt, then one call per token.
// With --prompt the generated text goes to stdout; with --ids / --golden the
// generated ids do (space-separated). Timings go to stderr.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <nlohmann/json.hpp>

#include "engine/engine.hpp"
#include "model.hpp"
#include "tokenizer/tokenizer.hpp"

namespace {

struct Options {
    std::string model_dir;
    std::vector<int64_t> ids;
    std::string prompt;
    bool have_prompt = false;
    std::string golden;
    int64_t max_new_tokens = 64;
    std::string device = "xpu";
    int threads = 0;
    bool ignore_eos = false;
    bool stats = false;
};

[[noreturn]] void usage(const char* msg = nullptr) {
    if (msg) std::fprintf(stderr, "kivi-run: %s\n\n", msg);
    std::fprintf(stderr,
        "usage: kivi-run --model DIR (--prompt TEXT | --prompt-file F | --ids 1,2,3 |\n"
        "                --ids-file F | --golden F) [options]\n"
        "  --model DIR           artifact directory (python -m kivi_sycl.export)\n"
        "  --prompt TEXT         prompt text (tokenized with DIR/tokenizer.json)\n"
        "  --prompt-file F       prompt text from a file\n"
        "  --ids LIST            prompt token ids, comma-separated\n"
        "  --ids-file F          prompt token ids, whitespace/comma-separated\n"
        "  --golden F            JSON {prompt_ids, tokens}: generate len(tokens) greedy\n"
        "                        tokens from prompt_ids and compare (exit 1 on mismatch)\n"
        "  --max-new-tokens N    default 64\n"
        "  --device xpu|cpu      where the KV cache engine runs (default xpu)\n"
        "  --threads N           intra-op CPU threads (default: torch's choice)\n"
        "  --ignore-eos          keep generating past an EOS token\n"
        "  --stats               print KV-cache statistics\n");
    std::exit(2);
}

std::vector<int64_t> parse_ids(const std::string& s) {
    std::vector<int64_t> out;
    std::string tok;
    std::stringstream ss(s);
    while (ss >> std::ws && !ss.eof()) {
        if (ss.peek() == ',') { ss.get(); continue; }
        long long v;
        if (!(ss >> v)) throw std::runtime_error("bad token id list: " + s);
        out.push_back(v);
    }
    return out;
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(("missing value for " + a).c_str());
            return argv[++i];
        };
        if (a == "--model") o.model_dir = next();
        else if (a == "--prompt") { o.prompt = next(); o.have_prompt = true; }
        else if (a == "--prompt-file") {
            std::ifstream in(next());
            if (!in) usage("cannot open the --prompt-file");
            std::stringstream b; b << in.rdbuf();
            o.prompt = b.str();
            o.have_prompt = true;
        }
        else if (a == "--ids") o.ids = parse_ids(next());
        else if (a == "--ids-file") {
            std::ifstream in(next());
            std::stringstream b; b << in.rdbuf();
            o.ids = parse_ids(b.str());
        }
        else if (a == "--golden") o.golden = next();
        else if (a == "--max-new-tokens") o.max_new_tokens = std::stoll(next());
        else if (a == "--device") o.device = next();
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--ignore-eos") o.ignore_eos = true;
        else if (a == "--stats") o.stats = true;
        else if (a == "-h" || a == "--help") usage();
        else usage(("unknown argument " + a).c_str());
    }
    if (o.model_dir.empty()) usage("--model is required");
    const int inputs = !o.golden.empty() + !o.ids.empty() + o.have_prompt;
    if (inputs != 1) usage("give exactly one of --prompt, --prompt-file, --ids, --ids-file, --golden");
    return o;
}

int64_t argmax_last(const at::Tensor& logits) {
    return logits.select(1, logits.size(1) - 1).argmax(-1).item<int64_t>();
}

double secs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    std::vector<int64_t> expected;
    if (!opt.golden.empty()) {
        std::ifstream in(opt.golden);
        if (!in) { std::fprintf(stderr, "kivi-run: cannot open %s\n", opt.golden.c_str()); return 2; }
        auto g = nlohmann::json::parse(in);
        opt.ids = g.at("prompt_ids").get<std::vector<int64_t>>();
        expected = g.at("tokens").get<std::vector<int64_t>>();
        opt.max_new_tokens = static_cast<int64_t>(expected.size());
        opt.ignore_eos = true;   // the golden run generates a fixed number of tokens
    }
    if (opt.threads > 0) at::set_num_threads(opt.threads);

    try {
        at::InferenceMode guard;
        kivi::runtime::Model model(opt.model_dir, opt.device);
        const auto& cfg = model.config();
        std::optional<kivi::runtime::Tokenizer> tok;
        if (opt.have_prompt) {
            tok = kivi::runtime::Tokenizer::from_file(opt.model_dir + "/tokenizer.json");
            opt.ids = tok->encode(opt.prompt);
        }
        std::fprintf(stderr, "kivi-run: loaded %s (%s, %lld layers) in %.2fs, %d threads\n",
                     opt.model_dir.c_str(), cfg.model_type.c_str(),
                     static_cast<long long>(cfg.num_layers), model.load_seconds(),
                     at::get_num_threads());
        const auto prompt_len = static_cast<int64_t>(opt.ids.size());
        model.reserve(prompt_len + opt.max_new_tokens);

        std::vector<int64_t> out;
        out.reserve(opt.max_new_tokens);
        const auto t0 = std::chrono::steady_clock::now();
        int64_t next = argmax_last(model.forward(opt.ids, 0));
        const auto t1 = std::chrono::steady_clock::now();
        auto is_eos = [&](int64_t t) {
            for (auto e : cfg.eos_token_ids) if (e == t) return true;
            return false;
        };
        while (static_cast<int64_t>(out.size()) < opt.max_new_tokens) {
            out.push_back(next);
            if (!opt.ignore_eos && is_eos(next)) break;
            if (static_cast<int64_t>(out.size()) == opt.max_new_tokens) break;
            next = argmax_last(model.forward({next}, model.position()));
        }
        const auto t2 = std::chrono::steady_clock::now();

        if (tok) {
            std::printf("%s\n", tok->decode(out, /*skip_special_tokens=*/true).c_str());
        } else {
            for (size_t i = 0; i < out.size(); ++i)
                std::printf(i ? " %lld" : "%lld", static_cast<long long>(out[i]));
            std::printf("\n");
        }
        const double decode_s = secs(t1, t2);
        const auto decode_n = static_cast<double>(out.size() > 0 ? out.size() - 1 : 0);
        std::fprintf(stderr, "kivi-run: prefill %lld tok in %.3fs (%.1f tok/s); decode %zu tok, %.2f tok/s\n",
                     static_cast<long long>(prompt_len), secs(t0, t1), prompt_len / secs(t0, t1),
                     out.size(), decode_n > 0 ? decode_n / decode_s : 0.0);
        if (opt.stats) {
            auto s = model.engine()->stats(0);
            std::fprintf(stderr, "kivi-run: layer 0 KV: %lld tokens (%lld quantized, %lld residual), "
                         "%.2fx vs fp16\n", static_cast<long long>(s.total_tokens),
                         static_cast<long long>(s.quantized_tokens),
                         static_cast<long long>(s.residual_tokens), s.compression);
        }

        if (!expected.empty()) {
            size_t same = 0;
            while (same < out.size() && same < expected.size() && out[same] == expected[same]) ++same;
            std::fprintf(stderr, "kivi-run: golden %zu/%zu tokens identical%s\n", same,
                         expected.size(), same == expected.size() ? "" : " — MISMATCH");
            return same == expected.size() && out.size() == expected.size() ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "kivi-run: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
