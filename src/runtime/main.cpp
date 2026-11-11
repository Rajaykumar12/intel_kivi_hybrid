// kivi-run — Python-free KIVI inference (Phase 5 Route A).
//
//   kivi-run --model artifacts/tinyllama --prompt "Once upon a time" --max-new-tokens 64
//   kivi-run --model artifacts/tinyllama --prompt "..." --temperature 0.8 --top-p 0.95 --seed 7
//   kivi-run --model artifacts/tinyllama --ids 1,9038,2501,263,931 --max-new-tokens 64
//   kivi-run --model artifacts/tinyllama --golden golden.json     # exit 0 iff identical
//
// One prefill call over the prompt, then one call per token. With --prompt
// the generated text streams to stdout as it is produced; with --ids /
// --golden the generated ids go to stdout (space-separated). Timings and
// --stats go to stderr.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
#include "sampler.hpp"
#include "tokenizer/tokenizer.hpp"

namespace {

using Clock = std::chrono::steady_clock;

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
    bool stream = true;
    kivi::runtime::SamplerConfig sampling;
};

[[noreturn]] void usage(const char* msg = nullptr) {
    if (msg) std::fprintf(stderr, "kivi-run: %s\n\n", msg);
    std::fprintf(stderr,
        "usage: kivi-run --model DIR (--prompt TEXT | --prompt-file F | --ids 1,2,3 |\n"
        "                --ids-file F | --golden F) [options]\n"
        "input:\n"
        "  --model DIR             artifact directory (python -m kivi_sycl.export)\n"
        "  --prompt TEXT           prompt text (tokenized with DIR/tokenizer.json)\n"
        "  --prompt-file F         prompt text from a file\n"
        "  --ids LIST              prompt token ids, comma-separated\n"
        "  --ids-file F            prompt token ids, whitespace/comma-separated\n"
        "  --golden F              JSON {prompt_ids, tokens}: generate len(tokens) greedy\n"
        "                          tokens from prompt_ids and compare (exit 1 on mismatch)\n"
        "generation:\n"
        "  --max-new-tokens N      default 64\n"
        "  --greedy                argmax decoding (the default)\n"
        "  --temperature T         sample with temperature T (> 0); 0 = greedy\n"
        "  --top-k K               keep the K most likely tokens (0 = off)\n"
        "  --top-p P               nucleus sampling, keep mass P (1 = off)\n"
        "  --repetition-penalty R  HF-style penalty on seen tokens (1 = off)\n"
        "  --seed S                RNG seed (default 0)\n"
        "  --ignore-eos            keep generating past an EOS token\n"
        "runtime:\n"
        "  --device xpu|cpu        where the KV cache engine runs (default xpu)\n"
        "  --threads N             intra-op CPU threads (default: torch's choice)\n"
        "  --no-stream             print the text once at the end\n"
        "  --stats                 KV-cache, memory and latency statistics\n");
    std::exit(2);
}

std::vector<int64_t> parse_ids(const std::string& s) {
    std::vector<int64_t> out;
    std::stringstream ss(s);
    while (ss >> std::ws && !ss.eof()) {
        if (ss.peek() == ',') { ss.get(); continue; }
        long long v;
        if (!(ss >> v)) throw std::runtime_error("bad token id list: " + s);
        out.push_back(v);
    }
    return out;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) usage(("cannot open " + path).c_str());
    std::stringstream b;
    b << in.rdbuf();
    return b.str();
}

Options parse_args(int argc, char** argv) {
    Options o;
    bool greedy = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage(("missing value for " + a).c_str());
            return argv[++i];
        };
        try {
            if (a == "--model") o.model_dir = next();
            else if (a == "--prompt") { o.prompt = next(); o.have_prompt = true; }
            else if (a == "--prompt-file") { o.prompt = read_file(next()); o.have_prompt = true; }
            else if (a == "--ids") o.ids = parse_ids(next());
            else if (a == "--ids-file") o.ids = parse_ids(read_file(next()));
            else if (a == "--golden") o.golden = next();
            else if (a == "--max-new-tokens") o.max_new_tokens = std::stoll(next());
            else if (a == "--greedy") greedy = true;
            else if (a == "--temperature") o.sampling.temperature = std::stod(next());
            else if (a == "--top-k") o.sampling.top_k = std::stoll(next());
            else if (a == "--top-p") o.sampling.top_p = std::stod(next());
            else if (a == "--repetition-penalty") o.sampling.repetition_penalty = std::stod(next());
            else if (a == "--seed") o.sampling.seed = std::stoull(next());
            else if (a == "--ignore-eos") o.ignore_eos = true;
            else if (a == "--device") o.device = next();
            else if (a == "--threads") o.threads = std::stoi(next());
            else if (a == "--no-stream") o.stream = false;
            else if (a == "--stats") o.stats = true;
            else if (a == "-h" || a == "--help") usage();
            else usage(("unknown argument " + a).c_str());
        } catch (const std::logic_error&) {      // stoll/stod on a non-number
            usage(("bad value for " + a).c_str());
        }
    }
    if (o.model_dir.empty()) usage("--model is required");
    const int inputs = !o.golden.empty() + !o.ids.empty() + o.have_prompt;
    if (inputs != 1) usage("give exactly one of --prompt, --prompt-file, --ids, --ids-file, --golden");
    if (o.max_new_tokens < 1) usage("--max-new-tokens must be >= 1");
    if (greedy) o.sampling.temperature = 0;
    return o;
}

double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// VmHWM / VmRSS in MiB from /proc/self/status.
std::pair<double, double> rss_mib() {
    std::ifstream in("/proc/self/status");
    std::string line;
    double hwm = 0, rss = 0;
    while (std::getline(in, line)) {
        auto kb = [&](const char* key) { return std::stod(line.substr(std::string(key).size())) / 1024.0; };
        if (line.rfind("VmHWM:", 0) == 0) hwm = kb("VmHWM:");
        else if (line.rfind("VmRSS:", 0) == 0) rss = kb("VmRSS:");
    }
    return {hwm, rss};
}

// Streams decoded text as tokens arrive. Decoding the whole output each step
// keeps the text identical to a one-shot decode (decoders such as Strip act
// on the full sequence); output is held back while it ends in U+FFFD, i.e.
// while a multi-byte character is still split across byte-fallback tokens.
class TextStream {
public:
    explicit TextStream(const kivi::runtime::Tokenizer& tok) : tok_(tok) {}
    void update(const std::vector<int64_t>& ids, bool final) {
        const std::string text = tok_.decode(ids, /*skip_special_tokens=*/true);
        static const std::string kReplacement = "\xEF\xBF\xBD";
        if (!final && text.size() >= 3 && text.compare(text.size() - 3, 3, kReplacement) == 0) return;
        if (text.size() > printed_) {
            std::fwrite(text.data() + printed_, 1, text.size() - printed_, stdout);
            std::fflush(stdout);
            printed_ = text.size();
        }
    }

private:
    const kivi::runtime::Tokenizer& tok_;
    size_t printed_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    const auto t_start = Clock::now();
    Options opt = parse_args(argc, argv);
    std::vector<int64_t> expected;
    if (!opt.golden.empty()) {
        std::ifstream in(opt.golden);
        if (!in) { std::fprintf(stderr, "kivi-run: cannot open %s\n", opt.golden.c_str()); return 2; }
        auto g = nlohmann::json::parse(in);
        opt.ids = g.at("prompt_ids").get<std::vector<int64_t>>();
        expected = g.at("tokens").get<std::vector<int64_t>>();
        opt.max_new_tokens = static_cast<int64_t>(expected.size());
        opt.ignore_eos = true;              // the golden run generates a fixed number of tokens
        opt.sampling = {};                  // and is greedy
    }
    if (opt.threads > 0) at::set_num_threads(opt.threads);

    try {
        at::InferenceMode guard;
        kivi::runtime::Sampler sampler(opt.sampling);
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

        auto is_eos = [&](int64_t t) {
            for (auto e : cfg.eos_token_ids) if (e == t) return true;
            return false;
        };
        auto last_row = [](const at::Tensor& logits) { return logits[0][logits.size(1) - 1]; };

        std::vector<int64_t> history = opt.ids;   // for the repetition penalty
        std::vector<int64_t> out;
        out.reserve(opt.max_new_tokens);
        std::optional<TextStream> stream;
        if (tok && opt.stream) stream.emplace(*tok);

        const auto t0 = Clock::now();
        int64_t next = sampler.next(last_row(model.forward(opt.ids, 0)), history);
        const auto t1 = Clock::now();
        while (true) {
            out.push_back(next);
            history.push_back(next);
            const bool stop = (!opt.ignore_eos && is_eos(next)) ||
                              static_cast<int64_t>(out.size()) == opt.max_new_tokens;
            if (stream) stream->update(out, stop);
            if (stop) break;
            next = sampler.next(last_row(model.forward({next}, model.position())), history);
        }
        const auto t2 = Clock::now();

        if (stream) {
            std::printf("\n");
        } else if (tok) {
            std::printf("%s\n", tok->decode(out, /*skip_special_tokens=*/true).c_str());
        } else {
            for (size_t i = 0; i < out.size(); ++i)
                std::printf(i ? " %lld" : "%lld", static_cast<long long>(out[i]));
            std::printf("\n");
        }
        std::fflush(stdout);

        const double decode_s = secs(t1, t2);
        const auto decode_n = static_cast<double>(out.size() - 1);   // the first token came from prefill
        std::fprintf(stderr, "kivi-run: prefill %lld tok in %.3fs (%.1f tok/s); decode %zu tok, %.2f tok/s\n",
                     static_cast<long long>(prompt_len), secs(t0, t1), prompt_len / secs(t0, t1),
                     out.size(), decode_n > 0 ? decode_n / decode_s : 0.0);
        if (opt.stats) {
            int64_t kivi = 0, fp = 0;
            for (int64_t l = 0; l < cfg.num_layers; ++l) {
                auto s = model.engine()->stats(l);
                kivi += s.kivi_bytes;
                fp += s.fp_baseline_bytes;
            }
            auto s0 = model.engine()->stats(0);
            const auto [hwm, rss] = rss_mib();
            std::fprintf(stderr,
                         "kivi-run: KV cache %lld tokens/layer (%lld 2-bit, %lld fp residual): "
                         "%.2f MiB vs %.2f MiB fp16 (%.2fx)\n",
                         static_cast<long long>(s0.total_tokens),
                         static_cast<long long>(s0.quantized_tokens),
                         static_cast<long long>(s0.residual_tokens), kivi / 1048576.0, fp / 1048576.0,
                         kivi > 0 ? static_cast<double>(fp) / kivi : 0.0);
            std::fprintf(stderr, "kivi-run: first token %.2fs after start; RSS %.0f MiB (peak %.0f MiB); "
                         "sampling %s\n", secs(t_start, t1), rss, hwm,
                         sampler.greedy() ? "greedy" : "stochastic");
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
