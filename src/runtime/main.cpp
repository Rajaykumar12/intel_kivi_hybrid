// kivi-run — Python-free KIVI inference (Phase 5 Route A).
//
//   kivi-run --model artifacts/qwen25 --chat                        # interactive chat
//   kivi-run --model artifacts/qwen25 --chat --prompt "Hi" --system "Be brief."
//   kivi-run --model artifacts/tinyllama --prompt "Once upon a time" --max-new-tokens 64
//   kivi-run --model artifacts/tinyllama --prompt "..." --temperature 0.8 --top-p 0.95 --seed 7
//   kivi-run --model artifacts/tinyllama --ids 1,9038,2501,263,931 --max-new-tokens 64
//   kivi-run --model artifacts/tinyllama --golden golden_run.json   # exit 0 iff identical
//
// One prefill call per prompt (or chat turn), then one call per token. Text
// streams to stdout as it is produced; with --ids / --golden the generated ids
// go to stdout instead. Timings and --stats go to stderr.
//
// Chat keeps the KV cache across turns and feeds only each turn's new tokens
// (see chat.hpp); "/reset" starts a new conversation, "/quit" or EOF ends.
// Chat samples with the model's generation_config defaults unless a sampling
// flag (or --greedy) is given.
#include <unistd.h>

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

#include "chat.hpp"
#include "engine/engine.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "tokenizer/tokenizer.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using kivi::runtime::Model;
using kivi::runtime::Sampler;
using kivi::runtime::Tokenizer;

struct Options {
    std::string model_dir;
    std::vector<int64_t> ids;
    std::string prompt;
    bool have_prompt = false;
    std::string golden;
    bool chat = false;
    std::string system;
    bool have_system = false;
    int64_t max_new_tokens = 64;
    bool max_new_set = false;
    std::string device = "xpu";
    int threads = 0;
    bool ignore_eos = false;
    bool stats = false;
    bool stream = true;
    int64_t prefill_chunk = 0;
    bool sampling_set = false;
    kivi::runtime::SamplerConfig sampling;
    // Chat golden (tools/runtime_golden.py --chat-turn): scripted turns and
    // the replies they must produce.
    std::vector<std::string> chat_turns, chat_expected;
};

[[noreturn]] void usage(const char* msg = nullptr) {
    if (msg) std::fprintf(stderr, "kivi-run: %s\n\n", msg);
    std::fprintf(stderr,
        "usage: kivi-run --model DIR (--chat | --prompt TEXT | --prompt-file F | --ids 1,2,3 |\n"
        "                --ids-file F | --golden F) [options]\n"
        "input:\n"
        "  --model DIR             artifact directory (python -m kivi_sycl.export)\n"
        "  --chat                  chat with the model's template; reads turns from stdin\n"
        "                          (or one turn from --prompt). /reset, /quit\n"
        "  --system TEXT           chat system message (default: the template's)\n"
        "  --prompt TEXT           prompt text (tokenized with DIR/tokenizer.json)\n"
        "  --prompt-file F         prompt text from a file\n"
        "  --ids LIST              prompt token ids, comma-separated\n"
        "  --ids-file F            prompt token ids, whitespace/comma-separated\n"
        "  --golden F              JSON {prompt_ids, tokens}: generate len(tokens) greedy\n"
        "                          tokens from prompt_ids and compare (exit 1 on mismatch);\n"
        "                          or {chat, replies}: replay the chat turns greedily and\n"
        "                          compare the reply texts\n"
        "generation:\n"
        "  --max-new-tokens N      default 64 (chat: 512)\n"
        "  --greedy                argmax decoding (default outside chat)\n"
        "  --temperature T         sample with temperature T (> 0); 0 = greedy\n"
        "  --top-k K               keep the K most likely tokens (0 = off)\n"
        "  --top-p P               nucleus sampling, keep mass P (1 = off)\n"
        "  --repetition-penalty R  HF-style penalty on seen tokens (1 = off)\n"
        "  --seed S                RNG seed (default 0)\n"
        "  --ignore-eos            keep generating past an EOS token\n"
        "runtime:\n"
        "  --device xpu|cpu        where the KV cache engine runs (default xpu)\n"
        "  --threads N             intra-op CPU threads (default: torch's choice)\n"
        "  --prefill-chunk N       prefill in chunks of N tokens (0 = one call)\n"
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
            else if (a == "--chat") o.chat = true;
            else if (a == "--system") { o.system = next(); o.have_system = true; }
            else if (a == "--prompt") { o.prompt = next(); o.have_prompt = true; }
            else if (a == "--prompt-file") { o.prompt = read_file(next()); o.have_prompt = true; }
            else if (a == "--ids") o.ids = parse_ids(next());
            else if (a == "--ids-file") o.ids = parse_ids(read_file(next()));
            else if (a == "--golden") o.golden = next();
            else if (a == "--max-new-tokens") { o.max_new_tokens = std::stoll(next()); o.max_new_set = true; }
            else if (a == "--greedy") { greedy = true; o.sampling_set = true; }
            else if (a == "--temperature") { o.sampling.temperature = std::stod(next()); o.sampling_set = true; }
            else if (a == "--top-k") { o.sampling.top_k = std::stoll(next()); o.sampling_set = true; }
            else if (a == "--top-p") { o.sampling.top_p = std::stod(next()); o.sampling_set = true; }
            else if (a == "--repetition-penalty") {
                o.sampling.repetition_penalty = std::stod(next());
                o.sampling_set = true;
            }
            else if (a == "--seed") o.sampling.seed = std::stoull(next());
            else if (a == "--ignore-eos") o.ignore_eos = true;
            else if (a == "--device") o.device = next();
            else if (a == "--threads") o.threads = std::stoi(next());
            else if (a == "--prefill-chunk") o.prefill_chunk = std::stoll(next());
            else if (a == "--no-stream") o.stream = false;
            else if (a == "--stats") o.stats = true;
            else if (a == "-h" || a == "--help") usage();
            else usage(("unknown argument " + a).c_str());
        } catch (const std::logic_error&) {      // stoll/stod on a non-number
            usage(("bad value for " + a).c_str());
        }
    }
    if (o.model_dir.empty()) usage("--model is required");
    if (o.chat) {
        if (!o.golden.empty() || !o.ids.empty()) usage("--chat takes text (--prompt or stdin), not ids");
        if (!o.max_new_set) o.max_new_tokens = 512;
    } else {
        const int inputs = !o.golden.empty() + !o.ids.empty() + o.have_prompt;
        if (inputs != 1) usage("give exactly one of --chat, --prompt, --prompt-file, --ids, --ids-file, --golden");
        if (o.have_system) usage("--system needs --chat");
    }
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
// while a multi-byte character is still split across tokens.
class TextStream {
public:
    explicit TextStream(const Tokenizer& tok) : tok_(tok) {}
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
    const Tokenizer& tok_;
    size_t printed_ = 0;
};

struct Generation {
    std::vector<int64_t> tokens;
    bool ended_with_eos = false;
    double prefill_s = 0, decode_s = 0;
    int64_t prompt_tokens = 0;
};

// Feed `prompt` (new tokens, continuing at model.position()), then sample up
// to max_new tokens. `history` accumulates every token for the repetition
// penalty. Streams through `stream` if given.
Generation generate(Model& model, Sampler& sampler, const std::vector<int64_t>& prompt,
                    int64_t max_new, const std::vector<int64_t>& eos, bool ignore_eos,
                    int64_t prefill_chunk, std::vector<int64_t>& history, TextStream* stream) {
    Generation g;
    g.prompt_tokens = static_cast<int64_t>(prompt.size());
    auto last_row = [](const at::Tensor& logits) { return logits[0][logits.size(1) - 1]; };
    auto is_eos = [&](int64_t t) {
        for (auto e : eos) if (e == t) return true;
        return false;
    };
    history.insert(history.end(), prompt.begin(), prompt.end());
    const auto t0 = Clock::now();
    at::Tensor logits;
    const auto n = static_cast<int64_t>(prompt.size());
    const int64_t chunk = prefill_chunk > 0 ? prefill_chunk : n;
    for (int64_t b = 0; b < n; b += chunk) {
        std::vector<int64_t> part(prompt.begin() + b, prompt.begin() + std::min(n, b + chunk));
        logits = model.forward(part, model.position());
    }
    int64_t next = sampler.next(last_row(logits), history);
    const auto t1 = Clock::now();
    while (true) {
        g.tokens.push_back(next);
        history.push_back(next);
        const bool eos_hit = is_eos(next);
        const bool stop = (!ignore_eos && eos_hit) || static_cast<int64_t>(g.tokens.size()) == max_new;
        if (stream) stream->update(g.tokens, stop);
        if (stop) {
            g.ended_with_eos = eos_hit;
            break;
        }
        next = sampler.next(last_row(model.forward({next}, model.position())), history);
    }
    g.prefill_s = secs(t0, t1);
    g.decode_s = secs(t1, Clock::now());
    return g;
}

void print_timing(const Generation& g) {
    const auto decode_n = static_cast<double>(g.tokens.size() - 1);   // first token came from prefill
    std::fprintf(stderr, "kivi-run: prefill %lld tok in %.3fs (%.1f tok/s); decode %zu tok, %.2f tok/s\n",
                 static_cast<long long>(g.prompt_tokens), g.prefill_s, g.prompt_tokens / g.prefill_s,
                 g.tokens.size(), decode_n > 0 ? decode_n / g.decode_s : 0.0);
}

void print_stats(Model& model, const Sampler& sampler, double first_token_s) {
    const auto& cfg = model.config();
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
                 static_cast<long long>(s0.total_tokens), static_cast<long long>(s0.quantized_tokens),
                 static_cast<long long>(s0.residual_tokens), kivi / 1048576.0, fp / 1048576.0,
                 kivi > 0 ? static_cast<double>(fp) / kivi : 0.0);
    std::fprintf(stderr, "kivi-run: first token %.2fs after start; RSS %.0f MiB (peak %.0f MiB); "
                 "sampling %s\n", first_token_s, rss, hwm, sampler.greedy() ? "greedy" : "stochastic");
}

int run_chat(const Options& opt, Model& model, Sampler& sampler, const Tokenizer& tok,
             Clock::time_point t_start) {
    const auto& cfg = model.config();
    std::string err;
    auto tmpl = kivi::runtime::ChatTemplate::load(opt.model_dir + "/generation.json", &err);
    if (!tmpl) {
        std::fprintf(stderr, "kivi-run: --chat unavailable for this model: %s\n", err.c_str());
        return 1;
    }
    model.reserve(cfg.max_tokens);
    const bool scripted = !opt.chat_turns.empty();
    const bool interactive = !opt.have_prompt && !scripted;
    size_t turn = 0, chat_mismatch = 0;
    const bool tty = isatty(STDIN_FILENO);
    std::vector<int64_t> history;
    bool first = true, ended_with_eos = false, printed_first = false;
    // The last token of a reply is sampled but never fed through the model
    // (generation stops right after sampling it), so it is not in the KV
    // cache yet: it leads the next turn's input. Without it the cache would
    // miss e.g. the reply's </s> that the template's rendering contains.
    std::optional<int64_t> pending;

    std::string line;
    while (true) {
        std::string user;
        if (interactive) {
            if (tty) { std::fprintf(stderr, "> "); std::fflush(stderr); }
            if (!std::getline(std::cin, line)) break;
            if (line == "/quit" || line == "/exit") break;
            if (line == "/reset") {
                model.reset();
                model.reserve(cfg.max_tokens);
                history.clear();
                pending.reset();
                first = true;
                std::fprintf(stderr, "kivi-run: new conversation\n");
                continue;
            }
            if (line.empty()) continue;
            user = line;
        } else if (scripted) {
            if (turn == opt.chat_turns.size()) break;
            user = opt.chat_turns[turn];
        } else {
            user = opt.prompt;
        }

        std::vector<int64_t> ids;
        if (first) {
            std::vector<kivi::runtime::ChatMessage> msgs;
            if (opt.have_system) msgs.push_back({"system", opt.system});
            msgs.push_back({"user", user});
            ids = tok.encode(tmpl->render(msgs), /*add_special_tokens=*/false);
        } else {
            ids = tok.encode(tmpl->next_turn(user, ended_with_eos), /*add_special_tokens=*/false,
                             /*continuation=*/true);
            if (pending) ids.insert(ids.begin(), *pending);
        }
        const int64_t room = cfg.max_tokens - model.position() - static_cast<int64_t>(ids.size());
        if (room < 1) {
            std::fprintf(stderr, "kivi-run: context full (%lld tokens); /reset to start over\n",
                         static_cast<long long>(cfg.max_tokens));
            if (!interactive) return 1;
            continue;
        }
        TextStream stream(tok);
        auto g = generate(model, sampler, ids, std::min(opt.max_new_tokens, room), cfg.eos_token_ids,
                          false, opt.prefill_chunk, history, opt.stream ? &stream : nullptr);
        if (!opt.stream) std::printf("%s", tok.decode(g.tokens, true).c_str());
        std::printf("\n");
        std::fflush(stdout);
        if (scripted) {
            const auto reply = tok.decode(g.tokens, true);
            const bool same = reply == opt.chat_expected[turn];
            chat_mismatch += !same;
            std::fprintf(stderr, "kivi-run: golden chat turn %zu: %s\n", turn + 1,
                         same ? "identical" : ("MISMATCH, want: " + opt.chat_expected[turn]).c_str());
            ++turn;
        }
        ended_with_eos = g.ended_with_eos;
        pending = g.tokens.back();
        history.pop_back();          // re-added with the next turn's input
        first = false;
        if (opt.stats) {
            print_timing(g);
            print_stats(model, sampler, printed_first ? 0.0 : secs(t_start, Clock::now()) - g.decode_s);
            printed_first = true;
        }
        if (!interactive && !scripted) break;
    }
    if (scripted) {
        std::fprintf(stderr, "kivi-run: golden chat %zu/%zu turns identical\n",
                     opt.chat_turns.size() - chat_mismatch, opt.chat_turns.size());
        return chat_mismatch == 0 && turn == opt.chat_turns.size() ? 0 : 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const auto t_start = Clock::now();
    Options opt = parse_args(argc, argv);
    std::vector<int64_t> expected;
    if (!opt.golden.empty()) {
        std::ifstream in(opt.golden);
        if (!in) { std::fprintf(stderr, "kivi-run: cannot open %s\n", opt.golden.c_str()); return 2; }
        auto g = nlohmann::json::parse(in);
        if (g.contains("chat")) {           // chat golden: scripted greedy turns
            const auto& c = g["chat"];
            opt.chat = true;
            if (!c.value("system", nlohmann::json()).is_null()) {
                opt.system = c["system"].get<std::string>();
                opt.have_system = true;
            }
            opt.chat_turns = c.at("turns").get<std::vector<std::string>>();
            opt.chat_expected = g.at("replies").get<std::vector<std::string>>();
            opt.max_new_tokens = c.value("reply_tokens", 40);
            opt.ids.clear();
        } else {
            opt.ids = g.at("prompt_ids").get<std::vector<int64_t>>();
            expected = g.at("tokens").get<std::vector<int64_t>>();
            opt.max_new_tokens = static_cast<int64_t>(expected.size());
            opt.ignore_eos = true;          // the golden run generates a fixed number of tokens
        }
        opt.sampling = {};                  // golden runs are greedy
        opt.sampling_set = true;
        opt.stream = false;
    }
    if (opt.threads > 0) at::set_num_threads(opt.threads);
    if (opt.chat && !opt.sampling_set) {    // the model's own defaults, as HF generate() uses
        if (auto d = kivi::runtime::load_sampling_defaults(opt.model_dir + "/generation.json")) {
            opt.sampling.temperature = d->temperature;
            opt.sampling.top_k = d->top_k;
            opt.sampling.top_p = d->top_p;
            opt.sampling.repetition_penalty = d->repetition_penalty;
        }
    }

    try {
        at::InferenceMode guard;
        Sampler sampler(opt.sampling);
        Model model(opt.model_dir, opt.device);
        const auto& cfg = model.config();
        std::optional<Tokenizer> tok;
        if (opt.have_prompt || opt.chat) tok = Tokenizer::from_file(opt.model_dir + "/tokenizer.json");
        std::fprintf(stderr, "kivi-run: loaded %s (%s, %lld layers) in %.2fs, %d threads\n",
                     opt.model_dir.c_str(), cfg.model_type.c_str(),
                     static_cast<long long>(cfg.num_layers), model.load_seconds(),
                     at::get_num_threads());
        if (opt.chat) return run_chat(opt, model, sampler, *tok, t_start);

        if (tok) opt.ids = tok->encode(opt.prompt);
        model.reserve(static_cast<int64_t>(opt.ids.size()) + opt.max_new_tokens);
        std::optional<TextStream> stream;
        if (tok && opt.stream) stream.emplace(*tok);
        std::vector<int64_t> history;
        auto g = generate(model, sampler, opt.ids, opt.max_new_tokens, cfg.eos_token_ids,
                          opt.ignore_eos, opt.prefill_chunk, history, stream ? &*stream : nullptr);

        if (stream) {
            std::printf("\n");
        } else if (tok) {
            std::printf("%s\n", tok->decode(g.tokens, /*skip_special_tokens=*/true).c_str());
        } else {
            for (size_t i = 0; i < g.tokens.size(); ++i)
                std::printf(i ? " %lld" : "%lld", static_cast<long long>(g.tokens[i]));
            std::printf("\n");
        }
        std::fflush(stdout);
        print_timing(g);
        if (opt.stats) print_stats(model, sampler, secs(t_start, Clock::now()) - g.decode_s);

        if (!expected.empty()) {
            const auto& out = g.tokens;
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
