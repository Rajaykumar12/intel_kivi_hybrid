#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <functional>
#include <queue>
#include <stdexcept>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <unicode/normalizer2.h>
#include <unicode/unistr.h>
#include <nlohmann/json.hpp>

namespace kivi::runtime {

using json = nlohmann::json;

namespace {

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("tokenizer.json: unsupported " + what);
}

// Length of the UTF-8 sequence starting with lead byte `c` (1 for invalid
// bytes, so malformed input still advances).
size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;
}

bool valid_utf8(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        auto c = static_cast<unsigned char>(s[i]);
        size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        if (n == 0 || i + n > s.size()) return false;
        uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
        for (size_t k = 1; k < n; ++k) {
            auto d = static_cast<unsigned char>(s[i + k]);
            if ((d >> 6) != 0x2) return false;
            cp = (cp << 6) | (d & 0x3F);
        }
        // overlong encodings, surrogates, out of range
        if ((n == 2 && cp < 0x80) || (n == 3 && cp < 0x800) || (n == 4 && cp < 0x10000) ||
            (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF)
            return false;
        i += n;
    }
    return true;
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    for (size_t j; (j = s.find(from, i)) != std::string::npos; i = j + from.size()) {
        out.append(s, i, j - i);
        out += to;
    }
    out.append(s, i, std::string::npos);
    s.swap(out);
}

bool is_ws(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

std::string string_pattern(const json& j, const char* where) {
    const auto& p = j.at("pattern");
    if (!p.contains("String")) unsupported(std::string(where) + " pattern (only String)");
    return p["String"].get<std::string>();
}

// A text segment between added tokens; `offset` is its byte offset in the
// original input (Metaspace prepend_scheme "first" looks at it).
struct Segment {
    std::string text;
    size_t offset;
};

using SegmentFn = std::function<void(std::vector<Segment>&)>;

// ---- byte-level (GPT-2) alphabet -------------------------------------------
// Each byte maps to one printable code point: '!'..'~', '¡'..'¬', '®'..'ÿ' map
// to themselves, the other 68 bytes to U+0100.. in byte order.
std::string utf8_encode(uint32_t cp) {
    std::string o;
    if (cp < 0x80) o += static_cast<char>(cp);
    else if (cp < 0x800) { o += static_cast<char>(0xC0 | (cp >> 6)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        o += static_cast<char>(0xE0 | (cp >> 12));
        o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        o += static_cast<char>(0xF0 | (cp >> 18));
        o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        o += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return o;
}

struct ByteAlphabet {
    std::string to_char[256];          // byte -> UTF-8 of its code point
    int16_t from_cp[512];              // code point -> byte, -1 if none
    ByteAlphabet() {
        std::fill(std::begin(from_cp), std::end(from_cp), int16_t{-1});
        uint32_t extra = 256;
        for (int b = 0; b < 256; ++b) {
            const bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE);
            const uint32_t cp = printable ? static_cast<uint32_t>(b) : extra++;
            to_char[b] = utf8_encode(cp);
            from_cp[cp] = static_cast<int16_t>(b);
        }
    }
};
const ByteAlphabet& byte_alphabet() {
    static const ByteAlphabet a;
    return a;
}

// Decode one UTF-8 code point at s[i] (assumes valid UTF-8; returns the
// lead byte for anything else). Advances i.
uint32_t next_cp(const std::string& s, size_t& i) {
    auto c = static_cast<unsigned char>(s[i]);
    size_t n = utf8_len(c);
    if (n == 1 || i + n > s.size()) { ++i; return c; }
    uint32_t cp = n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    i += n;
    return cp;
}

// Rust's String::from_utf8_lossy: one U+FFFD per maximal invalid subpart
// (the Unicode-recommended substitution), which is what tokenizers' ByteLevel
// decoder produces.
std::string utf8_lossy(const std::string& b) {
    static const std::string kRep = "\xEF\xBF\xBD";
    std::string out;
    out.reserve(b.size());
    size_t i = 0;
    const size_t n = b.size();
    auto byte = [&](size_t k) { return static_cast<unsigned char>(b[k]); };
    while (i < n) {
        const unsigned char c = byte(i);
        if (c < 0x80) { out += static_cast<char>(c); ++i; continue; }
        size_t need;
        unsigned char lo = 0x80, hi = 0xBF;              // range of the 2nd byte
        if (c >= 0xC2 && c <= 0xDF) need = 1;
        else if (c == 0xE0) { need = 2; lo = 0xA0; }
        else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) need = 2;
        else if (c == 0xED) { need = 2; hi = 0x9F; }
        else if (c == 0xF0) { need = 3; lo = 0x90; }
        else if (c >= 0xF1 && c <= 0xF3) need = 3;
        else if (c == 0xF4) { need = 3; hi = 0x8F; }
        else { out += kRep; ++i; continue; }          // 80..C1, F5..FF
        size_t k = 1;
        bool ok = true;
        for (; k <= need; ++k) {
            if (i + k >= n) { ok = false; break; }
            const unsigned char d = byte(i + k);
            const bool in = k == 1 ? (d >= lo && d <= hi) : (d >= 0x80 && d <= 0xBF);
            if (!in) { ok = false; break; }
        }
        if (ok) { out.append(b, i, need + 1); i += need + 1; }
        else { out += kRep; i += k; }                  // the valid prefix, as one U+FFFD
    }
    return out;
}

// ---- regex (tokenizers' Split pre-tokenizer) --------------------------------
// tokenizers uses Oniguruma; PCRE2 in UTF + UCP mode accepts the same
// patterns (\p{L}, \p{N}, (?i:..), lookahead) with the same leftmost-first
// semantics. MATCH_INVALID_UTF keeps arbitrary input bytes safe.
class Regex {
public:
    explicit Regex(const std::string& pattern) {
        int err = 0;
        PCRE2_SIZE off = 0;
        code_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(),
                              PCRE2_UTF | PCRE2_UCP | PCRE2_MATCH_INVALID_UTF, &err, &off, nullptr);
        if (!code_) {
            PCRE2_UCHAR msg[256];
            pcre2_get_error_message(err, msg, sizeof msg);
            throw std::runtime_error("tokenizer.json: regex error at " + std::to_string(off) + ": " +
                                     reinterpret_cast<const char*>(msg));
        }
        jit_ = pcre2_jit_compile(code_, PCRE2_JIT_COMPLETE) == 0;
    }
    ~Regex() { pcre2_code_free(code_); }
    Regex(const Regex&) = delete;
    Regex& operator=(const Regex&) = delete;

    // All non-overlapping matches as [begin, end) byte ranges.
    std::vector<std::pair<size_t, size_t>> find_all(const std::string& s) const {
        std::vector<std::pair<size_t, size_t>> out;
        pcre2_match_data* md = pcre2_match_data_create_from_pattern(code_, nullptr);
        size_t start = 0;
        while (start <= s.size()) {
            const int rc = (jit_ ? pcre2_jit_match : pcre2_match)(
                code_, reinterpret_cast<PCRE2_SPTR>(s.data()), s.size(), start, 0, md, nullptr);
            if (rc == PCRE2_ERROR_NOMATCH) break;
            if (rc < 0) {
                pcre2_match_data_free(md);
                throw std::runtime_error("tokenizer: regex match error " + std::to_string(rc));
            }
            const PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
            if (ov[1] > ov[0]) out.emplace_back(ov[0], ov[1]);
            if (ov[1] == ov[0]) {                      // empty match: step one character
                if (ov[1] >= s.size()) break;
                start = ov[1] + utf8_len(static_cast<unsigned char>(s[ov[1]]));
            } else {
                start = ov[1];
            }
        }
        pcre2_match_data_free(md);
        return out;
    }

private:
    pcre2_code* code_ = nullptr;
    bool jit_ = false;
};

std::string nfc(const std::string& s) {
    bool ascii = true;
    for (unsigned char c : s) if (c >= 0x80) { ascii = false; break; }
    if (ascii) return s;                               // ASCII is always NFC
    UErrorCode st = U_ZERO_ERROR;
    const icu::Normalizer2* n = icu::Normalizer2::getNFCInstance(st);
    if (U_FAILURE(st)) throw std::runtime_error("tokenizer: ICU NFC unavailable");
    icu::UnicodeString u = icu::UnicodeString::fromUTF8(s);
    icu::UnicodeString r = n->normalize(u, st);
    if (U_FAILURE(st)) throw std::runtime_error("tokenizer: ICU NFC failed");
    std::string out;
    r.toUTF8String(out);
    return out;
}

// Split::Isolated: matches and the text between them each become a piece.
void split_isolated(const Regex& re, std::vector<Segment>& segs) {
    std::vector<Segment> out;
    for (auto& s : segs) {
        size_t last = 0;
        for (auto [b, e] : re.find_all(s.text)) {
            if (b > last) out.push_back({s.text.substr(last, b - last), s.offset + last});
            out.push_back({s.text.substr(b, e - b), s.offset + b});
            last = e;
        }
        if (last < s.text.size()) out.push_back({s.text.substr(last), s.offset + last});
    }
    segs.swap(out);
}

}  // namespace

struct Tokenizer::Impl {
    struct Added {
        std::string content;
        int64_t id;
        bool special, lstrip, rstrip;
    };

    std::unordered_map<std::string, int64_t> vocab;
    std::vector<std::string> id_to_tok;
    std::vector<Added> added;               // sorted longest first
    std::vector<bool> special;              // per id
    std::unordered_map<uint64_t, std::pair<int32_t, int64_t>> merges;   // (a,b) -> (rank, merged)
    int64_t unk_id = -1;
    bool byte_fallback = false, fuse_unk = false, ignore_merges = false;
    std::vector<int64_t> byte_ids;          // <0xNN> ids, -1 if absent

    std::vector<SegmentFn> normalizers;     // per segment, in place
    std::vector<SegmentFn> pre_tokenizers;  // may split segments
    std::vector<int64_t> prefix_ids, suffix_ids;   // TemplateProcessing "single"
    std::vector<std::function<void(std::vector<std::string>&)>> decoders;

    static uint64_t key(int64_t a, int64_t b) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
    }

    // ---- loading -------------------------------------------------------------
    void load_model(const json& m) {
        if (m.at("type").get<std::string>() != "BPE") unsupported("model " + m["type"].dump());
        auto set = [&](const char* k) {
            const auto& v = m.value(k, json());
            return !v.is_null() && !v.get<std::string>().empty();
        };
        if (set("continuing_subword_prefix") || set("end_of_word_suffix"))
            unsupported("BPE continuing_subword_prefix / end_of_word_suffix");
        if (!m.value("dropout", json()).is_null()) unsupported("BPE dropout");
        byte_fallback = m.value("byte_fallback", false);
        fuse_unk = m.value("fuse_unk", false);
        ignore_merges = m.value("ignore_merges", false);
        int64_t max_id = -1;
        for (auto& [tok, id] : m.at("vocab").items()) {
            vocab.emplace(tok, id.get<int64_t>());
            max_id = std::max(max_id, id.get<int64_t>());
        }
        id_to_tok.assign(max_id + 1, "");
        for (auto& [tok, id] : vocab) id_to_tok[id] = tok;
        const auto& unk = m.value("unk_token", json());
        if (!unk.is_null()) {
            auto it = vocab.find(unk.get<std::string>());
            if (it == vocab.end()) throw std::runtime_error("tokenizer.json: unk_token not in vocab");
            unk_id = it->second;
        }
        int32_t rank = 0;
        for (const auto& mg : m.at("merges")) {
            std::string a, b;
            if (mg.is_array()) {
                a = mg.at(0).get<std::string>();
                b = mg.at(1).get<std::string>();
            } else {                                   // legacy "a b"
                auto s = mg.get<std::string>();
                auto sp = s.find(' ', 1);
                if (sp == std::string::npos) throw std::runtime_error("tokenizer.json: bad merge " + s);
                a = s.substr(0, sp);
                b = s.substr(sp + 1);
            }
            auto ia = vocab.find(a), ib = vocab.find(b), iab = vocab.find(a + b);
            if (ia == vocab.end() || ib == vocab.end() || iab == vocab.end())
                throw std::runtime_error("tokenizer.json: merge of unknown tokens " + a + " " + b);
            merges.emplace(key(ia->second, ib->second), std::make_pair(rank++, iab->second));
        }
        byte_ids.assign(256, -1);
        for (int b = 0; b < 256; ++b) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "<0x%02X>", b);
            auto it = vocab.find(buf);
            if (it != vocab.end()) byte_ids[b] = it->second;
        }
    }

    void load_added(const json& arr) {
        for (const auto& a : arr) {
            if (a.value("normalized", false)) unsupported("normalized added tokens");
            if (a.value("single_word", false)) unsupported("single_word added tokens");
            Added t{a.at("content").get<std::string>(), a.at("id").get<int64_t>(),
                    a.value("special", false), a.value("lstrip", false), a.value("rstrip", false)};
            if (t.id >= static_cast<int64_t>(id_to_tok.size())) id_to_tok.resize(t.id + 1);
            id_to_tok[t.id] = t.content;
            vocab[t.content] = t.id;
            added.push_back(std::move(t));
        }
        std::stable_sort(added.begin(), added.end(), [](const Added& x, const Added& y) {
            return x.content.size() > y.content.size();
        });
        special.assign(id_to_tok.size(), false);
        for (auto& t : added) special[t.id] = t.special;
    }

    void load_normalizer(const json& n) {
        if (n.is_null()) return;
        const auto type = n.at("type").get<std::string>();
        if (type == "Sequence") {
            for (const auto& c : n.at("normalizers")) load_normalizer(c);
        } else if (type == "Prepend") {
            auto p = n.at("prepend").get<std::string>();
            normalizers.push_back([p](std::vector<Segment>& segs) {
                for (auto& s : segs) if (!s.text.empty()) s.text = p + s.text;
            });
        } else if (type == "Replace") {
            auto from = string_pattern(n, "Replace normalizer");
            auto to = n.at("content").get<std::string>();
            normalizers.push_back([from, to](std::vector<Segment>& segs) {
                for (auto& s : segs) replace_all(s.text, from, to);
            });
        } else {
            if (type == "NFC") {
                normalizers.push_back([](std::vector<Segment>& segs) {
                    for (auto& s : segs) s.text = nfc(s.text);
                });
                return;
            }
            unsupported("normalizer " + type);
        }
    }

    void load_pre_tokenizer(const json& p) {
        if (p.is_null()) return;
        const auto type = p.at("type").get<std::string>();
        if (type == "Sequence") {
            for (const auto& c : p.at("pretokenizers")) load_pre_tokenizer(c);
        } else if (type == "Metaspace") {
            const auto rep = p.at("replacement").get<std::string>();
            const auto scheme = p.value("prepend_scheme", std::string("always"));
            const bool split = p.value("split", true);
            if (scheme != "always" && scheme != "first" && scheme != "never")
                unsupported("Metaspace prepend_scheme " + scheme);
            pre_tokenizers.push_back([rep, scheme, split](std::vector<Segment>& segs) {
                std::vector<Segment> out;
                for (auto& s : segs) {
                    replace_all(s.text, " ", rep);
                    const bool starts = s.text.compare(0, rep.size(), rep) == 0;
                    if (!starts && (scheme == "always" || (scheme == "first" && s.offset == 0)))
                        s.text = rep + s.text;
                    if (!split) { out.push_back(std::move(s)); continue; }
                    // MergedWithNext: each piece starts at a replacement char.
                    size_t start = 0;
                    for (size_t i = rep.size(); i <= s.text.size(); ++i) {
                        if (i == s.text.size() || s.text.compare(i, rep.size(), rep) == 0) {
                            if (i > start) out.push_back({s.text.substr(start, i - start), s.offset});
                            start = i;
                        }
                    }
                }
                segs.swap(out);
            });
        } else {
            if (type == "Split") {
                const auto& pat = p.at("pattern");
                const auto behavior = p.value("behavior", std::string());
                if (behavior != "Isolated") unsupported("Split behavior " + behavior);
                if (p.value("invert", false)) unsupported("Split invert");
                std::shared_ptr<Regex> re;
                if (pat.contains("Regex")) re = std::make_shared<Regex>(pat["Regex"].get<std::string>());
                else if (pat.contains("String")) {
                    // A literal: escape it for PCRE2 with \Q..\E.
                    re = std::make_shared<Regex>("\\Q" + pat["String"].get<std::string>() + "\\E");
                } else unsupported("Split pattern");
                pre_tokenizers.push_back([re](std::vector<Segment>& segs) { split_isolated(*re, segs); });
                return;
            }
            if (type == "ByteLevel") {
                const bool prefix = p.value("add_prefix_space", false);
                std::shared_ptr<Regex> re;
                if (p.value("use_regex", true))   // GPT-2's own split
                    re = std::make_shared<Regex>(
                        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
                pre_tokenizers.push_back([prefix, re](std::vector<Segment>& segs) {
                    if (prefix)
                        for (auto& s : segs) if (s.text.empty() || s.text[0] != ' ') s.text = " " + s.text;
                    if (re) split_isolated(*re, segs);
                    const auto& a = byte_alphabet();
                    for (auto& s : segs) {
                        std::string m;
                        m.reserve(s.text.size() * 2);
                        for (unsigned char c : s.text) m += a.to_char[c];
                        s.text.swap(m);
                    }
                });
                return;
            }
            unsupported("pre_tokenizer " + type);
        }
    }

    void load_post_processor(const json& p) {
        if (p.is_null()) return;
        const auto type = p.at("type").get<std::string>();
        if (type == "ByteLevel") return;   // only adjusts offsets; ids unchanged
        if (type != "TemplateProcessing") unsupported("post_processor " + type);
        const auto& specials = p.at("special_tokens");
        bool seen_seq = false;
        for (const auto& item : p.at("single")) {
            if (item.contains("Sequence")) {
                if (item["Sequence"].at("id").get<std::string>() != "A") unsupported("template sequence");
                seen_seq = true;
            } else if (item.contains("SpecialToken")) {
                const auto name = item["SpecialToken"].at("id").get<std::string>();
                for (const auto& id : specials.at(name).at("ids"))
                    (seen_seq ? suffix_ids : prefix_ids).push_back(id.get<int64_t>());
            } else {
                unsupported("template item " + item.dump());
            }
        }
    }

    void load_decoder(const json& d) {
        if (d.is_null()) return;
        const auto type = d.at("type").get<std::string>();
        if (type == "Sequence") {
            for (const auto& c : d.at("decoders")) load_decoder(c);
        } else if (type == "Replace") {
            auto from = string_pattern(d, "Replace decoder");
            auto to = d.at("content").get<std::string>();
            decoders.push_back([from, to](std::vector<std::string>& toks) {
                for (auto& t : toks) replace_all(t, from, to);
            });
        } else if (type == "ByteFallback") {
            decoders.push_back([](std::vector<std::string>& toks) {
                std::vector<std::string> out;
                std::string bytes;
                size_t nbytes = 0;
                auto flush = [&] {
                    if (!nbytes) return;
                    if (valid_utf8(bytes)) out.push_back(bytes);
                    else for (size_t i = 0; i < nbytes; ++i) out.push_back("\xEF\xBF\xBD");   // U+FFFD
                    bytes.clear();
                    nbytes = 0;
                };
                for (auto& t : toks) {
                    if (t.size() == 6 && t.compare(0, 3, "<0x") == 0 && t[5] == '>' &&
                        std::isxdigit(static_cast<unsigned char>(t[3])) &&
                        std::isxdigit(static_cast<unsigned char>(t[4]))) {
                        bytes.push_back(static_cast<char>(std::stoi(t.substr(3, 2), nullptr, 16)));
                        ++nbytes;
                    } else {
                        flush();
                        out.push_back(std::move(t));
                    }
                }
                flush();
                toks.swap(out);
            });
        } else if (type == "Fuse") {
            decoders.push_back([](std::vector<std::string>& toks) {
                std::string all;
                for (auto& t : toks) all += t;
                toks.assign(1, std::move(all));
            });
        } else if (type == "Strip") {
            const auto content = d.at("content").get<std::string>();
            const auto start = d.value("start", 0), stop = d.value("stop", 0);
            if (content.size() != 1) unsupported("Strip content (one ASCII char only)");
            const char c = content[0];
            decoders.push_back([c, start, stop](std::vector<std::string>& toks) {
                for (auto& t : toks) {
                    size_t b = 0, e = t.size();
                    for (int i = 0; i < start && b < e && t[b] == c; ++i) ++b;
                    for (int i = 0; i < stop && e > b && t[e - 1] == c; ++i) --e;
                    t = t.substr(b, e - b);
                }
            });
        } else {
            if (type == "ByteLevel") {
                // Every char of a token maps back to a byte; a token with any
                // char outside the alphabet is kept as its raw UTF-8. Then all
                // bytes are decoded at once, invalid sequences -> U+FFFD.
                decoders.push_back([](std::vector<std::string>& toks) {
                    const auto& a = byte_alphabet();
                    std::string bytes;
                    for (auto& t : toks) {
                        std::string tb;
                        bool ok = true;
                        for (size_t i = 0; i < t.size() && ok;) {
                            const uint32_t cp = next_cp(t, i);
                            if (cp < 512 && a.from_cp[cp] >= 0) tb += static_cast<char>(a.from_cp[cp]);
                            else ok = false;
                        }
                        bytes += ok ? tb : t;
                    }
                    toks.assign(1, utf8_lossy(bytes));
                });
                return;
            }
            unsupported("decoder " + type);
        }
    }

    // ---- encoding -------------------------------------------------------------
    // Split `text` around added tokens; segments go to `segs`, and the
    // interleaving order is recorded in `order` (>= 0: added id, -1: next segment).
    void split_added(const std::string& text, std::vector<Segment>& segs,
                     std::vector<int64_t>& order) const {
        size_t seg_start = 0, i = 0;
        auto emit_segment = [&](size_t end) {
            if (end > seg_start) {
                segs.push_back({text.substr(seg_start, end - seg_start), seg_start});
                order.push_back(-1);
            }
        };
        while (i < text.size()) {
            const Added* hit = nullptr;
            for (const auto& a : added)                 // longest first
                if (text.compare(i, a.content.size(), a.content) == 0) { hit = &a; break; }
            if (!hit) { ++i; continue; }
            size_t seg_end = i;
            if (hit->lstrip) while (seg_end > seg_start && is_ws(text[seg_end - 1])) --seg_end;
            emit_segment(seg_end);
            order.push_back(hit->id);
            i += hit->content.size();
            if (hit->rstrip) while (i < text.size() && is_ws(text[i])) ++i;
            seg_start = i;
        }
        emit_segment(text.size());
    }

    void bpe(const std::string& word, std::vector<int64_t>& out) const {
        if (ignore_merges) {
            auto it = vocab.find(word);
            if (it != vocab.end()) { out.push_back(it->second); return; }
        }
        struct Sym { int64_t id; int prev, next; bool live; };
        std::vector<Sym> syms;
        auto push = [&](int64_t id) {
            int n = static_cast<int>(syms.size());
            syms.push_back({id, n - 1, n + 1, true});
        };
        for (size_t i = 0; i < word.size();) {
            const size_t n = std::min(utf8_len(static_cast<unsigned char>(word[i])), word.size() - i);
            auto it = vocab.find(word.substr(i, n));
            if (it != vocab.end()) {
                push(it->second);
            } else {
                bool done = false;
                if (byte_fallback) {
                    bool all = true;
                    for (size_t k = 0; k < n; ++k)
                        all = all && byte_ids[static_cast<unsigned char>(word[i + k])] >= 0;
                    if (all) {
                        for (size_t k = 0; k < n; ++k) push(byte_ids[static_cast<unsigned char>(word[i + k])]);
                        done = true;
                    }
                }
                if (!done) {
                    if (unk_id < 0) throw std::runtime_error("tokenizer: character not in vocab and no unk_token");
                    if (!(fuse_unk && !syms.empty() && syms.back().id == unk_id)) push(unk_id);
                }
            }
            i += n;
        }
        if (syms.empty()) return;
        syms.back().next = -1;

        // HF tokenizers' merge order: lowest rank first, ties -> leftmost.
        struct Cand { int32_t rank; int pos; int64_t merged; };
        auto cmp = [](const Cand& a, const Cand& b) {
            return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;
        };
        std::priority_queue<Cand, std::vector<Cand>, decltype(cmp)> q(cmp);
        auto consider = [&](int pos) {
            if (pos < 0 || syms[pos].next < 0) return;
            auto it = merges.find(key(syms[pos].id, syms[syms[pos].next].id));
            if (it != merges.end()) q.push({it->second.first, pos, it->second.second});
        };
        for (int p = 0; p + 1 < static_cast<int>(syms.size()); ++p) consider(p);
        while (!q.empty()) {
            const Cand c = q.top();
            q.pop();
            Sym& left = syms[c.pos];
            if (!left.live || left.next < 0) continue;
            Sym& right = syms[left.next];
            auto it = merges.find(key(left.id, right.id));
            if (it == merges.end() || it->second.second != c.merged) continue;   // stale
            left.id = c.merged;
            right.live = false;
            left.next = right.next;
            if (right.next >= 0) syms[right.next].prev = c.pos;
            consider(left.prev);
            consider(c.pos);
        }
        for (int p = 0; p >= 0; p = syms[p].next) out.push_back(syms[p].id);
    }
};

Tokenizer::Tokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer Tokenizer::from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    const json j = json::parse(in);
    auto impl = std::make_unique<Impl>();
    if (j.contains("truncation") && !j["truncation"].is_null()) unsupported("truncation");
    if (j.contains("padding") && !j["padding"].is_null()) unsupported("padding");
    impl->load_model(j.at("model"));
    impl->load_added(j.value("added_tokens", json::array()));
    impl->load_normalizer(j.value("normalizer", json()));
    impl->load_pre_tokenizer(j.value("pre_tokenizer", json()));
    impl->load_post_processor(j.value("post_processor", json()));
    impl->load_decoder(j.value("decoder", json()));
    return Tokenizer(std::move(impl));
}

std::vector<int64_t> Tokenizer::encode(const std::string& text, bool add_special_tokens) const {
    const Impl& m = *impl_;
    std::vector<Segment> segs;
    std::vector<int64_t> order;
    m.split_added(text, segs, order);
    std::vector<int64_t> ids;
    if (add_special_tokens) ids = m.prefix_ids;
    size_t next_seg = 0;
    for (int64_t o : order) {
        if (o >= 0) { ids.push_back(o); continue; }
        std::vector<Segment> pieces{segs[next_seg++]};
        for (auto& f : m.normalizers) f(pieces);
        for (auto& f : m.pre_tokenizers) f(pieces);
        for (auto& p : pieces)
            if (!p.text.empty()) m.bpe(p.text, ids);
    }
    if (add_special_tokens) ids.insert(ids.end(), m.suffix_ids.begin(), m.suffix_ids.end());
    return ids;
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids, bool skip_special_tokens) const {
    const Impl& m = *impl_;
    std::vector<std::string> toks;
    toks.reserve(ids.size());
    for (int64_t id : ids) {
        if (id < 0 || id >= static_cast<int64_t>(m.id_to_tok.size()))
            throw std::runtime_error("tokenizer: id " + std::to_string(id) + " out of range");
        if (skip_special_tokens && m.special[id]) continue;
        toks.push_back(m.id_to_tok[id]);
    }
    for (auto& f : m.decoders) f(toks);
    std::string out;
    for (auto& t : toks) out += t;
    return out;
}

int64_t Tokenizer::vocab_size() const { return static_cast<int64_t>(impl_->id_to_tok.size()); }

std::string Tokenizer::id_to_token(int64_t id) const {
    return id >= 0 && id < vocab_size() ? impl_->id_to_tok[id] : std::string();
}

int64_t Tokenizer::token_to_id(const std::string& token) const {
    auto it = impl_->vocab.find(token);
    return it == impl_->vocab.end() ? -1 : it->second;
}

bool Tokenizer::is_special(int64_t id) const {
    return id >= 0 && id < vocab_size() && impl_->special[id];
}

}  // namespace kivi::runtime
