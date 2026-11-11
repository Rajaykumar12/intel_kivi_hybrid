// tokenizer.json reader + BPE (Phase 5 A.5), no Python, no Rust.
//
// Implements the subset of the HuggingFace `tokenizers` pipeline that the
// supported models use, exactly as the library runs it:
//
//   encode:  split out added tokens (leftmost-longest, on the raw text)
//            -> per remaining segment: normalizer -> pre-tokenizer -> BPE
//            -> post-processor (TemplateProcessing: e.g. prepend <s>)
//   decode:  ids -> tokens (optionally dropping special tokens) -> decoder
//
// Supported components (anything else is rejected at load time, never
// silently approximated):
//   normalizer     null, Sequence, Prepend, Replace(String)
//   pre_tokenizer  null, Metaspace
//   model          BPE (byte_fallback, fuse_unk, ignore_merges)
//   decoder        Sequence, Replace(String), ByteFallback, Fuse, Strip
//   post_processor null, TemplateProcessing
// Byte-level (GPT-2 / Qwen) components are added in A.7.
//
// The exporter writes the pipeline transformers actually runs
// (backend_tokenizer.save), which can differ from the tokenizer.json a model
// repo ships (kivi_sycl.export.artifacts.write_effective_tokenizer).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace kivi::runtime {

class Tokenizer {
public:
    static Tokenizer from_file(const std::string& tokenizer_json_path);
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    std::vector<int64_t> encode(const std::string& text, bool add_special_tokens = true) const;
    std::string decode(const std::vector<int64_t>& ids, bool skip_special_tokens = false) const;

    int64_t vocab_size() const;
    // Token string for `id` ("" if unknown) / id for a token string (-1 if unknown).
    std::string id_to_token(int64_t id) const;
    int64_t token_to_id(const std::string& token) const;
    bool is_special(int64_t id) const;

    struct Impl;

private:
    explicit Tokenizer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace kivi::runtime
