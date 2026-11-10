// An artifact directory loaded for inference (Phase 5 A.4) — the C++ twin of
// kivi_sycl.export.artifacts.load_artifacts:
//   kivi_config.json  -> kivi::Engine, registered as the id baked into the graph
//   model.pt2         -> AOTIModelPackageLoader (code only)
//   constants.json    -> each package constant read from its safetensors file,
//                        converted to the dtype the package expects, and handed
//                        to the package as user-managed constants
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/Tensor.h>

namespace torch::inductor {
class AOTIModelPackageLoader;
}

namespace kivi {
class Engine;
}

namespace kivi::runtime {

struct KiviConfig {
    int64_t artifact_version = 0;
    std::string torch_version;
    int64_t engine_id = 0;
    int64_t num_layers = 0, num_kv_heads = 0, head_dim = 0;
    int64_t residual_length = 0;
    std::vector<int64_t> group_size;
    std::string metadata_dtype = "float16";
    int64_t max_tokens = 0;
    int64_t vocab_size = 0;
    std::optional<int64_t> bos_token_id;
    std::vector<int64_t> eos_token_ids;
    std::string model_type;

    static KiviConfig load(const std::string& path);
};

class Model {
public:
    // device: where the KV engine keeps its cache ("xpu" or "cpu").
    Model(const std::string& artifact_dir, const std::string& device = "xpu");
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    const KiviConfig& config() const { return cfg_; }
    // Size the engine for `total_tokens` (prompt + generated) up front.
    void reserve(int64_t total_tokens);
    // Run tokens `ids` at positions start_pos.. ; returns logits [1, T, V].
    at::Tensor forward(const std::vector<int64_t>& ids, int64_t start_pos);
    // Tokens seen so far (the next position).
    int64_t position() const { return pos_; }
    std::shared_ptr<kivi::Engine> engine() const { return engine_; }
    double load_seconds() const { return load_seconds_; }

private:
    KiviConfig cfg_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> package_;
    std::unordered_map<std::string, at::Tensor> constants_;   // user-managed: kept alive here
    std::shared_ptr<kivi::Engine> engine_;
    std::optional<int64_t> engine_id_;
    at::Tensor state_;
    int64_t pos_ = 0;
    double load_seconds_ = 0;
};

}  // namespace kivi::runtime
