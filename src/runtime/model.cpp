#include "model.hpp"

#include <chrono>
#include <fstream>
#include <set>
#include <stdexcept>

#include <ATen/ATen.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/version.h>

#include <cstdio>

#include "engine/engine.hpp"
#include "engine/registry.hpp"
#include "safetensors.hpp"

namespace kivi::runtime {

namespace {

constexpr int64_t kArtifactVersion = 1;

nlohmann::json read_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    return nlohmann::json::parse(in);
}

at::ScalarType dtype_from_name(const std::string& s) {
    if (s == "float32") return at::kFloat;
    if (s == "float16") return at::kHalf;
    if (s == "bfloat16") return at::kBFloat16;
    throw std::runtime_error("constants.json: unsupported dtype " + s);
}

}  // namespace

KiviConfig KiviConfig::load(const std::string& path) {
    auto j = read_json(path);
    KiviConfig c;
    c.artifact_version = j.at("artifact_version").get<int64_t>();
    if (c.artifact_version != kArtifactVersion)
        throw std::runtime_error(path + ": artifact_version " + std::to_string(c.artifact_version) +
                                 ", this runtime reads version " + std::to_string(kArtifactVersion));
    c.torch_version = j.value("torch_version", "");
    c.engine_id = j.at("engine_id").get<int64_t>();
    c.num_layers = j.at("num_layers").get<int64_t>();
    c.num_kv_heads = j.at("num_kv_heads").get<int64_t>();
    c.head_dim = j.at("head_dim").get<int64_t>();
    c.residual_length = j.at("residual_length").get<int64_t>();
    c.group_size = j.at("group_size").get<std::vector<int64_t>>();
    c.metadata_dtype = j.value("metadata_dtype", "float16");
    c.max_tokens = j.at("max_tokens").get<int64_t>();
    c.vocab_size = j.at("vocab_size").get<int64_t>();
    if (j.contains("bos_token_id") && !j["bos_token_id"].is_null())
        c.bos_token_id = j["bos_token_id"].get<int64_t>();
    c.eos_token_ids = j.value("eos_token_ids", std::vector<int64_t>{});
    c.model_type = j.value("model_type", "");
    if (static_cast<int64_t>(c.group_size.size()) != c.num_layers)
        throw std::runtime_error(path + ": group_size must list one value per layer");
    return c;
}

Model::Model(const std::string& dir, const std::string& device) {
    const auto t0 = std::chrono::steady_clock::now();
    cfg_ = KiviConfig::load(dir + "/kivi_config.json");
    // AOTInductor packages are tied to the torch release that compiled them.
    // "2.13.0+xpu" (artifact) vs TORCH_VERSION "2.13.0" (headers): compare
    // the release part.
    const std::string want = cfg_.torch_version.substr(0, cfg_.torch_version.find('+'));
    if (!want.empty() && want != TORCH_VERSION)
        std::fprintf(stderr, "kivi-run: warning: %s was exported with torch %s, this runtime "
                     "is built on %s; re-export if loading fails\n", dir.c_str(),
                     cfg_.torch_version.c_str(), TORCH_VERSION);

    // Engine first: the package's kivi.layer_attention calls look it up by id.
    EngineConfig ec;
    ec.num_layers = cfg_.num_layers;
    ec.head_dim = cfg_.head_dim;
    ec.num_kv_heads = cfg_.num_kv_heads;
    ec.residual_length = cfg_.residual_length;
    ec.group_size = cfg_.group_size;
    if (cfg_.metadata_dtype == "float16") ec.metadata_dtype = at::kHalf;
    else if (cfg_.metadata_dtype == "float32") ec.metadata_dtype = at::kFloat;
    else if (cfg_.metadata_dtype != "auto")
        throw std::runtime_error("kivi_config.json: metadata_dtype " + cfg_.metadata_dtype);
    ec.device = device;
    engine_ = std::make_shared<Engine>(ec);
    engine_id_ = register_engine(engine_, cfg_.engine_id);

    package_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(dir + "/model.pt2");

    // Constants: exactly the set the package reads.
    auto manifest = read_json(dir + "/constants.json");
    std::set<std::string> needed;
    for (auto& f : package_->get_constant_fqns()) needed.insert(f);
    std::set<std::string> listed;
    for (auto& [fqn, _] : manifest.items()) listed.insert(fqn);
    if (needed != listed) {
        std::string msg = "constants.json does not match model.pt2:";
        for (auto& f : needed) if (!listed.count(f)) { msg += " missing " + f; break; }
        for (auto& f : listed) if (!needed.count(f)) { msg += " unexpected " + f; break; }
        throw std::runtime_error(msg);
    }
    std::map<std::string, std::unique_ptr<SafeTensorsFile>> files;
    // A source tensor read by several constants (tied weights) is released
    // after its last conversion, so the later ones do not re-read the file.
    std::map<std::string, int> uses;
    for (auto& [fqn, m] : manifest.items())
        ++uses[m.at("file").get<std::string>() + "/" + m.at("tensor").get<std::string>()];
    for (auto& [fqn, m] : manifest.items()) {
        const auto file = m.at("file").get<std::string>();
        auto& f = files[file];
        if (!f) f = std::make_unique<SafeTensorsFile>(dir + "/" + file);
        at::Tensor t = f->tensor(m.at("tensor").get<std::string>());
        auto shape = m.at("shape").get<std::vector<int64_t>>();
        if (t.sizes() != at::IntArrayRef(shape))
            throw std::runtime_error("constants.json: shape mismatch for " + fqn);
        // Same dtype: aliases the mapping (no copy). Otherwise e.g. bf16 ->
        // fp32, which is exact; the bf16 source pages are then released at
        // once, so peak memory is the fp32 weights plus one tensor rather
        // than fp32 + the whole bf16 file (TinyLlama: 6.8 -> ~4.7 GB).
        const auto tensor_name = m.at("tensor").get<std::string>();
        at::Tensor c = t.to(dtype_from_name(m.at("dtype").get<std::string>())).contiguous();
        if (--uses[file + "/" + tensor_name] == 0 && c.data_ptr() != t.data_ptr())
            f->release(tensor_name);
        constants_[fqn] = std::move(c);
    }
    package_->load_constants(constants_, /*use_inactive=*/false, /*check_full_update=*/true,
                             /*user_managed=*/true);

    state_ = at::zeros({1}, at::kLong);
    load_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

Model::~Model() {
    package_.reset();
    if (engine_id_) {
        try { unregister_engine(*engine_id_); } catch (...) {}
    }
}

void Model::reserve(int64_t total_tokens) { engine_->reserve(total_tokens); }

void Model::reset() {
    engine_->reset();
    pos_ = 0;
}

at::Tensor Model::forward(const std::vector<int64_t>& ids, int64_t start_pos) {
    const auto T = static_cast<int64_t>(ids.size());
    if (T < 1) throw std::runtime_error("forward: no tokens");
    if (start_pos + T > cfg_.max_tokens)
        throw std::runtime_error("forward: " + std::to_string(start_pos + T) +
                                 " tokens exceed the exported max_tokens " +
                                 std::to_string(cfg_.max_tokens));
    at::Tensor input_ids = at::tensor(ids, at::kLong).view({1, T});
    at::Tensor position_ids = at::arange(start_pos, start_pos + T, at::kLong).view({1, T});
    auto out = package_->run({input_ids, position_ids, state_});
    for (auto& o : out)
        if (o.dim() == 3) {
            pos_ = start_pos + T;
            return o;
        }
    throw std::runtime_error("forward: the package returned no [1, T, V] logits");
}

}  // namespace kivi::runtime
