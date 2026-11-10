#include "safetensors.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

#include <ATen/ATen.h>
#include <nlohmann/json.hpp>

namespace kivi::runtime {

struct SafeTensorsFile::Mapping {
    void* addr = MAP_FAILED;
    size_t size = 0;
    ~Mapping() {
        if (addr != MAP_FAILED) munmap(addr, size);
    }
};

namespace {

at::ScalarType parse_dtype(const std::string& s) {
    if (s == "F32") return at::kFloat;
    if (s == "F16") return at::kHalf;
    if (s == "BF16") return at::kBFloat16;
    if (s == "F64") return at::kDouble;
    if (s == "I64") return at::kLong;
    if (s == "I32") return at::kInt;
    if (s == "I16") return at::kShort;
    if (s == "I8") return at::kChar;
    if (s == "U8") return at::kByte;
    if (s == "BOOL") return at::kBool;
    throw std::runtime_error("safetensors: unsupported dtype " + s);
}

}  // namespace

SafeTensorsFile::SafeTensorsFile(const std::string& path) : path_(path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("safetensors: cannot open " + path + ": " + std::strerror(errno));
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("safetensors: cannot stat " + path);
    }
    map_ = std::make_shared<Mapping>();
    map_->size = static_cast<size_t>(st.st_size);
    if (map_->size < 8) {
        ::close(fd);
        throw std::runtime_error("safetensors: truncated file " + path);
    }
    map_->addr = mmap(nullptr, map_->size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (map_->addr == MAP_FAILED) throw std::runtime_error("safetensors: mmap failed for " + path);

    const auto* base = static_cast<const unsigned char*>(map_->addr);
    uint64_t n = 0;
    for (int i = 7; i >= 0; --i) n = (n << 8) | base[i];   // little-endian u64
    if (8 + n > map_->size) throw std::runtime_error("safetensors: bad header length in " + path);
    data_offset_ = 8 + n;

    auto header = nlohmann::json::parse(base + 8, base + 8 + n);
    const size_t data_size = map_->size - data_offset_;
    for (auto& [name, v] : header.items()) {
        if (name == "__metadata__") continue;
        Entry e;
        e.dtype = parse_dtype(v.at("dtype").get<std::string>());
        e.shape = v.at("shape").get<std::vector<int64_t>>();
        auto off = v.at("data_offsets").get<std::vector<size_t>>();
        if (off.size() != 2 || off[0] > off[1] || off[1] > data_size)
            throw std::runtime_error("safetensors: bad offsets for " + name + " in " + path);
        e.begin = off[0];
        e.end = off[1];
        int64_t numel = 1;
        for (auto d : e.shape) numel *= d;
        if (static_cast<size_t>(numel) * at::elementSize(e.dtype) != e.end - e.begin)
            throw std::runtime_error("safetensors: size mismatch for " + name + " in " + path);
        entries_.emplace(name, std::move(e));
    }
}

std::vector<std::string> SafeTensorsFile::names() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (auto& kv : entries_) out.push_back(kv.first);
    return out;
}

at::Tensor SafeTensorsFile::tensor(const std::string& name) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) throw std::runtime_error("safetensors: no tensor " + name + " in " + path_);
    const Entry& e = it->second;
    auto* p = static_cast<char*>(map_->addr) + data_offset_ + e.begin;
    auto keep = map_;   // the tensor keeps the mapping alive
    return at::from_blob(p, e.shape, [keep](void*) {}, at::TensorOptions().dtype(e.dtype));
}

}  // namespace kivi::runtime
