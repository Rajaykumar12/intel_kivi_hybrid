// Minimal read-only safetensors reader (Phase 5 A.4).
//
// Format: u64 little-endian header length N, N bytes of JSON
//   {"name": {"dtype": "BF16", "shape": [..], "data_offsets": [begin, end]}, ...}
// then the raw tensor bytes (offsets are relative to the end of the header).
//
// The file is mmapped once; tensor() returns a CPU tensor that aliases the
// mapping (MAP_PRIVATE, so an accidental write is copy-on-write rather than a
// write to the artifact) and keeps the mapping alive through its deleter.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ATen/Tensor.h>

namespace kivi::runtime {

class SafeTensorsFile {
public:
    explicit SafeTensorsFile(const std::string& path);

    bool contains(const std::string& name) const { return entries_.count(name) != 0; }
    std::vector<std::string> names() const;
    // Zero-copy view of `name` in its stored dtype.
    at::Tensor tensor(const std::string& name) const;
    // Drop `name`'s pages from this process's memory (madvise DONTNEED) once
    // a converted copy exists. Only the pages wholly inside the tensor are
    // released; the mapping is private and never written, so the pages are
    // clean and any later access just re-reads them from the file — a view
    // returned by tensor() stays valid, merely slower.
    void release(const std::string& name) const;
    const std::string& path() const { return path_; }

private:
    struct Entry {
        at::ScalarType dtype;
        std::vector<int64_t> shape;
        size_t begin, end;
    };
    struct Mapping;
    std::string path_;
    std::shared_ptr<Mapping> map_;
    size_t data_offset_ = 0;
    std::map<std::string, Entry> entries_;
};

}  // namespace kivi::runtime
