// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 the YASAKANI authors
// Lightweight safetensors reader with small staging buffer + POSIX_FADV_DONTNEED so uploading ~19 GB checkpoints to GPU
// leaves minimal Linux page cache / host RSS behind.
#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace yskn {

struct StTensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t file_offset = 0;  // absolute byte offset inside file (8 + hdr_len + data_offsets[0])
  size_t nbytes = 0;
  size_t file_idx = 0;
  size_t numel() const {
    size_t n = 1;
    for (int64_t d : shape) n *= size_t(d);
    return n;
  }
};

// Bytes per element of a safetensors dtype; 0 for a dtype this reader does not know.
inline size_t st_dtype_bytes(const std::string& dt) {
  if (dt == "F64" || dt == "I64" || dt == "U64") return 8;
  if (dt == "F32" || dt == "I32" || dt == "U32") return 4;
  if (dt == "BF16" || dt == "F16" || dt == "I16" || dt == "U16") return 2;
  if (dt == "F8_E4M3" || dt == "F8_E5M2" || dt == "F8_E8M0" || dt == "U8" || dt == "I8" || dt == "BOOL") return 1;
  return 0;
}

// Minimal recursive-descent JSON parser (sufficient for safetensors header and config.json). Nesting deeper than kMaxDepth is
// refused: a crafted header could otherwise overflow the stack, which no catch can stop.
namespace stjson {

constexpr int kMaxDepth = 64;

struct Val {
  enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
  bool b = false;
  double num = 0.0;
  int64_t i = 0;
  std::string s;
  std::vector<Val> arr;
  std::unordered_map<std::string, Val> obj;

  bool has(const char* k) const { return kind == Obj && obj.count(k) > 0; }
  const Val& at(const char* k) const {
    static const Val empty;
    if (kind != Obj) return empty;
    auto it = obj.find(k);
    return it == obj.end() ? empty : it->second;
  }
};

inline void skip_ws(const char* p, size_t n, size_t& i) {
  while (i < n && (p[i] == ' ' || p[i] == '\t' || p[i] == '\n' || p[i] == '\r')) ++i;
}

inline std::string parse_str(const char* p, size_t n, size_t& i) {
  if (i >= n || p[i] != '"') throw std::runtime_error("json: expected '\"'");
  ++i;
  std::string out;
  while (i < n) {
    const char c = p[i++];
    if (c == '"') return out;
    if (c == '\\' && i < n) {
      const char e = p[i++];
      if (e == 'n') out += '\n';
      else if (e == 'r') out += '\r';
      else if (e == 't') out += '\t';
      else if (e == 'u' && i + 4 <= n) {
        unsigned cp = 0;
        for (int k = 0; k < 4; ++k) {
          const char h = p[i + k];
          cp = (cp << 4) | unsigned(h >= 'a' ? 10 + h - 'a' : h >= 'A' ? 10 + h - 'A' : h - '0');
        }
        i += 4;
        if (cp < 0x80) out += char(cp);
        else if (cp < 0x800) {
          out += char(0xC0 | (cp >> 6));
          out += char(0x80 | (cp & 0x3F));
        } else {
          out += char(0xE0 | (cp >> 12));
          out += char(0x80 | ((cp >> 6) & 0x3F));
          out += char(0x80 | (cp & 0x3F));
        }
      } else {
        out += e;
      }
    } else {
      out += c;
    }
  }
  throw std::runtime_error("json: unterminated string");
}

inline Val parse_val(const char* p, size_t n, size_t& i, int depth = 0) {
  if (depth > kMaxDepth) throw std::runtime_error("json: nested deeper than " + std::to_string(kMaxDepth) + " levels");
  skip_ws(p, n, i);
  if (i >= n) throw std::runtime_error("json: unexpected eof");
  const char c = p[i];
  if (c == '"') {
    Val v;
    v.kind = Val::Str;
    v.s = parse_str(p, n, i);
    return v;
  }
  if (c == '{') {
    ++i;
    Val v;
    v.kind = Val::Obj;
    skip_ws(p, n, i);
    if (i < n && p[i] == '}') {
      ++i;
      return v;
    }
    while (true) {
      skip_ws(p, n, i);
      std::string k = parse_str(p, n, i);
      skip_ws(p, n, i);
      if (i >= n || p[i] != ':') throw std::runtime_error("json: expected ':'");
      ++i;
      v.obj.emplace(std::move(k), parse_val(p, n, i, depth + 1));
      skip_ws(p, n, i);
      if (i < n && p[i] == ',') {
        ++i;
        continue;
      }
      if (i < n && p[i] == '}') {
        ++i;
        return v;
      }
      throw std::runtime_error("json: expected ',' or '}'");
    }
  }
  if (c == '[') {
    ++i;
    Val v;
    v.kind = Val::Arr;
    skip_ws(p, n, i);
    if (i < n && p[i] == ']') {
      ++i;
      return v;
    }
    while (true) {
      v.arr.push_back(parse_val(p, n, i, depth + 1));
      skip_ws(p, n, i);
      if (i < n && p[i] == ',') {
        ++i;
        continue;
      }
      if (i < n && p[i] == ']') {
        ++i;
        return v;
      }
      throw std::runtime_error("json: expected ',' or ']'");
    }
  }
  if (c == 't' && i + 4 <= n && memcmp(p + i, "true", 4) == 0) {
    i += 4;
    Val v;
    v.kind = Val::Bool;
    v.b = true;
    return v;
  }
  if (c == 'f' && i + 5 <= n && memcmp(p + i, "false", 5) == 0) {
    i += 5;
    Val v;
    v.kind = Val::Bool;
    v.b = false;
    return v;
  }
  if (c == 'n' && i + 4 <= n && memcmp(p + i, "null", 4) == 0) {
    i += 4;
    return Val{};
  }
  size_t s = i;
  while (i < n && (p[i] == '-' || p[i] == '+' || p[i] == '.' || p[i] == 'e' || p[i] == 'E' || (p[i] >= '0' && p[i] <= '9')))
    ++i;
  if (i == s) throw std::runtime_error("json: unexpected char");
  Val v;
  v.kind = Val::Num;
  const std::string sub(p + s, i - s);
  v.num = std::stod(sub);
  v.i = (sub.find('.') == std::string::npos && sub.find('e') == std::string::npos && sub.find('E') == std::string::npos)
            ? std::stoll(sub)
            : static_cast<int64_t>(v.num);
  return v;
}

inline Val parse(const std::string& text) {
  size_t i = 0;
  return parse_val(text.data(), text.size(), i);
}

}  // namespace stjson

class SafetensorsReader {
 public:
  static constexpr size_t kChunkBytes = 32u << 20;

  SafetensorsReader() { staging_.resize(kChunkBytes); }
  ~SafetensorsReader() { close_all(); }

  void add_file(const std::string& path, const char* key_prefix) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size < 8) {
      ::close(fd);
      throw std::runtime_error("cannot read the size of " + path + " (or it is shorter than a safetensors header)");
    }
    const uint64_t fsize = uint64_t(st.st_size);
    uint64_t hlen = 0;
    if (::pread(fd, &hlen, 8, 0) != 8 || hlen == 0 || hlen > (64u << 20) || 8u + hlen > fsize) {
      ::close(fd);
      throw std::runtime_error("bad safetensors header in " + path);
    }
    std::string hdr(size_t(hlen), '\0');
    if (::pread(fd, hdr.data(), size_t(hlen), 8) != ssize_t(hlen)) {
      ::close(fd);
      throw std::runtime_error("truncated safetensors header in " + path);
    }
    const size_t fidx = fds_.size();
    fds_.push_back(fd);
    const stjson::Val root = stjson::parse(hdr);
    for (const auto& [name, val] : root.obj) {
      if (name == "__metadata__") continue;
      if (key_prefix && name.rfind(key_prefix, 0) != 0) continue;
      StTensorInfo info;
      info.dtype = val.at("dtype").s;
      uint64_t numel = 1;
      for (const auto& d : val.at("shape").arr) {
        if (d.kind != stjson::Val::Num || d.i < 0 || (d.i > 0 && numel > (uint64_t(1) << 48) / uint64_t(d.i)))
          throw std::runtime_error("bad shape for " + name + " in " + path);
        numel *= uint64_t(d.i);
        info.shape.push_back(d.i);
      }
      const auto& offs = val.at("data_offsets").arr;
      if (offs.size() != 2 || offs[0].i < 0 || offs[1].i < offs[0].i) throw std::runtime_error("bad offsets for " + name);
      info.file_offset = 8u + hlen + uint64_t(offs[0].i);
      info.nbytes = size_t(offs[1].i - offs[0].i);
      info.file_idx = fidx;
      // the data must lie inside the file, and be as large as dtype x shape says (the engine sizes its buffers by the shape)
      if (uint64_t(offs[1].i) > fsize - 8u - hlen)
        throw std::runtime_error("tensor " + name + " lies past the end of " + path + " (truncated or damaged file)");
      const size_t eb = st_dtype_bytes(info.dtype);
      if (eb != 0 && uint64_t(info.nbytes) != numel * eb)
        throw std::runtime_error("tensor " + name + " in " + path + ": " + std::to_string(info.nbytes) + " bytes of data, but dtype " +
                                 info.dtype + " and its shape need " + std::to_string(numel * eb));
      tensors_.emplace(name, std::move(info));
    }
  }

  void open_file(const std::string& path, const char* key_prefix = nullptr) {
    close_all();
    tensors_.clear();
    add_file(path, key_prefix);
  }

  void open_dir(const std::string& dir, const char* key_prefix = "model.llm.") {
    close_all();
    tensors_.clear();
    namespace fs = std::filesystem;
    std::vector<std::string> files;
    for (const auto& ent : fs::directory_iterator(dir)) {
      const std::string p = ent.path().string();
      if (p.size() > 12 && p.compare(p.size() - 12, 12, ".safetensors") == 0) files.push_back(p);
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error("no .safetensors files in " + dir);
    for (const std::string& path : files) add_file(path, key_prefix);
  }

  void close_all() {
    for (int fd : fds_) {
#if defined(POSIX_FADV_DONTNEED)
      ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
      if (fd >= 0) ::close(fd);
    }
    fds_.clear();
    staging_.shrink_to_fit();
  }

  const StTensorInfo& get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("missing tensor: " + name);
    return it->second;
  }

  void read_host(const StTensorInfo& t, void* dst) const {
    const int fd = fds_.at(t.file_idx);
    size_t done = 0;
    while (done < t.nbytes) {
      const ssize_t got = ::pread(fd, static_cast<uint8_t*>(dst) + done, t.nbytes - done, off_t(t.file_offset + done));
      if (got <= 0) throw std::runtime_error("pread failed");
      done += size_t(got);
    }
#if defined(POSIX_FADV_DONTNEED)
    ::posix_fadvise(fd, off_t(t.file_offset), off_t(t.nbytes), POSIX_FADV_DONTNEED);
#endif
  }

  void upload_gpu(const StTensorInfo& t, void* d_dst, cudaStream_t stream = nullptr) {
    if (staging_.size() < kChunkBytes) staging_.resize(kChunkBytes);
    const int fd = fds_.at(t.file_idx);
    size_t done = 0;
    while (done < t.nbytes) {
      const size_t n = std::min(kChunkBytes, t.nbytes - done);
      size_t got_tot = 0;
      while (got_tot < n) {
        const ssize_t got = ::pread(fd, staging_.data() + got_tot, n - got_tot, off_t(t.file_offset + done + got_tot));
        if (got <= 0) throw std::runtime_error("pread failed during GPU upload");
        got_tot += size_t(got);
      }
      const cudaError_t err = stream
          ? cudaMemcpyAsync(static_cast<uint8_t*>(d_dst) + done, staging_.data(), n, cudaMemcpyHostToDevice, stream)
          : cudaMemcpy(static_cast<uint8_t*>(d_dst) + done, staging_.data(), n, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
      if (stream) cudaStreamSynchronize(stream);
      done += n;
    }
#if defined(POSIX_FADV_DONTNEED)
    ::posix_fadvise(fd, off_t(t.file_offset), off_t(t.nbytes), POSIX_FADV_DONTNEED);
#endif
  }

 private:
  std::vector<int> fds_;
  std::unordered_map<std::string, StTensorInfo> tensors_;
  std::vector<uint8_t> staging_;
};

}  // namespace yskn
