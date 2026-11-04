#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "types.h"

namespace ftl {

enum class Op : uint8_t { kRead, kWrite, kTrim };

// One host request in page units.
struct IoRequest {
  Op op = Op::kWrite;
  Lpn lpn = 0;
  uint32_t pages = 1;
};

class Workload {
 public:
  virtual ~Workload() = default;
  // Returns false when the workload is exhausted (traces only).
  virtual bool next(IoRequest& req) = 0;
};

class SequentialWorkload : public Workload {
 public:
  SequentialWorkload(uint64_t logical_pages, uint32_t req_pages);
  bool next(IoRequest& req) override;

 private:
  uint64_t n_;
  uint32_t req_pages_;
  uint64_t pos_ = 0;
};

class UniformWorkload : public Workload {
 public:
  UniformWorkload(uint64_t logical_pages, uint64_t seed);
  bool next(IoRequest& req) override;

 private:
  std::mt19937_64 rng_;
  std::uniform_int_distribution<uint64_t> dist_;
};

// hot_access of the writes go to the first hot_frac of the address space,
// e.g. 0.8 / 0.2 for the classic 80/20 workload.
class HotColdWorkload : public Workload {
 public:
  HotColdWorkload(uint64_t logical_pages, double hot_frac, double hot_access,
                  uint64_t seed);
  bool next(IoRequest& req) override;

 private:
  std::mt19937_64 rng_;
  uint64_t hot_pages_;
  uint64_t n_;
  double hot_access_;
};

// Zipf(theta) over pages; ranks are shuffled so hot pages are scattered.
class ZipfWorkload : public Workload {
 public:
  ZipfWorkload(uint64_t logical_pages, double theta, uint64_t seed);
  bool next(IoRequest& req) override;

 private:
  std::mt19937_64 rng_;
  std::vector<double> cdf_;
  std::vector<Lpn> rank_to_lpn_;
};

// Unified trace format (see tools/convert_trace.py): CSV lines
//   timestamp_us,op,offset_bytes,size_bytes
// op is R, W or T. Addresses beyond the device wrap around (counted).
class TraceWorkload : public Workload {
 public:
  TraceWorkload(const std::string& path, uint64_t logical_pages, uint32_t page_size);
  bool next(IoRequest& req) override;
  // One parsed line in bytes, before page conversion.
  bool next_raw(Op& op, uint64_t& offset, uint64_t& bytes);
  uint64_t wrapped() const { return wrapped_; }
  uint64_t bad_lines() const { return bad_lines_; }

 private:
  std::ifstream in_;
  uint64_t n_;
  uint32_t page_size_;
  uint64_t wrapped_ = 0;
  uint64_t bad_lines_ = 0;
};

// Highest byte address touched by a unified trace (for sizing the device).
uint64_t trace_span_bytes(const std::string& path);

std::unique_ptr<Workload> make_synthetic(const std::string& name, uint64_t logical_pages,
                                         uint64_t seed, double zipf_theta,
                                         double hot_frac, double hot_access,
                                         uint32_t req_pages);

}  // namespace ftl
