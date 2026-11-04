#include "workload.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ftl {

SequentialWorkload::SequentialWorkload(uint64_t logical_pages, uint32_t req_pages)
    : n_(logical_pages), req_pages_(req_pages) {}

bool SequentialWorkload::next(IoRequest& req) {
  req.op = Op::kWrite;
  req.lpn = static_cast<Lpn>(pos_);
  req.pages = static_cast<uint32_t>(std::min<uint64_t>(req_pages_, n_ - pos_));
  pos_ += req.pages;
  if (pos_ >= n_) pos_ = 0;
  return true;
}

UniformWorkload::UniformWorkload(uint64_t logical_pages, uint64_t seed)
    : rng_(seed), dist_(0, logical_pages - 1) {}

bool UniformWorkload::next(IoRequest& req) {
  req.op = Op::kWrite;
  req.lpn = static_cast<Lpn>(dist_(rng_));
  req.pages = 1;
  return true;
}

HotColdWorkload::HotColdWorkload(uint64_t logical_pages, double hot_frac,
                                 double hot_access, uint64_t seed)
    : rng_(seed),
      hot_pages_(std::max<uint64_t>(1, static_cast<uint64_t>(logical_pages * hot_frac))),
      n_(logical_pages),
      hot_access_(hot_access) {}

bool HotColdWorkload::next(IoRequest& req) {
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  req.op = Op::kWrite;
  req.pages = 1;
  if (coin(rng_) < hot_access_ || hot_pages_ == n_) {
    req.lpn = static_cast<Lpn>(std::uniform_int_distribution<uint64_t>(0, hot_pages_ - 1)(rng_));
  } else {
    req.lpn = static_cast<Lpn>(
        std::uniform_int_distribution<uint64_t>(hot_pages_, n_ - 1)(rng_));
  }
  return true;
}

ZipfWorkload::ZipfWorkload(uint64_t logical_pages, double theta, uint64_t seed)
    : rng_(seed), cdf_(logical_pages), rank_to_lpn_(logical_pages) {
  double sum = 0.0;
  for (uint64_t i = 0; i < logical_pages; ++i) {
    sum += 1.0 / std::pow(static_cast<double>(i + 1), theta);
    cdf_[i] = sum;
  }
  for (auto& c : cdf_) c /= sum;
  std::iota(rank_to_lpn_.begin(), rank_to_lpn_.end(), 0);
  std::shuffle(rank_to_lpn_.begin(), rank_to_lpn_.end(), rng_);
}

bool ZipfWorkload::next(IoRequest& req) {
  double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
  size_t rank = std::lower_bound(cdf_.begin(), cdf_.end(), u) - cdf_.begin();
  if (rank >= cdf_.size()) rank = cdf_.size() - 1;
  req.op = Op::kWrite;
  req.lpn = rank_to_lpn_[rank];
  req.pages = 1;
  return true;
}

TraceWorkload::TraceWorkload(const std::string& path, uint64_t logical_pages,
                             uint32_t page_size)
    : in_(path), n_(logical_pages), page_size_(page_size) {
  if (!in_) throw std::runtime_error("cannot open trace: " + path);
}

bool TraceWorkload::next_raw(Op& op, uint64_t& offset, uint64_t& bytes) {
  std::string line;
  while (std::getline(in_, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string ts, ops, off, size;
    if (!std::getline(ss, ts, ',') || !std::getline(ss, ops, ',') ||
        !std::getline(ss, off, ',') || !std::getline(ss, size, ',')) {
      ++bad_lines_;
      continue;
    }
    try {
      offset = std::stoull(off);
      bytes = std::stoull(size);
    } catch (const std::exception&) {
      ++bad_lines_;  // header line or garbage
      continue;
    }
    if (bytes == 0) continue;
    try {
      last_time_s_ = static_cast<uint32_t>(std::stoull(ts) / 1000000) + time_offset_s_;
    } catch (const std::exception&) {
      last_time_s_ = time_offset_s_;
    }
    switch (ops.empty() ? '?' : ops[0]) {
      case 'R': case 'r': op = Op::kRead; break;
      case 'W': case 'w': op = Op::kWrite; break;
      case 'T': case 't': op = Op::kTrim; break;
      default: ++bad_lines_; continue;
    }
    return true;
  }
  return false;
}

bool TraceWorkload::next(IoRequest& req) {
  uint64_t offset, bytes;
  if (!next_raw(req.op, offset, bytes)) return false;
  uint64_t first = offset / page_size_;
  uint64_t last = (offset + bytes - 1) / page_size_;
  if (last >= n_) ++wrapped_;
  req.lpn = static_cast<Lpn>(first % n_);
  req.pages = static_cast<uint32_t>(last - first + 1);
  req.time_s = last_time_s_;
  return true;
}

uint64_t trace_span_bytes(const std::string& path) {
  TraceWorkload tw(path, 1, 1);
  uint64_t span = 0, offset, bytes;
  Op op;
  while (tw.next_raw(op, offset, bytes)) span = std::max(span, offset + bytes);
  return span;
}

std::unique_ptr<Workload> make_synthetic(const std::string& name, uint64_t logical_pages,
                                         uint64_t seed, double zipf_theta,
                                         double hot_frac, double hot_access,
                                         uint32_t req_pages) {
  if (name == "seq") return std::make_unique<SequentialWorkload>(logical_pages, req_pages);
  if (name == "uniform") return std::make_unique<UniformWorkload>(logical_pages, seed);
  if (name == "hotcold") {
    return std::make_unique<HotColdWorkload>(logical_pages, hot_frac, hot_access, seed);
  }
  if (name == "zipf") return std::make_unique<ZipfWorkload>(logical_pages, zipf_theta, seed);
  throw std::invalid_argument("unknown workload: " + name);
}

}  // namespace ftl
