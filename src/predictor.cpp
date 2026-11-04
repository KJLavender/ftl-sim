#include "predictor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <stdexcept>

namespace ftl {

const char* FeatureExtractor::feature_name(int i) {
  static const char* names[kNumFeatures] = {
      "prev_lifetime", "first_write", "write_count",
      "prev2_lifetime", "region_heat", "req_pages_log2"};
  return names[i];
}

void FeatureExtractor::init(uint64_t logical_pages) {
  capacity_ = std::max<uint64_t>(1, logical_pages);
  count_.assign(logical_pages, 0);
  last_bucket_.assign(logical_pages, 0);
  region_heat_.assign((logical_pages >> kRegionShift) + 1, 0);
  next_decay_ = capacity_;
}

// log2(1 + 1024 * lifetime / capacity): 0 = instant, 10 = one capacity.
float FeatureExtractor::norm_log_lifetime(Time lifetime) const {
  double x = 1024.0 * static_cast<double>(lifetime) / static_cast<double>(capacity_);
  return static_cast<float>(std::log2(1.0 + x));
}

FeatureExtractor::Features FeatureExtractor::extract(const HostWrite& w) const {
  Features f{};
  const bool first = w.prev_write_time == kNever;
  f[0] = first ? -1.0f : norm_log_lifetime(w.now - w.prev_write_time);
  f[1] = first ? 1.0f : 0.0f;
  f[2] = count_[w.lpn];
  f[3] = last_bucket_[w.lpn] ? last_bucket_[w.lpn] - 1.0f : -1.0f;
  f[4] = static_cast<float>(std::log2(1.0 + region_heat_[w.lpn >> kRegionShift]));
  f[5] = static_cast<float>(std::log2(static_cast<double>(std::max(1u, w.req_pages))));
  return f;
}

void FeatureExtractor::decay(Time now) {
  // Halve all counters once per capacity written (an exponential window).
  for (auto& c : count_) c >>= 1;
  for (auto& h : region_heat_) h >>= 1;
  next_decay_ = now + capacity_;
}

void FeatureExtractor::update(const HostWrite& w) {
  if (w.now >= next_decay_) decay(w.now);
  if (count_[w.lpn] < 255) ++count_[w.lpn];
  uint16_t& heat = region_heat_[w.lpn >> kRegionShift];
  if (heat < UINT16_MAX) ++heat;
  if (w.prev_write_time != kNever) {
    // Store floor(norm log lifetime) + 1 in one byte; 0 means unknown.
    float b = norm_log_lifetime(w.now - w.prev_write_time);
    last_bucket_[w.lpn] = static_cast<uint8_t>(std::min(254.0f, std::floor(b)) + 1);
  }
}

uint64_t FeatureExtractor::state_bytes() const {
  return count_.size() + last_bucket_.size() + region_heat_.size() * sizeof(uint16_t);
}

void DecisionTree::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open model: " + path);
  std::string magic, version;
  size_t n = 0;
  int nf = 0;
  in >> magic >> version >> n >> nf;
  if (magic != "ftl-tree" || version != "v1") throw std::runtime_error("bad model header");
  if (nf != FeatureExtractor::kNumFeatures) {
    throw std::runtime_error("model feature count does not match extractor");
  }
  nodes_.resize(n);
  for (auto& nd : nodes_) {
    if (!(in >> nd.feature >> nd.threshold >> nd.left >> nd.right >> nd.value)) {
      throw std::runtime_error("truncated model file");
    }
  }
  if (nodes_.empty()) throw std::runtime_error("empty model");
}

int DecisionTree::predict(const FeatureExtractor::Features& x) const {
  int i = 0;
  while (nodes_[i].feature >= 0) {
    i = x[nodes_[i].feature] <= nodes_[i].threshold ? nodes_[i].left : nodes_[i].right;
  }
  return nodes_[i].value;
}

int DecisionTree::depth() const {
  std::function<int(int)> rec = [&](int i) -> int {
    if (nodes_[i].feature < 0) return 0;
    return 1 + std::max(rec(nodes_[i].left), rec(nodes_[i].right));
  };
  return rec(0);
}

MlPlacement::MlPlacement(const std::string& model_path, const std::string& gc_mode) {
  if (gc_mode != "age" && gc_mode != "sepbit" && gc_mode != "hybrid") {
    throw std::invalid_argument("ml gc mode must be age, sepbit or hybrid");
  }
  hybrid_ = gc_mode == "hybrid";
  sepbit_gc_ = gc_mode == "sepbit" || hybrid_;
  tree_.load(model_path);
}

void MlPlacement::init(uint64_t logical_pages) {
  capacity_ = std::max<uint64_t>(1, logical_pages);
  fx_.init(logical_pages);
}

int MlPlacement::host_stream(const HostWrite& w) {
  FeatureExtractor::Features x = fx_.extract(w);
  fx_.update(w);
  if (hybrid_) {
    // SepBIT's own test first: a short previous lifetime goes to stream 0.
    if (w.prev_write_time != kNever &&
        static_cast<double>(w.now - w.prev_write_time) < sepbit_.ell()) {
      return 0;
    }
    // Everything SepBIT would lump into one class is split by the tree.
    return std::clamp(tree_.predict(x), 1, kLifetimeClasses - 1);
  }
  int cls = tree_.predict(x);
  return std::clamp(cls, 0, kLifetimeClasses - 1);
}

int MlPlacement::as_sepbit_class(int stream) const {
  if (stream < kLifetimeClasses) return stream == 0 ? 0 : 1;  // host classes
  return stream - kLifetimeClasses + 2;                       // GC classes 2..5
}

int MlPlacement::gc_stream(Lpn lpn, Time now, Time data_time, int victim_stream) {
  if (!sepbit_gc_) return gc_age_stream(now, data_time, capacity_);
  int cls = sepbit_.gc_stream(lpn, now, data_time, as_sepbit_class(victim_stream));
  return kLifetimeClasses + cls - 2;
}

void MlPlacement::on_victim(int victim_stream, Time open_time, Time now) {
  if (sepbit_gc_) sepbit_.on_victim(as_sepbit_class(victim_stream), open_time, now);
}

}  // namespace ftl
