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
      "prev2_lifetime", "region_heat", "req_pages_log2",
      "hour", "lba_frac", "prev_gap_s_log2", "ell_log"};
  return names[i];
}

float FeatureExtractor::norm_log(double lifetime, uint64_t capacity) {
  const double x = 1024.0 * lifetime / static_cast<double>(std::max<uint64_t>(1, capacity));
  return static_cast<float>(std::log2(1.0 + x));
}

void FeatureExtractor::set_ell(double ell) {
  ell_log_ = ell >= kEllUnknown ? -1.0f : norm_log(ell, capacity_);
}

void FeatureExtractor::init(uint64_t logical_pages, bool track_wall_time) {
  capacity_ = std::max<uint64_t>(1, logical_pages);
  count_.assign(logical_pages, 0);
  last_bucket_.assign(logical_pages, 0);
  region_heat_.assign((logical_pages >> kRegionShift) + 1, 0);
  last_time_s_.assign(track_wall_time ? logical_pages : 0, 0);
  next_decay_ = capacity_;
}

float FeatureExtractor::norm_log_lifetime(Time lifetime) const {
  return norm_log(static_cast<double>(lifetime), capacity_);
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
  f[6] = static_cast<float>((w.time_s / 3600) % 24);
  f[7] = static_cast<float>(static_cast<double>(w.lpn) / static_cast<double>(capacity_));
  f[8] = -1.0f;
  if (!first && !last_time_s_.empty() && w.time_s >= last_time_s_[w.lpn]) {
    f[8] = static_cast<float>(std::log2(1.0 + (w.time_s - last_time_s_[w.lpn])));
  }
  f[9] = ell_log_;
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
  if (!last_time_s_.empty()) last_time_s_[w.lpn] = w.time_s;
  if (w.prev_write_time != kNever) {
    // Store floor(norm log lifetime) + 1 in one byte; 0 means unknown.
    float b = norm_log_lifetime(w.now - w.prev_write_time);
    last_bucket_[w.lpn] = static_cast<uint8_t>(std::min(254.0f, std::floor(b)) + 1);
  }
}

uint64_t FeatureExtractor::state_bytes() const {
  return count_.size() + last_bucket_.size() + region_heat_.size() * sizeof(uint16_t) +
         last_time_s_.size() * sizeof(uint32_t);
}

void DecisionTree::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open model: " + path);
  std::string magic, version;
  size_t n = 0;
  int nf = 0;
  in >> magic >> version >> n >> nf;
  if ((magic != "ftl-tree" && magic != "ftl-gbm") || version != "v1") {
    throw std::runtime_error("bad model header");
  }
  if (nf > FeatureExtractor::kNumFeatures) {
    throw std::runtime_error("model uses more features than the extractor has");
  }
  ensemble_ = magic == "ftl-gbm";
  size_t num_trees = 1;
  if (ensemble_) {
    num_trees = n;
    in >> base_;
  }
  trees_.assign(num_trees, {});
  for (Tree& t : trees_) {
    if (ensemble_) in >> n;
    t.resize(n);
    for (Node& nd : t) {
      if (!(in >> nd.feature >> nd.threshold >> nd.left >> nd.right >> nd.value)) {
        throw std::runtime_error("truncated model file");
      }
      if (nd.feature >= nf) throw std::runtime_error("bad feature index in model");
    }
    if (t.empty()) throw std::runtime_error("empty tree in model");
  }
  if (trees_.empty()) throw std::runtime_error("empty model");
}

float DecisionTree::walk(const Tree& t, const FeatureExtractor::Features& x) {
  int i = 0;
  while (t[i].feature >= 0) {
    i = x[t[i].feature] <= t[i].threshold ? t[i].left : t[i].right;
  }
  return t[i].value;
}

float DecisionTree::predict_value(const FeatureExtractor::Features& x) const {
  float sum = base_;
  for (const Tree& t : trees_) sum += walk(t, x);
  return sum;
}

int DecisionTree::predict(const FeatureExtractor::Features& x) const {
  if (!ensemble_) return static_cast<int>(walk(trees_[0], x));
  const float sum = predict_value(x);
  // Class boundaries C/8, C/2, 2C, 8C in the normalized log lifetime
  // log2(1 + 1024 * L / C), same as lifetime_class().
  static const float bounds[] = {7.0112f, 9.0028f, 11.0007f, 13.0002f};
  int cls = 0;
  while (cls < 4 && sum >= bounds[cls]) ++cls;
  return cls;
}

size_t DecisionTree::num_nodes() const {
  size_t n = 0;
  for (const Tree& t : trees_) n += t.size();
  return n;
}

int DecisionTree::depth() const {
  int best = 0;
  for (const Tree& t : trees_) {
    std::function<int(int)> rec = [&](int i) -> int {
      if (t[i].feature < 0) return 0;
      return 1 + std::max(rec(t[i].left), rec(t[i].right));
    };
    best = std::max(best, rec(0));
  }
  return best;
}

bool DecisionTree::uses_feature(int f) const {
  for (const Tree& t : trees_) {
    for (const Node& nd : t) {
      if (nd.feature == f) return true;
    }
  }
  return false;
}

MlPlacement::MlPlacement(const std::string& model_path, const std::string& gc_mode,
                         std::vector<double> ell_splits) {
  if (gc_mode != "age" && gc_mode != "sepbit" && gc_mode != "hybrid" &&
      gc_mode != "ell2" && gc_mode != "ell4" && gc_mode != "ell" &&
      gc_mode != "ellrel" && gc_mode != "ellcls") {
    throw std::invalid_argument(
        "ml gc mode must be age, sepbit, hybrid, ell2, ell4, ell, ellrel or ellcls");
  }
  ell_target_ = gc_mode == "ellrel" ? EllTarget::kRelative
              : gc_mode == "ellcls" ? EllTarget::kGroup
                                    : EllTarget::kAbsolute;
  hybrid_ = gc_mode == "hybrid";
  ell_mode_ = gc_mode.rfind("ell", 0) == 0;
  if (ell_mode_) {
    ell_name_ = gc_mode;
    ell_splits_ = gc_mode == "ell2" ? std::vector<double>{}
                : gc_mode == "ell4" ? std::vector<double>{4, 16}
                                    : std::move(ell_splits);
    host_streams_ = 2 + static_cast<int>(ell_splits_.size());
  }
  sepbit_gc_ = gc_mode != "age";
  tree_.load(model_path);
  if (ell_target_ == EllTarget::kGroup) {
    if (tree_.regression()) {
      throw std::invalid_argument("ellcls needs a class tree (train with --target ellgroup)");
    }
  } else if (ell_mode_ && !tree_.regression()) {
    throw std::invalid_argument("ell modes need a lifetime model (train with --raw or gbm)");
  }
}

void MlPlacement::init(uint64_t logical_pages) {
  capacity_ = std::max<uint64_t>(1, logical_pages);
  fx_.init(logical_pages, tree_.uses_feature(FeatureExtractor::kWallGapFeature));
  predicted_life_.assign(model_gc_ ? logical_pages : 0, 0.0f);
}

void MlPlacement::set_model_gc(bool on) {
  if (on && (!ell_mode_ || ell_target_ == EllTarget::kGroup)) {
    throw std::invalid_argument("model GC needs ell, ell2, ell4 or ellrel");
  }
  model_gc_ = on;
}

int MlPlacement::host_stream(const HostWrite& w) {
  fx_.set_ell(sepbit_.ell());
  FeatureExtractor::Features x = fx_.extract(w);
  fx_.update(w);
  if (ell_mode_) {
    const double ell = sepbit_.ell();
    if (ell_target_ != EllTarget::kAbsolute) {
      // Before the first measurement SepBIT treats everything as short-lived.
      if (ell >= FeatureExtractor::kEllUnknown) {
        if (model_gc_) predicted_life_[w.lpn] = 0.0f;
        return 0;
      }
      if (ell_target_ == EllTarget::kGroup) {
        return std::clamp(tree_.predict(x), 0, host_streams_ - 1);
      }
      // Model output is log2(lifetime / ell).
      const double life = std::exp2(tree_.predict_value(x)) * ell;
      if (model_gc_) predicted_life_[w.lpn] = static_cast<float>(life);
      return SepBitPlacement::ell_group(life, ell, ell_splits_);
    }
    // Back from log2(1 + 1024 L / C) to host writes, then compare with ell.
    const double v = tree_.predict_value(x);
    const double life = (std::exp2(v) - 1.0) * static_cast<double>(capacity_) / 1024.0;
    if (model_gc_) predicted_life_[w.lpn] = static_cast<float>(life);
    return SepBitPlacement::ell_group(life, ell, ell_splits_);
  }
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
  if (stream < host_streams_) return stream == 0 ? 0 : 1;  // host classes
  return stream - host_streams_ + 2;                       // GC classes 2..5
}

int MlPlacement::gc_stream(Lpn lpn, Time now, Time data_time, int victim_stream) {
  if (!sepbit_gc_) return gc_age_stream(now, data_time, capacity_);
  if (model_gc_ && as_sepbit_class(victim_stream) != 0) {
    // Same classes as SepBIT's age rule, but with the expected remaining life.
    const double age = static_cast<double>(now - data_time);
    const double predicted = predicted_life_[lpn];
    const double remaining = predicted > age ? predicted - age : age;
    const double ell = sepbit_.ell();
    const int cls = remaining < 4 * ell ? 3 : remaining < 16 * ell ? 4 : 5;
    return host_streams_ + cls - 2;
  }
  int cls = sepbit_.gc_stream(lpn, now, data_time, as_sepbit_class(victim_stream));
  return host_streams_ + cls - 2;
}

void MlPlacement::on_victim(int victim_stream, Time open_time, Time now) {
  if (sepbit_gc_) sepbit_.on_victim(as_sepbit_class(victim_stream), open_time, now);
}

}  // namespace ftl
