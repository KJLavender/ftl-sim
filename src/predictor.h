#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "placement.h"

namespace ftl {

// Online features for a host write, computable inside the SSD.
// Per-LPN state is 2 bytes (decayed write count + last lifetime bucket);
// per-region state is 2 bytes per 256 pages. Lifetimes are normalized by
// user capacity so a model transfers across device sizes.
class FeatureExtractor {
 public:
  static constexpr int kNumFeatures = 6;
  static constexpr int kRegionShift = 8;  // 256 pages = 1 MiB per region
  using Features = std::array<float, kNumFeatures>;
  static const char* feature_name(int i);

  void init(uint64_t logical_pages);
  // Features of w, computed from state before w. Call update(w) afterwards.
  Features extract(const HostWrite& w) const;
  void update(const HostWrite& w);
  // Firmware memory needed for the feature state.
  uint64_t state_bytes() const;

 private:
  float norm_log_lifetime(Time lifetime) const;
  void decay(Time now);

  uint64_t capacity_ = 1;
  std::vector<uint8_t> count_;        // decayed writes per LPN
  std::vector<uint8_t> last_bucket_;  // previous lifetime bucket, 0 = unknown
  std::vector<uint16_t> region_heat_; // decayed writes per region
  Time next_decay_ = 0;
};

// Decision tree exported by scripts/train_predictor.py. Text format:
//   ftl-tree v1 <num_nodes> <num_features>
//   <feature> <threshold> <left> <right> <class>     (one line per node)
// feature = -1 marks a leaf; go left when x[feature] <= threshold.
class DecisionTree {
 public:
  void load(const std::string& path);
  int predict(const FeatureExtractor::Features& x) const;
  size_t num_nodes() const { return nodes_.size(); }
  int depth() const;
  // Packed firmware size: feature (1B) + threshold (4B) + 2 children (2B each)
  // + class (1B) per node.
  size_t firmware_bytes() const { return nodes_.size() * 10; }

 private:
  struct Node {
    int feature;
    float threshold;
    int left, right;
    int value;
  };
  std::vector<Node> nodes_;
};

// Device-side learned placement: host writes go to the predicted lifetime
// class (streams 0..4). GC writes are regrouped either
//   "age":    by data age, same as FDP initial isolation, so fdp-initial with
//             zero noise is the perfect-host counterpart; or
//   "sepbit": with SepBIT's GC rules, so the comparison with SepBIT isolates
//             the host-side predictor. ML class 0 plays SepBIT's class 0.
class MlPlacement : public Placement {
 public:
  MlPlacement(const std::string& model_path, const std::string& gc_mode);
  std::string name() const override { return sepbit_gc_ ? "ml-sepbitgc" : "ml"; }
  int num_streams() const override { return kLifetimeClasses + (sepbit_gc_ ? 4 : 3); }
  void init(uint64_t logical_pages) override;
  int host_stream(const HostWrite& w) override;
  int gc_stream(Lpn lpn, Time now, Time data_time, int victim_stream) override;
  void on_victim(int victim_stream, Time open_time, Time now) override;

  const DecisionTree& tree() const { return tree_; }
  const FeatureExtractor& features() const { return fx_; }

 private:
  // Maps one of our streams to the SepBIT class it corresponds to.
  int as_sepbit_class(int stream) const;

  DecisionTree tree_;
  FeatureExtractor fx_;
  SepBitPlacement sepbit_;
  bool sepbit_gc_;
  uint64_t capacity_ = 1;
};

}  // namespace ftl
