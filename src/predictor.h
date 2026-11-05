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
// Features 6..8 are extras for testing whether more information helps:
// hour of day, position on the drive, and wall-clock time since the page was
// last written. The last one needs 4 more bytes per LPN, so it is only
// tracked when asked for.
class FeatureExtractor {
 public:
  static constexpr int kNumFeatures = 9;
  static constexpr int kWallGapFeature = 8;
  static constexpr int kRegionShift = 8;  // 256 pages = 1 MiB per region
  using Features = std::array<float, kNumFeatures>;
  static const char* feature_name(int i);

  void init(uint64_t logical_pages, bool track_wall_time = true);
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
  std::vector<uint32_t> last_time_s_; // wall clock of the last write (optional)
  Time next_decay_ = 0;
};

// Model exported by scripts/train_predictor.py. Two text formats:
//   ftl-tree v1 <num_nodes> <num_features>          one tree, leaves = class
//   <feature> <threshold> <left> <right> <class>     (one line per node)
// or a gradient-boosted ensemble whose leaves add up to a log lifetime:
//   ftl-gbm v1 <num_trees> <num_features> <base>
//   then per tree: <num_nodes> and node lines with a float leaf value
// feature = -1 marks a leaf; go left when x[feature] <= threshold.
class DecisionTree {
 public:
  void load(const std::string& path);
  int predict(const FeatureExtractor::Features& x) const;
  // Ensemble only: the predicted normalized log lifetime itself.
  float predict_value(const FeatureExtractor::Features& x) const;
  bool regression() const { return ensemble_; }
  size_t num_nodes() const;
  size_t num_trees() const { return trees_.size(); }
  int depth() const;
  bool uses_feature(int f) const;
  // Packed firmware size per node: feature (1B) + threshold (4B) + 2 children
  // (2B each) + class (1B), or a 4B leaf value for the ensemble.
  size_t firmware_bytes() const { return num_nodes() * (ensemble_ ? 13 : 10); }

 private:
  struct Node {
    int feature;
    float threshold;
    int left, right;
    float value;
  };
  using Tree = std::vector<Node>;
  static float walk(const Tree& t, const FeatureExtractor::Features& x);
  std::vector<Tree> trees_;
  bool ensemble_ = false;
  float base_ = 0.0f;
};

// Device-side learned placement: host writes go to the predicted lifetime
// class (streams 0..4). GC writes are regrouped either
//   "age":    by data age, same as FDP initial isolation, so fdp-initial with
//             zero noise is the perfect-host counterpart; or
//   "sepbit": with SepBIT's GC rules, so the comparison with SepBIT isolates
//             the host-side predictor. ML class 0 plays SepBIT's class 0.
//   "hybrid": SepBIT GC rules, and SepBIT's host test (previous lifetime < ell
//             -> stream 0) runs first; the tree only splits what SepBIT
//             would put in its single long-lived class into streams 1..4.
//   "ell2":   SepBIT with the predicted lifetime in place of the previous
//             one: predicted < ell -> stream 0, else stream 1. SepBIT GC.
//   "ell4":   same, but the rest is split at 4 ell and 16 ell (streams 1..3).
//             Both need a model that outputs a lifetime (ftl-gbm format).
class MlPlacement : public Placement {
 public:
  MlPlacement(const std::string& model_path, const std::string& gc_mode);
  std::string name() const override {
    if (ell_groups_) return ell_groups_ == 2 ? "ml-ell2" : "ml-ell4";
    return hybrid_ ? "ml-hybrid" : sepbit_gc_ ? "ml-sepbitgc" : "ml";
  }
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
  bool hybrid_ = false;  // SepBIT host test first, tree splits the rest
  int ell_groups_ = 0;   // 2 or 4: compare the predicted lifetime with ell
  uint64_t capacity_ = 1;
};

}  // namespace ftl
