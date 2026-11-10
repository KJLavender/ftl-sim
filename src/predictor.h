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
// Feature 9 is SepBIT's current ell (same log scale, -1 before it is known),
// set by the caller, so a model can predict lifetime relative to ell.
class FeatureExtractor {
 public:
  static constexpr int kNumFeatures = 10;
  static constexpr int kWallGapFeature = 8;
  static constexpr int kEllFeature = 9;
  static constexpr int kRegionShift = 8;  // 256 pages = 1 MiB per region
  using Features = std::array<float, kNumFeatures>;
  static const char* feature_name(int i);

  void init(uint64_t logical_pages, bool track_wall_time = true);
  // Features of w, computed from state before w. Call update(w) afterwards.
  Features extract(const HostWrite& w) const;
  void update(const HostWrite& w);
  // Firmware memory needed for the feature state.
  uint64_t state_bytes() const;
  // Current SepBIT ell in host writes; values >= 1e299 mean not measured yet.
  void set_ell(double ell);
  // log2(1 + 1024 * lifetime / capacity): 0 = instant, 10 = one capacity.
  static float norm_log(double lifetime, uint64_t capacity);
  static constexpr double kEllUnknown = 1e299;

 private:
  float norm_log_lifetime(Time lifetime) const;
  void decay(Time now);

  uint64_t capacity_ = 1;
  std::vector<uint8_t> count_;        // decayed writes per LPN
  std::vector<uint8_t> last_bucket_;  // previous lifetime bucket, 0 = unknown
  std::vector<uint16_t> region_heat_; // decayed writes per region
  std::vector<uint32_t> last_time_s_; // wall clock of the last write (optional)
  Time next_decay_ = 0;
  float ell_log_ = -1.0f;
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
//   "ell":    same, split at the given multiples of ell.
//             These need a model that outputs a lifetime (ftl-gbm format).
//   "ellrel": the model predicts log2(lifetime / ell) directly, with the
//             current ell as a feature; grouped at the given splits.
//   "ellcls": the model is a class tree whose class is the host group.
class MlPlacement : public Placement {
 public:
  // ell_splits is used by gc_mode "ell" (ell2 / ell4 fix it to {} / {4, 16}).
  MlPlacement(const std::string& model_path, const std::string& gc_mode,
              std::vector<double> ell_splits = {4, 16});
  std::string name() const override {
    if (ell_mode_) return "ml-" + ell_name_;
    return hybrid_ ? "ml-hybrid" : sepbit_gc_ ? "ml-sepbitgc" : "ml";
  }
  int num_streams() const override { return host_streams_ + (sepbit_gc_ ? 4 : 3); }
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
  bool ell_mode_ = false;  // compare the predicted lifetime with ell
  // What the model outputs in the ell modes: lifetime on the capacity scale
  // (ell, ell2, ell4), log2(lifetime / ell) (ellrel), or the group (ellcls).
  enum class EllTarget { kAbsolute, kRelative, kGroup };
  EllTarget ell_target_ = EllTarget::kAbsolute;
  std::string ell_name_;
  std::vector<double> ell_splits_;
  // Host streams in use. Exact in the ell modes, so the GC free-block
  // threshold (streams + 2) matches the sepbit control with the same groups.
  int host_streams_ = kLifetimeClasses;
  uint64_t capacity_ = 1;
};

}  // namespace ftl
