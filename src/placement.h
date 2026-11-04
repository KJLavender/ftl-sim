#pragma once

#include <deque>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "types.h"

namespace ftl {

// One host page write as seen by the placement policy.
struct HostWrite {
  Lpn lpn;
  Time now;
  Time prev_write_time;  // overwritten version's write time, kNever if none
  uint32_t req_pages;    // size of the host request this page belongs to
};

// Decides which write stream (open block) a page goes to. Pages in the same
// stream share blocks, so a good placement groups pages that die together.
class Placement {
 public:
  virtual ~Placement() = default;
  virtual std::string name() const = 0;
  virtual int num_streams() const = 0;
  virtual void init(uint64_t logical_pages) { (void)logical_pages; }

  virtual int host_stream(const HostWrite& w) = 0;

  // GC migration of a valid page. data_time is its original host write time;
  // victim_stream is the stream of the block being collected.
  virtual int gc_stream(Lpn lpn, Time now, Time data_time, int victim_stream) = 0;

  // Called once per GC victim before its pages are migrated.
  virtual void on_victim(int victim_stream, Time open_time, Time now) {
    (void)victim_stream; (void)open_time; (void)now;
  }
};

// No separation: host and GC writes share a single stream.
class NoPlacement : public Placement {
 public:
  std::string name() const override { return "none"; }
  int num_streams() const override { return 1; }
  int host_stream(const HostWrite&) override { return 0; }
  int gc_stream(Lpn, Time, Time, int) override { return 0; }
};

// Host writes in stream 0, GC writes in stream 1. GC-surviving data is
// already "colder" than fresh writes, so keeping them apart helps.
class SepGcPlacement : public Placement {
 public:
  std::string name() const override { return "sepgc"; }
  int num_streams() const override { return 2; }
  int host_stream(const HostWrite&) override { return 0; }
  int gc_stream(Lpn, Time, Time, int) override { return 1; }
};

// Rule-based baseline: per-LPN write counter with periodic halving.
// Hot (count >= threshold) -> stream 0, cold host writes -> stream 1,
// GC writes -> stream 1 (survivors are treated as cold).
class WriteCountPlacement : public Placement {
 public:
  explicit WriteCountPlacement(uint8_t threshold = 2) : threshold_(threshold) {}
  std::string name() const override { return "wcount"; }
  int num_streams() const override { return 2; }
  void init(uint64_t logical_pages) override;
  int host_stream(const HostWrite& w) override;
  int gc_stream(Lpn, Time, Time, int) override { return 1; }

 private:
  uint8_t threshold_;
  std::vector<uint8_t> count_;
  uint64_t decay_period_ = 0;
  Time next_decay_ = 0;
};

// SepBIT (Wang et al., FAST'22): infer block invalidation time.
//   Host writes: class 0 if the overwritten version lived shorter than
//   ell (avg lifespan of recently collected class-0 blocks), else class 1.
//   GC writes: from a class-0 victim -> class 2; otherwise by data age:
//   < 4*ell -> 3, < 16*ell -> 4, else 5.
class SepBitPlacement : public Placement {
 public:
  std::string name() const override { return "sepbit"; }
  int num_streams() const override { return 6; }
  int host_stream(const HostWrite& w) override;
  int gc_stream(Lpn lpn, Time now, Time data_time, int victim_stream) override;
  void on_victim(int victim_stream, Time open_time, Time now) override;

  double ell() const { return ell_; }

 private:
  static constexpr size_t kWindow = 16;  // recent class-0 victims to average
  double ell_ = 1e300;                   // infinite until first measurement
  std::deque<Time> lifespans_;
  Time lifespan_sum_ = 0;
};

// Future knowledge for oracle-style placements: death[t] is the time at which
// the data written at host time t is overwritten or trimmed (kNever if never).
using DeathTimes = std::vector<Time>;

// Buckets a (remaining) lifetime into 5 classes relative to user capacity C:
// < C/8, < C/2, < 2C, < 8C, longer or never.
constexpr int kLifetimeClasses = 5;
int lifetime_class(Time lifetime, uint64_t capacity);

// GC regrouping by data age (< C/2, < 2C, older) into streams
// kLifetimeClasses .. kLifetimeClasses+2. Shared by FDP initial isolation
// and the learned placement.
int gc_age_stream(Time now, Time data_time, uint64_t capacity);

// Upper bound: knows every page's true lifetime. Host and GC writes go to
// the class of their remaining lifetime, so blocks hold data that dies together.
class OraclePlacement : public Placement {
 public:
  OraclePlacement(std::shared_ptr<const DeathTimes> death, uint64_t capacity)
      : death_(std::move(death)), capacity_(capacity) {}
  std::string name() const override { return "oracle"; }
  int num_streams() const override { return kLifetimeClasses; }
  int host_stream(const HostWrite& w) override;
  int gc_stream(Lpn lpn, Time now, Time data_time, int victim_stream) override;

 private:
  int remaining_class(Time written, Time now) const;
  std::shared_ptr<const DeathTimes> death_;
  uint64_t capacity_;
};

// NVMe FDP model: the host tags each write with a reclaim unit handle (RUH)
// derived from the true lifetime class, optionally degraded:
//   mode "random": with probability noise, the hint is a random class;
//   mode "coarse": the host only tells short-lived (< C/2) from the rest.
// GC semantics (both exist in the FDP spec):
//   persistent: GC keeps data inside its RUH;
//   initial:    GC data leaves the RUH and the device regroups it by age
//               (< C/2, < 2C, older), so the SSD can correct bad hints.
class FdpPlacement : public Placement {
 public:
  FdpPlacement(std::shared_ptr<const DeathTimes> death, uint64_t capacity,
               double noise, const std::string& mode, bool persistent, uint64_t seed);
  std::string name() const override { return persistent_ ? "fdp" : "fdp-initial"; }
  int num_streams() const override { return kLifetimeClasses + (persistent_ ? 0 : 3); }
  int host_stream(const HostWrite& w) override;
  int gc_stream(Lpn lpn, Time now, Time data_time, int victim_stream) override;

 private:
  std::shared_ptr<const DeathTimes> death_;
  uint64_t capacity_;
  double noise_;
  bool coarse_;
  bool persistent_;
  std::mt19937_64 rng_;
};

struct PlacementOptions {
  std::shared_ptr<const DeathTimes> death;  // required by oracle / fdp
  uint64_t logical_pages = 0;
  double hint_noise = 0.0;
  std::string hint_mode = "random";  // random | coarse
  bool fdp_persistent = true;
  std::string model_path;  // required by ml
  std::string ml_gc = "age";  // ml: age | sepbit
  uint64_t seed = 1;
};

bool placement_needs_future(const std::string& name);
std::unique_ptr<Placement> make_placement(const std::string& name,
                                          const PlacementOptions& opt = {});

}  // namespace ftl
