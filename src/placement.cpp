#include "placement.h"

#include "predictor.h"

#include <stdexcept>

namespace ftl {

void WriteCountPlacement::init(uint64_t logical_pages) {
  count_.assign(logical_pages, 0);
  decay_period_ = logical_pages;  // halve all counters once per capacity written
  next_decay_ = decay_period_;
}

int WriteCountPlacement::host_stream(const HostWrite& w) {
  if (w.now >= next_decay_) {
    for (auto& c : count_) c >>= 1;
    next_decay_ = w.now + decay_period_;
  }
  if (count_[w.lpn] < 255) ++count_[w.lpn];
  return count_[w.lpn] >= threshold_ ? 0 : 1;
}

int SepBitPlacement::host_stream(const HostWrite& w) {
  const int last = host_groups_ - 1;
  if (w.prev_write_time == kNever) return last;  // first write: no lifespan known
  double lifespan = static_cast<double>(w.now - w.prev_write_time);
  if (lifespan < ell_) return 0;
  if (host_groups_ == 2) return 1;
  return lifespan < 4 * ell_ ? 1 : lifespan < 16 * ell_ ? 2 : 3;
}

int SepBitPlacement::gc_stream(Lpn, Time now, Time data_time, int victim_stream) {
  const int gc = host_groups_;  // first GC stream
  if (victim_stream == 0) return gc;
  double age = static_cast<double>(now - data_time);
  if (age < 4 * ell_) return gc + 1;
  if (age < 16 * ell_) return gc + 2;
  return gc + 3;
}

void SepBitPlacement::on_victim(int victim_stream, Time open_time, Time now) {
  if (victim_stream != 0) return;
  Time lifespan = now - open_time;
  lifespans_.push_back(lifespan);
  lifespan_sum_ += lifespan;
  if (lifespans_.size() > kWindow) {
    lifespan_sum_ -= lifespans_.front();
    lifespans_.pop_front();
  }
  ell_ = static_cast<double>(lifespan_sum_) / lifespans_.size();
}

int lifetime_class(Time lifetime, uint64_t capacity) {
  if (lifetime == kNever) return kLifetimeClasses - 1;
  const double c = static_cast<double>(capacity);
  const double l = static_cast<double>(lifetime);
  if (l < c / 8) return 0;
  if (l < c / 2) return 1;
  if (l < 2 * c) return 2;
  if (l < 8 * c) return 3;
  return 4;
}

int gc_age_stream(Time now, Time data_time, uint64_t capacity) {
  const double age = static_cast<double>(now - data_time);
  const double c = static_cast<double>(capacity);
  if (age < c / 2) return kLifetimeClasses;
  if (age < 2 * c) return kLifetimeClasses + 1;
  return kLifetimeClasses + 2;
}

int OraclePlacement::remaining_class(Time written, Time now) const {
  Time death = (*death_)[written];
  if (death == kNever) return kLifetimeClasses - 1;
  return lifetime_class(death > now ? death - now : 0, capacity_);
}

int OraclePlacement::host_stream(const HostWrite& w) {
  return remaining_class(w.now, w.now);
}

int OraclePlacement::gc_stream(Lpn, Time now, Time data_time, int) {
  return remaining_class(data_time, now);
}

FdpPlacement::FdpPlacement(std::shared_ptr<const DeathTimes> death, uint64_t capacity,
                           double noise, const std::string& mode, bool persistent,
                           uint64_t seed)
    : death_(std::move(death)),
      capacity_(capacity),
      noise_(noise),
      persistent_(persistent),
      rng_(seed) {
  if (mode != "random" && mode != "coarse") {
    throw std::invalid_argument("hint mode must be random or coarse");
  }
  coarse_ = mode == "coarse";
}

int FdpPlacement::host_stream(const HostWrite& w) {
  Time death = (*death_)[w.now];
  int truth = lifetime_class(death == kNever ? kNever : death - w.now, capacity_);
  if (coarse_) truth = truth <= 1 ? 0 : kLifetimeClasses - 1;
  if (std::uniform_real_distribution<double>(0.0, 1.0)(rng_) >= noise_) return truth;
  return std::uniform_int_distribution<int>(0, kLifetimeClasses - 1)(rng_);
}

int FdpPlacement::gc_stream(Lpn, Time now, Time data_time, int victim_stream) {
  if (persistent_) return victim_stream;
  return gc_age_stream(now, data_time, capacity_);
}

bool placement_needs_future(const std::string& name) {
  return name == "oracle" || name == "fdp";
}

std::unique_ptr<Placement> make_placement(const std::string& name,
                                          const PlacementOptions& opt) {
  if (placement_needs_future(name) && !opt.death) {
    throw std::invalid_argument(name + " placement needs recorded death times");
  }
  if (name == "none") return std::make_unique<NoPlacement>();
  if (name == "sepgc") return std::make_unique<SepGcPlacement>();
  if (name == "wcount") return std::make_unique<WriteCountPlacement>();
  if (name == "sepbit") return std::make_unique<SepBitPlacement>();
  if (name == "sepbit4") return std::make_unique<SepBitPlacement>(4);
  if (name == "oracle") return std::make_unique<OraclePlacement>(opt.death, opt.logical_pages);
  if (name == "fdp") {
    return std::make_unique<FdpPlacement>(opt.death, opt.logical_pages, opt.hint_noise,
                                          opt.hint_mode, opt.fdp_persistent, opt.seed);
  }
  if (name == "ml") {
    if (opt.model_path.empty()) throw std::invalid_argument("ml placement needs --model");
    return std::make_unique<MlPlacement>(opt.model_path, opt.ml_gc);
  }
  throw std::invalid_argument("unknown placement: " + name);
}

}  // namespace ftl
