#include "latency.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ftl {

LatencyModel::LatencyModel(const LatencyConfig& cfg) : cfg_(cfg) {
  if (cfg.load <= 0 || cfg.dies == 0) throw std::invalid_argument("latency: load and dies must be > 0");
  gap_us_ = cfg.t_prog_us / cfg.dies / cfg.load;
}

void LatencyModel::on_write(uint64_t copies, uint64_t erases, uint64_t map_reads,
                            uint64_t map_writes) {
  const double work = cfg_.t_prog_us + copies * (cfg_.t_read_us + cfg_.t_prog_us) +
                      erases * cfg_.t_erase_us + map_reads * cfg_.t_read_us +
                      map_writes * cfg_.t_prog_us;
  const double service = work / cfg_.dies;
  samples_.push_back(static_cast<float>(wait_us_ + service));
  busy_us_ += service;
  max_copies_ = std::max(max_copies_, copies);
  max_erases_ = std::max(max_erases_, erases);
  wait_us_ = std::max(0.0, wait_us_ + service - gap_us_);
}

double LatencyModel::mean_us() const {
  if (samples_.empty()) return 0;
  return std::accumulate(samples_.begin(), samples_.end(), 0.0) / samples_.size();
}

double LatencyModel::percentile_us(double p) const {
  if (samples_.empty()) return 0;
  std::vector<float> v = samples_;
  const size_t k = std::min(v.size() - 1, static_cast<size_t>(std::ceil(p / 100 * v.size())) -
                                              (p > 0 ? 1 : 0));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

double LatencyModel::max_us() const {
  return samples_.empty() ? 0 : *std::max_element(samples_.begin(), samples_.end());
}

double LatencyModel::busy_frac() const {
  return samples_.empty() ? 0 : busy_us_ / (samples_.size() * gap_us_);
}

}  // namespace ftl
