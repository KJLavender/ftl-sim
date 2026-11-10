#pragma once

#include <cstdint>
#include <vector>

namespace ftl {

// Host write latency under foreground GC, as a single FIFO queue.
// Writes arrive at a fixed rate. The flash back end is lumped into one server
// that works on `dies` operations at a time, so each write occupies it for
//   (t_prog + copies * (t_read + t_prog) + erases * t_erase
//    + map_reads * t_read + map_writes * t_prog) / dies
// where copies / erases / map traffic are what the FTL did inside that write
// (GC runs in the write path when free blocks run out). No write buffer, no
// background GC: this is the worst case firmware tries to hide.
struct LatencyConfig {
  double t_read_us = 50;     // TLC page read
  double t_prog_us = 600;    // TLC page program
  double t_erase_us = 3000;  // block erase
  uint32_t dies = 8;
  // Arrival rate as a fraction of the no-GC peak (dies / t_prog).
  double load = 0;  // 0 = off
};

class LatencyModel {
 public:
  explicit LatencyModel(const LatencyConfig& cfg);

  void on_write(uint64_t copies, uint64_t erases, uint64_t map_reads, uint64_t map_writes);
  void reset_samples() {
    samples_.clear();
    busy_us_ = 0;
    max_copies_ = max_erases_ = 0;
  }

  uint64_t count() const { return samples_.size(); }
  double mean_us() const;
  double percentile_us(double p) const;  // p in [0, 100]
  double max_us() const;
  double busy_frac() const;  // back-end busy time / elapsed time
  // Most GC work any single write had to wait for.
  uint64_t max_copies() const { return max_copies_; }
  uint64_t max_erases() const { return max_erases_; }

 private:
  LatencyConfig cfg_;
  double gap_us_;        // time between arrivals
  double wait_us_ = 0;   // queueing delay of the next arrival (Lindley)
  double busy_us_ = 0;
  uint64_t max_copies_ = 0, max_erases_ = 0;
  std::vector<float> samples_;
};

}  // namespace ftl
