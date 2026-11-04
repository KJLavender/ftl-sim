#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "config.h"
#include "gc_policy.h"
#include "nand.h"
#include "placement.h"

namespace ftl {

struct Stats {
  uint64_t host_writes = 0;  // pages
  uint64_t host_reads = 0;
  uint64_t host_trims = 0;
  uint64_t nand_writes = 0;  // host + GC programs
  uint64_t gc_copies = 0;
  uint64_t gc_count = 0;     // victims collected
  uint64_t wl_copies = 0;    // pages moved by static wear leveling
  uint64_t wl_count = 0;     // blocks relocated by static wear leveling
  uint64_t erases = 0;
  uint64_t unmapped_reads = 0;

  double waf() const {
    return host_writes ? static_cast<double>(nand_writes) / host_writes : 0.0;
  }
  double avg_valid_per_victim() const {
    return gc_count ? static_cast<double>(gc_copies) / gc_count : 0.0;
  }
};

// Page-level mapping FTL with pluggable GC victim policy and data placement.
class Ftl {
 public:
  Ftl(const FtlConfig& cfg, std::unique_ptr<VictimPolicy> victim,
      std::unique_ptr<Placement> placement);
  explicit Ftl(const FtlConfig& cfg);

  void write(Lpn lpn, uint32_t req_pages = 1);
  void read(Lpn lpn);
  void trim(Lpn lpn);

  uint64_t logical_pages() const { return logical_pages_; }
  // User capacity in pages for a config: physical / (1 + OP).
  static uint64_t logical_pages_for(const FtlConfig& cfg);
  Time now() const { return now_; }
  const Stats& stats() const { return stats_; }
  void reset_stats() { stats_ = Stats{}; }
  const Nand& nand() const { return nand_; }
  const Placement& placement() const { return *placement_; }
  const VictimPolicy& victim_policy() const { return *victim_; }
  uint32_t gc_free_threshold() const { return gc_threshold_; }
  size_t free_blocks() const { return free_.size(); }
  // physical / logical after removing space GC can never use: the free-block
  // watermark and the unfilled part of open blocks (~half a block each).
  double effective_alpha() const;

  // Full consistency check of L2P vs. OOB and per-block counters. Throws.
  void check_invariants() const;

 private:
  Ppn append(int stream, Lpn lpn, uint64_t tag, Time data_time);
  BlockId allocate_block(bool most_worn = false);
  void ensure_free_space();
  void collect_one();
  void maybe_static_wear_level();
  // Moves every valid page out of the block, erases it, returns pages moved.
  // forced_stream >= 0 bypasses the placement policy.
  uint32_t relocate_and_erase(BlockId victim, int forced_stream = -1);
  void unmap(Lpn lpn);

  FtlConfig cfg_;
  Nand nand_;
  std::unique_ptr<VictimPolicy> victim_;
  std::unique_ptr<Placement> placement_;
  uint64_t logical_pages_;
  uint32_t gc_threshold_;

  std::vector<Ppn> l2p_;
  std::vector<uint64_t> shadow_;  // latest tag per LPN, 0 = unmapped (verify only)
  std::vector<BlockId> open_;     // open block per stream
  std::deque<BlockId> free_;
  bool in_gc_ = false;
  Time now_ = 0;
  uint64_t tag_seq_ = 0;
  uint64_t open_seq_ = 0;
  uint32_t gc_since_wl_check_ = 0;
  int wl_stream_ = -1;  // internal stream for static WL, -1 if disabled
  Stats stats_;
};

}  // namespace ftl
