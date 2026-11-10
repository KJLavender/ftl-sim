#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "ftl.h"
#include "workload.h"

namespace ftl {

// Applies one request page by page. Addresses wrap at the logical capacity.
void apply(Ftl& ftl, const IoRequest& req);

struct RunOptions {
  bool verify_reads = false;  // read a random LPN after every request
  uint64_t check_every = 0;   // run check_invariants every N requests (0 = off)
  uint64_t seed = 1;
};

// Drives the FTL until `host_writes` more host pages have been written
// or the workload ends. Returns the number of host pages written.
uint64_t run(Ftl& ftl, Workload& wl, uint64_t host_writes, const RunOptions& opt = {});

// Writes every logical page once, in order (preconditioning).
void fill_sequential(Ftl& ftl);

// A whole run materialized as page-level ops, so every placement sees the
// exact same sequence and oracle-style placements can look into the future.
struct PageOp {
  Op op;
  Lpn lpn;
  uint32_t req_pages;
  uint32_t time_s;
};
struct Recording {
  std::vector<PageOp> ops;
  size_t measure_from = 0;  // stats are reset before this op
  uint64_t writes = 0;
  uint64_t trace_wrapped = 0;
  void add(const IoRequest& req, uint64_t logical_pages);
};

// fill (optional sequential pass) -> warmup_writes -> measure_writes host pages.
Recording record_synthetic(Workload& wl, uint64_t logical_pages, bool fill,
                           uint64_t warmup_writes, uint64_t measure_writes);
// fill -> trace replayed `loops` times; the first `warmup_loops` are not measured.
Recording record_trace(const std::string& path, uint64_t logical_pages, uint32_t page_size,
                       bool fill, int loops, int warmup_loops);

// death[t] for every host write time t (1-based, matching Ftl::now()).
DeathTimes death_times(const Recording& rec, uint64_t logical_pages);

void replay(Ftl& ftl, const Recording& rec, const RunOptions& opt = {});

// Writes one CSV row per sampled host write: the online features, the true
// lifetime class (label) and whether the write is in the measured phase.
// Returns rows written.
// ell_log (optional, from ell_trajectory) fills the ell feature and adds an
// ell_log column, so the trainer can learn lifetime relative to ell.
uint64_t dump_features(const Recording& rec, const DeathTimes& death,
                       uint64_t logical_pages, const std::string& path,
                       double sample, uint64_t seed,
                       const std::vector<float>* ell_log = nullptr);

// Replays rec under SepBIT and records its ell (FeatureExtractor log scale,
// -1 while unknown) at every host write time t, as seen by that write.
std::vector<float> ell_trajectory(const FtlConfig& cfg, const Recording& rec,
                                  uint64_t logical_pages);

struct EraseSummary {
  uint32_t min = 0, max = 0;
  double mean = 0.0, stddev = 0.0;
};
EraseSummary erase_summary(const Nand& nand);

// Desnoyers (SYSTOR'12) WAF for FIFO cleaning under uniform random writes:
//   WAF = alpha / (alpha + W0(-alpha * e^-alpha)),  alpha = physical / logical.
double fifo_uniform_waf(double alpha);

}  // namespace ftl
