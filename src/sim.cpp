#include "sim.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "predictor.h"

namespace ftl {

void apply(Ftl& ftl, const IoRequest& req) {
  const uint64_t n = ftl.logical_pages();
  for (uint32_t i = 0; i < req.pages; ++i) {
    Lpn lpn = static_cast<Lpn>((static_cast<uint64_t>(req.lpn) + i) % n);
    switch (req.op) {
      case Op::kWrite: ftl.write(lpn, req.pages); break;
      case Op::kRead: ftl.read(lpn); break;
      case Op::kTrim: ftl.trim(lpn); break;
    }
  }
}

uint64_t run(Ftl& ftl, Workload& wl, uint64_t host_writes, const RunOptions& opt) {
  std::mt19937_64 rng(opt.seed ^ 0x9e3779b97f4a7c15ULL);
  std::uniform_int_distribution<uint64_t> any_lpn(0, ftl.logical_pages() - 1);
  const uint64_t start = ftl.stats().host_writes;
  uint64_t requests = 0;
  IoRequest req;
  while (ftl.stats().host_writes - start < host_writes && wl.next(req)) {
    apply(ftl, req);
    ++requests;
    if (opt.verify_reads) ftl.read(static_cast<Lpn>(any_lpn(rng)));
    if (opt.check_every && requests % opt.check_every == 0) ftl.check_invariants();
  }
  return ftl.stats().host_writes - start;
}

void fill_sequential(Ftl& ftl) {
  for (Lpn lpn = 0; lpn < ftl.logical_pages(); ++lpn) ftl.write(lpn);
}

void Recording::add(const IoRequest& req, uint64_t logical_pages) {
  for (uint32_t i = 0; i < req.pages; ++i) {
    Lpn lpn = static_cast<Lpn>((static_cast<uint64_t>(req.lpn) + i) % logical_pages);
    ops.push_back({req.op, lpn, req.pages});
    if (req.op == Op::kWrite) ++writes;
  }
}

Recording record_synthetic(Workload& wl, uint64_t logical_pages, bool fill,
                           uint64_t warmup_writes, uint64_t measure_writes) {
  Recording rec;
  if (fill) {
    for (Lpn l = 0; l < logical_pages; ++l) rec.add({Op::kWrite, l, 1}, logical_pages);
  }
  IoRequest req;
  uint64_t target = rec.writes + warmup_writes;
  while (rec.writes < target && wl.next(req)) rec.add(req, logical_pages);
  rec.measure_from = rec.ops.size();
  target = rec.writes + measure_writes;
  while (rec.writes < target && wl.next(req)) rec.add(req, logical_pages);
  return rec;
}

Recording record_trace(const std::string& path, uint64_t logical_pages, uint32_t page_size,
                       bool fill, int loops, int warmup_loops) {
  Recording rec;
  if (fill) {
    for (Lpn l = 0; l < logical_pages; ++l) rec.add({Op::kWrite, l, 1}, logical_pages);
  }
  rec.measure_from = rec.ops.size();
  for (int loop = 0; loop < loops; ++loop) {
    if (loop == warmup_loops) rec.measure_from = rec.ops.size();
    TraceWorkload tw(path, logical_pages, page_size);
    IoRequest req;
    while (tw.next(req)) rec.add(req, logical_pages);
    rec.trace_wrapped = tw.wrapped();
  }
  if (warmup_loops >= loops) rec.measure_from = rec.ops.size();
  return rec;
}

DeathTimes death_times(const Recording& rec, uint64_t logical_pages) {
  DeathTimes death(rec.writes + 2, kNever);
  std::vector<Time> last(logical_pages, kNever);
  Time t = 0;
  for (const PageOp& op : rec.ops) {
    if (op.op == Op::kWrite) {
      ++t;
      if (last[op.lpn] != kNever) death[last[op.lpn]] = t;
      last[op.lpn] = t;
    } else if (op.op == Op::kTrim && last[op.lpn] != kNever) {
      death[last[op.lpn]] = t + 1;  // dies before the next write
      last[op.lpn] = kNever;
    }
  }
  return death;
}

void replay(Ftl& ftl, const Recording& rec, const RunOptions& opt) {
  std::mt19937_64 rng(opt.seed ^ 0x9e3779b97f4a7c15ULL);
  std::uniform_int_distribution<uint64_t> any_lpn(0, ftl.logical_pages() - 1);
  for (size_t i = 0; i < rec.ops.size(); ++i) {
    if (i == rec.measure_from) ftl.reset_stats();
    const PageOp& op = rec.ops[i];
    switch (op.op) {
      case Op::kWrite: ftl.write(op.lpn, op.req_pages); break;
      case Op::kRead: ftl.read(op.lpn); break;
      case Op::kTrim: ftl.trim(op.lpn); break;
    }
    if (opt.verify_reads) ftl.read(static_cast<Lpn>(any_lpn(rng)));
    if (opt.check_every && (i + 1) % opt.check_every == 0) ftl.check_invariants();
  }
  if (rec.measure_from >= rec.ops.size()) ftl.reset_stats();
}

uint64_t dump_features(const Recording& rec, const DeathTimes& death,
                       uint64_t logical_pages, const std::string& path,
                       double sample, uint64_t seed) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throw std::runtime_error("cannot write " + path);
  for (int i = 0; i < FeatureExtractor::kNumFeatures; ++i) {
    std::fprintf(f, "%s,", FeatureExtractor::feature_name(i));
  }
  std::fprintf(f, "label,lifetime_log,measured\n");

  FeatureExtractor fx;
  fx.init(logical_pages);
  std::vector<Time> last(logical_pages, kNever);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> coin(0.0, 1.0);
  Time t = 0;
  uint64_t rows = 0;
  for (size_t i = 0; i < rec.ops.size(); ++i) {
    const PageOp& op = rec.ops[i];
    if (op.op == Op::kTrim) last[op.lpn] = kNever;
    if (op.op != Op::kWrite) continue;
    HostWrite w{op.lpn, ++t, last[op.lpn], op.req_pages};
    auto x = fx.extract(w);
    fx.update(w);
    last[op.lpn] = t;
    if (coin(rng) >= sample) continue;
    Time d = death[t];
    int label = lifetime_class(d == kNever ? kNever : d - t, logical_pages);
    // Same normalization as the features: log2(1 + 1024 * L / C). Data never
    // overwritten before the run ends is censored at 16 capacities.
    double l = d == kNever ? 16.0 * logical_pages : static_cast<double>(d - t);
    double lifetime_log = std::log2(1.0 + 1024.0 * std::min(l, 16.0 * logical_pages) /
                                              static_cast<double>(logical_pages));
    for (float v : x) std::fprintf(f, "%g,", v);
    std::fprintf(f, "%d,%.4f,%d\n", label, lifetime_log, i >= rec.measure_from ? 1 : 0);
    ++rows;
  }
  std::fclose(f);
  return rows;
}

EraseSummary erase_summary(const Nand& nand) {
  EraseSummary s;
  s.min = UINT32_MAX;
  double sum = 0.0, sq = 0.0;
  for (BlockId b = 0; b < nand.num_blocks(); ++b) {
    uint32_t e = nand.block(b).erase_count;
    s.min = std::min(s.min, e);
    s.max = std::max(s.max, e);
    sum += e;
    sq += static_cast<double>(e) * e;
  }
  const double n = nand.num_blocks();
  s.mean = sum / n;
  s.stddev = std::sqrt(std::max(0.0, sq / n - s.mean * s.mean));
  return s;
}

double fifo_uniform_waf(double alpha) {
  if (alpha <= 1.0) throw std::invalid_argument("alpha must be > 1");
  // Principal branch W0(x) for x in (-1/e, 0), via Newton from w = -0.5.
  const double x = -alpha * std::exp(-alpha);
  double w = -0.5;
  for (int i = 0; i < 100; ++i) {
    double ew = std::exp(w);
    double f = w * ew - x;
    double step = f / (ew * (w + 1.0));
    w -= step;
    if (w <= -1.0) w = -1.0 + 1e-9;  // stay on the principal branch
    if (std::fabs(step) < 1e-14) break;
  }
  return alpha / (alpha + w);
}

}  // namespace ftl
