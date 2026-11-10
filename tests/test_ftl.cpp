// Correctness tests: shadow map, invariants, and theory checks on WAF.
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "ftl.h"
#include "predictor.h"
#include "sim.h"
#include "workload.h"

using namespace ftl;

namespace {

int g_failed = 0;

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::printf("    CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failed;                                                       \
    }                                                                   \
  } while (0)

FtlConfig small_config(const std::string& gc, const std::string& placement, double op) {
  FtlConfig cfg;
  cfg.geo.num_blocks = 256;
  cfg.geo.pages_per_block = 64;
  cfg.op_ratio = op;
  cfg.gc_policy = gc;
  cfg.placement = placement;
  cfg.verify = true;
  return cfg;
}

void test_sequential_waf_is_one() {
  for (const char* gc : {"greedy", "cost-benefit", "fifo"}) {
    Ftl ftl(small_config(gc, "none", 0.07));
    fill_sequential(ftl);
    ftl.reset_stats();
    SequentialWorkload wl(ftl.logical_pages(), 8);
    run(ftl, wl, 5 * ftl.logical_pages(), {true, 1000, 1});
    ftl.check_invariants();
    std::printf("    %-13s seq WAF = %.5f\n", gc, ftl.stats().waf());
    CHECK(ftl.stats().waf() == 1.0);
    CHECK(ftl.stats().gc_copies == 0);
  }
}

void test_random_with_shadow_all_configs() {
  for (const char* gc : {"greedy", "cost-benefit", "fifo"}) {
    for (const char* pl : {"none", "sepgc", "wcount", "sepbit", "sepbit4", "sepbit-split"}) {
      for (bool wear : {false, true}) {
        FtlConfig cfg = small_config(gc, pl, 0.14);
        cfg.wear_aware_alloc = wear;
        Ftl ftl(cfg);
        fill_sequential(ftl);
        ZipfWorkload wl(ftl.logical_pages(), 0.9, 7);
        run(ftl, wl, 6 * ftl.logical_pages(), {true, 500, 3});
        // Read every page back against the shadow map.
        for (Lpn l = 0; l < ftl.logical_pages(); ++l) ftl.read(l);
        ftl.check_invariants();
        CHECK(ftl.stats().waf() >= 1.0);
      }
    }
  }
}

void test_trim() {
  Ftl ftl(small_config("greedy", "none", 0.07));
  fill_sequential(ftl);
  for (Lpn l = 0; l < ftl.logical_pages(); l += 2) ftl.trim(l);
  for (Lpn l = 0; l < ftl.logical_pages(); ++l) ftl.read(l);
  CHECK(ftl.stats().unmapped_reads == (ftl.logical_pages() + 1) / 2);
  ftl.check_invariants();

  // With half the device trimmed, uniform writes to the other half see an
  // effective OP far above 7%, so WAF must drop well below the full-device case.
  ftl.reset_stats();
  std::mt19937_64 rng(5);
  std::uniform_int_distribution<uint64_t> d(0, ftl.logical_pages() / 2 - 1);
  for (uint64_t i = 0; i < 4 * ftl.logical_pages(); ++i) ftl.write(static_cast<Lpn>(2 * d(rng) + 1));
  double trimmed_waf = ftl.stats().waf();

  Ftl full(small_config("greedy", "none", 0.07));
  fill_sequential(full);
  UniformWorkload wl(full.logical_pages(), 5);
  run(full, wl, 4 * full.logical_pages());
  full.reset_stats();
  run(full, wl, 4 * full.logical_pages());
  std::printf("    WAF trimmed-half = %.3f, full device = %.3f\n", trimmed_waf, full.stats().waf());
  CHECK(trimmed_waf < full.stats().waf() * 0.6);
  ftl.check_invariants();
}

// FIFO under uniform random writes has a closed form; the simulator must match.
void test_fifo_matches_model() {
  for (double op : {0.07, 0.14, 0.28}) {
    FtlConfig cfg;
    cfg.geo.num_blocks = 1024;
    cfg.geo.pages_per_block = 256;
    cfg.op_ratio = op;
    cfg.gc_policy = "fifo";
    Ftl ftl(cfg);
    fill_sequential(ftl);
    UniformWorkload wl(ftl.logical_pages(), 11);
    run(ftl, wl, 3 * ftl.logical_pages());
    ftl.reset_stats();
    run(ftl, wl, 5 * ftl.logical_pages());
    double model = fifo_uniform_waf(ftl.effective_alpha());
    double sim = ftl.stats().waf();
    double err = std::fabs(sim - model) / model;
    std::printf("    OP %.2f  sim %.3f  model %.3f  err %.1f%%\n", op, sim, model, 100 * err);
    CHECK(err < 0.02);
  }
}

// Greedy is never worse than FIFO on uniform random writes.
void test_greedy_beats_fifo() {
  double waf[2];
  const char* gcs[2] = {"greedy", "fifo"};
  for (int i = 0; i < 2; ++i) {
    FtlConfig cfg = small_config(gcs[i], "none", 0.07);
    cfg.verify = false;
    Ftl ftl(cfg);
    fill_sequential(ftl);
    UniformWorkload wl(ftl.logical_pages(), 3);
    run(ftl, wl, 3 * ftl.logical_pages());
    ftl.reset_stats();
    run(ftl, wl, 5 * ftl.logical_pages());
    waf[i] = ftl.stats().waf();
  }
  std::printf("    greedy %.3f  fifo %.3f\n", waf[0], waf[1]);
  CHECK(waf[0] <= waf[1]);
}

void test_too_small_op_rejected() {
  bool threw = false;
  try {
    FtlConfig cfg = small_config("greedy", "sepbit", 0.001);
    Ftl ftl(cfg);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

void test_analytic_model_values() {
  // alpha = 1 + OP. Larger OP must give lower WAF, and WAF > 1.
  double prev = 1e9;
  for (double op : {0.07, 0.14, 0.28, 1.0}) {
    double w = fifo_uniform_waf(1.0 + op);
    CHECK(w > 1.0 && w < prev);
    prev = w;
  }
}

// Cold data that is never rewritten pins blocks at a low erase count.
// Static WL must bound the spread; it costs extra copies (counted in WAF).
void test_static_wear_leveling() {
  struct Result { uint32_t spread; double waf; uint64_t wl; };
  auto run_case = [](uint32_t delta) {
    FtlConfig cfg = small_config("greedy", "none", 0.14);
    cfg.static_wl_delta = delta;
    cfg.static_wl_period = 4;
    Ftl ftl(cfg);
    fill_sequential(ftl);
    // 10% of the space takes every write; the other 90% stays cold forever.
    HotColdWorkload wl(ftl.logical_pages(), 0.1, 1.0, 9);
    run(ftl, wl, 20 * ftl.logical_pages(), {true, 2000, 4});
    for (Lpn l = 0; l < ftl.logical_pages(); ++l) ftl.read(l);
    ftl.check_invariants();
    EraseSummary e = erase_summary(ftl.nand());
    return Result{e.max - e.min, ftl.stats().waf(), ftl.stats().wl_copies};
  };
  Result off = run_case(0), on = run_case(10);
  std::printf("    spread off %u / on %u, WAF off %.3f / on %.3f, wl copies %llu\n",
              off.spread, on.spread, off.waf, on.waf, (unsigned long long)on.wl);
  CHECK(off.wl == 0);
  CHECK(on.wl > 0);
  CHECK(on.spread < off.spread / 2);
  CHECK(on.spread <= 2 * 10);  // delta plus slack for the check period
  CHECK(on.waf > off.waf);
}

void test_death_times() {
  // t=1 W0, t=2 W1, T0, t=3 W1, t=4 W0
  Recording rec;
  rec.add({Op::kWrite, 0, 1}, 4);
  rec.add({Op::kWrite, 1, 1}, 4);
  rec.add({Op::kTrim, 0, 1}, 4);
  rec.add({Op::kWrite, 1, 1}, 4);
  rec.add({Op::kWrite, 0, 1}, 4);
  DeathTimes d = death_times(rec, 4);
  CHECK(d[1] == 3);       // trimmed after t=2, dies before the write at t=3
  CHECK(d[2] == 3);       // overwritten at t=3
  CHECK(d[3] == kNever);
  CHECK(d[4] == kNever);
}

// Runs one placement over an identical recorded Zipf stream.
double zipf_waf(const std::string& placement, double noise = 0.0,
                const std::string& mode = "random", bool persistent = true,
                bool verify = false) {
  FtlConfig cfg;
  cfg.geo.num_blocks = 1024;
  cfg.geo.pages_per_block = 128;
  cfg.op_ratio = 0.14;
  cfg.placement = placement;
  cfg.verify = verify;
  const uint64_t n = Ftl::logical_pages_for(cfg);
  ZipfWorkload wl(n, 0.99, 21);
  Recording rec = record_synthetic(wl, n, true, 2 * n, 4 * n);
  PlacementOptions popt;
  popt.logical_pages = n;
  popt.hint_noise = noise;
  popt.hint_mode = mode;
  popt.fdp_persistent = persistent;
  if (placement_needs_future(placement)) {
    popt.death = std::make_shared<const DeathTimes>(death_times(rec, n));
  }
  Ftl ftl(cfg, make_victim_policy("greedy"), make_placement(placement, popt));
  replay(ftl, rec, {verify, verify ? 50000u : 0u, 2});
  if (verify) ftl.check_invariants();
  return ftl.stats().waf();
}

void test_oracle_and_hints() {
  double none = zipf_waf("none"), sepbit = zipf_waf("sepbit");
  double oracle = zipf_waf("oracle", 0.0, "random", true, true);
  double fdp = zipf_waf("fdp", 0.0, "random", true, true);
  double fdp_noisy = zipf_waf("fdp", 0.5);
  double fdp_noisy_initial = zipf_waf("fdp", 0.5, "random", false, true);
  double fdp_coarse = zipf_waf("fdp", 0.0, "coarse");
  std::printf("    none %.3f  sepbit %.3f  oracle %.3f\n", none, sepbit, oracle);
  std::printf("    fdp %.3f  coarse %.3f  50%% noise: persistent %.3f / initial %.3f\n",
              fdp, fdp_coarse, fdp_noisy, fdp_noisy_initial);
  CHECK(oracle < sepbit);
  CHECK(sepbit < none);
  CHECK(oracle <= fdp);
  CHECK(fdp < fdp_coarse);
  CHECK(fdp < fdp_noisy);
  CHECK(fdp_noisy_initial < fdp_noisy);  // device-side regrouping repairs bad hints
}

// Hand-written stump: write_count (feature 2) <= 3 -> class 4, else class 0.
void test_ml_placement() {
  const std::string path = "build/test_stump.tree";
  {
    std::FILE* f = std::fopen(path.c_str(), "w");
    std::fprintf(f, "ftl-tree v1 3 %d\n", FeatureExtractor::kNumFeatures);
    std::fprintf(f, "2 3.5 1 2 0\n-1 0 -1 -1 4\n-1 0 -1 -1 0\n");
    std::fclose(f);
  }
  DecisionTree tree;
  tree.load(path);
  FeatureExtractor::Features x{};
  x[2] = 1;
  CHECK(tree.predict(x) == 4);
  x[2] = 9;
  CHECK(tree.predict(x) == 0);
  CHECK(tree.depth() == 1);

  // Two-tree ensemble on the wall-clock gap (feature 8): leaves add up to a
  // log lifetime, which is bucketed into a class at 7.01 / 9.00 / 11.0 / 13.0.
  const std::string gbm_path = "build/test_gbm.tree";
  {
    std::FILE* f = std::fopen(gbm_path.c_str(), "w");
    std::fprintf(f, "ftl-gbm v1 2 %d 8.0\n", FeatureExtractor::kNumFeatures);
    std::fprintf(f, "3\n8 10 1 2 0\n-1 0 -1 -1 -2.0\n-1 0 -1 -1 2.0\n");
    std::fprintf(f, "1\n-1 0 -1 -1 0.5\n");
    std::fclose(f);
  }
  DecisionTree gbm;
  gbm.load(gbm_path);
  CHECK(gbm.num_trees() == 2 && gbm.num_nodes() == 4);
  CHECK(gbm.uses_feature(8) && !gbm.uses_feature(2));
  x[8] = 3;   // 8 - 2 + 0.5 = 6.5  -> class 0
  CHECK(gbm.predict(x) == 0);
  x[8] = 12;  // 8 + 2 + 0.5 = 10.5 -> class 2
  CHECK(gbm.predict(x) == 2);
  CHECK(std::abs(gbm.predict_value(x) - 10.5f) < 1e-5f);

  bool rejected = false;
  try {
    MlPlacement bad(path, "ell2");  // class-only tree has no lifetime output
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);

  // Host groups by lifetime relative to ell: <ell, then one per split passed.
  const std::vector<double> splits = {4, 16};
  CHECK(SepBitPlacement::ell_group(5, 10, splits) == 0);
  CHECK(SepBitPlacement::ell_group(10, 10, splits) == 1);
  CHECK(SepBitPlacement::ell_group(40, 10, splits) == 2);
  CHECK(SepBitPlacement::ell_group(1000, 10, splits) == 3);
  CHECK(SepBitPlacement::ell_group(1000, 10, {}) == 1);
  // The learned ell modes use exactly as many streams as the rule control, so
  // both reserve the same number of free blocks for GC.
  CHECK(MlPlacement(gbm_path, "ell4").num_streams() ==
        SepBitPlacement(std::vector<double>{4, 16}).num_streams());
  CHECK(MlPlacement(gbm_path, "ell2").num_streams() == SepBitPlacement().num_streams());
  CHECK(MlPlacement(gbm_path, "ell", {2, 8, 32, 128}).num_streams() ==
        SepBitPlacement(std::vector<double>{2, 8, 32, 128}).num_streams());

  // ellcls needs a class tree, ellrel a lifetime (regression) model.
  bool cls_rejects_gbm = false, rel_rejects_tree = false;
  try { MlPlacement p(gbm_path, "ellcls"); } catch (const std::invalid_argument&) { cls_rejects_gbm = true; }
  try { MlPlacement p(path, "ellrel"); } catch (const std::invalid_argument&) { rel_rejects_tree = true; }
  CHECK(cls_rejects_gbm && rel_rejects_tree);

  for (const char* gc : {"age", "sepbit", "hybrid", "ell2", "ell4", "ell", "ellrel", "ellcls"}) {
    FtlConfig cfg = small_config("greedy", "ml", 0.14);
    const std::string mode = gc;
    const bool lifetime_model = mode.rfind("ell", 0) == 0 && mode != "ellcls";
    auto placement = std::make_unique<MlPlacement>(lifetime_model ? gbm_path : path, gc,
                                                   std::vector<double>{2, 8, 32, 128});
    Ftl ftl(cfg, make_victim_policy("greedy"), std::move(placement));
    fill_sequential(ftl);
    ZipfWorkload wl(ftl.logical_pages(), 0.99, 4);
    run(ftl, wl, 6 * ftl.logical_pages(), {true, 500, 6});
    for (Lpn l = 0; l < ftl.logical_pages(); ++l) ftl.read(l);
    ftl.check_invariants();
  }

  // Model-driven GC grouping: same invariants, and only for lifetime models.
  for (const char* gc : {"ell4", "ellrel"}) {
    FtlConfig cfg = small_config("greedy", "ml", 0.14);
    auto placement = std::make_unique<MlPlacement>(gbm_path, gc);
    placement->set_model_gc(true);
    Ftl ftl(cfg, make_victim_policy("greedy"), std::move(placement));
    fill_sequential(ftl);
    ZipfWorkload wl(ftl.logical_pages(), 0.99, 5);
    run(ftl, wl, 6 * ftl.logical_pages(), {true, 500, 6});
    for (Lpn l = 0; l < ftl.logical_pages(); ++l) ftl.read(l);
    ftl.check_invariants();
  }
  bool gc_rejects_cls = false;
  try { MlPlacement(path, "ellcls").set_model_gc(true); } catch (const std::invalid_argument&) { gc_rejects_cls = true; }
  CHECK(gc_rejects_cls);

  // ell trajectory: unknown (-1) until SepBIT collects its first hot block,
  // then a finite value on the feature scale for every later write.
  {
    FtlConfig cfg = small_config("greedy", "sepbit", 0.14);
    const uint64_t n = Ftl::logical_pages_for(cfg);
    ZipfWorkload wl(n, 0.99, 7);
    Recording rec = record_synthetic(wl, n, true, 2 * n, 2 * n);
    std::vector<float> ell = ell_trajectory(cfg, rec, n);
    CHECK(ell.size() >= rec.writes + 1);
    CHECK(ell[1] < 0);              // first write: nothing collected yet
    CHECK(ell[rec.writes] > 0);     // by the end ell is known
    size_t first_known = 1;
    while (first_known <= rec.writes && ell[first_known] < 0) ++first_known;
    bool stays_known = true;
    for (size_t t = first_known; t <= rec.writes; ++t) stays_known &= ell[t] >= 0;
    CHECK(stays_known);
  }
}

// Recorded replay must reproduce the streaming driver exactly.
void test_replay_matches_run() {
  FtlConfig cfg = small_config("greedy", "sepbit", 0.14);
  cfg.verify = false;
  const uint64_t n = Ftl::logical_pages_for(cfg);
  Ftl a(cfg);
  fill_sequential(a);
  ZipfWorkload wa(n, 0.9, 5);
  run(a, wa, 2 * n);
  a.reset_stats();
  run(a, wa, 3 * n);

  ZipfWorkload wb(n, 0.9, 5);
  Recording rec = record_synthetic(wb, n, true, 2 * n, 3 * n);
  Ftl b(cfg);
  replay(b, rec);
  CHECK(a.stats().nand_writes == b.stats().nand_writes);
  CHECK(a.stats().host_writes == b.stats().host_writes);
}

}  // namespace

int main() {
  struct Test { const char* name; std::function<void()> fn; };
  std::vector<Test> tests = {
      {"sequential WAF == 1", test_sequential_waf_is_one},
      {"shadow map, all policy combos", test_random_with_shadow_all_configs},
      {"TRIM", test_trim},
      {"FIFO matches Desnoyers model", test_fifo_matches_model},
      {"greedy <= FIFO", test_greedy_beats_fifo},
      {"too-small OP rejected", test_too_small_op_rejected},
      {"analytic model sanity", test_analytic_model_values},
      {"static wear leveling", test_static_wear_leveling},
      {"death times", test_death_times},
      {"replay == streaming run", test_replay_matches_run},
      {"oracle and FDP hints", test_oracle_and_hints},
      {"decision tree + ml placement", test_ml_placement},
  };
  for (auto& t : tests) {
    int before = g_failed;
    std::printf("[ RUN  ] %s\n", t.name);
    try {
      t.fn();
    } catch (const std::exception& ex) {
      std::printf("    exception: %s\n", ex.what());
      ++g_failed;
    }
    std::printf("[ %s ] %s\n", g_failed == before ? " OK " : "FAIL", t.name);
  }
  std::printf("%s\n", g_failed ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
  return g_failed ? 1 : 0;
}
