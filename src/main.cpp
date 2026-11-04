// ftlsim: trace-driven SSD FTL simulator. Prints one JSON object to stdout.
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include "ftl.h"
#include "predictor.h"
#include "sim.h"
#include "workload.h"

using namespace ftl;

namespace {

const char* kUsage = R"(usage: ftlsim [--key=value ...]
device
  --blocks=N            physical blocks (default 1024)
  --ppb=N               pages per block (default 256)
  --page-size=N         bytes (default 4096)
  --op=F                over-provisioning, (phys-user)/user (default 0.07)
ftl
  --gc=NAME             greedy | cost-benefit | fifo (default greedy)
  --placement=NAME      none | sepgc | wcount | sepbit | oracle | fdp | ml (default none)
  --model=PATH          ml: decision tree from scripts/train_predictor.py
  --ml-gc=NAME          ml: age | sepbit (GC regrouping) | hybrid (SepBIT host test,
                        tree splits the rest, SepBIT GC) (default age)
  --dump-features=PATH  write predictor training rows (features + true class)
  --dump-sample=F       fraction of host writes to dump (default 0.1)
  --dump-erase=PATH     write each block's final erase count, one per line
  --hint-noise=F        fdp: probability a host hint is wrong (default 0)
  --hint-mode=NAME      fdp: random | coarse (default random)
  --fdp-gc=NAME         fdp: persistent | initial isolation (default persistent)
  --gc-threshold=N      free-block low watermark, 0 = auto
  --wear-aware          dynamic wear leveling on block allocation
  --static-wl=N         static wear leveling, max erase-count spread (0 = off)
  --static-wl-period=N  check every N GC victims (default 16)
workload
  --workload=NAME       seq | uniform | hotcold | zipf (default uniform)
  --trace=PATH          unified CSV trace instead of a synthetic workload
  --trace-loops=N       replay the trace N times (default 1)
  --trace-warmup-loops=N  first N loops are not measured (default 0)
  --size-to-trace       set --blocks so user capacity covers the trace's address span
  --zipf-theta=F        (default 0.99)
  --hot-frac=F --hot-access=F   hotcold split (default 0.2 / 0.8)
  --req-pages=N         pages per sequential request (default 1)
  --seed=N
run
  --fill=0|1            sequential fill before warmup (default 1)
  --warmup=F            warmup writes, in multiples of user capacity (default 2)
  --writes=F            measured writes, in multiples of user capacity (default 4)
  --verify              shadow map + random read-back after each request
  --check-every=N       full invariant check every N requests (with --verify)
)";

std::map<std::string, std::string> parse_args(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      std::cout << kUsage;
      std::exit(0);
    }
    if (a.rfind("--", 0) != 0) throw std::invalid_argument("bad argument: " + a);
    a = a.substr(2);
    auto eq = a.find('=');
    if (eq == std::string::npos) args[a] = "1";
    else args[a.substr(0, eq)] = a.substr(eq + 1);
  }
  return args;
}

struct Args {
  std::map<std::string, std::string> m;
  std::string str(const std::string& k, const std::string& d) {
    auto it = m.find(k);
    if (it == m.end()) return d;
    std::string v = it->second;
    m.erase(it);
    return v;
  }
  double num(const std::string& k, double d) {
    std::string v = str(k, "");
    return v.empty() ? d : std::stod(v);
  }
};

void print_json(const FtlConfig& cfg, const Ftl& ftl, const std::string& workload,
                double alpha, uint64_t wrapped) {
  const Stats& s = ftl.stats();
  EraseSummary e = erase_summary(ftl.nand());
  std::printf("{\n");
  std::printf("  \"workload\": \"%s\",\n", workload.c_str());
  std::printf("  \"gc\": \"%s\",\n", ftl.victim_policy().name().c_str());
  std::printf("  \"placement\": \"%s\",\n", ftl.placement().name().c_str());
  std::printf("  \"op\": %.4f,\n", cfg.op_ratio);
  std::printf("  \"blocks\": %u,\n", cfg.geo.num_blocks);
  std::printf("  \"ppb\": %u,\n", cfg.geo.pages_per_block);
  std::printf("  \"logical_pages\": %llu,\n", (unsigned long long)ftl.logical_pages());
  std::printf("  \"wear_aware\": %s,\n", cfg.wear_aware_alloc ? "true" : "false");
  std::printf("  \"host_writes\": %llu,\n", (unsigned long long)s.host_writes);
  std::printf("  \"host_reads\": %llu,\n", (unsigned long long)s.host_reads);
  std::printf("  \"host_trims\": %llu,\n", (unsigned long long)s.host_trims);
  std::printf("  \"nand_writes\": %llu,\n", (unsigned long long)s.nand_writes);
  std::printf("  \"gc_copies\": %llu,\n", (unsigned long long)s.gc_copies);
  std::printf("  \"gc_count\": %llu,\n", (unsigned long long)s.gc_count);
  std::printf("  \"wl_copies\": %llu,\n", (unsigned long long)s.wl_copies);
  std::printf("  \"wl_count\": %llu,\n", (unsigned long long)s.wl_count);
  std::printf("  \"static_wl\": %u,\n", cfg.static_wl_delta);
  std::printf("  \"avg_valid_per_victim\": %.3f,\n", s.avg_valid_per_victim());
  std::printf("  \"waf\": %.5f,\n", s.waf());
  std::printf("  \"fifo_uniform_model_waf\": %.5f,\n", fifo_uniform_waf(alpha));
  std::printf("  \"fifo_uniform_model_waf_effective\": %.5f,\n",
              fifo_uniform_waf(ftl.effective_alpha()));
  std::printf("  \"erase_min\": %u,\n", e.min);
  std::printf("  \"erase_max\": %u,\n", e.max);
  std::printf("  \"erase_mean\": %.3f,\n", e.mean);
  std::printf("  \"erase_std\": %.3f,\n", e.stddev);
  if (auto* ml = dynamic_cast<const MlPlacement*>(&ftl.placement())) {
    std::printf("  \"model_nodes\": %zu,\n", ml->tree().num_nodes());
    std::printf("  \"model_depth\": %d,\n", ml->tree().depth());
    std::printf("  \"model_bytes\": %zu,\n", ml->tree().firmware_bytes());
    std::printf("  \"feature_state_bytes\": %llu,\n",
                (unsigned long long)ml->features().state_bytes());
  }
  std::printf("  \"trace_wrapped_requests\": %llu\n", (unsigned long long)wrapped);
  std::printf("}\n");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a{parse_args(argc, argv)};
    FtlConfig cfg;
    cfg.geo.num_blocks = static_cast<uint32_t>(a.num("blocks", 1024));
    cfg.geo.pages_per_block = static_cast<uint32_t>(a.num("ppb", 256));
    cfg.geo.page_size = static_cast<uint32_t>(a.num("page-size", 4096));
    cfg.op_ratio = a.num("op", 0.07);
    cfg.gc_policy = a.str("gc", "greedy");
    cfg.placement = a.str("placement", "none");
    cfg.gc_free_threshold = static_cast<uint32_t>(a.num("gc-threshold", 0));
    cfg.wear_aware_alloc = a.num("wear-aware", 0) != 0;
    cfg.static_wl_delta = static_cast<uint32_t>(a.num("static-wl", 0));
    cfg.static_wl_period = static_cast<uint32_t>(a.num("static-wl-period", 16));
    cfg.verify = a.num("verify", 0) != 0;

    const std::string workload = a.str("workload", "uniform");
    const std::string trace = a.str("trace", "");
    const uint64_t seed = static_cast<uint64_t>(a.num("seed", 1));
    const double theta = a.num("zipf-theta", 0.99);
    const double hot_frac = a.num("hot-frac", 0.2);
    const double hot_access = a.num("hot-access", 0.8);
    const uint32_t req_pages = static_cast<uint32_t>(a.num("req-pages", 1));
    const bool fill = a.num("fill", 1) != 0;
    const double warmup = a.num("warmup", 2);
    const double writes = a.num("writes", 4);
    const int trace_loops = static_cast<int>(a.num("trace-loops", 1));
    const int trace_warmup = static_cast<int>(a.num("trace-warmup-loops", 0));
    const double hint_noise = a.num("hint-noise", 0.0);
    const std::string hint_mode = a.str("hint-mode", "random");
    const std::string fdp_gc = a.str("fdp-gc", "persistent");
    if (fdp_gc != "persistent" && fdp_gc != "initial") {
      throw std::invalid_argument("--fdp-gc must be persistent or initial");
    }
    const std::string model = a.str("model", "");
    const std::string ml_gc = a.str("ml-gc", "age");
    const std::string dump_path = a.str("dump-features", "");
    const double dump_sample = a.num("dump-sample", 0.1);
    const std::string erase_path = a.str("dump-erase", "");

    RunOptions opt;
    opt.verify_reads = cfg.verify;
    opt.check_every = static_cast<uint64_t>(a.num("check-every", 0));
    opt.seed = seed;
    if (a.num("size-to-trace", 0) != 0) {
      if (trace.empty()) throw std::invalid_argument("--size-to-trace needs --trace");
      const uint64_t page = cfg.geo.page_size;
      const uint64_t span_pages = (trace_span_bytes(trace) + page - 1) / page;
      const double phys_pages = static_cast<double>(span_pages) * (1.0 + cfg.op_ratio);
      cfg.geo.num_blocks =
          static_cast<uint32_t>(phys_pages / cfg.geo.pages_per_block) + 1;
    }
    if (!a.m.empty()) throw std::invalid_argument("unknown option --" + a.m.begin()->first);

    const uint64_t n = Ftl::logical_pages_for(cfg);
    Recording rec;
    std::string label = workload;
    if (!trace.empty()) {
      label = trace;
      rec = record_trace(trace, n, cfg.geo.page_size, fill, trace_loops, trace_warmup);
    } else {
      auto wl = make_synthetic(workload, n, seed, theta, hot_frac, hot_access, req_pages);
      rec = record_synthetic(*wl, n, fill, static_cast<uint64_t>(warmup * n),
                             static_cast<uint64_t>(writes * n));
    }

    PlacementOptions popt;
    popt.logical_pages = n;
    popt.hint_noise = hint_noise;
    popt.hint_mode = hint_mode;
    popt.fdp_persistent = fdp_gc == "persistent";
    popt.seed = seed;
    popt.model_path = model;
    popt.ml_gc = ml_gc;
    if (placement_needs_future(cfg.placement) || !dump_path.empty()) {
      popt.death = std::make_shared<const DeathTimes>(death_times(rec, n));
    }
    if (!dump_path.empty()) {
      uint64_t rows = dump_features(rec, *popt.death, n, dump_path, dump_sample, seed);
      std::fprintf(stderr, "dumped %llu rows to %s\n", (unsigned long long)rows,
                   dump_path.c_str());
    }
    Ftl ftl(cfg, make_victim_policy(cfg.gc_policy), make_placement(cfg.placement, popt));
    const double alpha = static_cast<double>(cfg.geo.physical_pages()) / n;
    replay(ftl, rec, opt);
    const uint64_t wrapped = rec.trace_wrapped;
    if (cfg.verify) ftl.check_invariants();
    if (!erase_path.empty()) {
      std::FILE* f = std::fopen(erase_path.c_str(), "w");
      if (!f) throw std::runtime_error("cannot write " + erase_path);
      for (BlockId b = 0; b < ftl.nand().num_blocks(); ++b) {
        std::fprintf(f, "%u\n", ftl.nand().block(b).erase_count);
      }
      std::fclose(f);
    }
    print_json(cfg, ftl, label, alpha, wrapped);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "ftlsim: %s\n", ex.what());
    return 1;
  }
  return 0;
}
