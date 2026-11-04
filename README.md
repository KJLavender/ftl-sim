# ftlsim — SSD FTL Simulator

**English** | [繁體中文](README.zh-TW.md)

A trace-driven SSD Flash Translation Layer simulator written from scratch in C++17.
It measures how GC policy, over-provisioning, data placement (hot/cold separation)
and wear leveling affect write amplification (WAF). The end goal is to compare
rule-based placement (SepBIT) with a firmware-sized ML lifetime predictor.
See [01-ftl-simulator-plan.md](01-ftl-simulator-plan.md) for the full plan.

## Features

| Area | Implemented |
|---|---|
| NAND model | page program / block erase, no in-place update, per-block erase count, OOB LPN |
| Mapping | page-level L2P |
| GC victim | `greedy`, `cost-benefit`, `fifo` |
| Placement | `none`, `sepgc` (GC writes in their own stream), `wcount` (write-count rule), `sepbit` (FAST'22), `oracle` (true lifetimes), `fdp` (host hints, noisy / coarse, persistent / initial isolation), `ml` (decision-tree lifetime predictor) |
| Wear leveling | dynamic (`--wear-aware`: allocate least-erased free block); static (`--static-wl=N`: relocate cold data to the most-worn free block when the erase-count spread exceeds N) |
| Learned placement | online features with 2 B/LPN state, decision tree trained offline (`scripts/train_predictor.py`), loaded as a ~1 KB node table |
| Host ops | write, read, TRIM |
| Workloads | `seq`, `uniform`, `hotcold` (80/20), `zipf`, real traces (MSR / Alibaba / Tencent) |
| Correctness | shadow map on every read + full L2P/OOB invariant checks |

## Build and test

Requires `g++` (C++17) and `make`; developed on WSL2 Ubuntu.

```bash
make            # build/ftlsim and build/test_ftl
make test       # unit + theory tests
bash scripts/smoke_trace.sh   # end-to-end trace path with --verify
```

## Usage

```bash
./build/ftlsim --workload=zipf --gc=cost-benefit --placement=sepbit --op=0.14
./build/ftlsim --help
```

Output is a single JSON object (WAF, GC count, valid pages per victim, erase-count
min/max/mean/std, and the analytic FIFO model WAF for comparison).

Run order: sequential fill of the whole device → `--warmup` × capacity writes
(not measured) → `--writes` × capacity writes (measured). The whole run is
recorded first and then replayed, so every placement sees the identical
request stream and `oracle` / `fdp` can look up each page's true lifetime.

### Learned placement

```bash
bash scripts/ml_pipeline.sh zipf 0.14     # dump features (seed 1) -> train -> evaluate on seed 2
./build/ftlsim --placement=ml --model=results/ml/zipf.tree --ml-gc=sepbit
```

Training needs [uv](https://docs.astral.sh/uv/); dependencies are declared inline in the script.

### Real traces

```bash
python3 tools/convert_trace.py alibaba io_traces.csv --volume 28 -o vol28.csv
./build/ftlsim --trace=vol28.csv --size-to-trace --trace-loops=2 --trace-warmup-loops=1
```

Unified trace format: `timestamp_us,op,offset_bytes,size_bytes` with `op` ∈ {R, W, T}.

### Experiment grids

```bash
python3 scripts/experiment.py --grid op_sweep  --out results/op_sweep.csv
python3 scripts/experiment.py --grid placement --out results/placement.csv
python3 scripts/summarize.py results/placement.csv --rows workload,gc,op --cols placement
```

## Validation

| Check | Result |
|---|---|
| Sequential overwrite, all GC policies | WAF = 1.00000, zero GC copies |
| Shadow map, 3 GC × 4 placement × 2 wear-leveling settings, Zipf writes | every read returns the latest version |
| FIFO under uniform random vs. Desnoyers (SYSTOR'12) closed form | error 0.7% / 0.3% / 0.1% at OP 7% / 14% / 28% |
| Greedy ≤ FIFO on uniform random | yes |
| Recorded replay vs. streaming driver | identical NAND write counts |
| Static WL, oracle, FDP and ML placements under the shadow map | every read returns the latest version |

The analytic model is compared against the *effective* OP, which excludes the
GC free-block watermark and the unfilled part of open blocks.

## Early results

2048 blocks × 256 pages × 4 KiB (2 GiB physical), 3× warmup + 5× measured writes.

**WAF vs OP and GC policy (no separation)**

| Workload | GC | 7% | 10% | 14% | 20% | 28% |
|---|---|---|---|---|---|---|
| uniform | greedy | 7.73 | 5.63 | 4.23 | 3.17 | 2.47 |
| uniform | cost-benefit | 9.17 | 6.43 | 4.68 | 3.39 | 2.58 |
| zipf 0.99 | greedy | 7.50 | 5.85 | 4.62 | 3.59 | 2.84 |
| zipf 0.99 | cost-benefit | 5.79 | 4.36 | 3.42 | 2.74 | 2.29 |

**WAF by placement (greedy GC, OP 14%)**

| Workload | none | sepgc | wcount | sepbit |
|---|---|---|---|---|
| uniform | 4.23 | 4.24 | 4.25 | 4.33 |
| hotcold 80/20 | 4.53 | 4.53 | 4.46 | 4.43 |
| zipf 0.99 | 4.62 | 4.59 | 4.22 | 3.55 |

Takeaways so far: cost-benefit only pays off when the workload is skewed;
separation cannot help uniform writes (extra open streams just consume OP);
SepBIT cuts WAF by 23% on Zipf at OP 14%.

**Wear leveling (hot/cold, cold 80% never rewritten, OP 7%, greedy)**

| Setting | WAF | max erase | erase std |
|---|---|---|---|
| off | 2.15 | 106 | 44.7 |
| static, spread ≤ 20 | 2.30 | 53 | 4.9 |
| static, spread ≤ 50 | 2.24 | 80 | 19.2 |

Static WL costs 7% extra WAF here. Where the relocated cold data goes matters:
writing it back into the shared host stream raised WAF from 1.16 to 3.68 in a
unit test; parking it on the most-worn free block keeps it at 1.33.

**Lifetime prediction vs. bounds (Zipf 0.99, OP 14%, train seed 1 → test seed 2)**

| Placement | WAF |
|---|---|
| none | 4.62 |
| GC age regrouping only (FDP initial, random hints) | 4.07 |
| ml, decision tree depth 6, ~1.2 KB | 3.78 – 3.97 |
| sepbit | 3.55 |
| FDP, perfect host hints, persistent isolation | 1.93 |
| perfect host classes + GC age regrouping | 1.74 |
| oracle | 1.56 |

On a memoryless synthetic workload the tree does not beat SepBIT: what is
learnable is each page's write rate, and SepBIT's previous-lifetime test with
an adaptive threshold already captures most of it. The oracle's 1.56 depends on
knowing random inter-write gaps that no history-based predictor can see, so it
is a loose bound here. Real traces, which have temporal and spatial structure,
are the actual test.

**FDP host hints (same setup)**

| Hints | Persistent isolation (GC stays in RUH) | Initial isolation (device regroups GC data by age) |
|---|---|---|
| perfect | 1.93 | 1.74 |
| hot vs. cold only | 3.83 | – |
| 20% wrong | 4.42 | 4.02 |
| 50% wrong | 5.03 (worse than no separation, 4.62) | 4.14 |

Wrong hints under persistent isolation are worse than no hints at all;
letting the device regroup GC data recovers most of the loss.

## Layout

```
src/
  nand.{h,cpp}        NAND array: pages, blocks, erase counts, OOB
  ftl.{h,cpp}         page-mapped FTL: write/read/trim, GC, free pool, invariants
  gc_policy.{h,cpp}   victim selection: greedy / cost-benefit / fifo
  placement.{h,cpp}   stream assignment: none / sepgc / wcount / sepbit / oracle / fdp
  predictor.{h,cpp}   online features, decision tree, ml placement
  workload.{h,cpp}    synthetic generators + unified trace reader
  sim.{h,cpp}         record / replay, death times, feature dump, erase stats, FIFO model
  main.cpp            CLI, JSON output
tests/test_ftl.cpp    correctness and theory tests
tools/convert_trace.py  MSR / Alibaba / Tencent -> unified format
scripts/              experiment grids, summaries, smoke test
```

## Roadmap

- [x] Week 1: NAND model, L2P, shadow map, sequential WAF = 1
- [x] Week 2: greedy / cost-benefit / FIFO GC, OP sweep, analytic validation, SepBIT, GC-write separation
- [x] Week 3 (code): trace converters, static wear leveling
- [ ] Week 3 (data): run MSR / Alibaba / Tencent traces, compare SepBIT trends with the paper
- [x] Week 4 (synthetic): oracle, FDP hints (noisy / coarse, persistent / initial), ML predictor pipeline
- [ ] Week 4 (traces): train on volume A, test on volume B; time-of-day feature; online retraining
- [ ] Week 5: plots, write-up
