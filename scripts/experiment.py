#!/usr/bin/env python3
"""Run ftlsim over a parameter grid and write results as CSV.

Example:
  python3 scripts/experiment.py --grid op_sweep --out results/op_sweep.csv
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "build", "ftlsim")

# Each grid: dict of option -> list of values; the cartesian product is run.
GRIDS = {
    # Q1 + Q2: GC policy x OP x workload
    "op_sweep": {
        "workload": ["uniform", "hotcold", "zipf"],
        "gc": ["greedy", "cost-benefit", "fifo"],
        "op": [0.05, 0.07, 0.10, 0.14, 0.20, 0.28, 0.40],
    },
    # Q3 + Q5: placement schemes on skewed workloads
    "placement": {
        "workload": ["uniform", "hotcold", "zipf"],
        "gc": ["greedy", "cost-benefit"],
        "placement": ["none", "sepgc", "wcount", "sepbit"],
        "op": [0.07, 0.14, 0.28],
    },
    # Real traces (use with --traces): placement schemes incl. the oracle bound
    "trace": {
        "gc": ["greedy", "cost-benefit"],
        "placement": ["none", "sepgc", "wcount", "sepbit", "oracle"],
        "op": [0.07, 0.14, 0.28],
    },
    # Q2 on real traces (use with --traces): where does extra OP stop paying off?
    "trace_op": {
        "gc": ["greedy", "cost-benefit"],
        "placement": ["none", "sepbit"],
        "op": [0.05, 0.07, 0.10, 0.14, 0.20, 0.28, 0.40],
    },
    # Q4: wear leveling. hot-access 1.0 = the cold 80% is never rewritten,
    # the worst case for erase-count spread.
    "wear": {
        "workload": ["hotcold"],
        "hot-access": [0.8, 1.0],
        "gc": ["greedy"],
        "wear-aware": [0, 1],
        "static-wl": [0, 20, 50],
        "op": [0.07, 0.28],
    },
}


def run_one(params, common):
    args = [SIM] + [f"--{k}={v}" for k, v in {**common, **params}.items()]
    out = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        return {**params, "error": out.stderr.strip()}
    return {**params, **json.loads(out.stdout)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=GRIDS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=2048)
    ap.add_argument("--ppb", type=int, default=256)
    ap.add_argument("--warmup", type=float, default=3)
    ap.add_argument("--writes", type=float, default=5)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--traces", nargs="+",
                    help="unified trace files; adds a trace axis, sizes the device to "
                         "each trace and replays it twice (first pass = warmup)")
    a = ap.parse_args()

    grid = dict(GRIDS[a.grid])
    common = {"blocks": a.blocks, "ppb": a.ppb, "warmup": a.warmup, "writes": a.writes}
    if a.traces:
        grid.pop("workload", None)
        grid = {"trace": a.traces, **grid}
        common = {"ppb": a.ppb, "size-to-trace": 1, "trace-loops": 2,
                  "trace-warmup-loops": 1}
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]

    with ThreadPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(lambda p: run_one(p, common), combos))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in keys, k))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    errors = [r for r in rows if "error" in r]
    print(f"{len(rows)} runs -> {a.out}" + (f", {len(errors)} errors" if errors else ""))
    for r in errors:
        print("  ", {k: r[k] for k in keys}, r["error"], file=sys.stderr)


if __name__ == "__main__":
    main()
