"""Summarize scripts/latency.sh: burst vs. paced GC, host write latency.

  python scripts/latency_stats.py results/latency/latency_op0.07_load0.2.csv

Latencies are medians over volumes (one overloaded volume would swamp a mean).
"""
import csv
import sys
from collections import defaultdict

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
by = defaultdict(list)
for r in rows:
    by[(r["placement"], int(r["soft"]))].append(r)
vols = sorted({r["volume"] for r in rows})
print(f"{len(vols)} volumes. WAF / busy: mean over volumes; latency (ms): median over volumes.")
print(f"{'placement':<9} {'GC':<6} {'WAF':>6} {'busy':>5} {'mean':>7} {'p99':>7} "
      f"{'p99.9':>7} {'max':>7} {'max copies':>10} {'busy>1':>6}")
for p in ("none", "sepgc", "sepbit"):
    for k in (0, 2):
        rs = by.get((p, k), [])
        if len(rs) != len(vols):
            continue
        avg = lambda c: np.mean([float(r[c]) for r in rs])
        med = lambda c: np.median([float(r[c]) for r in rs]) / 1000
        over = sum(float(r["busy"]) >= 1 for r in rs)
        print(f"{p:<9} {'paced' if k else 'burst':<6} {avg('waf'):>6.3f} {avg('busy'):>5.2f} "
              f"{med('mean_us'):>7.2f} {med('p99_us'):>7.2f} {med('p999_us'):>7.2f} "
              f"{med('max_us'):>7.2f} {np.median([int(r['max_copies']) for r in rs]):>10.0f} "
              f"{over:>6}")
