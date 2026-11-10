"""Summarize scripts/map_cache.sh: cost of a partial L2P cache (DRAM-less).

  python scripts/map_cache_stats.py results/map_cache/map_op0.07.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
by = defaultdict(list)
for r in rows:
    by[(r["placement"], float(r["ratio"]))].append(r)
vols = sorted({r["volume"] for r in rows})
print(f"{len(vols)} volumes; means over volumes. Extra = map write-backs per host write.")
print(f"{'cache':>8} {'placement':<9} {'hit rate':>8} {'map reads/w':>11} "
      f"{'WAF data':>8} {'WAF +map':>8} {'extra':>6}")
for ratio in sorted({k[1] for k in by}):
    for p in ("none", "sepbit", "sepbit4"):
        rs = by.get((p, ratio), [])
        if len(rs) != len(vols):
            continue
        f = lambda k: np.mean([float(r[k]) for r in rs])
        print(f"1/{round(1 / ratio):<6} {p:<9} {f('map_hit_rate'):>8.3f} "
              f"{f('map_reads_per_write'):>11.3f} {f('waf'):>8.3f} {f('waf_with_map'):>8.3f} "
              f"{f('waf_with_map') - f('waf'):>6.3f}")
