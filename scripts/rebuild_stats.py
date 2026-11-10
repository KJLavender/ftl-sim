"""Summarize scripts/rebuild.sh: checkpoint cost vs. rebuild scan after a crash.

  python scripts/rebuild_stats.py results/rebuild/rebuild_op0.07.csv

scan = OOB pages read per rebuild, as a fraction of all physical pages.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
by = defaultdict(list)
for r in rows:
    by[(r["placement"], float(r["frac"]))].append(r)
vols = sorted({r["volume"] for r in rows})
print(f"{len(vols)} volumes; means over volumes. Checkpoint every F x capacity host writes.")
print(f"{'F':>6} {'placement':<9} {'WAF':>6} {'cp writes/w':>11} {'scan avg':>8} "
      f"{'no-cp scan':>10} {'rebuilds':>8} {'mismatch':>8} {'resurrect':>9}")
for frac in sorted({k[1] for k in by}):
    for p in ("none", "sepbit"):
        rs = by.get((p, frac), [])
        if len(rs) != len(vols):
            continue
        f = lambda k: np.mean([float(r[k]) for r in rs])
        print(f"{frac:>6g} {p:<9} {f('waf'):>6.3f} {f('cp_writes_per_write'):>11.5f} "
              f"{f('scan_frac_avg'):>8.3f} "
              f"{f('full_scan_frac'):>10.3f} {f('rebuilds'):>8.0f} "
              f"{sum(int(r['mismatches']) for r in rs):>8} "
              f"{sum(int(r['resurrected']) for r in rs):>9}")
