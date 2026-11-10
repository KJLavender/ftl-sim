"""Summarize scripts/ml_splits.sh: mean WAF per split setting, rule vs model.

  python scripts/splits_stats.py results/ml_splits/waf.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
waf = defaultdict(dict)  # (op, splits, groups, method) -> volume -> waf
for r in rows:
    waf[(r["op"], r["splits"], int(r["groups"]), r["method"])][r["volume"]] = float(r["waf"])

for op in sorted({k[0] for k in waf}):
    settings = sorted({(k[2], k[1]) for k in waf if k[0] == op}, key=lambda s: (s[0], s[1]))
    base = waf.get((op, "", 2, "rule"), {})
    print(f"\nOP {op}  (SepBIT = rule with 2 groups: {np.mean(list(base.values())):.3f})")
    print(f"{'groups':>6} {'splits (x ell)':<16} {'rule':>7} {'model':>7} "
          f"{'model gain':>10} {'model wins':>10}")
    for g, sp in settings:
        r = waf.get((op, sp, g, "rule"), {})
        m = waf.get((op, sp, g, "model"), {})
        vols = sorted(set(r) & set(m))
        if not vols:
            continue
        rm = np.mean([r[v] for v in vols])
        mm = np.mean([m[v] for v in vols])
        wins = sum(m[v] < r[v] for v in vols)
        print(f"{g:>6} {sp or '-':<16} {rm:>7.3f} {mm:>7.3f} {rm - mm:>+10.3f} "
              f"{wins:>7}/{len(vols)}")
