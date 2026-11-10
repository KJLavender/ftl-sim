"""Summarize scripts/ml_gc_model.sh: SepBIT's GC age rule vs model-driven GC.

  python scripts/gc_model_stats.py results/ml_gc_model/waf_op0.07.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
w = defaultdict(dict)  # (model, rule, seed) -> volume -> waf
for r in rows:
    w[(r["model"], r["gc_rule"], r["seed"])][r["volume"]] = float(r["waf"])
vols = sorted({r["volume"] for r in rows})

for m in sorted({r["model"] for r in rows}):
    seeds = sorted({k[2] for k in w if k[0] == m
                    and set(w[(m, "sepbit", k[2])]) >= set(vols)
                    and set(w.get((m, "model", k[2]), {})) >= set(vols)}, key=int)
    if not seeds:
        continue
    age = np.array([np.mean([w[(m, "sepbit", s)][v] for v in vols]) for s in seeds])
    mod = np.array([np.mean([w[(m, "model", s)][v] for v in vols]) for s in seeds])
    diff = age - mod
    per_vol = [np.mean([w[(m, "sepbit", s)][v] - w[(m, "model", s)][v] for s in seeds])
               for v in vols]
    p = stats.ttest_1samp(per_vol, 0).pvalue
    print(f"{m}: GC by SepBIT age {age.mean():.3f} +- {age.std(ddof=1):.3f}   "
          f"GC by model {mod.mean():.3f} +- {mod.std(ddof=1):.3f}   "
          f"gain {diff.mean():+.3f} +- {diff.std(ddof=1):.3f}  ({len(seeds)} seeds)")
    print("   per volume (age - model): " +
          "  ".join(f"{v}:{d:+.3f}" for v, d in zip(vols, per_vol)) +
          f"   better on {sum(d > 0 for d in per_vol)}/{len(vols)}, paired p={p:.3f}")
