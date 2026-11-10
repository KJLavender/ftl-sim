"""Summarize scripts/ml_seeds.sh output: is the model's gain over the control real?

  python scripts/seed_stats.py results/ml_seeds/waf_op0.07.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
base = {(r["volume"], r["method"]): float(r["waf"]) for r in rows if r["seed"] == "-"}
model = defaultdict(dict)  # seed -> volume -> waf
for r in rows:
    if r["method"] == "ell4":
        model[r["seed"]][r["volume"]] = float(r["waf"])
vols = sorted({v for v, _ in base})
seeds = sorted(model, key=int)

sep = np.mean([base[(v, "sepbit")] for v in vols])
ctl = np.mean([base[(v, "sepbit4")] for v in vols])
per_seed = np.array([np.mean([model[s][v] for v in vols]) for s in seeds])
gains = ctl - per_seed
print(f"volumes {len(vols)}, seeds {len(seeds)}")
print(f"SepBIT {sep:.3f}   4-group control {ctl:.3f}")
print("model per seed: " + "  ".join(f"{s}:{w:.3f}" for s, w in zip(seeds, per_seed)))
print(f"model mean {per_seed.mean():.3f} +- {per_seed.std(ddof=1):.3f} (std over seeds)")
print(f"gain over control {gains.mean():.3f} +- {gains.std(ddof=1):.3f}; "
      f"min {gains.min():.3f}, max {gains.max():.3f}")
print(f"share of the SepBIT -> model gain from the model: "
      f"{gains.mean() / (sep - per_seed.mean()):.2f}")

# Per volume: average the model over seeds, compare with the control.
diff = np.array([base[(v, "sepbit4")] - np.mean([model[s][v] for s in seeds]) for v in vols])
wins = int((diff > 0).sum())
t = stats.ttest_rel([base[(v, "sepbit4")] for v in vols],
                    [np.mean([model[s][v] for s in seeds]) for v in vols])
w = stats.wilcoxon(diff) if len(vols) >= 6 else None
print("per volume (control - model, seed mean): " +
      "  ".join(f"{v}:{d:+.3f}" for v, d in zip(vols, diff)))
print(f"model beats control on {wins}/{len(vols)} volumes; "
      f"paired t-test p={t.pvalue:.3f}" + (f", Wilcoxon p={w.pvalue:.3f}" if w else ""))
# Every (seed, volume) pair.
pairs = [(base[(v, "sepbit4")] - model[s][v]) for s in seeds for v in vols]
print(f"(seed, volume) pairs where the model wins: {sum(p > 0 for p in pairs)}/{len(pairs)}")
