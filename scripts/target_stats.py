"""Summarize scripts/ml_ell_target.sh: does training relative to ell help?

  python scripts/target_stats.py results/ml_ell_target/waf_op0.07.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
ctl = {r["volume"]: float(r["waf"]) for r in rows if r["method"] == "sepbit4"}
vols = sorted(ctl)
by = defaultdict(lambda: defaultdict(dict))  # method -> seed -> volume -> waf
for r in rows:
    if r["seed"] != "-":
        by[r["method"]][r["seed"]][r["volume"]] = float(r["waf"])

names = {"abs": "lifetime (capacity scale), as before",
         "rel": "log2(lifetime / ell)",
         "grp": "host group directly"}
print(f"4-group control (no model): {np.mean(list(ctl.values())):.3f}")
for m in ("abs", "rel", "grp"):
    seeds = sorted(s for s in by[m] if set(by[m][s]) >= set(vols))
    if not seeds:
        continue
    means = np.array([np.mean([by[m][s][v] for v in vols]) for s in seeds])
    gain = np.mean(list(ctl.values())) - means
    per_vol = np.array([np.mean([by[m][s][v] for s in seeds]) for v in vols])
    wins = int((per_vol < np.array([ctl[v] for v in vols])).sum())
    p = stats.ttest_rel([ctl[v] for v in vols], per_vol).pvalue
    sd = f" +- {means.std(ddof=1):.3f}" if len(seeds) > 1 else ""
    print(f"{m:<4} {names[m]:<40} WAF {means.mean():.3f}{sd}  gain over control "
          f"{gain.mean():+.3f}  wins {wins}/{len(vols)}  paired p={p:.3f}  ({len(seeds)} seeds)")
# Head to head: relative-target vs absolute-target model, same seeds.
for m in ("rel", "grp"):
    common = sorted(set(by[m]) & set(by["abs"]))
    if not common:
        continue
    d = [np.mean([by["abs"][s][v] - by[m][s][v] for s in common]) for v in vols]
    print(f"{m} vs abs: mean improvement {np.mean(d):+.3f}, better on "
          f"{sum(x > 0 for x in d)}/{len(vols)} volumes, paired p="
          f"{stats.ttest_1samp(d, 0).pvalue:.3f}")
