#!/usr/bin/env bash
# Would an SSD gain from retraining on recent data? Split each volume's trace
# into four time quarters and always measure WAF on the last one (Q4). Only
# the training data changes:
#   stale  : Q1 only        (trained once, never updated)
#   recent : Q3 only        (retrained just before Q4)
#   all    : Q1-Q3          (retrained, keeping all history)
# Host groups by predicted lifetime vs ell (4 groups), SepBIT GC.
#   bash scripts/ml_retrain.sh [trace dir] [op]
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:-trace/unified}
op=${2:-0.07}
test="mds_0 rsrch_0 src1_2 ts_0 prxy_0 wdev_0 web_0"
out=results/ml_retrain
mkdir -p "$out"
run="--size-to-trace --trace-loops=2 --trace-warmup-loops=1 --op=$op"
make -s build/ftlsim
bin=$out/ftlsim.bin
cp build/ftlsim "$bin"
waf() { "$bin" --trace="$1" $run "${@:2}" | grep '"waf"' | tr -dc '0-9.'; }

csv=$out/waf_op$op.csv
[[ -s $csv ]] || echo "volume,method,waf" > "$csv"
for v in $test; do
  q=$out/$v
  mkdir -p "$q"
  if [[ ! -s $q/q4.csv ]]; then
    end=$(tail -n 1 "$dir/$v.csv" | cut -d, -f1)
    awk -F, -v e="$end" -v d="$q" '{ k = int(4 * $1 / (e + 1)) + 1; print > (d "/q" k ".csv") }' "$dir/$v.csv"
    cat "$q/q1.csv" "$q/q2.csv" "$q/q3.csv" > "$q/q123.csv"
  fi
  for part in q1 q3 q123; do
    [[ -s $q/$part.feat ]] || "$bin" --trace="$q/$part.csv" $run \
        --dump-features="$q/$part.feat" --dump-sample=0.2 >/dev/null 2>&1
    [[ -s $q/$part.model ]] || uv run -q scripts/train_predictor.py "$q/$part.feat" \
        -o "$q/$part.model" --raw >/dev/null
  done
  t=$q/q4.csv
  grep -q "^$v,sepbit," "$csv" || echo "$v,sepbit,$(waf $t --placement=sepbit)" >> "$csv"
  grep -q "^$v,sepbit4," "$csv" || echo "$v,sepbit4,$(waf $t --placement=sepbit4)" >> "$csv"
  for part in q1 q3 q123; do
    name=$([[ $part == q1 ]] && echo stale || ([[ $part == q3 ]] && echo recent || echo all))
    grep -q "^$v,$name," "$csv" || echo "$v,$name,$(waf $t --placement=ml \
        --model=$q/$part.model --ml-gc=ell4)" >> "$csv"
  done
  echo "$v done"
done
uv run -q --with numpy python - "$csv" <<'PY' | tee "$out/summary_op$op.txt"
import csv, sys
from collections import defaultdict
import numpy as np
w = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    w[r["method"]][r["volume"]] = float(r["waf"])
vols = sorted(w["sepbit"])
ctl = w["sepbit4"]
print(f"{'method':<8} {'mean WAF':>8} {'beats 4-group control':>22}")
for m in ("sepbit", "sepbit4", "stale", "recent", "all"):
    if set(w[m]) >= set(vols):
        wins = sum(w[m][v] < ctl[v] for v in vols)
        print(f"{m:<8} {np.mean([w[m][v] for v in vols]):>8.3f} {wins:>19}/{len(vols)}")
print("per volume: " + "  ".join(
    f"{v}: stale {w['stale'][v]:.3f} recent {w['recent'][v]:.3f} all {w['all'][v]:.3f}"
    for v in vols if all(v in w[m] for m in ("stale", "recent", "all"))))
PY
