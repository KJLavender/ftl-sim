#!/usr/bin/env bash
# Do more host groups (finer splits of ell) keep helping? Sweep the split
# points for the rule (sepbit-split, previous lifetime) and the learned model
# (--ml-gc=ell, predicted lifetime) on the 7 test volumes, at two OPs.
#   bash scripts/ml_splits.sh [trace dir] [model]
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:-trace/unified}
model=${2:-results/ml_cost/tree.model}   # trained by ml_cost.sh
test="mds_0 rsrch_0 src1_2 ts_0 prxy_0 wdev_0 web_0"
splits=("" "4" "4,16" "2,8" "8,64" "2,4,8,16" "2,4,8,16,32,64")
out=results/ml_splits
mkdir -p "$out"
[[ -s $model ]] || { echo "missing $model (run scripts/ml_cost.sh first)"; exit 1; }
make -s build/ftlsim
bin=$out/ftlsim.bin
cp build/ftlsim "$bin"
csv=$out/waf.csv
[[ -s $csv ]] || echo "op,splits,groups,volume,method,waf" > "$csv"
for op in 0.07 0.14; do
  run="--size-to-trace --trace-loops=2 --trace-warmup-loops=1 --op=$op"
  for sp in "${splits[@]}"; do
    groups=$(( 2 + $(echo "$sp" | tr ',' '\n' | grep -c . || true) ))
    for v in $test; do
      key="$op,\"$sp\",$groups,$v"
      grep -q "^$key,rule," "$csv" || echo "$key,rule,$("$bin" --trace="$dir/$v.csv" $run \
          --placement=sepbit-split --ell-splits="$sp" | grep '"waf"' | tr -dc '0-9.')" >> "$csv"
      grep -q "^$key,model," "$csv" || echo "$key,model,$("$bin" --trace="$dir/$v.csv" $run \
          --placement=ml --model="$model" --ml-gc=ell --ell-splits="$sp" | grep '"waf"' | tr -dc '0-9.')" >> "$csv"
    done
    echo "op $op splits [$sp] done"
  done
done
uv run -q --with numpy python scripts/splits_stats.py "$csv" | tee "$out/summary.txt"
