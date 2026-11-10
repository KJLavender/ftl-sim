#!/usr/bin/env bash
# How much of the model's gain survives a different random seed?
# For each seed: re-sample the training rows, retrain the tree, run ell4 on the
# 7 test volumes. SepBIT and the 4-group control are deterministic, so they are
# run once. Prints the gain over the control as mean +- std across seeds and a
# paired test across volumes.
#   bash scripts/ml_seeds.sh [trace dir] [op] [seeds]
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:-trace/unified}
op=${2:-0.07}
seeds=${3:-"1 2 3 4 5"}
train="hm_0 proj_0 stg_0 usr_0 src2_0"
test="mds_0 rsrch_0 src1_2 ts_0 prxy_0 wdev_0 web_0"
out=results/ml_seeds
mkdir -p "$out"
run="--size-to-trace --trace-loops=2 --trace-warmup-loops=1 --op=$op"
make -s build/ftlsim
bin=$out/ftlsim.bin   # private copy: rebuilding build/ftlsim mid-run is safe
cp build/ftlsim "$bin"
waf() { "$bin" --trace="$dir/$1.csv" $run "${@:2}" | grep '"waf"' | tr -dc '0-9.'; }

csv=$out/waf_op$op.csv
if [[ ! -s $csv ]]; then
  echo "seed,volume,method,waf" > "$csv"
  for v in $test; do
    echo "-,$v,sepbit,$(waf $v --placement=sepbit)" >> "$csv"
    echo "-,$v,sepbit4,$(waf $v --placement=sepbit4)" >> "$csv"
  done
fi
for s in $seeds; do
  grep -q "^$s,.*,ell4," "$csv" && continue
  tr=""
  for v in $train; do
    f=$out/seed$s/$v.csv
    mkdir -p "$out/seed$s"
    [[ -s $f ]] || "$bin" --trace="$dir/$v.csv" $run --seed=$s \
        --dump-features="$f" --dump-sample=0.05 >/dev/null 2>&1
    tr+="${tr:+,}$f"
  done
  model=$out/seed$s/tree.model
  uv run -q scripts/train_predictor.py "$tr" -o "$model" --raw --seed "$s" >/dev/null
  for v in $test; do
    echo "$s,$v,ell4,$(waf $v --placement=ml --model=$model --ml-gc=ell4)" >> "$csv"
  done
  echo "seed $s done"
done
uv run -q --with numpy --with scipy python scripts/seed_stats.py "$csv" | tee "$out/summary_op$op.txt"
