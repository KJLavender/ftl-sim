#!/usr/bin/env bash
# Cross-volume generalization: train one lifetime predictor on some trace
# volumes, then evaluate WAF on volumes it has never seen.
#   bash scripts/ml_traces.sh <unified trace dir> "<train vols>" "<test vols>" [op] [depth] [split]
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:?trace dir}
train=${2:?train volumes}
test=${3:?test volumes}
op=${4:-0.14}
depth=${5:-6}
split=${6:-}
out=results/ml_traces
mkdir -p "$out"
run="--size-to-trace --trace-loops=2 --trace-warmup-loops=1 --op=$op"

make -s build/ftlsim
# 1. Dump features from the training volumes and concatenate them.
merged="$out/train.csv"
rm -f "$merged"
for v in $train; do
  [[ -s $out/$v.csv ]] || ./build/ftlsim --trace="$dir/$v.csv" $run \
      --dump-features="$out/$v.csv" --dump-sample=0.05 >/dev/null
  if [[ -s $merged ]]; then tail -n +2 "$out/$v.csv" >> "$merged"; else cp "$out/$v.csv" "$merged"; fi
done

# 2. Train.
model="$out/msr.tree"
uv run -q scripts/train_predictor.py "$merged" -o "$model" --max-depth "$depth" \
  ${split:+--split "$split"}

# 3. Evaluate on unseen volumes.
waf() { ./build/ftlsim --trace="$dir/$1.csv" $run "${@:2}" | grep '"waf"' | tr -dc '0-9.'; }
printf "\n%-8s %8s %8s %8s %12s %8s\n" volume none sepbit ml ml+sepbitGC oracle
for v in $test; do
  printf "%-8s %8.3f %8.3f %8.3f %12.3f %8.3f\n" "$v" \
    "$(waf $v --placement=none)" "$(waf $v --placement=sepbit)" \
    "$(waf $v --placement=ml --model=$model)" \
    "$(waf $v --placement=ml --model=$model --ml-gc=sepbit)" \
    "$(waf $v --placement=oracle)"
done
