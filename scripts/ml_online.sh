#!/usr/bin/env bash
# Honest "learn on the drive" test: split each volume's trace in time, train a
# tree only on the first half, and measure WAF only on the second half.
# Compared with SepBIT, the 4-group SepBIT control, and the cross-volume model
# from ml_cost.sh. All learned runs use --ml-gc=ell4.
#   bash scripts/ml_online.sh [trace dir] [op]
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:-trace/unified}
op=${2:-0.07}
test="mds_0 rsrch_0 src1_2 ts_0 prxy_0 wdev_0 web_0"
cross=results/ml_cost/tree.model   # trained on other volumes (ml_cost.sh)
out=results/ml_online
mkdir -p "$out"
run="--size-to-trace --trace-loops=2 --trace-warmup-loops=1 --op=$op"
make -s build/ftlsim
[[ -s $cross ]] || { echo "run scripts/ml_cost.sh first ($cross missing)"; exit 1; }

waf() { ./build/ftlsim --trace="$1" $run "${@:2}" | grep '"waf"' | tr -dc '0-9.'; }
printf "%-8s %8s %8s %10s %10s\n" volume sepbit sepbit4 cross-vol own-past | tee "$out/waf.txt"
for v in $test; do
  a=$out/${v}_first.csv b=$out/${v}_second.csv
  if [[ ! -s $b ]]; then
    # Split at half of the trace's time span.
    end=$(tail -n 1 "$dir/$v.csv" | cut -d, -f1)
    awk -F, -v mid=$((end / 2)) -v a="$a" -v b="$b" '{ print > ($1 < mid ? a : b) }' "$dir/$v.csv"
  fi
  feats=$out/${v}_first.feat.csv
  [[ -s $feats ]] || ./build/ftlsim --trace="$a" $run \
      --dump-features="$feats" --dump-sample=0.2 >/dev/null 2>&1
  uv run -q scripts/train_predictor.py "$feats" -o "$out/$v.model" --raw >/dev/null
  printf "%-8s %8.3f %8.3f %10.3f %10.3f\n" "$v" \
    "$(waf $b --placement=sepbit)" "$(waf $b --placement=sepbit4)" \
    "$(waf $b --placement=ml --model=$cross --ml-gc=ell4)" \
    "$(waf $b --placement=ml --model=$out/$v.model --ml-gc=ell4)" | tee -a "$out/waf.txt"
done
