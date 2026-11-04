#!/usr/bin/env bash
# Is the learned predictor limited by the model or by the information it gets?
#   1. Same six features, bigger models (deeper tree, 100-tree boosted ensemble).
#   2. Same models, three extra features (hour of day, position on the drive,
#      wall-clock time since the last write).
#   3. Learning curve: train on 1%..100% of the rows.
# Train on some volumes, score on others (held-out R^2 / accuracy and WAF).
#   bash scripts/ml_compare.sh [trace dir] [op]
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:-trace/unified}
op=${2:-0.07}
train="hm_0 proj_0 stg_0 usr_0 src2_0"
test="mds_0 rsrch_0 src1_2 ts_0 prxy_0 wdev_0 web_0"
out=results/ml_cmp
mkdir -p "$out"
run="--size-to-trace --trace-loops=2 --trace-warmup-loops=1 --op=$op"
make -s build/ftlsim

csvs() { local s=""; for v in $1; do s+="${s:+,}$out/$v.csv"; done; echo "$s"; }
for v in $train $test; do
  [[ -s $out/$v.csv ]] || ./build/ftlsim --trace="$dir/$v.csv" $run \
      --dump-features="$out/$v.csv" --dump-sample=0.05 >/dev/null 2>&1
done
tr=$(csvs "$train")
te=$(csvs "$test")
train_model() {  # name, extra args
  local name=$1; shift
  echo "== $name"
  uv run -q scripts/train_predictor.py "$tr" --test "$te" -o "$out/$name.model" "$@" \
    | grep -E "accuracy|R\^2|model:|importance"
}

# 1 + 2: models.
models="tree6 tree12 gbm tree6+ gbm+"
train_model tree6  --max-depth 6
train_model tree12 --max-depth 12
train_model gbm    --model gbm
train_model tree6+ --max-depth 6 --features all
train_model gbm+   --model gbm --features all

waf() { ./build/ftlsim --trace="$dir/$1.csv" $run "${@:2}" | grep '"waf"' | tr -dc '0-9.'; }
{
  printf "%-8s %7s %7s" volume none sepbit
  for m in $models; do printf " %7s" "$m"; done
  printf " %7s\n" oracle
  for v in $test; do
    printf "%-8s %7.3f %7.3f" "$v" "$(waf $v --placement=none)" "$(waf $v --placement=sepbit)"
    for m in $models; do
      printf " %7.3f" "$(waf $v --placement=ml --model=$out/$m.model --ml-gc=sepbit)"
    done
    printf " %7.3f\n" "$(waf $v --placement=oracle)"
  done
} | tee "$out/waf.txt"

# 3. Learning curve (held-out R^2 only).
echo "== learning curve (gbm+, held-out volumes)"
for f in 0.01 0.03 0.1 0.3 1.0; do
  printf "train-frac %-5s " "$f"
  uv run -q scripts/train_predictor.py "$tr" --test "$te" -o "$out/lc.model" \
      --model gbm --features all --train-frac "$f" | grep -E "train rows|R\^2" \
      | sed 's/  test rows.*//' | tr '\n' ' '
  echo
done
