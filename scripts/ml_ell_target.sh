#!/usr/bin/env bash
# Train the model on the question it actually answers. The ell modes compare a
# predicted lifetime with SepBIT's moving threshold ell, but the model in
# ml_cost.sh learns lifetime on a fixed (capacity) scale. Here it gets the
# current ell as a feature and learns either log2(lifetime / ell) (ellrel) or
# the host group itself (ellcls). Same 5 training / 7 test volumes, splits 4,16.
#   bash scripts/ml_ell_target.sh [trace dir] [op] [seeds]
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:-trace/unified}
op=${2:-0.07}
seeds=${3:-"1 2 3"}
train="hm_0 proj_0 stg_0 usr_0 src2_0"
test="mds_0 rsrch_0 src1_2 ts_0 prxy_0 wdev_0 web_0"
out=results/ml_ell_target
mkdir -p "$out"
run="--size-to-trace --trace-loops=2 --trace-warmup-loops=1 --op=$op"
make -s build/ftlsim
bin=$out/ftlsim.bin
cp build/ftlsim "$bin"
waf() { "$bin" --trace="$dir/$1.csv" $run "${@:2}" | grep '"waf"' | tr -dc '0-9.'; }

csv=$out/waf_op$op.csv
[[ -s $csv ]] || echo "seed,volume,method,waf" > "$csv"
for v in $test; do
  grep -q "^-,$v,sepbit4," "$csv" || echo "-,$v,sepbit4,$(waf $v --placement=sepbit4)" >> "$csv"
done
for s in $seeds; do
  tr=""
  for v in $train; do
    f=$out/seed$s/$v.csv
    mkdir -p "$out/seed$s"
    [[ -s $f ]] || "$bin" --trace="$dir/$v.csv" $run --seed=$s \
        --dump-features="$f" --dump-sample=0.05 >/dev/null 2>&1
    tr+="${tr:+,}$f"
  done
  # Same rows, three ways to train: absolute lifetime (as before), lifetime
  # relative to ell, and the group directly.
  [[ -s $out/seed$s/abs.model ]] || uv run -q scripts/train_predictor.py "$tr" \
      -o "$out/seed$s/abs.model" --raw --seed "$s" > "$out/seed$s/abs.txt"
  [[ -s $out/seed$s/rel.model ]] || uv run -q scripts/train_predictor.py "$tr" \
      -o "$out/seed$s/rel.model" --target ellrel --seed "$s" > "$out/seed$s/rel.txt"
  [[ -s $out/seed$s/grp.model ]] || uv run -q scripts/train_predictor.py "$tr" \
      -o "$out/seed$s/grp.model" --target ellgroup --seed "$s" > "$out/seed$s/grp.txt"
  for v in $test; do
    grep -q "^$s,$v,abs," "$csv" || echo "$s,$v,abs,$(waf $v --placement=ml \
        --model=$out/seed$s/abs.model --ml-gc=ell4)" >> "$csv"
    grep -q "^$s,$v,rel," "$csv" || echo "$s,$v,rel,$(waf $v --placement=ml \
        --model=$out/seed$s/rel.model --ml-gc=ellrel --ell-splits=4,16)" >> "$csv"
    grep -q "^$s,$v,grp," "$csv" || echo "$s,$v,grp,$(waf $v --placement=ml \
        --model=$out/seed$s/grp.model --ml-gc=ellcls --ell-splits=4,16)" >> "$csv"
  done
  echo "seed $s done"
done
uv run -q --with numpy --with scipy python scripts/target_stats.py "$csv" | tee "$out/summary_op$op.txt"
