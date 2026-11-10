#!/usr/bin/env bash
# Let the model drive GC grouping too. Host writes stay as in ml_seeds.sh /
# ml_ell_target.sh (4 groups by predicted lifetime vs ell); only the grouping
# of GC'd data changes, from SepBIT's age rule to the predicted remaining
# lifetime. Reuses the per-seed models those scripts trained.
#   bash scripts/ml_gc_model.sh [trace dir] [op]
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:-trace/unified}
op=${2:-0.07}
test="mds_0 rsrch_0 src1_2 ts_0 prxy_0 wdev_0 web_0"
out=results/ml_gc_model
mkdir -p "$out"
run="--size-to-trace --trace-loops=2 --trace-warmup-loops=1 --op=$op"
make -s build/ftlsim
bin=$out/ftlsim.bin
cp build/ftlsim "$bin"
waf() { "$bin" --trace="$dir/$1.csv" $run "${@:2}" | grep '"waf"' | tr -dc '0-9.'; }

csv=$out/waf_op$op.csv
[[ -s $csv ]] || echo "seed,volume,model,gc_rule,waf" > "$csv"
add() {  # seed model_name model_path ml_gc
  local s=$1 name=$2 path=$3 mode=$4
  [[ -s $path ]] || { echo "skip $path (missing)"; return; }
  for v in $test; do
    for rule in sepbit model; do
      grep -q "^$s,$v,$name,$rule," "$csv" || echo "$s,$v,$name,$rule,$(waf $v \
          --placement=ml --model="$path" --ml-gc="$mode" --ml-gc-rule=$rule)" >> "$csv"
    done
  done
  echo "seed $s $name done"
}
for s in 1 2 3 4 5; do add $s abs results/ml_seeds/seed$s/tree.model ell4; done
for s in 1 2 3; do add $s rel results/ml_ell_target/seed$s/rel.model ellrel; done
uv run -q --with numpy --with scipy python scripts/gc_model_stats.py "$csv" | tee "$out/summary_op$op.txt"
