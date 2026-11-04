#!/usr/bin/env bash
# Train the lifetime predictor on one run (seed 1) and evaluate WAF on an
# unseen run (seed 2) against the rule baselines and the perfect predictor.
#   bash scripts/ml_pipeline.sh [workload] [op] [max_depth] [classify|regress]
set -euo pipefail
cd "$(dirname "$0")/.."
wl=${1:-zipf}
op=${2:-0.14}
depth=${3:-6}
target=${4:-regress}
out=results/ml
mkdir -p "$out"
dev="--blocks=2048 --ppb=256 --op=$op --warmup=3 --writes=5 --workload=$wl"

make -s build/ftlsim
./build/ftlsim $dev --seed=1 --dump-features="$out/${wl}_s1.csv" --dump-sample=0.1 >/dev/null
uv run -q scripts/train_predictor.py "$out/${wl}_s1.csv" -o "$out/${wl}.tree" --max-depth "$depth" --target "$target"

printf "\n%-26s %s\n" "placement (seed 2)" "WAF"
for p in none sepbit "ml --model=$out/${wl}.tree" \
         "ml --model=$out/${wl}.tree --ml-gc=sepbit" "fdp --fdp-gc=initial" oracle; do
  waf=$(./build/ftlsim $dev --seed=2 --placement=$p | grep '"waf"' | tr -dc '0-9.')
  label=${p%% *}
  [[ $p == *ml-gc=sepbit* ]] && label="ml + sepbit GC"
  [[ $p == fdp* ]] && label="perfect-host (fdp-initial)"
  printf "%-26s %s\n" "$label" "$waf"
done
./build/ftlsim $dev --seed=2 --placement=ml --model="$out/${wl}.tree" \
  | grep -E '"(model_nodes|model_depth|model_bytes|feature_state_bytes)"'
