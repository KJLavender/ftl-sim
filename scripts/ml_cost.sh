#!/usr/bin/env bash
# Two changes to the learned predictor, tested separately and together:
#   * asymmetric loss: predict a high quantile of the lifetime (q = 0.7, 0.85)
#     so the model is slow to call data hot;
#   * SepBIT's moving threshold ell instead of fixed class boundaries:
#     predicted lifetime < ell -> hot group (ell2), plus 4 ell / 16 ell
#     splits for the rest (ell4). sepbit4 is the rule-only control with the
#     same four groups, so ell4 vs sepbit4 is what the model itself adds.
# Same 5 training / 7 test volumes as ml_compare.sh, SepBIT GC everywhere.
#   bash scripts/ml_cost.sh [trace dir] [op]
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:-trace/unified}
op=${2:-0.07}
train="hm_0 proj_0 stg_0 usr_0 src2_0"
test="mds_0 rsrch_0 src1_2 ts_0 prxy_0 wdev_0 web_0"
feat=results/ml_cmp   # feature dumps, shared with ml_compare.sh
out=results/ml_cost
mkdir -p "$feat" "$out"
run="--size-to-trace --trace-loops=2 --trace-warmup-loops=1 --op=$op"
make -s build/ftlsim

csvs() { local s=""; for v in $1; do s+="${s:+,}$feat/$v.csv"; done; echo "$s"; }
for v in $train $test; do
  [[ -s $feat/$v.csv ]] || ./build/ftlsim --trace="$dir/$v.csv" $run \
      --dump-features="$feat/$v.csv" --dump-sample=0.05 >/dev/null 2>&1
done
tr=$(csvs "$train")
te=$(csvs "$test")
train_model() {  # name, extra args
  local name=$1; shift
  echo "== $name"
  uv run -q scripts/train_predictor.py "$tr" --test "$te" -o "$out/$name.model" "$@" \
    | grep -E "accuracy|R\^2|too short|model:"
}

models="tree tree_q70 tree_q85 gbm gbm_q70"
train_model tree     --raw
train_model tree_q70 --raw --quantile 0.7
train_model tree_q85 --raw --quantile 0.85
train_model gbm      --model gbm
train_model gbm_q70  --model gbm --quantile 0.7

waf() { ./build/ftlsim --trace="$dir/$1.csv" $run "${@:2}" | grep '"waf"' | tr -dc '0-9.'; }
csv=$out/waf.csv
echo "volume,model,mode,waf" > "$csv"
for v in $test; do
  echo "$v,none,-,$(waf $v --placement=none)" >> "$csv"
  echo "$v,sepbit,-,$(waf $v --placement=sepbit)" >> "$csv"
  # Control for ell4: same four host groups, split by previous lifetime.
  echo "$v,sepbit4,-,$(waf $v --placement=sepbit4)" >> "$csv"
  for m in $models; do
    for mode in sepbit ell2 ell4; do
      echo "$v,$m,$mode,$(waf $v --placement=ml --model=$out/$m.model --ml-gc=$mode)" >> "$csv"
    done
  done
done

# Mean WAF per (model, mode) and how many volumes beat SepBIT.
awk -F, 'NR > 1 {
  key = $2 "," $3; sum[key] += $4; n[key]++
  if ($2 == "sepbit") sep[$1] = $4; else w[key "," $1] = $4
} END {
  printf "%-10s %-7s %6s %s\n", "model", "mode", "waf", "beats-sepbit"
  for (k in sum) {
    split(k, p, ","); wins = 0
    for (v in sep) if ((k "," v) in w && w[k "," v] < sep[v]) wins++
    printf "%-10s %-7s %6.3f %d\n", p[1], (p[2] == "sepbit" ? "fixed" : p[2]), sum[k] / n[k], wins
  }
}' "$csv" | sort -k3 -n | tee "$out/summary.txt"

# Same tree at 14% OP: does the model still add anything over the control?
run14="--size-to-trace --trace-loops=2 --trace-warmup-loops=1 --op=0.14"
{
  printf "%-8s %8s %8s %8s %9s\n" volume none sepbit sepbit4 tree-ell4
  for v in $test; do
    w14() { ./build/ftlsim --trace="$dir/$v.csv" $run14 "$@" | grep '"waf"' | tr -dc '0-9.'; }
    printf "%-8s %8.3f %8.3f %8.3f %9.3f\n" "$v" "$(w14 --placement=none)" \
      "$(w14 --placement=sepbit)" "$(w14 --placement=sepbit4)" \
      "$(w14 --placement=ml --model=$out/tree.model --ml-gc=ell4)"
  done
} | tee "$out/op14.txt"
