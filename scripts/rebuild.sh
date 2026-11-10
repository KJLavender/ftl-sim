#!/usr/bin/env bash
# Power-loss recovery: how often should the L2P table be checkpointed?
# More checkpoints = more map pages written, but after a crash fewer blocks
# have to be scanned (OOB) to rebuild the table. Every rebuild is compared
# with the live table, so the run also checks the recovery is exact.
#   bash scripts/rebuild.sh [trace dir] [op] [jobs]
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:-trace/unified}
op=${2:-0.07}
jobs=${3:-4}
out=results/rebuild
mkdir -p "$out"
make -s build/ftlsim
bin=$out/ftlsim.bin
cp build/ftlsim "$bin"
export bin dir op out
csv=$out/rebuild_op$op.csv

one() {  # volume placement fraction -> one CSV row
  local v=$1 p=$2 f=$3
  "$bin" --trace="$dir/$v.csv" --size-to-trace --trace-loops=2 --trace-warmup-loops=1 \
      --op="$op" --placement="$p" --checkpoint="$f" |
    awk -v v="$v" -v p="$p" -v f="$f" -F': ' '
      /"waf"/ {w=$2} /"checkpoint_writes_per_host_write"/ {cw=$2}
      /"rebuild_scan_frac_avg"/ {sa=$2} /"rebuild_pages_max"/ {pm=$2}
      /"rebuild_mismatches"/ {mm=$2} /"rebuild_resurrected_at_end"/ {rs=$2}
      /"full_scan_frac"/ {fs=$2} /"rebuilds"/ {rb=$2}
      END {gsub(/,/,"",w); gsub(/,/,"",cw); gsub(/,/,"",sa); gsub(/,/,"",pm);
           gsub(/,/,"",mm); gsub(/,/,"",rs); gsub(/,/,"",fs); gsub(/,/,"",rb);
           print v "," p "," f "," w "," cw "," sa "," pm "," mm "," rs "," fs "," rb}'
}
export -f one

if [[ ! -s $csv ]]; then
  echo "volume,placement,frac,waf,cp_writes_per_write,scan_frac_avg,scan_pages_max,mismatches,resurrected,full_scan_frac,rebuilds" > "$csv"
  for v in $(ls "$dir" | sed -n 's/\.csv$//p'); do
    for p in none sepbit; do
      for f in 0.0625 0.25 1 4; do
        echo "$v $p $f"
      done
    done
  done | xargs -P "$jobs" -n 3 bash -c 'one "$@"' _ >> "$csv"
fi
uv run -q --with numpy python scripts/rebuild_stats.py "$csv" | tee "$out/summary_op$op.txt"
