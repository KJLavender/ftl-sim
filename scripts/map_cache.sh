#!/usr/bin/env bash
# DRAM-less controllers keep only part of the L2P table in SRAM / HMB. How
# much extra flash traffic does that cost, and does hot/cold placement help?
# Sweeps the cached fraction of the map on the MSR volumes.
#   bash scripts/map_cache.sh [trace dir] [op] [jobs]
# For scale: a 1 TB drive's full table is ~1 GB; a 64 MB HMB is ~1/16 of it,
# a few MB of SRAM ~1/256 to 1/512.
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:-trace/unified}
op=${2:-0.07}
jobs=${3:-4}
out=results/map_cache
mkdir -p "$out"
make -s build/ftlsim
bin=$out/ftlsim.bin
cp build/ftlsim "$bin"
export bin dir op out
csv=$out/map_op$op.csv

one() {  # volume placement ratio -> one CSV row
  local v=$1 p=$2 r=$3
  "$bin" --trace="$dir/$v.csv" --size-to-trace --trace-loops=2 --trace-warmup-loops=1 \
      --op="$op" --placement="$p" --map-cache="$r" |
    awk -v v="$v" -v p="$p" -v r="$r" -F': ' '
      /"waf"/ {w=$2} /"waf_with_map"/ {wm=$2} /"map_hit_rate"/ {h=$2}
      /"map_reads_per_host_write"/ {mr=$2}
      END {gsub(/,/,"",w); gsub(/,/,"",wm); gsub(/,/,"",h); gsub(/,/,"",mr);
           print v "," p "," r "," w "," wm "," h "," mr}'
}
export -f one

if [[ ! -s $csv ]]; then
  echo "volume,placement,ratio,waf,waf_with_map,map_hit_rate,map_reads_per_write" > "$csv"
  for v in $(ls "$dir" | sed -n 's/\.csv$//p'); do
    for p in none sepbit sepbit4; do
      for r in 0.0009765625 0.00390625 0.015625 0.0625 0.25; do
        echo "$v $p $r"
      done
    done
  done | xargs -P "$jobs" -n 3 bash -c 'one "$@"' _ >> "$csv"
fi
uv run -q --with numpy python scripts/map_cache_stats.py "$csv" | tee "$out/summary_op$op.txt"
