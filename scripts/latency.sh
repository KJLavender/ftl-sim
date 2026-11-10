#!/usr/bin/env bash
# Host write latency when GC runs in the write path. Compares collecting
# whole victims when free blocks run out (burst) with paced GC (a few pages
# of the current victim per host write, --gc-soft) on the MSR volumes.
#   bash scripts/latency.sh [trace dir] [op] [load] [jobs]
# load = arrival rate as a fraction of the no-GC peak (8 dies / 600 us).
set -euo pipefail
cd "$(dirname "$0")/.."
dir=${1:-trace/unified}
op=${2:-0.07}
load=${3:-0.2}
jobs=${4:-4}
out=results/latency
mkdir -p "$out"
make -s build/ftlsim
bin=$out/ftlsim.bin
cp build/ftlsim "$bin"
export bin dir op load out
csv=$out/latency_op${op}_load$load.csv

one() {  # volume placement soft -> one CSV row
  local v=$1 p=$2 k=$3
  "$bin" --trace="$dir/$v.csv" --size-to-trace --trace-loops=2 --trace-warmup-loops=1 \
      --op="$op" --placement="$p" --latency="$load" --gc-soft="$k" |
    awk -v v="$v" -v p="$p" -v k="$k" -F': ' '
      /"waf"/ {w=$2} /"latency_busy"/ {b=$2} /"latency_mean_us"/ {m=$2}
      /"latency_p99_us"/ {p99=$2} /"latency_p999_us"/ {p999=$2} /"latency_max_us"/ {mx=$2}
      /"latency_max_gc_copies"/ {mc=$2}
      END {gsub(/,/,"",w); gsub(/,/,"",b); gsub(/,/,"",m); gsub(/,/,"",p99);
           gsub(/,/,"",p999); gsub(/,/,"",mx); gsub(/,/,"",mc);
           print v "," p "," k "," w "," b "," m "," p99 "," p999 "," mx "," mc}'
}
export -f one

if [[ ! -s $csv ]]; then
  echo "volume,placement,soft,waf,busy,mean_us,p99_us,p999_us,max_us,max_copies" > "$csv"
  for v in $(ls "$dir" | sed -n 's/\.csv$//p'); do
    for p in none sepgc sepbit; do
      for k in 0 2; do
        echo "$v $p $k"
      done
    done
  done | xargs -P "$jobs" -n 3 bash -c 'one "$@"' _ >> "$csv"
fi
uv run -q --with numpy python scripts/latency_stats.py "$csv" | tee "$out/summary_op${op}_load$load.txt"
