#!/usr/bin/env bash
# Produce every CSV the figures need. Real-trace parts are skipped if
# trace/unified/ is missing.
#   bash scripts/figure_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
make -s
mkdir -p results/erase

python3 scripts/experiment.py --grid op_sweep --out results/op_sweep.csv
python3 scripts/experiment.py --grid placement --out results/placement.csv

# Erase-count distribution: 80% of the data is never rewritten after the fill.
dev="--blocks=2048 --ppb=256 --op=0.07 --warmup=3 --writes=10 --workload=hotcold --hot-access=1.0"
./build/ftlsim $dev --dump-erase=results/erase/off.txt > /dev/null
./build/ftlsim $dev --wear-aware --dump-erase=results/erase/dynamic.txt > /dev/null
./build/ftlsim $dev --static-wl=20 --dump-erase=results/erase/static20.txt > /dev/null

if ls trace/unified/*.csv >/dev/null 2>&1; then
  python3 scripts/experiment.py --grid trace_op --traces trace/unified/*.csv \
    --out results/msr_op_raw.csv --jobs 8
  python3 scripts/merge_results.py results/msr_op_raw.csv -o results/msr_op.csv
fi
echo figure-data-done
