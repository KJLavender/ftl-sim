#!/usr/bin/env bash
# Unpack the SNIA MSR Cambridge tarballs and convert selected volumes to the
# unified format (writes only, 1 MiB regions compacted to the touched footprint).
#   bash scripts/prep_msr.sh <dir with msr-cambridge*.tar> [volume ...]
set -euo pipefail
cd "$(dirname "$0")/.."
src=${1:?usage: prep_msr.sh <tar dir> [volume ...]}
shift
# Write-heavy volumes commonly used in FTL papers.
vols=("$@")
[[ ${#vols[@]} -eq 0 ]] && vols=(hm_0 prxy_0 proj_0 src1_2 stg_0 usr_0 wdev_0 web_0 mds_0 rsrch_0 ts_0 src2_0)

raw="$src/raw"
out="$src/unified"
mkdir -p "$raw" "$out"
for t in "$src"/msr-cambridge*.tar; do
  tar -xf "$t" -C "$raw" --skip-old-files
done

for v in "${vols[@]}"; do
  f="$raw/MSR-Cambridge/$v.csv.gz"
  if [[ ! -f $f ]]; then
    echo "skip $v (not found)" >&2
    continue
  fi
  [[ -s $out/$v.csv ]] && { echo "have $v"; continue; }
  echo -n "$v: " >&2
  python3 tools/convert_trace.py msr "$f" --writes-only --compact -o "$out/$v.csv"
done
ls -la "$out"
