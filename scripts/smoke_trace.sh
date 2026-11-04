#!/usr/bin/env bash
# End-to-end check of the trace path: fake Alibaba trace -> converter -> ftlsim --verify.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
python3 - "$tmp/ali.csv" <<'PY'
import random, sys
random.seed(1)
with open(sys.argv[1], "w") as f:
    t = 1_000_000
    for i in range(200_000):
        dev = random.choice([7, 7, 7, 9])
        hot = random.random() < 0.8
        page = random.randrange(0, 4096) if hot else random.randrange(4096, 65536)
        op = "W" if random.random() < 0.7 else "R"
        size = 4096 * random.choice([1, 1, 2, 8])
        t += random.randrange(1, 50)
        f.write(f"{dev},{op},{page * 4096},{size},{t}\n")
PY
python3 tools/convert_trace.py alibaba "$tmp/ali.csv" --volume 7 -o "$tmp/vol7.csv"
./build/ftlsim --trace="$tmp/vol7.csv" --size-to-trace --ppb=64 --op=0.14 \
  --trace-loops=3 --trace-warmup-loops=1 --placement=sepbit --verify --check-every=20000 \
  | grep -E '"(logical_pages|host_writes|host_reads|waf|trace_wrapped_requests)"'
rm -rf "$tmp"
echo "smoke_trace OK"
