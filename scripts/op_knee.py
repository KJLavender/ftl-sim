#!/usr/bin/env python3
"""Q2: how much WAF does each extra percent of spare space buy?

Prints, per series, the mean WAF at each OP and the WAF saved per +1% OP
between neighbouring points. The "knee" is the first OP after which one
more percent saves less than a quarter of what it saved at the start.

  python3 scripts/op_knee.py results/msr_op.csv
  python3 scripts/op_knee.py results/op_sweep.csv --group workload
"""
import argparse
import csv
from collections import defaultdict
from statistics import mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--group", default="placement",
                    help="column that separates series besides gc (default placement)")
    ap.add_argument("--ratio", type=float, default=0.25)
    a = ap.parse_args()

    with open(a.csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r.get("error")]
    series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        series[(r["gc"], r[a.group])][float(r["op"])].append(float(r["waf"]))

    for (gc, grp), by_op in sorted(series.items()):
        ops = sorted(by_op)
        waf = [mean(by_op[o]) for o in ops]
        gains = [(waf[i] - waf[i + 1]) / (100 * (ops[i + 1] - ops[i]))
                 for i in range(len(ops) - 1)]
        knee = next((ops[i] for i, g in enumerate(gains) if g < a.ratio * gains[0]), None)
        print(f"{gc} / {grp}")
        print("  OP %      " + "".join(f"{100 * o:>7.0f}" for o in ops))
        print("  WAF       " + "".join(f"{w:>7.3f}" for w in waf))
        print("  saved/1%  " + "       " + "".join(f"{g:>7.3f}" for g in gains))
        print(f"  knee: {'%.0f%%' % (100 * knee) if knee is not None else 'none in range'}")


if __name__ == "__main__":
    main()
