#!/usr/bin/env python3
"""Print a WAF pivot table from an experiment CSV.

Example:
  python3 scripts/summarize.py results/placement.csv --rows workload,gc,op --cols placement
"""
import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--rows", required=True, help="comma-separated row keys")
    ap.add_argument("--cols", required=True, help="column key")
    ap.add_argument("--metric", default="waf")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    row_keys = a.rows.split(",")
    table = defaultdict(dict)
    cols = []
    for r in rows:
        rk = tuple(r[k] for k in row_keys)
        c = r[a.cols]
        if c not in cols:
            cols.append(c)
        table[rk][c] = r.get("error") and "ERR" or f"{float(r[a.metric]):.3f}"

    widths = [max(len(k), *(len(rk[i]) for rk in table)) for i, k in enumerate(row_keys)]
    cw = max(8, *(len(c) for c in cols))
    print("  ".join(k.ljust(w) for k, w in zip(row_keys, widths)) + "  " +
          "  ".join(c.rjust(cw) for c in cols))
    for rk in table:
        print("  ".join(v.ljust(w) for v, w in zip(rk, widths)) + "  " +
              "  ".join(table[rk].get(c, "-").rjust(cw) for c in cols))


if __name__ == "__main__":
    main()
