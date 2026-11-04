#!/usr/bin/env python3
"""Merge experiment CSVs and shorten trace paths to volume names.

Example:
  python3 scripts/merge_results.py results/msr_grid.csv results/msr_grid2.csv -o results/msr.csv
"""
import argparse
import csv
import os


def volume(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    rows, fields = [], []
    for p in a.inputs:
        with open(p, encoding="utf-8") as f:
            r = csv.DictReader(f)
            fields += [k for k in r.fieldnames if k not in fields]
            rows += list(r)
    for r in rows:
        for k in ("trace", "workload"):
            if r.get(k):
                r[k] = volume(r[k])
    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {a.output}")


if __name__ == "__main__":
    main()
