#!/usr/bin/env python3
"""Convert public block I/O traces to the unified ftlsim format.

Unified format (CSV, no header):
  timestamp_us,op,offset_bytes,size_bytes        op in {R, W, T}

Supported inputs:
  msr      MSR Cambridge (SNIA IOTTA):
           Timestamp(100ns FILETIME),Hostname,DiskNumber,Type,Offset,Size,ResponseTime
  alibaba  Alibaba Cloud block traces (github.com/alibaba/block-traces):
           device_id,opcode,offset,length,timestamp(us)
  tencent  Tencent CBS (SNIA IOTTA):
           Timestamp(s),Offset(512B sectors),Size(sectors),IOType(0=R,1=W),VolumeID

Example:
  python3 tools/convert_trace.py alibaba io_traces.csv --volume 28 --limit 5000000 -o vol28.csv
"""
import argparse
import gzip
import sys


def open_any(path):
    if path == "-":
        return sys.stdin
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def parse_msr(f, p):
    # MSR 0, 1, 2... disks share one file per host; volume = DiskNumber.
    for line in f:
        p_ = line.rstrip("\n").split(",")
        if len(p_) < 6:
            continue
        ts, _host, disk, typ, off, size = p_[:6]
        if p.volume is not None and disk != p.volume:
            continue
        op = "R" if typ.lower().startswith("r") else "W"
        yield int(ts) // 10, op, int(off), int(size)


def parse_alibaba(f, p):
    for line in f:
        p_ = line.rstrip("\n").split(",")
        if len(p_) < 5:
            continue
        dev, opcode, off, length, ts = p_[:5]
        if p.volume is not None and dev != p.volume:
            continue
        op = "R" if opcode.upper().startswith("R") else "W"
        yield int(ts), op, int(off), int(length)


def parse_tencent(f, p):
    for line in f:
        p_ = line.rstrip("\n").split(",")
        if len(p_) < 5:
            continue
        ts, off, size, iotype, vol = p_[:5]
        if p.volume is not None and vol != p.volume:
            continue
        op = "W" if iotype.strip() == "1" else "R"
        yield int(ts) * 1_000_000, op, int(off) * 512, int(size) * 512


PARSERS = {"msr": parse_msr, "alibaba": parse_alibaba, "tencent": parse_tencent}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("format", choices=PARSERS)
    ap.add_argument("input", help="trace file (.csv or .csv.gz), '-' for stdin")
    ap.add_argument("-o", "--output", default="-")
    ap.add_argument("--volume", help="keep only this volume / device / disk id")
    ap.add_argument("--limit", type=int, default=0, help="stop after N requests")
    ap.add_argument("--writes-only", action="store_true")
    ap.add_argument("--rebase", action="store_true",
                    help="shift offsets so the smallest one becomes 0 (two passes)")
    a = ap.parse_args()
    if a.rebase and a.input == "-":
        ap.error("--rebase needs a file input (it reads the trace twice)")

    def records():
        n = 0
        with open_any(a.input) as f:
            for ts, op, off, size in PARSERS[a.format](f, a):
                if size <= 0 or (a.writes_only and op != "W"):
                    continue
                yield ts, op, off, size
                n += 1
                if a.limit and n >= a.limit:
                    return

    base = 0
    if a.rebase:
        base = min((r[2] for r in records()), default=0)

    out = sys.stdout if a.output == "-" else open(a.output, "w", encoding="utf-8", newline="\n")
    n, max_end, t0 = 0, 0, None
    stats = {"R": 0, "W": 0, "T": 0}
    for ts, op, off, size in records():
        t0 = ts if t0 is None else t0
        off -= base
        out.write(f"{ts - t0},{op},{off},{size}\n")
        stats[op] += 1
        max_end = max(max_end, off + size)
        n += 1
    if out is not sys.stdout:
        out.close()
    print(f"{n} requests (R {stats['R']}, W {stats['W']}), "
          f"address span {max_end / 2**30:.2f} GiB", file=sys.stderr)


if __name__ == "__main__":
    main()
