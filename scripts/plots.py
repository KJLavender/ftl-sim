#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Draw the README figures from the CSVs in results/.

  uv run scripts/plots.py            # writes docs/figures/*.png
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "figures")

# Reference palette, fixed order (validated: scripts/validate_palette.js).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e",
                                         "#898781", "#e1e0d9", "#c3c2b7")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "lines.markersize": 6, "font.size": 10,
})


def rows(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not r.get("error")]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join("docs", "figures", name))


def fig_validation():
    """FIFO + uniform random: simulator vs. Desnoyers' closed form."""
    r = [x for x in rows("op_sweep.csv") if x["workload"] == "uniform" and x["gc"] == "fifo"]
    if not r:
        return
    r.sort(key=lambda x: float(x["op"]))
    op = [100 * float(x["op"]) for x in r]
    sim = [float(x["waf"]) for x in r]
    model = [float(x["fifo_uniform_model_waf_effective"]) for x in r]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(op, model, color=ORANGE, linestyle="--", label="formula (Desnoyers 2012)")
    ax.plot(op, sim, color=BLUE, marker="o", linestyle="none", markersize=8,
            label="simulator")
    ax.set(xlabel="spare space (OP, %)", ylabel="WAF",
           title="Simulator matches the known formula")
    ax.set_ylim(bottom=1)
    ax.legend(loc="upper right")
    save(fig, "validation.png")


def fig_op_msr():
    """Q2: mean WAF over the MSR volumes as spare space grows."""
    r = rows("msr_op.csv")
    if not r:
        return
    series = [("greedy", "none", BLUE, "greedy, no separation"),
              ("greedy", "sepbit", ORANGE, "greedy + SepBIT"),
              ("cost-benefit", "none", AQUA, "cost-benefit, no separation"),
              ("cost-benefit", "sepbit", YELLOW, "cost-benefit + SepBIT")]
    vols = sorted({x["trace"] for x in r})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for gc, pl, color, label in series:
        by_op = defaultdict(list)
        for x in r:
            if x["gc"] == gc and x["placement"] == pl:
                by_op[float(x["op"])].append(float(x["waf"]))
        ops = sorted(by_op)
        mean = [np.mean(by_op[o]) for o in ops]
        ax.plot([100 * o for o in ops], mean, color=color, marker="o", label=label)
    ax.axhline(1, color=AXIS, linewidth=1)
    ax.set(xlabel="spare space (OP, %)", ylabel=f"mean WAF over {len(vols)} volumes",
           title="More spare space helps, but less and less")
    ax.set_ylim(bottom=0.9)
    ax.legend(loc="upper right")
    save(fig, "op_msr.png")


def fig_msr_placement():
    """Per-volume WAF for each placement, greedy GC, OP 7%."""
    r = [x for x in rows("msr.csv") if x["gc"] == "greedy" and x["op"] == "0.07"]
    if not r:
        return
    series = [("none", BLUE, "no separation"), ("wcount", ORANGE, "write-count rule"),
              ("sepbit", AQUA, "SepBIT"), ("oracle", YELLOW, "oracle (knows the future)")]
    waf = {(x["trace"], x["placement"]): float(x["waf"]) for x in r}
    vols = sorted({x["trace"] for x in r}, key=lambda v: -waf[(v, "none")])
    x = np.arange(len(vols))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (pl, color, label) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, [waf[(v, pl)] for v in vols], width=w * 0.9,
               color=color, label=label, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, vols, rotation=0)
    ax.set_ylim(0.9, max(waf.values()) * 1.08)
    ax.set(ylabel="WAF", title="Real servers (MSR Cambridge), greedy GC, 7% spare space")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=4)
    save(fig, "msr_placement.png")


def fig_erase():
    """Erase-count spread with wear leveling off, dynamic and static."""
    files = [("off.txt", "off", BLUE), ("dynamic.txt", "dynamic", ORANGE),
             ("static20.txt", "static (spread ≤ 20)", AQUA)]
    data = []
    for f, label, color in files:
        p = os.path.join(RES, "erase", f)
        if not os.path.exists(p):
            return
        data.append((np.loadtxt(p), label, color))
    lo = min(d.min() for d, _, _ in data)
    hi = max(d.max() for d, _, _ in data)
    bins = np.linspace(lo, hi + 1, 40)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, (d, label, color) in zip(axes, data):
        ax.hist(d, bins=bins, color=color, edgecolor=SURFACE, linewidth=1)
        ax.set_title(f"{label}\nstd {d.std():.1f}, max {int(d.max())}", fontsize=10)
        ax.set_xlabel("erase count")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("blocks")
    fig.suptitle("Wear leveling: 80% of the data is never rewritten", x=0.01,
                 ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "erase.png")


def mean_table(name):
    """Whitespace table (volume + one column per method) -> {method: mean}."""
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        lines = [ln.split() for ln in f if ln.strip()]
    head, body = lines[0][1:], lines[1:]
    return {h: float(np.mean([float(r[i + 1]) for r in body])) for i, h in enumerate(head)}


def fig_ml():
    """Learned placement vs. SepBIT and the 4-group control, three settings."""
    cost = rows(os.path.join("ml_cost", "waf.csv"))
    online = mean_table(os.path.join("ml_online", "waf.txt"))
    op14 = mean_table(os.path.join("ml_cost", "op14.txt"))
    if not cost or not online or not op14:
        return
    by = defaultdict(list)
    for x in cost:
        by[(x["model"], x["mode"])].append(float(x["waf"]))
    m = {k: float(np.mean(v)) for k, v in by.items()}

    # Colors follow the method, the same in every panel.
    fixed = ("tree, fixed boundaries", MUTED)
    sep = ("SepBIT", BLUE)
    sep4 = ("SepBIT split into 4 groups (no model)", ORANGE)
    tree = ("tree + ell, 4 groups", AQUA)
    own = ("tree + ell, trained on own first half", YELLOW)
    panels = [
        ("7% spare space, volumes the model never saw",
         [(fixed, m[("tree", "sepbit")]), (sep, m[("sepbit", "-")]),
          (sep4, m[("sepbit4", "-")]), (tree, m[("tree", "ell4")])]),
        ("7% spare space, trained on the first half, measured on the second",
         [(sep, online["sepbit"]), (sep4, online["sepbit4"]),
          (("tree + ell, trained on other volumes", AQUA), online["cross-vol"]),
          (own, online["own-past"])]),
        ("14% spare space, volumes the model never saw",
         [(sep, op14["sepbit"]), (sep4, op14["sepbit4"]), (tree, op14["tree-ell4"])]),
    ]
    heights = [len(p[1]) + 1.2 for p in panels]
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": heights})
    for ax, (title, items) in zip(axes, panels):
        y = np.arange(len(items))[::-1]
        for yi, ((label, color), v) in zip(y, items):
            ax.plot([v], [yi], "o", color=color, markersize=10,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate(f"{v:.2f}", (v, yi), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
        ax.set_yticks(y, [lab for (lab, _), _ in items])
        ax.set_ylim(-0.7, len(items) - 0.3)
        ax.set_title(title, fontsize=10, color=INK)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0, labelcolor=INK2)
    axes[-1].set_xlabel("mean WAF over 7 MSR volumes (lower is better)")
    fig.suptitle("Learned placement: where the gain comes from", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "ml.png")


if __name__ == "__main__":
    fig_validation()
    fig_op_msr()
    fig_msr_placement()
    fig_erase()
    fig_ml()
