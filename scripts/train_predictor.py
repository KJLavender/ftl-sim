#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""Train a small decision tree that predicts a write's lifetime class.

Input is the CSV from `ftlsim --dump-features`; output is the text model that
`ftlsim --placement=ml --model=...` loads. Prints accuracy next to two cheap
baselines so the tree has to earn its place.

Targets:
  classify  predict the realized lifetime class directly (argmax per leaf)
  regress   predict the expected log lifetime, then bucket it into a class.
            For memoryless data this groups pages by hotness instead of
            chasing the noisy realized class.

Example:
  uv run scripts/train_predictor.py results/ml/zipf_s1.csv -o results/ml/zipf.tree
"""
import argparse
import math

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

NUM_CLASSES = 5
# Class boundaries in the normalized log lifetime used by the C++ extractor:
# log2(1 + 1024 * L / C) at L = C/8, C/2, 2C, 8C.
BOUNDS = [math.log2(1 + 1024 * f) for f in (1 / 8, 1 / 2, 2, 8)]


def to_class(log_lifetime):
    return np.searchsorted(BOUNDS, log_lifetime, side="right")


def load(path):
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    feat_names = header[:-3]
    X = data[:, : len(feat_names)]
    y = data[:, -3].astype(int)
    life = data[:, -2]
    return feat_names, X, y, life


def prev_lifetime_rule(X, feat_names):
    """Predict that the page lives as long as its previous version did."""
    x = X[:, feat_names.index("prev_lifetime")]
    cls = to_class(x)
    cls[x < 0] = NUM_CLASSES - 1  # first write: assume long-lived
    return cls


def leaf_classes(tree, target, split):
    t = tree.tree_
    if target == "classify":
        return [int(tree.classes_[int(np.argmax(t.value[i][0]))]) for i in range(t.node_count)]
    if split:
        # Custom grouping of the predicted lifetime: boundaries in capacities.
        bounds = [math.log2(1 + 1024 * f) for f in split]
        return [int(np.searchsorted(bounds, t.value[i][0][0], side="right"))
                for i in range(t.node_count)]
    return [int(to_class(t.value[i][0][0])) for i in range(t.node_count)]


def export(tree, target, path, num_features, split):
    t = tree.tree_
    values = leaf_classes(tree, target, split)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"ftl-tree v1 {t.node_count} {num_features}\n")
        for i in range(t.node_count):
            if t.children_left[i] == -1:
                f.write(f"-1 0 -1 -1 {values[i]}\n")
            else:
                f.write(f"{t.feature[i]} {t.threshold[i]:.9g} "
                        f"{t.children_left[i]} {t.children_right[i]} {values[i]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--target", choices=["classify", "regress"], default="regress")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="",
                    help="regress: comma-separated stream boundaries in capacities "
                         "(max 4), e.g. 0.25 for two streams; default = label classes")
    a = ap.parse_args()
    split = [float(v) for v in a.split.split(",")] if a.split else []
    if len(split) > NUM_CLASSES - 1:
        ap.error("--split allows at most 4 boundaries")

    feat_names, X, y, life = load(a.csv)
    Xtr, Xte, ytr, yte, ltr, _ = train_test_split(X, y, life, test_size=0.2,
                                                  random_state=a.seed)
    if a.target == "classify":
        tree = DecisionTreeClassifier(max_depth=a.max_depth, min_samples_leaf=a.min_leaf,
                                      random_state=a.seed).fit(Xtr, ytr)
        pred = tree.predict(Xte)
    else:
        tree = DecisionTreeRegressor(max_depth=a.max_depth, min_samples_leaf=a.min_leaf,
                                     random_state=a.seed).fit(Xtr, ltr)
        pred = to_class(tree.predict(Xte))
    export(tree, a.target, a.output, X.shape[1], split)

    acc = (pred == yte).mean()
    majority = (yte == np.bincount(ytr).argmax()).mean()
    rule = (prev_lifetime_rule(Xte, feat_names) == yte).mean()
    nodes = tree.tree_.node_count
    print(f"rows {len(y)}  class mix {np.bincount(y, minlength=NUM_CLASSES).tolist()}")
    print(f"target {a.target}: predicted mix {np.bincount(pred, minlength=NUM_CLASSES).tolist()}")
    print(f"accuracy: tree {acc:.3f}  prev-lifetime rule {rule:.3f}  majority {majority:.3f}")
    print(f"tree: {nodes} nodes, depth {tree.get_depth()}, ~{nodes * 10} bytes packed")
    imp = sorted(zip(tree.feature_importances_, feat_names), reverse=True)
    print("importance: " + ", ".join(f"{n} {v:.2f}" for v, n in imp if v > 0))
    print(f"model -> {a.output}")


if __name__ == "__main__":
    main()
