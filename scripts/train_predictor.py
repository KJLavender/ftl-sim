#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn"]
# ///
"""Train a small model that predicts a write's lifetime class.

Input is the CSV from `ftlsim --dump-features`; output is the text model that
`ftlsim --placement=ml --model=...` loads. Prints accuracy next to two cheap
baselines so the model has to earn its place.

Targets:
  classify  predict the realized lifetime class directly (argmax per leaf)
  regress   predict the expected log lifetime, then bucket it into a class.
            For memoryless data this groups pages by hotness instead of
            chasing the noisy realized class.

Models:
  tree  one decision tree (default, what a controller could afford)
  gbm   gradient-boosted trees (sklearn's LightGBM-style HistGradientBoosting),
        regress only; used to check whether a much bigger model helps

Asymmetric cost (--quantile q, regress only): predict the q-th quantile of the
lifetime instead of the mean. Guessing too short costs q per unit, guessing
too long costs 1 - q, so q > 0.5 makes the model slow to call data hot. That
matches the drive: long-lived data in a hot block has to be copied by GC,
short-lived data in a cold block only wastes a page for a while.

Example:
  uv run scripts/train_predictor.py results/ml/zipf_s1.csv -o results/ml/zipf.tree
  uv run scripts/train_predictor.py train.csv -o m.gbm --model gbm \\
      --features all --test a.csv,b.csv
"""
import argparse
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

NUM_CLASSES = 5
# Class boundaries in the normalized log lifetime used by the C++ extractor:
# log2(1 + 1024 * L / C) at L = C/8, C/2, 2C, 8C.
BOUNDS = [math.log2(1 + 1024 * f) for f in (1 / 8, 1 / 2, 2, 8)]
# The six features the original predictor used; the rest are extras.
BASE_FEATURES = ["prev_lifetime", "first_write", "write_count",
                 "prev2_lifetime", "region_heat", "req_pages_log2"]


def to_class(log_lifetime):
    return np.searchsorted(BOUNDS, log_lifetime, side="right")


def load(paths):
    frames = [pd.read_csv(p, dtype=np.float32) for p in paths.split(",")]
    df = pd.concat(frames, ignore_index=True)
    all_feats = list(df.columns[:-3])
    return all_feats, df


def prev_lifetime_rule(x):
    """Predict that the page lives as long as its previous version did."""
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


def export_tree(tree, target, path, cols, num_features, split):
    t = tree.tree_
    values = leaf_classes(tree, target, split)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"ftl-tree v1 {t.node_count} {num_features}\n")
        for i in range(t.node_count):
            if t.children_left[i] == -1:
                f.write(f"-1 0 -1 -1 {values[i]}\n")
            else:
                f.write(f"{cols[t.feature[i]]} {t.threshold[i]:.9g} "
                        f"{t.children_left[i]} {t.children_right[i]} {values[i]}\n")


def export_raw_tree(tree, leaf_values, path, cols, num_features):
    """One regression tree in the ensemble format, so leaves hold a lifetime."""
    t = tree.tree_
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"ftl-gbm v1 1 {num_features} 0\n{t.node_count}\n")
        for i in range(t.node_count):
            if t.children_left[i] == -1:
                f.write(f"-1 0 -1 -1 {leaf_values[i]:.9g}\n")
            else:
                f.write(f"{cols[t.feature[i]]} {t.threshold[i]:.9g} "
                        f"{t.children_left[i]} {t.children_right[i]} 0\n")


def leaf_quantiles(tree, X, y, q):
    """Replace each leaf's mean with the q-th quantile of its training rows."""
    t = tree.tree_
    values = t.value[:, 0, 0].astype(np.float64).copy()
    leaves = tree.apply(X)
    order = np.argsort(leaves, kind="stable")
    ids, starts = np.unique(leaves[order], return_index=True)
    for leaf, part in zip(ids, np.split(y[order], starts[1:])):
        values[leaf] = np.quantile(part, q)
    return values


def gbm_trees(gbm):
    """(base, [node arrays]) of a fitted HistGradientBoostingRegressor."""
    base = float(np.ravel(gbm._baseline_prediction)[0])
    return base, [it[0].nodes for it in gbm._predictors]


def gbm_predict_exported(base, trees, X):
    """Re-evaluate the exported trees the way the C++ side does."""
    out = np.full(len(X), base, dtype=np.float64)
    for nodes in trees:
        idx = np.zeros(len(X), dtype=np.int64)
        while True:
            leaf = nodes["is_leaf"][idx].astype(bool)
            if leaf.all():
                break
            feat = nodes["feature_idx"][idx]
            thr = nodes["num_threshold"][idx]
            go_left = X[np.arange(len(X)), feat] <= thr
            nxt = np.where(go_left, nodes["left"][idx], nodes["right"][idx])
            idx = np.where(leaf, idx, nxt)
        out += nodes["value"][idx]
    return out


def export_gbm(gbm, path, cols, num_features):
    base, trees = gbm_trees(gbm)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"ftl-gbm v1 {len(trees)} {num_features} {base:.9g}\n")
        for nodes in trees:
            f.write(f"{len(nodes)}\n")
            for nd in nodes:
                if nd["is_leaf"]:
                    f.write(f"-1 0 -1 -1 {nd['value']:.9g}\n")
                else:
                    f.write(f"{cols[nd['feature_idx']]} {nd['num_threshold']:.9g} "
                            f"{nd['left']} {nd['right']} 0\n")
    return sum(len(n) for n in trees)


def r2(y, p):
    return 1.0 - np.mean((y - p) ** 2) / np.var(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="training CSV(s), comma-separated")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--target", choices=["classify", "regress"], default="regress")
    ap.add_argument("--model", choices=["tree", "gbm"], default="tree")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--gbm-trees", type=int, default=100)
    ap.add_argument("--gbm-leaves", type=int, default=31)
    ap.add_argument("--features", default="base",
                    help="'base' (the original six), 'all', or a comma-separated list")
    ap.add_argument("--test", default="",
                    help="CSV(s) from other volumes to score on instead of a random 20%%")
    ap.add_argument("--quantile", type=float, default=0.0,
                    help="regress: predict this quantile of the lifetime (0 = mean)")
    ap.add_argument("--raw", action="store_true",
                    help="tree, regress: export the predicted lifetime instead of a "
                         "class (needed for --ml-gc=ell2/ell4)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="train on this random fraction of the training rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="",
                    help="regress: comma-separated stream boundaries in capacities "
                         "(max 4), e.g. 0.25 for two streams; default = label classes")
    a = ap.parse_args()
    split = [float(v) for v in a.split.split(",")] if a.split else []
    if len(split) > NUM_CLASSES - 1:
        ap.error("--split allows at most 4 boundaries")
    if a.model == "gbm" and (a.target != "regress" or split):
        ap.error("--model gbm supports only --target regress without --split")
    if (a.quantile or a.raw) and a.target != "regress":
        ap.error("--quantile and --raw need --target regress")
    if a.raw and split:
        ap.error("--raw exports a lifetime, --split does not apply")

    all_feats, df = load(a.csv)
    if a.features == "base":
        names = BASE_FEATURES
    elif a.features == "all":
        names = all_feats
    else:
        names = a.features.split(",")
    cols = [all_feats.index(n) for n in names]  # model index -> extractor index

    if a.test:
        _, te = load(a.test)
        tr = df
    else:
        tr, te = train_test_split(df, test_size=0.2, random_state=a.seed)
    if a.train_frac < 1.0:
        tr = tr.sample(frac=a.train_frac, random_state=a.seed)
    Xtr, Xte = tr[names].to_numpy(), te[names].to_numpy()
    ytr, yte = tr["label"].to_numpy().astype(int), te["label"].to_numpy().astype(int)
    ltr, lte = tr["lifetime_log"].to_numpy(), te["lifetime_log"].to_numpy()

    if a.model == "gbm":
        loss = {"loss": "quantile", "quantile": a.quantile} if a.quantile else {}
        model = HistGradientBoostingRegressor(
            max_iter=a.gbm_trees, max_leaf_nodes=a.gbm_leaves, learning_rate=0.1,
            min_samples_leaf=a.min_leaf, early_stopping=False, random_state=a.seed, **loss)
        model.fit(Xtr, ltr)
        raw = model.predict(Xte)
        base, trees = gbm_trees(model)
        sample = Xte[: min(len(Xte), 20000)].astype(np.float32)
        diff = np.abs(gbm_predict_exported(base, trees, sample) - raw[: len(sample)]).max()
        if diff > 1e-3:
            raise SystemExit(f"exported ensemble disagrees with sklearn (max diff {diff})")
        nodes = export_gbm(model, a.output, cols, len(all_feats))
        size = f"{len(trees)} trees, {nodes} nodes, ~{nodes * 13} bytes packed"
        pred = to_class(raw)
    elif a.target == "classify":
        model = DecisionTreeClassifier(max_depth=a.max_depth, min_samples_leaf=a.min_leaf,
                                       random_state=a.seed).fit(Xtr, ytr)
        raw = None
        pred = model.predict(Xte)
    else:
        model = DecisionTreeRegressor(max_depth=a.max_depth, min_samples_leaf=a.min_leaf,
                                      random_state=a.seed).fit(Xtr, ltr)
        if a.quantile:
            leaf_values = leaf_quantiles(model, Xtr, ltr, a.quantile)
            model.tree_.value[:, 0, 0] = leaf_values  # export_tree buckets these
        else:
            leaf_values = model.tree_.value[:, 0, 0].copy()
        raw = leaf_values[model.apply(Xte)]
        pred = to_class(raw)
    if a.model == "tree" and a.raw:
        export_raw_tree(model, leaf_values, a.output, cols, len(all_feats))
    if a.model == "tree":
        if not a.raw:
            export_tree(model, a.target, a.output, cols, len(all_feats), split)
        n = model.tree_.node_count
        size = f"{n} nodes, depth {model.get_depth()}, ~{n * 10} bytes packed"

    acc = (pred == yte).mean()
    majority = (yte == np.bincount(ytr).argmax()).mean()
    rule = (prev_lifetime_rule(te["prev_lifetime"].to_numpy()) == yte).mean()
    print(f"train rows {len(tr)}  test rows {len(te)}  features {','.join(names)}")
    print(f"test class mix {np.bincount(yte, minlength=NUM_CLASSES).tolist()}")
    print(f"predicted mix {np.bincount(pred, minlength=NUM_CLASSES).tolist()}")
    print(f"accuracy: model {acc:.3f}  prev-lifetime rule {rule:.3f}  majority {majority:.3f}")
    if raw is not None:
        print(f"log-lifetime R^2: model {r2(lte, raw):.3f}")
        # One unit of log lifetime is a factor of ~2.
        print(f"guessed >2x too short {np.mean(raw < lte - 1):.3f}  "
              f">2x too long {np.mean(raw > lte + 1):.3f}")
    print(f"model: {size}")
    if a.model == "tree":
        imp = sorted(zip(model.feature_importances_, names), reverse=True)
        print("importance: " + ", ".join(f"{n} {v:.2f}" for v, n in imp if v > 0))
    print(f"model -> {a.output}")


if __name__ == "__main__":
    main()
