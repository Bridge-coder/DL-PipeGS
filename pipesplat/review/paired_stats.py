"""Paired comparison of two scored models on the same test frames (per_frame*.csv of train_gs.py): median of the
per-frame differences, its bootstrap 95 % confidence interval and the Wilcoxon signed-rank test (numpy only).

usage: python paired_stats.py RUN_A/per_frame_SUFFIX.csv RUN_B/per_frame_SUFFIX.csv [--metric G_psnr] [--tags held block ...]
       [--names A B] [--boot 10000]
For every tag (test set) the frames present in both files are paired by frame id. Reported: n, median of A and B,
median of A - B with its bootstrap 95 % interval (percentile method, resampling frames), the share of frames where
A is better, and the two-sided p-value of the Wilcoxon signed-rank test (normal approximation with the tie and zero
corrections of Pratt / Wilcoxon; adequate for n >= 20).
For metrics where lower is better (lpips, shift_px, depth_rel_err) "better" means lower.
"""
import argparse
import csv
import math

import numpy as np

LOWER = ("lpips", "shift_px", "depth_rel_err", "mse_low", "mse_high")


def read(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def wilcoxon_p(d):
    """two-sided p of the Wilcoxon signed-rank test, normal approximation; zeros dropped (Wilcoxon), ties averaged"""
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    a = np.abs(d)
    order = np.argsort(a)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    for v in np.unique(a):                          # average ranks of ties
        m = a == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    w = ranks[d > 0].sum()
    mu = n * (n + 1) / 4
    _, cnt = np.unique(a, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - (cnt ** 3 - cnt).sum() / 48
    z = (w - mu) / math.sqrt(max(var, 1e-12))
    return float(math.erfc(abs(z) / math.sqrt(2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a"), ap.add_argument("b"), ap.add_argument("--metric", default="G_psnr")
    ap.add_argument("--tags", nargs="+"), ap.add_argument("--names", nargs=2, default=("A", "B"))
    ap.add_argument("--boot", type=int, default=10000), ap.add_argument("--metric-b", default=None,
                                                                        help="column of file B (default: --metric)")
    ap.add_argument("--pool", nargs="+", default=[],
                    help="also pool all tags starting with each of these prefixes into one set (e.g. xrun_ xblk_)")
    ap.add_argument("--held-outside", action="store_true",
                    help="drop held frames that lie inside the blocks (the main tables' held set, train_gs "
                         "held_outside_blocks)")
    a = ap.parse_args()
    A, B = read(a.a), read(a.b)
    mb = a.metric_b or a.metric
    tags = a.tags or sorted({r["tag"] for r in A} & {r["tag"] for r in B})
    tags = list(tags) + [p + "*" for p in a.pool]
    lower = any(k in a.metric for k in LOWER)
    rng = np.random.default_rng(0)
    print(f"{a.metric}: {a.names[0]} = {a.a}\n{' ' * len(a.metric)}  {a.names[1]} = {a.b}")

    def take(rows, t, col):
        sel = (lambda r: r["tag"].startswith(t[:-1])) if t.endswith("*") else (lambda r: r["tag"] == t)
        drop = (lambda r: r["tag"] == "held" and r.get("in_block") == "True") if a.held_outside else (lambda r: False)
        return {(r["tag"], r["frame"]): float(r[col]) for r in rows
                if sel(r) and not drop(r) and r.get(col) not in (None, "", "nan")}
    for t in tags:
        da, db = take(A, t, a.metric), take(B, t, mb)
        keys = sorted(set(da) & set(db))
        if len(keys) < 3:
            continue
        x, y = np.array([da[k] for k in keys]), np.array([db[k] for k in keys])
        d = x - y
        bs = np.median(d[rng.integers(0, len(d), (a.boot, len(d)))], axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        better = float(np.mean(d < 0) if lower else np.mean(d > 0))
        print(f"  {t:22s} n {len(d):3d}  median {a.names[0]} {np.median(x):.3f}  {a.names[1]} {np.median(y):.3f}  "
              f"diff {np.median(d):+.3f} [{lo:+.3f}, {hi:+.3f}]  {a.names[0]} better on {better:.0%}  "
              f"Wilcoxon p {wilcoxon_p(d):.2g}")


if __name__ == "__main__":
    main()
