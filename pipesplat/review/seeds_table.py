"""Main-table numbers over random seeds (docs/EVALUATION.md protocol: 3 seeds per method): for every method and test group the
median over the group's frames per seed, reported as mean +- standard deviation over the seeds; and per method a
seed-averaged per-frame file (every frame's metric averaged over the seeds) for paired tests between methods
(review/paired_stats.py). numpy only.

usage: python seeds_table.py DATA_DIR NAME=RUN[:SUF[+SUF]],RUN[:SUF],... [NAME=...] [--metrics psnr lpips fid_coarse]
                             [--avg-dir DIR] [--md OUT.md]
       one RUN per seed; :SUF picks per_frame<SUF>.csv (default _r2); SUF1+SUF2 merges two files of the same run (a later
       file replaces rows with the same tag and frame), e.g. run_g0al_a01:_r2+_x5 = training-end scoring + re-scoring on
       the other runs' frames
Groups: held (tag held outside the blocks), block, xrun (tags xrun_* pooled), xblk (tags xblk_* pooled).
Writes DIR/<NAME>_seedavg.csv (tag, frame, in_block, G_<metric> averaged over the seeds that have the frame).
"""
import argparse
import csv
import os
import sys

import numpy as np

GROUPS = (("held", lambda r: r["tag"] == "held" and r["in_block"] == "False"), ("block", lambda r: r["tag"] == "block"),
          ("xrun", lambda r: r["tag"].startswith("xrun_")), ("xblk", lambda r: r["tag"].startswith("xblk_")))


def read_run(data, spec):
    run, sufs = (spec.split(":", 1) + ["_r2"])[:2]
    rows = {}
    for suf in sufs.split("+"):
        p = os.path.join(data, run, f"per_frame{suf}.csv")
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        for r in csv.DictReader(open(p, encoding="utf-8")):
            rows[(r["tag"], r["frame"])] = r
    return list(rows.values())


def val(r, k):
    v = r.get(k)
    return float(v) if v not in (None, "", "nan") else np.nan


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data"), ap.add_argument("methods", nargs="+")
    ap.add_argument("--metrics", nargs="+", default=["psnr", "lpips", "fid_coarse"])
    ap.add_argument("--avg-dir", default=None), ap.add_argument("--md", default=None)
    a = ap.parse_args()
    avg_dir = a.avg_dir or os.path.join(a.data, "seedavg")
    os.makedirs(avg_dir, exist_ok=True)
    lines = ["| method | seeds | " + " | ".join(f"{g} {m}" for g, _ in GROUPS for m in a.metrics) + " |",
             "|" + "---|" * (2 + len(GROUPS) * len(a.metrics))]
    for spec in a.methods:
        name, runs = spec.split("=", 1)
        seeds = [read_run(a.data, s) for s in runs.split(",")]
        cells = []
        for gname, sel in GROUPS:
            for m in a.metrics:
                meds = [np.nanmedian([val(r, f"G_{m}") for r in rows if sel(r)]) for rows in seeds
                        if any(sel(r) for r in rows)]
                cells.append(f"{np.mean(meds):.3f} ± {np.std(meds, ddof=1) if len(meds) > 1 else 0:.3f}" if meds else "—")
        lines.append(f"| {name} | {len(seeds)} | " + " | ".join(cells) + " |")
        # seed-averaged per-frame file
        keys = {}
        for rows in seeds:
            for r in rows:
                keys.setdefault((r["tag"], r["frame"]), []).append(r)
        out = os.path.join(avg_dir, f"{name}_seedavg.csv")
        cols = [f"G_{m}" for m in ("psnr", "ssim", "lpips", "shift_px", "fid_fine", "fid_mid", "fid_coarse")] + ["depth_rel_err"]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["tag", "frame", "in_block", "n_seeds"] + cols)
            for (tag, frame), rs in sorted(keys.items()):
                wr.writerow([tag, frame, rs[0]["in_block"], len(rs)] + [f"{np.nanmean([val(r, c) for r in rs]):.5f}" for c in cols])
    text = "\n".join(lines)
    print(text)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(text + "\n")
    print(f"seed-averaged per-frame files in {avg_dir}")


if __name__ == "__main__":
    main()
