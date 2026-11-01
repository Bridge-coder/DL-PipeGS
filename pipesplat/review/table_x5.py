"""Markdown table of scored runs over the test groups of docs/EVALUATION.md: in-sequence held-out frames, block frames, frames of
the other inspection runs (pooled and per run) and other-run frames inside the reference run's block stretches.
Reads RUN/per_frame<SUFFIX>.csv (train_gs.py / score_xrun.py); numpy only.

usage: python table_x5.py DATA_DIR RUN [RUN ...] [--suffix _r2] [--metric psnr] [--md OUT.md] [--per-run]
       a RUN may be RUN:SUFFIX to read another file of that run (e.g. run_g0a:_x5 for a re-scored older model)
Groups: held = tag "held" outside the blocks; block = tag "block"; xrun = tags xrun_* pooled; xblk = tags xblk_* pooled.
Every cell is the median over the group's frames; the first run also gives B (OpenMVS mesh x flat-field) and Nal
(aligned next frame, the video's noise ceiling) on the same frames.
"""
import argparse
import csv
import os

import numpy as np

COLS = (("psnr", "PSNR", "{:.2f}"), ("ssim", "SSIM", "{:.3f}"), ("lpips", "LPIPS", "{:.3f}"), ("shift_px", "shift px", "{:.2f}"),
        ("fid_fine", "fine", "{:.3f}"), ("fid_mid", "mid", "{:.3f}"), ("fid_coarse", "coarse", "{:.3f}"))


def groups(rows, per_run):
    g = {"held": [r for r in rows if r["tag"] == "held" and r["in_block"] == "False"],
         "block": [r for r in rows if r["tag"] == "block"],
         "xrun": [r for r in rows if r["tag"].startswith("xrun_")],
         "xblk": [r for r in rows if r["tag"].startswith("xblk_")]}
    if per_run:
        for t in sorted({r["tag"] for r in rows if r["tag"].startswith(("xrun_", "xblk_"))}):
            g[t] = [r for r in rows if r["tag"] == t]
    return g


def med(rs, key):
    v = [float(r[key]) for r in rs if r.get(key) not in (None, "", "nan")]
    return float(np.median(v)) if v else float("nan")


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data"), ap.add_argument("runs", nargs="+")
    ap.add_argument("--suffix", default="_r2"), ap.add_argument("--md"), ap.add_argument("--per-run", action="store_true")
    a = ap.parse_args()
    out = []
    first = True
    for spec in a.runs:
        run, suf = (spec.split(":", 1) + [a.suffix])[:2]
        p = os.path.join(a.data, run, f"per_frame{suf}.csv")
        if not os.path.exists(p):
            print(f"missing {p}")
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        G = groups(rows, a.per_run)
        if first:
            out.append("| model | group | n | " + " | ".join(c[1] for c in COLS) + " | depth err |")
            out.append("|" + "---|" * (len(COLS) + 4))
        for tags in (("G",) if not first else ("B", "Nal", "G")):
            for gname, rs in G.items():
                if not rs:
                    continue
                label = run if tags == "G" else {"B": "OpenMVS mesh x flat-field (B)", "Nal": "aligned next frame"}[tags]
                cells = [c[2].format(med(rs, f"{tags}_{c[0]}")) for c in COLS]
                dep = f"{med(rs, 'depth_rel_err'):.1%}" if tags == "G" else "—"
                out.append(f"| {label} | {gname} | {len(rs)} | " + " | ".join(cells) + f" | {dep} |")
        first = False
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
