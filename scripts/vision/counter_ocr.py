"""Read the on-screen distance counter of a new Sheffield video by template matching (normalised cross-correlation)
against digit images taken from videos whose counter was already decoded and checked (their counter.csv), so a new
video needs no labelling by eye.

usage: counter_ocr.py VIDEO OUT_CSV --ref REF_VIDEO REF_COUNTER_CSV [--ref ...] [--min-ncc 0.6] [--check COUNTER_CSV]
Digit cells are cut as evaluate.cluster_digits does ("D.DD m" below 10 m, "DD.DD m" from 10 m on). From every
reference frame whose reading is new (the counter just changed, so the label belongs to that picture), each cell is
normalised to zero mean and unit variance; the per-digit median of those patches is that digit's template. A cell of
the new video takes the digit with the highest normalised cross-correlation, if it exceeds --min-ncc; a frame with an
unreadable cell keeps the previous reading. Clean-up as `evaluate.py counter --nearest`: a reading more than 0.1 m
off the median of the 51 frames around it is replaced by that median, then the series is made monotonic.
--check: compare the result with an existing counter.csv of the same video (validation).
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def cells(frame):
    """normalised digit patches of one frame and whether it shows the 10 m layout; None if the strip is unreadable"""
    g = cv2.cvtColor(frame[ev.Y0:ev.Y1, ev.X0:ev.X1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    p50, p99 = np.percentile(g, [50, 99])
    b = g > (140 if p50 < 100 else 0.5 * (p50 + p99))
    if b.mean() > 0.4:
        return None, None
    ten = b[6:, ev.DOT_CELL[0]:ev.DOT_CELL[1]].sum() < 12
    out = []
    for a, c in (ev.CELLS_10M if ten else ev.CELLS):
        p = g[6:, a:c]
        p = (p - p.mean()) / max(float(p.std()), 1e-3)
        out.append(p)
    return out, ten


def digits_of(v, ten):
    s = f"{v:05.2f}" if ten else f"{v:04.2f}"
    return [int(ch) for ch in s.replace(".", "")]


def templates(refs):
    bank = {d: [] for d in range(10)}
    for video, ccsv in refs:
        cnt = np.array([float(r["counter_m"]) for r in csv.DictReader(open(ccsv, encoding="utf-8"))])
        cap = cv2.VideoCapture(video)
        i, used = 0, 0
        while True:
            ok, fr = cap.read()
            if not ok or i >= len(cnt):
                break
            if i > 0 and cnt[i] != cnt[i - 1]:          # reading just changed: the label belongs to this picture
                ps, ten = cells(fr)
                if ps is not None and ten == (cnt[i] >= 10) and len(ps) == len(digits_of(cnt[i], ten)):
                    for p, dg in zip(ps, digits_of(cnt[i], ten)):
                        bank[dg].append(p)
                    used += 1
            i += 1
        cap.release()
        print(f"reference {os.path.basename(video)}: {used} frames with a new reading")
    return {d: np.median(np.stack(v), 0) for d, v in bank.items() if v}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video"), ap.add_argument("out")
    ap.add_argument("--ref", nargs=2, action="append", required=True, metavar=("REF_VIDEO", "REF_COUNTER_CSV"))
    ap.add_argument("--min-ncc", type=float, default=0.6), ap.add_argument("--check", default=None)
    a = ap.parse_args()
    T = templates(a.ref)
    print(f"templates for digits {sorted(T)}")
    keys = sorted(T)
    TM = np.stack([T[k].ravel() / np.linalg.norm(T[k]) for k in keys])
    cap = cv2.VideoCapture(a.video)
    values, low = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ps, ten = cells(fr)
        v = np.nan
        if ps is not None:
            ds = []
            for p in ps:
                s = TM @ (p.ravel() / max(np.linalg.norm(p), 1e-6))
                k = int(np.argmax(s))
                if s[k] < a.min_ncc:
                    ds = None
                    low += 1
                    break
                ds.append(keys[k])
            if ds:
                v = int("".join(map(str, ds))) / 100
        values.append(v)
    cap.release()
    values = np.array(values)
    unreadable = int(np.isnan(values).sum())
    for i in range(len(values)):
        if np.isnan(values[i]):
            values[i] = values[i - 1] if i else 0.0
    pad = np.pad(values, 25, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 51), axis=1)
    fixed = np.abs(values - med) > 0.1
    values = np.maximum.accumulate(np.where(fixed, med, values))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["source_frame", "counter_m"])
        wr.writerows((i, f"{v:.2f}") for i, v in enumerate(values))
    d = np.round(np.diff(values), 2)
    steps = d[d != 0]
    print(f"frames: {len(values)}; unreadable: {unreadable} ({low} with a cell below NCC {a.min_ncc}); replaced by the "
          f"running median: {int(fixed.sum())}; counter first {values[0]:.2f} m, last {values[-1]:.2f} m; largest step "
          f"{steps.max() if len(steps) else 0:.2f} m")
    if a.check:
        ref = np.array([float(r["counter_m"]) for r in csv.DictReader(open(a.check, encoding="utf-8"))])
        n = min(len(ref), len(values))
        dd = np.abs(values[:n] - ref[:n])
        print(f"check against {a.check}: identical {np.mean(dd < 0.005):.2%} of {n} frames, off by more than 1 cm "
              f"{int((dd > 0.011).sum())}, max {dd.max():.2f} m")
    print("saved", a.out)


if __name__ == "__main__":
    main()
