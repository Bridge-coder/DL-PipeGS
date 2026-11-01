"""Geometry repeatability of several inspection runs of the same pipe, each reconstructed on its own (run_pipeline.py
steps 2-11): bore diameter and ovality along the pipe from every run's dense MVS cloud, with the scale taken from the
on-screen counter (not from the nominal bore), and their agreement between runs.

usage: python -I geometry_repeat.py OUT_PREFIX --run NAME WORK [--run NAME WORK ...] [--slice-m 0.05] [--from-m 0.25]
       WORK  a run_pipeline.py work folder: sfm/recon/sfm_data.bin, mvs/scene_dense.ply, frames.csv, counter.csv

Per run
  scale: one global fit between the camera positions along the camera-path axis and the counter readings (least
         squares, frames with counter >= FROM_M; as eval_gt_scale.py): metres per model unit and offset
  cross-sections every SLICE_M of counter distance (slab thickness SLICE_M), each perpendicular to the local
         camera-path direction (cameras within +-0.5 m); the dense points in the slab give
         - a circle (robust algebraic fit, evaluate.fit_circle): bore diameter D = 2 r x scale
         - an ellipse (Fitzgibbon direct least squares) on the circle inliers: ovality
           q = 100 (D_mean - D_min) / D_mean = 100 (a - b) / (a + b)   (ASTM F1216, appendix X1)
         kept when the points cover >= 75 % of the 10-degree sectors and >= 60 % of them are circle inliers;
         also the mean point brightness of the slab (joints are dark rings)
Between runs (common grid of counter positions, >= 3 runs at a position)
  repeatability standard deviation s_r = sqrt(mean over positions of the between-run variance) and repeatability
  limit r = 2.8 s_r (ISO 5725-2, ISO 5725-6), for D (as measured, and with each run's median removed = profile shape)
  and q; per-run median D (differences = global scale of each run); Pearson correlation of the de-meaned D profiles
Writes OUT_PREFIX.json, OUT_PREFIX_profiles.csv, OUT_PREFIX.png.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vision"))
import evaluate as ev  # noqa: E402
import paths  # noqa: E402
from pipe_geometry import fit_ellipse  # noqa: E402
from ply_io import read_ply_vertices  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def run_profile(name, work, grid, slice_m, from_m, shift=0.0, local_m=None):
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(work, "frames.csv"), encoding="utf-8"))}
    gt = {int(r["source_frame"]): float(r["counter_m"])
          for r in csv.DictReader(open(os.path.join(work, "counter.csv"), encoding="utf-8"))}
    views, _, _, poses, _, _, _ = ev.load_sfm(paths.CONVERTER, os.path.join(work, "sfm", "recon", "sfm_data.bin"))
    reg = sorted((frames[n], poses[p]) for n, p in views.values() if p in poses)
    cams = np.array([c for _, c in reg])
    src = np.array([f for f, _ in reg])
    mu = cams.mean(axis=0)
    _, _, vt = np.linalg.svd(cams - mu)
    d0 = vt[0] * np.sign((cams[-1] - cams[0]) @ vt[0])
    s_cam = (cams - mu) @ d0
    use = np.array([f in gt and gt[f] >= from_m for f in src])
    g = np.array([gt[f] for f in src[use]])
    (scale, off), *_ = np.linalg.lstsq(np.c_[s_cam[use], np.ones(use.sum())], g, rcond=None)
    err = scale * s_cam[use] + off - g
    off += shift                                                # counter -> common position (joint registration)
    x_cam = scale * s_cam + off
    s_use = s_cam[use]

    def scale_at(x):
        """metres per unit at counter position x: global, or the counter fit over +-local_m/2 of cameras"""
        if not local_m:
            return scale
        m = np.abs(g + shift - x) <= local_m / 2
        if m.sum() < 20 or np.ptp(g[m]) < local_m / 2:
            return np.nan
        return float(np.polyfit(s_use[m], g[m], 1)[0])
    V = read_ply_vertices(os.path.join(work, "mvs", "scene_dense.ply"))
    X = np.column_stack([V["x"], V["y"], V["z"]]).astype(float)
    bright = (V["red"].astype(float) + V["green"] + V["blue"]) / 3 if "red" in V.dtype.names else np.full(len(X), np.nan)
    x_pts = scale * ((X - mu) @ d0) + off
    order = np.argsort(x_pts)
    xs = x_pts[order]
    half = slice_m / 2 / scale                                  # slab half-thickness in model units
    rows = []
    for x in grid:
        near = np.abs(x_cam - x) < 0.5
        if near.sum() < 5 or x < x_cam.min() or x > x_cam.max():
            rows.append(None)
            continue
        q = cams[near]
        _, _, vv = np.linalg.svd(q - q.mean(0))
        dl = vv[0] * np.sign(vv[0] @ d0)
        p0 = mu + d0 * (x - off) / scale
        i0, i1 = np.searchsorted(xs, [x - 0.15, x + 0.15])
        cand = order[i0:i1]
        m = np.abs((X[cand] - p0) @ dl) < half
        P = X[cand][m]
        if len(P) < 200:
            rows.append(None)
            continue
        e1 = np.cross(dl, [0.0, 0.0, 1.0] if abs(dl[2]) < 0.9 else [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(dl, e1)
        uv = np.c_[(P - p0) @ e1, (P - p0) @ e2]
        c, r, keep = ev.fit_circle(uv)
        ang = np.arctan2(*(uv[keep] - c).T[::-1])
        cover = np.mean(np.bincount(((ang + np.pi) / (2 * np.pi) * 36).astype(int) % 36, minlength=36) >= 5)
        ok = cover >= 0.75 and keep.mean() >= 0.6
        try:
            _, _, a, b, _ = fit_ellipse(*(uv[keep] - c).T)
            ov = 100 * (a - b) / (a + b) if np.isfinite(a) and np.isfinite(b) else np.nan
        except Exception:  # noqa: BLE001
            ov = np.nan
        res = np.hypot(*(uv[keep] - c).T) - r
        sc = scale_at(x)
        rows.append(dict(x=float(x), D_mm=2000 * r * sc if ok else np.nan, ovality_pct=ov if ok else np.nan,
                         rms_mm=1000 * scale * float(np.sqrt(np.mean(res ** 2))), cover=float(cover),
                         inliers=float(keep.mean()), n=int(len(P)), bright=float(np.nanmean(bright[cand][m]))))
    info = dict(name=name, work=work, frames=len(frames), registered=len(reg), scale_m_per_unit=float(scale),
                travel_mae_cm=float(np.abs(err).mean() * 100), travel_max_cm=float(np.abs(err).max() * 100),
                counter_range_m=[float(g.min()), float(g.max())], dense_points=int(len(X)))
    return info, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--run", nargs=2, action="append", metavar=("NAME", "WORK"), required=True)
    ap.add_argument("--slice-m", type=float, default=0.05), ap.add_argument("--from-m", type=float, default=0.25)
    ap.add_argument("--to-m", type=float, default=None, help="last cross-section (default: end of the shortest run)")
    ap.add_argument("--shift", nargs=2, action="append", metavar=("NAME", "M"), default=[],
                    help="add M metres to this run's counter positions (common position from a joint registration: "
                         "the median of reference axial position - own counter)")
    ap.add_argument("--local-m", type=float, default=None,
                    help="scale each cross-section by the counter fit over this window of cameras instead of the "
                         "global fit (local scale drift of the monocular reconstruction)")
    a = ap.parse_args()
    grid = np.round(np.arange(a.from_m, (a.to_m or 30.0) + 1e-9, a.slice_m), 4)
    shifts = {n: float(m) for n, m in a.shift}
    infos, prof = [], {}
    for name, work in a.run:
        info, rows = run_profile(name, work, grid, a.slice_m, a.from_m, shifts.get(name, 0.0), a.local_m)
        info["shift_m"] = shifts.get(name, 0.0)
        D = np.array([r["D_mm"] if r else np.nan for r in rows])
        Q = np.array([r["ovality_pct"] if r else np.nan for r in rows])
        info.update(D_median_mm=float(np.nanmedian(D)), D_p10_p90_mm=[float(np.nanpercentile(D, 10)), float(np.nanpercentile(D, 90))],
                    ovality_median_pct=float(np.nanmedian(Q)), sections=int(np.isfinite(D).sum()))
        print(f"{name}: {info['registered']}/{info['frames']} frames, scale {info['scale_m_per_unit']:.4f} m/unit, travel MAE "
              f"{info['travel_mae_cm']:.1f} cm; {info['sections']} sections, D median {info['D_median_mm']:.1f} mm "
              f"(p10-p90 {info['D_p10_p90_mm'][0]:.1f}-{info['D_p10_p90_mm'][1]:.1f}), ovality median {info['ovality_median_pct']:.2f} %")
        infos.append(info)
        prof[name] = rows
    names = [n for n, _ in a.run]
    D = np.array([[r["D_mm"] if r else np.nan for r in prof[n]] for n in names])
    Q = np.array([[r["ovality_pct"] if r else np.nan for r in prof[n]] for n in names])
    Dc = D - np.nanmedian(D, axis=1, keepdims=True)
    ok = np.isfinite(D).sum(0) >= 3

    def s_r(M):
        return float(np.sqrt(np.nanmean(np.nanvar(M[:, ok], axis=0, ddof=1))))
    out = dict(runs=infos, slice_m=a.slice_m, positions=int(ok.sum()),
               s_r_D_mm=s_r(D), s_r_D_shape_mm=s_r(Dc), s_r_ovality_pct=s_r(Q))
    for k in ("s_r_D_mm", "s_r_D_shape_mm", "s_r_ovality_pct"):
        out[k.replace("s_r", "r_limit")] = 2.8 * out[k]
    corr = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m = np.isfinite(Dc[i]) & np.isfinite(Dc[j])
            if m.sum() >= 10:
                corr[f"{names[i]}-{names[j]}"] = float(np.corrcoef(Dc[i, m], Dc[j, m])[0, 1])
    out["profile_corr"] = corr
    out["profile_corr_median"] = float(np.median(list(corr.values()))) if corr else None
    print(f"{ok.sum()} positions with >= 3 runs: s_r(D) {out['s_r_D_mm']:.2f} mm, s_r(D, per-run median removed) "
          f"{out['s_r_D_shape_mm']:.2f} mm, s_r(ovality) {out['s_r_ovality_pct']:.2f} %-points; "
          f"profile correlation median {out['profile_corr_median']}")
    json.dump(out, open(a.out + ".json", "w", encoding="utf-8"), indent=1)
    with open(a.out + "_profiles.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        keys = ("D_mm", "ovality_pct", "rms_mm", "cover", "inliers", "n", "bright")
        wr.writerow(["x_m"] + [f"{n}_{k}" for n in names for k in keys])
        for k, x in enumerate(grid):
            if not any(prof[n][k] for n in names):
                continue
            wr.writerow([f"{x:.3f}"] + [(f"{prof[n][k][q]:.4g}" if prof[n][k] else "") for n in names for q in keys])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for n, d, q in zip(names, D, Q):
        ax[0].plot(grid, d, lw=1, label=n)
        ax[1].plot(grid, q, lw=1, label=n)
        b = np.array([r["bright"] if r else np.nan for r in prof[n]])
        ax[2].plot(grid, b, lw=1, label=n)
    ax[0].set(ylabel="管径 D [mm]（计数器定尺度）")
    ax[1].set(ylabel="椭圆度 q [%]（ASTM F1216）")
    ax[2].set(ylabel="切片平均亮度", xlabel="计数器距离 [m]")
    ax[0].legend(ncol=len(names), fontsize=8)
    ax[0].set_title(f"各次巡检单独重建：s_r(D) {out['s_r_D_mm']:.1f} mm，去掉各自中位数后 {out['s_r_D_shape_mm']:.1f} mm，"
                    f"s_r(q) {out['s_r_ovality_pct']:.2f} 个百分点")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=110)
    print("saved", a.out + ".json / _profiles.csv / .png")


if __name__ == "__main__":
    main()
