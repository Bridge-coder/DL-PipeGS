"""Compare unwrapped wall images (albedo_map.py) of models trained on different inspection runs of the same pipe, and
measure how much lamp shading is left in each. Runs anywhere with numpy + OpenCV.

usage: python albedo_compare.py MAP_A.npz MAP_B.npz [MAP_C.npz ...] [--names A B C ...] [--json OUT.json]
For every pair, over the texels both maps cover (same scene coordinates, same grid):
  ncc        normalised cross-correlation (Pearson) of the luminance
  ncc_detail the same after a high-pass (luminance minus its Gaussian blur, sigma 3 mm): texture only
  logratio   standard deviation of log(A / B) after a Gaussian blur of 20 mm: how much the two maps disagree in their
             smooth part (shading); 0 = one global factor apart
For every map:
  band       circumferential profile (mean luminance per angle over x, normalised by its mean): ratio of the darkest to
             the brightest 60-degree sector - the camera rides near the invert, so lamp shading left in a map shows up
             as a bright invert band
"""
import argparse
import itertools
import json

import cv2
import numpy as np


def lum(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114])


def load(p):
    """luminance on the video's value scale (rgb_display for models with a power-law response), mask, texel size"""
    z = np.load(p)
    rgb = z["rgb_display"] if "rgb_display" in z.files else z["rgb"]
    return lum(rgb.astype(np.float64)), z["weight"] > 1e-3, float(z["texel"])


def blur(x, m, sigma_px):
    w = cv2.GaussianBlur(m.astype(np.float64), (0, 0), sigma_px)
    return cv2.GaussianBlur(np.where(m, x, 0.0), (0, 0), sigma_px) / np.maximum(w, 1e-6)


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / max(np.sqrt((a @ a) * (b @ b)), 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("maps", nargs="+"), ap.add_argument("--names", nargs="+"), ap.add_argument("--json")
    a = ap.parse_args()
    names = a.names or [f"m{k}" for k in range(len(a.maps))]
    M = [load(p) for p in a.maps]
    t = M[0][2]
    out = dict(maps={}, pairs={})
    for n, (L, m, _) in zip(names, M):
        prof = np.array([L[i][m[i]].mean() if m[i].any() else np.nan for i in range(L.shape[0])])
        prof = prof / np.nanmean(prof)
        k = max(1, int(round(len(prof) / 6)))                 # 60-degree sectors (circular)
        sec = np.array([np.nanmean(np.take(prof, range(i, i + k), mode="wrap")) for i in range(len(prof))])
        out["maps"][n] = dict(band=float(np.nanmin(sec) / np.nanmax(sec)), covered=float(m.mean()),
                              brightest_deg=float(np.nanargmax(sec) * 360 / len(prof)))
        print(f"{n}: covered {m.mean():.1%}; circumferential profile darkest / brightest 60-deg sector "
              f"{out['maps'][n]['band']:.3f} (brightest at {out['maps'][n]['brightest_deg']:.0f} deg from the top)")
    for (na, (La, ma, _)), (nb, (Lb, mb, _)) in itertools.combinations(zip(names, M), 2):
        m = ma & mb
        hp = lambda L, mm: L - blur(L, mm, 3e-3 / t)        # noqa: E731
        lr = np.log(np.maximum(blur(La, m, 20e-3 / t), 1e-6)) - np.log(np.maximum(blur(Lb, m, 20e-3 / t), 1e-6))
        r = dict(ncc=ncc(La[m], Lb[m]), ncc_detail=ncc(hp(La, m)[m], hp(Lb, m)[m]),
                 logratio=float(np.std(lr[m])), common=float(m.mean()))
        out["pairs"][f"{na}|{nb}"] = r
        print(f"{na} vs {nb}: NCC {r['ncc']:.3f}, NCC of the texture {r['ncc_detail']:.3f}, "
              f"std of the smooth log ratio {r['logratio']:.3f} (common {r['common']:.1%})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
