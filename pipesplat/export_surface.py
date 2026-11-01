"""Export the measured pipe wall next to scene.npz (runs on Windows): the OpenMVS mesh of the front end, as the triangles
that train_gs.py renders for the dense depth supervision (--baseline-depth / --anchor) and for the lamp model's
distance and incidence angle (--lamp).

usage: python -I pipesplat/export_surface.py vision WORK_DIR OUT_DIR
writes OUT_DIR/surface.npz: centres, unit normals and size (sqrt(2 x area), ~ edge length) of every textured triangle
of the OpenMVS mesh, in the pipe coordinates of scene.npz (render_views.load_model; same PIPE3D_PIPE_DIAMETER_M as
export_g0.py). For dark, smooth or wet pipes whose mesh has holes use pipesplat/vision_surface.py instead.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (os.path.join(SCRIPTS, "vision"), SCRIPTS, HERE):
    sys.path.insert(0, p)

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def faces(xyz, tri):
    a, b, c = xyz[tri[:, 0]], xyz[tri[:, 1]], xyz[tri[:, 2]]
    n = np.cross(b - a, c - a)
    area2 = np.linalg.norm(n, axis=1)
    keep = area2 > 1e-14
    return (((a + b + c) / 3)[keep].astype(np.float32), (n[keep] / area2[keep, None]).astype(np.float32),
            np.sqrt(area2[keep]).astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    v = sub.add_parser("vision")
    v.add_argument("work"), v.add_argument("out")
    a = ap.parse_args()
    t0 = time.time()
    import paths
    import render_views as rv
    m = rv.load_model(paths.CONVERTER, a.work, [None], straighten=False)
    c, n, s = faces(m["xyz"], m["tri"][~m["blank"]])
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "surface.npz"), centres=c, normals=n, size=s)
    print(f"{len(c):,} triangles, size median {np.median(s) * 1000:.2f} mm, p95 {np.percentile(s, 95) * 1000:.2f} mm "
          f"[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
