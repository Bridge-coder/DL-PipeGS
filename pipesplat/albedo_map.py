"""Unwrapped (developed) wall image of a trained Gaussian model, made the way a camera sees the model: the model's
view-independent colour (degree-0 spherical-harmonic term, no lamp factor, no exposure) is rendered from training
poses with gsplat, and every texel of a regular grid on the measured wall (axial position x by arc length along the
circumference) takes the weighted mean of the renders that see it (multi-view texture baking). Models trained on
different inspection runs share the scene's pipe coordinates, so their maps can be compared texel by texel
(review/albedo_compare.py). Runs in WSL.

usage: $PY pipesplat/albedo_map.py SCENE_DIR RUN_DIR [RUN_DIR ...] [--texel-mm 1.0] [--x0 0.2 --x1 4.4] [--every 4]
       writes RUN_DIR/albedo_map.npz (rgb H x W x 3 in 0..1, weight H x W, x0, texel, radius) and albedo_map.png
       (rows = angle from the top of the pipe, clockwise looking in the direction of travel; columns = x)
Wall: texel (x, angle) -> the measured surface (SCENE_DIR/surface.npz, its triangle centres in pipe coordinates) at
that place (nearest centre in the unwrapped plane). A render counts for a texel when the texel projects inside the
valid canvas (margin 8 px), the rendered depth there agrees with the texel's depth (1.5 % + 2 mm) and the model is
opaque there (alpha > 0.9); weight = cos(incidence)^2 / depth^2 (the sharpest, most frontal views count most).
With a near-field lamp model (--lamp) the colour is the surface's albedo up to one global factor; without it, it is
the colour under the average lighting of the training frames.
"""
import argparse
import os

import cv2
import numpy as np
import torch
import gsplat
from scipy.spatial import cKDTree

C0 = 0.28209479177387814
dev = "cuda"


def canvas_of(K, k1, w, h):
    """the undistorted training canvas of train_gs.py: size, pinhole K (pixel centres at integers), valid mask"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if k1 == 0.0:
        return w, h, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), np.ones((h, w), bool)
    xd, yd = np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy)
    out = (xd ** 2 + yd ** 2 >= (0.95 * 2 / (3 * np.sqrt(-3 * k1))) ** 2) if k1 < 0 else np.zeros(xd.shape, bool)
    xu, yu = np.where(out, 0.0, xd), np.where(out, 0.0, yd)
    for _ in range(20):
        g = 1 + k1 * (xu ** 2 + yu ** 2)
        xu, yu = np.where(out, 0.0, xd) / g, np.where(out, 0.0, yd) / g
    xu[out], yu[out] = np.nan, np.nan
    CW, CH = 2 * int(np.ceil(np.nanmax(np.abs(xu)) * fx)) + 4, 2 * int(np.ceil(np.nanmax(np.abs(yu)) * fy)) + 4
    ccx, ccy = CW / 2 - 0.5, CH / 2 - 0.5
    xu, yu = np.meshgrid((np.arange(CW) - ccx) / fx, (np.arange(CH) - ccy) / fy)
    g = 1 + k1 * (xu ** 2 + yu ** 2)
    ud, vd = cx + fx * xu * g, cy + fy * yu * g
    valid = ((xu ** 2 + yu ** 2) < (0.95 / np.sqrt(-3 * k1)) ** 2) & (ud >= 0) & (ud <= w - 1) & (vd >= 0) & (vd <= h - 1)
    return CW, CH, np.array([[fx, 0, ccx], [0, fy, ccy], [0, 0, 1]]), valid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scene"), ap.add_argument("runs", nargs="+")
    ap.add_argument("--texel-mm", type=float, default=1.0), ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--x0", type=float, default=0.2), ap.add_argument("--x1", type=float, default=4.4)
    ap.add_argument("--radius-m", type=float, default=0.150)
    a = ap.parse_args()
    d = np.load(os.path.join(a.scene, "scene.npz"))
    K = d["K"].astype(np.float64)
    w, h = (int(v) for v in d["size"])
    CW, CH, Kc, valid = canvas_of(K, float(d["k1"]), w, h)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0
    Kt = torch.tensor([[Kc[0, 0], 0, Kc[0, 2] + 0.5], [0, Kc[1, 1], Kc[1, 2] + 0.5], [0, 0, 1]], device=dev, dtype=torch.float32)
    VAL = torch.from_numpy(valid).to(dev)
    views = torch.from_numpy(d["train_viewmats"]).to(dev)
    sel = np.arange(len(views))
    if "train_drop" in d.files:
        sel = np.setdiff1d(sel, d["train_drop"])
    sel = np.setdiff1d(sel, d["block_idx"])[:: a.every]
    # the wall grid on the measured surface
    sf = np.load(os.path.join(a.scene, "surface.npz"))
    cen = sf["centres"].astype(np.float64)
    rc = np.hypot(cen[:, 1], cen[:, 2])
    thc = np.mod(np.arctan2(-cen[:, 1], cen[:, 2]), 2 * np.pi)
    wall = np.abs(rc - a.radius_m) < 0.03
    t = a.texel_mm / 1000
    W, H = int(np.ceil((a.x1 - a.x0) / t)), int(np.ceil(2 * np.pi * a.radius_m / t))
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    xg, thg = a.x0 + (jj + 0.5) * t, (ii + 0.5) * t / a.radius_m
    tree = cKDTree(np.c_[cen[wall, 0], thc[wall] * a.radius_m])
    dist, idx = tree.query(np.c_[xg.ravel(), thg.ravel() * a.radius_m], k=1)
    rr = rc[wall][idx]
    ok_g = dist < 0.01
    P = np.c_[xg.ravel(), -rr * np.sin(thg.ravel()), rr * np.cos(thg.ravel())]
    Nw = -np.c_[np.zeros(len(P)), -np.sin(thg.ravel()), np.cos(thg.ravel())]       # inward wall normal
    Pt = torch.from_numpy(P[ok_g]).float().to(dev)
    Nt = torch.from_numpy(Nw[ok_g]).float().to(dev)
    print(f"grid {W} x {H} texels of {a.texel_mm:g} mm, {ok_g.mean():.1%} on the measured wall; {len(sel)} views", flush=True)
    for rd in a.runs:
        m = torch.load(os.path.join(rd, "model.pt"), map_location=dev)
        means, quats, scales = m["means"].to(dev), m["quats"].to(dev), torch.exp(m["scales"].to(dev))
        opac = torch.sigmoid(m["opacities"].to(dev))
        col = (0.5 + C0 * m["sh0"][:, 0].to(dev)).clamp(0, 1)
        acc = torch.zeros(len(Pt), 3, device=dev)
        wsum = torch.zeros(len(Pt), device=dev)
        for n in sel:
            v = views[n]
            out, al, _ = gsplat.rasterization(means, quats, scales, opac, col, v[None], Kt[None], CW, CH, packed=False,
                                              render_mode="RGB+ED", near_plane=0.01)
            img, dep, al = out[0, ..., :3], out[0, ..., 3], al[0, ..., 0]
            c = Pt @ v[:3, :3].T + v[:3, 3]
            z = c[:, 2]
            u = torch.round(Kc[0, 0] * c[:, 0] / z.clamp(min=1e-6) + Kc[0, 2]).long()
            q = torch.round(Kc[1, 1] * c[:, 1] / z.clamp(min=1e-6) + Kc[1, 2]).long()
            ins = (z > 0.03) & (u >= 0) & (u < CW) & (q >= 0) & (q < CH)
            u, q = u.clamp(0, CW - 1), q.clamp(0, CH - 1)
            ins &= VAL[q, u] & (al[q, u] > 0.9) & ((dep[q, u] - z).abs() < 0.015 * z + 0.002)
            ray = -(Pt @ v[:3, :3].T + v[:3, 3])                   # texel -> camera, camera axes
            ray = ray / ray.norm(dim=1, keepdim=True)
            cosi = ((Nt @ v[:3, :3].T) * ray).sum(1).clamp(min=0)
            wt = torch.where(ins, cosi ** 2 / z.clamp(min=1e-3) ** 2, torch.zeros_like(z))
            acc += wt[:, None] * img[q, u]
            wsum += wt
        rgb = np.zeros((H * W, 3), np.float32)
        wgt = np.zeros(H * W, np.float32)
        rgb[ok_g] = (acc / wsum.clamp(min=1e-9)[:, None]).cpu().numpy()
        wgt[ok_g] = wsum.cpu().numpy()
        rgb, wgt = rgb.reshape(H, W, 3), wgt.reshape(H, W)
        # a model with a power-law response (--lamp-gamma / --lamp-gamma-fixed) holds linear albedo; rgb_display = rgb^p is
        # on the scale of the video values, comparable with the colours of models without one
        p = 1.0
        if "lamp_logp" in m:
            p = float(torch.exp(m["lamp_logp"]).item())
        else:
            sj = os.path.join(rd, "summary_r2.json")
            if os.path.exists(sj):
                import json
                p = float(json.load(open(sj))["args"].get("lamp_gamma_fixed", 0.0)) or 1.0
        np.savez_compressed(os.path.join(rd, "albedo_map.npz"), rgb=rgb, rgb_display=np.clip(rgb, 0, None) ** p,
                            response_p=p, weight=wgt, x0=a.x0, texel=t, radius=a.radius_m)
        cov = wgt > 1e-3 * np.percentile(wgt[wgt > 0], 50)
        vis = np.where(cov[..., None], rgb, 0.5)
        lo, hi = np.percentile(vis[cov], [1, 99])
        vis = np.clip((vis - lo) / max(hi - lo, 1e-6), 0, 1)
        cv2.imwrite(os.path.join(rd, "albedo_map.png"), (vis[..., ::-1] * 255).astype(np.uint8))
        print(f"{rd}: covered {cov.mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
