"""Bake a Gaussian model onto the height-field tube of vision_surface.py and score the textured mesh (docs/11 Y2):
the viewer-ready deliverable (OBJ + PNG, colours from the original video through the Gaussian model) for a data set
whose wall is a height field r(u, theta) around a centerline - InPipe (vision poses) and Pipe1. Runs in the pipe3d
WSL distro. Same recipe as bake_gs_tex.py (docs/10 X8, concrete_2), simpler because the mesh is a height field:
the texture is the (u, theta) grid itself, every texel's surface point is centerline.point(u, theta, r(u, theta)).

1. every training pose (block frames left out), --source video (default): the RAW training frame itself, visibility
   from the tube's own depth (meshraster.py, exact triangles in the raw frame); --source model: the Gaussian model
   rendered on its training canvas (as train_gs.py: colour, expected depth, alpha >= --alpha-min). For a --lamp model
   the Gaussians' own colour is the de-lit colour, which only looks like the video after the lamp factor: on InPipe
   Y1 its bake was swirls (docs/11 Y2), so lamp models are baked from the video. Each texel near the camera is
   projected (with k1); kept where visible (|z - depth| < 1.5 % + 2 mm) and inside the frame; footprint = the size of
   one raw pixel on the wall (z / f / cos(incidence to the radius), times the stretch of k1).
   Before that (default; --mesh-surface skips it), as bake_gs_tex.py --model-surface (docs/10 X8): the tube is moved
   onto the model's own surface - per 2 mm cell the radial offset of the model's rendered depth (alpha > 0.99, within
   1.5 % + 4 mm of the tube) along the pixel's ray, weighted cos^2 / z^2 over the training views, smoothed
   (--disp-smooth-mm), clipped to +-5 mm. The fitted tube is smooth; the Gaussians, anchored within +-2 mm, sit
   where the images agree - ~1.4 mm off it on InPipe, which at grazing angles is pixels between views
   (docs/11 Y2). The moved tube is the delivered OBJ's geometry (mesh_bake.npz)
2. two passes: (a) the smallest footprint per texel, (b) weighted means: near w = exp(-(fp / fp_best - 1) / --near-tau)
   and ref w = exp(-((z - ref_z) / ref_sigma)^2) (+1e-4); texture = near x blur(ref) / blur(near) (Gaussian
   --lp-mm, wrapping in theta for a closed tube): detail of the sharpest views, shading of views at one distance
3. unseen texels filled from their neighbours (push-pull); texture.png (8 bit, 99.5th percentile at 250) + OBJ/MTL
   (vertices = the height-field mesh, UV = (u, theta); a closed tube gets a duplicated seam column)
4. scoring with ruler v2 on the held-out and block frames, exactly as train_gs.py --ruler2 / train_tex.py (B's mask,
   Nal, D): the mesh rasterised in the raw frame (meshraster.py), every pixel's surface point -> (u, theta) ->
   texture (bilinear), times a camera-fixed lamp map learnt on training frames (lampmap.learn_lamp_map's recipe,
   "x lamp map" = B's recipe), then the frame's gain + offset. summary_r2.json in summarize_r2.py's layout.

--two-light (docs/13 Z2; a --lamp --lamp-ambient model, InPipe Y1): the frames are lit by the robot's lamp (follows the
camera, 1/r^2) and by the room light through the slot and the open end (follows the world); one static texture times
a camera-fixed lamp map cannot hold both (docs/11 sec 8, item 1). The model's own decomposition is used instead:
frame = gain_n x (L x albedo + ambient) + offset_n with L = the model's lamp factor (camera-fixed map M x cos / (r /
r_ref)^2, train_gs.py) and gain_n / offset_n the model's exposure of training frame n.
  ambient E: the model's per-Gaussian ambient colour rendered in every training view, averaged per texel (all views);
  albedo A: from the video, least squares per texel over the near views (weights as pass 2b near):
            A = sum w L (y' - E) / sum w L^2 with y' = (frame - offset_n) / gain_n.
  albedo.png + ambient.png (+ lamp.npz: M, lamp offset, r_ref) are the relightable mesh; texture.png = A x L_typ + E
  (L_typ the median lamp factor of the near views) is the static look for viewers without a headlight.
  Scored twice: summary_r2.json = the mesh with the model's headlight (L x A + E, then the frame's gain + offset as
  every model), summary_r2_static.json = texture.png x a camera-fixed lamp map (the recipe above, comparable with Y2).

--two-light --source unwrap (11-01, the figures of the main model): albedo and ambient taken from the model's Gaussians
themselves instead of solved from the video: the albedo is the colour the lamp factor multiplies, the ambient the
added colour, so frame = gain_n x (L x albedo + ambient) + offset_n holds by construction wherever the model renders
the wall. Both are rendered by virtual pinhole cameras on the centerline, every --unwrap-step-mm along u, in
--unwrap-dirs directions around it, each looking radially at the wall (x = increasing theta, y = +u, z = outward):
every ray in a camera's cross-section plane is radial, so theta has no parallax and u only a little (texels are
blended from the two nearest stations and the two nearest directions, bilinear weights). Only the Gaussians whose
centre lies within -IN..+OUT mm (--unwrap-band-mm) of the tube (moved onto the model's surface) are drawn: floaters in
the pipe and Gaussians behind the wall, which the inspection camera never sees, are left out. Colour = the Gaussians'
spherical harmonics toward the centerline point --unwrap-view-z behind them (the direction the robot saw them from,
--unwrap-colour view) or their view-independent base colour (dc); normalised by the rendered alpha; texels kept where
alpha >= --unwrap-alpha and the rendered depth is the tube's (1.5 % + 3 mm). The video bake still runs first, for the
typical lamp factor of texture.png. Also writes unwrap.npz: alpha, the radius of the rendered surface and the seen
mask per texel (rows = theta). First result (InPipe): the 3DGS Gaussians, trained from views along the pipe only,
are needles when seen face-on - streaks in the texture.
--two-light --source delit: the same two textures from the model rendered at the training poses (as the robot saw
the wall, so the Gaussians look as trained): per texel the near-weighted mean of the model's own colour (its albedo,
spherical harmonics toward that camera) and of its ambient render, visible where the model is opaque at the tube.

usage: TORCH_HOME=<TORCH_CACHE of scripts/paths.py> /opt/gsenv/bin/python pipesplat/bake_vhf.py SCENE_DIR MODEL_RUN OUT_DIR
           [--name vhf] [--texel-mm 0.5] [--near-tau 1] [--ref-z auto] [--lp-mm 12] [--alpha-min 0.95] [--far 1.2]
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gsplat
from gsplat.cuda._torch_impl import _spherical_harmonics

HERE =os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lampmap"))
sys.path.insert(0, HERE)
import imgmetrics as im  # noqa: E402
from meshraster import MeshRaster  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("scene"), ap.add_argument("run"), ap.add_argument("out")
ap.add_argument("--name", default="vhf", help="SCENE_DIR/mesh_<name>.npz and centerline_<name>.npz (vision_surface.py)")
ap.add_argument("--texel-mm", type=float, default=0.5)
ap.add_argument("--near-tau", type=float, default=1.0)
ap.add_argument("--ref-z", type=float, default=None, help="m ahead of the camera; default 2.5 x the median radius")
ap.add_argument("--ref-sigma", type=float, default=None, help="default 0.4 x the median radius")
ap.add_argument("--lp-mm", type=float, default=12.0, help="0: the near mean alone (no shading from the ref views)")
ap.add_argument("--alpha-min", type=float, default=0.95)
ap.add_argument("--margin-px", type=int, default=3)
ap.add_argument("--far", type=float, default=1.2, help="texels at most this far ahead of a camera (m along u)")
ap.add_argument("--save", type=int, default=4)
ap.add_argument("--mesh-surface", action="store_true", help="bake on the fitted tube as it is (no move to the model's surface)")
ap.add_argument("--disp-smooth-mm", type=float, default=4.0)
ap.add_argument("--flat-iters", type=int, default=1,
                help="video source: learn the frames' camera-fixed shading (lamp x vignetting) against the baked texture, "
                     "divide it out of the frames and bake again, this many times (0 = off)")
ap.add_argument("--edge-px", type=float, default=30.0, help="samples closer than this to the frame border weigh less")
ap.add_argument("--source", choices=("video", "model", "delit", "unwrap"), default="video",
                help="texel colours from the raw training frames (default), from the Gaussian model's renders, or "
                     "(--two-light) the model's albedo / ambient rendered at the training poses (delit) or seen from "
                     "the centerline (unwrap)")
ap.add_argument("--two-light", action="store_true",
                help="albedo + ambient textures with the model's lamp (see the module docstring); --source video or "
                     "unwrap, no flat field")
ap.add_argument("--unwrap-step-mm", type=float, default=20.0, help="--source unwrap: stations along the centerline")
ap.add_argument("--unwrap-dirs", type=int, default=12, help="--source unwrap: cameras around the centerline per station")
ap.add_argument("--unwrap-ss", type=float, default=1.5, help="--source unwrap: rendered pixels per texel on the wall")
ap.add_argument("--unwrap-band-mm", type=float, nargs=2, default=(15.0, 10.0),
                help="--source unwrap: Gaussians drawn whose centre is IN mm inside .. OUT mm outside the tube")
ap.add_argument("--unwrap-colour", choices=("view", "dc"), default="view",
                help="--source unwrap: spherical harmonics toward the robot's typical view, or the base colour only")
ap.add_argument("--unwrap-view-z", type=float, default=None,
                help="--source unwrap, view: the robot's camera this far behind the Gaussian on the centerline (m); "
                     "default the ref distance")
ap.add_argument("--unwrap-alpha", type=float, default=0.5)
a = ap.parse_args()
dev = "cuda"
os.makedirs(os.path.join(a.out, "renders"), exist_ok=True)
os.makedirs(os.path.join(a.out, "obj"), exist_ok=True)
T0 = time.time()
_logf = open(os.path.join(a.out, "bake.log"), "a", encoding="utf-8")


def log(msg):
    line = f"{msg} [{time.time() - T0:.0f} s]"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


log(f"args {vars(a)}")
d = np.load(os.path.join(a.scene, "scene.npz"))
K = d["K"].astype(np.float64)
k1 = float(d["k1"])
w, h = (int(v) for v in d["size"])
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]


# ------------------------------------------------------------------------------ canvas (train_gs.py) ----------
def undistort_grid(xd, yd, k1):
    out = (xd ** 2 + yd ** 2 >= (0.95 * 2 / (3 * np.sqrt(-3 * k1))) ** 2) if k1 < 0 else np.zeros(xd.shape, bool)
    xd, yd = np.where(out, 0.0, xd), np.where(out, 0.0, yd)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        g = 1 + k1 * (xu ** 2 + yu ** 2)
        xu, yu = xd / g, yd / g
    xu[out], yu[out] = np.nan, np.nan
    return xu, yu


XU, YU = undistort_grid(*np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy), k1) if k1 != 0 else \
    np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy)
valid_raw = np.isfinite(XU)
if k1 == 0.0:
    CW, CH, ccx, ccy = w, h, cx, cy
    valid_canvas = np.ones((h, w), bool)
else:
    CW, CH = 2 * int(np.ceil(np.nanmax(np.abs(XU)) * fx)) + 4, 2 * int(np.ceil(np.nanmax(np.abs(YU)) * fy)) + 4
    ccx, ccy = CW / 2 - 0.5, CH / 2 - 0.5
    xu, yu = np.meshgrid((np.arange(CW) - ccx) / fx, (np.arange(CH) - ccy) / fy)
    g = 1 + k1 * (xu ** 2 + yu ** 2)
    ud, vd = cx + fx * xu * g, cy + fy * yu * g
    valid_canvas = ((xu ** 2 + yu ** 2) < (0.95 / np.sqrt(-3 * k1)) ** 2) & (ud >= 0) & (ud <= w - 1) & (vd >= 0) & (vd <= h - 1)
Kc = torch.tensor([[fx, 0, ccx + 0.5], [0, fy, ccy + 0.5], [0, 0, 1]], device=dev, dtype=torch.float32)
m_ = 2 * a.margin_px + 1
VALID = torch.from_numpy(cv2.erode(valid_canvas.astype(np.uint8), np.ones((m_, m_), np.uint8)) > 0).to(dev)

sv = torch.load(os.path.join(a.run, "model.pt"))
assert tuple(int(v) for v in sv["size"]) == (CW, CH), "canvas differs from the model's"
GS = {k: sv[k].to(dev) for k in ("means", "quats", "scales", "opacities", "sh0", "shN")}
lamp_model = "lamp_logM" in sv
if a.two_light:
    assert lamp_model and "amb" in sv, "--two-light needs a --lamp --lamp-ambient model (train_gs.py)"
    assert a.source != "model", "--two-light: the albedo from the video or the model's albedo (--source delit / unwrap)"
    a.flat_iters = 0                            # the lamp model replaces the flat field
    LP = {k: sv["lamp_" + k].to(dev) for k in ("logM", "t", "gain", "off")}
    AMB = torch.sigmoid(sv["amb"].to(dev))
    mr = re.search(r"r_ref (\d+(?:\.\d+)?) mm", open(os.path.join(a.run, "train.log"), encoding="utf-8").read())
    R_REF = float(mr.group(1)) / 1000
del sv
log(f"raw {w}x{h} k1 {k1}, canvas {CW}x{CH}; model {a.run}: {len(GS['means']):,} Gaussians"
    + (" (--lamp model)" if lamp_model else "") + f"; colours from the {a.source}")


@torch.no_grad()
def render_model(view):
    out, al, _ = gsplat.rasterization(GS["means"], GS["quats"], torch.exp(GS["scales"]), torch.sigmoid(GS["opacities"]),
                                      torch.cat([GS["sh0"], GS["shN"]], 1), view[None], Kc[None], CW, CH, sh_degree=3,
                                      packed=False, render_mode="RGB+ED", near_plane=0.01)
    return out[0, ..., :3].clamp(0, 1), out[0, ..., 3], al[0, ..., 0]


@torch.no_grad()
def render_ambient(view):
    """the model's per-Gaussian ambient colour on the canvas (train_gs.py ambient())"""
    out, _, _ = gsplat.rasterization(GS["means"], GS["quats"], torch.exp(GS["scales"]), torch.sigmoid(GS["opacities"]),
                                     AMB, view[None], Kc[None], CW, CH, packed=False, render_mode="RGB", near_plane=0.01)
    return out[0]


def lamp_L(Xc, nc):
    """the model's lamp factor M x cos(incidence) / (r / r_ref)^2 (train_gs.py lamp_factor) at camera-frame points Xc with
    unit normals nc; M bilinear on the canvas as train_gs.py (align_corners)"""
    z = Xc[:, 2].clamp(min=1e-6)
    px, py = ccx + fx * Xc[:, 0] / z, ccy + fy * Xc[:, 1] / z
    g = torch.stack([px / (CW - 1) * 2 - 1, py / (CH - 1) * 2 - 1], -1).view(1, 1, -1, 2)
    M = torch.exp(F.grid_sample(LP["logM"], g, align_corners=True, padding_mode="border")).view(-1)
    Lv = LP["t"] - Xc
    r = Lv.norm(dim=1).clamp(min=1e-3)
    cos = ((nc * Lv).sum(1) / r).abs().clamp(min=0.05)
    return M * cos / (r / R_REF) ** 2


# ------------------------------------------------------------------------------ height field + centerline ----
mz = np.load(os.path.join(a.scene, f"mesh_{a.name}.npz"))
cz = np.load(os.path.join(a.scene, f"centerline_{a.name}.npz"))
CLt = {k: torch.from_numpy(cz[k].astype(np.float32)).to(dev) for k in ("u", "C", "T", "N", "Bn")}
cl_u0, cl_ds, th0 = float(cz["u"][0]), float(cz["u"][1] - cz["u"][0]), math.radians(float(cz["th0"]))
RG = mz["r"].astype(np.float32)                         # grid radius [nu, nt] (cell centres)
OPEN = mz["open"]
g_u0, g_du, g_dth = float(mz["u0"]), float(mz["du"]), float(mz["dth"])
g_nu, g_nt = RG.shape
closed = not OPEN.any()
R_med = float(np.median(RG))
ref_z = a.ref_z or 2.5 * R_med
ref_sigma = a.ref_sigma or 0.4 * R_med


def cl_frame(u):
    i = ((u - cl_u0) / cl_ds).clamp(0, len(CLt["u"]) - 1)
    i0 = i.floor().long().clamp(max=len(CLt["u"]) - 2)   # (float32: n - 1.000001 rounds to n - 1 on a 12 m line)
    t = (i - i0).clamp(0, 1)[:, None]
    return [(1 - t) * CLt[k][i0] + t * CLt[k][i0 + 1] for k in ("C", "N", "Bn")]


def grid_r(u, th):
    """bilinear radius of the height field at (u, theta) (torch)"""
    gi = ((u - g_u0) / g_du - 0.5).clamp(0, g_nu - 1.0001)
    gj = (th + math.pi) / g_dth - 0.5
    i0 = gi.floor().long()
    j0 = torch.floor(gj).long()
    ti, tj = gi - i0, gj - j0
    RGt = RG_t
    j0w, j1w = j0 % g_nt, (j0 + 1) % g_nt
    i1 = (i0 + 1).clamp(max=g_nu - 1)
    return ((1 - ti) * ((1 - tj) * RGt[i0, j0w] + tj * RGt[i0, j1w]) + ti * ((1 - tj) * RGt[i1, j0w] + tj * RGt[i1, j1w]))


RG_t = torch.from_numpy(RG).to(dev)
OPEN_t = torch.from_numpy(OPEN).to(dev)
xyz, tri = mz["xyz"].astype(np.float64), mz["tri"].astype(np.int64)
MR = MeshRaster(xyz, tri, K, k1, w, h, np.nan_to_num(XU), np.nan_to_num(YU), dev)
VR = torch.from_numpy(valid_raw).to(dev)
VR_in = torch.from_numpy(cv2.erode(valid_raw.astype(np.uint8), np.ones((m_, m_), np.uint8)) > 0).to(dev)
raw_train = d["train_images"]
R2MAX = 0.95 ** 2 / (-3 * k1) if k1 < 0 else float("inf")
assert a.source != "model" or not lamp_model, "a --lamp model's own colour is de-lit: bake it with --source video"
assert a.source not in ("delit", "unwrap") or a.two_light, "--source delit / unwrap are --two-light bakes"


# texture grid
du = a.texel_mm / 1000
U0, U1 = g_u0, g_u0 + g_nu * g_du
NU = int(math.ceil((U1 - U0) / du))
UT1 = U0 + NU * du                                      # the texture's extent along u
NTH = int(math.ceil(2 * math.pi * R_med / du))
dth = 2 * math.pi / NTH
tu = torch.from_numpy((U0 + (np.arange(NU) + 0.5) * du).astype(np.float32)).to(dev)
tt = torch.from_numpy((-math.pi + (np.arange(NTH) + 0.5) * dth).astype(np.float32)).to(dev)
log(f"height field {g_nu} x {g_nt}, u {U0:.3f}..{U1:.3f} m, radius median {R_med * 1000:.1f} mm, "
    f"{'closed' if closed else 'open (slot / ends)'}; texture {NTH} x {NU} texels ({a.texel_mm} mm); ref {ref_z:.2f} m")


def texel_points(cols):
    """world points, outward unit normals (radial) and open mask of texture columns cols (all angles)"""
    U = tu[cols][:, None].expand(-1, NTH).reshape(-1)
    TH = tt[None, :].expand(len(cols), -1).reshape(-1)
    r = grid_r(U, TH)
    C, N, Bn = cl_frame(U)
    t = TH + th0
    rad = torch.cos(t)[:, None] * N + torch.sin(t)[:, None] * Bn
    gi = ((U - g_u0) / g_du).long().clamp(0, g_nu - 1)
    gj = ((TH + math.pi) / g_dth).long().clamp(0, g_nt - 1)
    return C + r[:, None] * rad, rad, OPEN_t[gi, gj]


def cam_u(view):
    R, t = view[:3, :3], view[:3, 3]
    Cw = -(R.T @ t)
    d2 = ((CLt["C"][::10] - Cw) ** 2).sum(1)
    return float(CLt["u"][::10][d2.argmin()])


views_all = torch.from_numpy(d["train_viewmats"]).to(dev)
train = np.setdiff1d(np.arange(len(views_all)), d["block_idx"])
COLS = torch.arange(NU, device=dev)


@torch.no_grad()
def project_view(view, cols, n):
    """per texel of cols: colour, footprint, z, ok (visible, inside; --source model: opaque) in training frame n"""
    X, nrm, op = texel_points(cols)
    R, t = view[:3, :3], view[:3, 3]
    Xc = X @ R.T + t
    z = Xc[:, 2]
    ok = (z > 0.02) & ~op
    xn, yn = Xc[:, 0] / z.clamp(min=1e-6), Xc[:, 1] / z.clamp(min=1e-6)
    if a.source in ("video", "unwrap"):         # the raw frame: distorted pixel, the tube's own depth
        r2 = xn ** 2 + yn ** 2
        ok &= r2 < R2MAX
        g_ = 1 + k1 * r2
        px, py = cx + fx * xn * g_, cy + fy * yn * g_
        ok &= (px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1)
        pi_, pj = px.round().long().clamp(0, w - 1), py.round().long().clamp(0, h - 1)
        ok &= VR_in[pj, pi_]
        dep, _ = MR(view)
        dz = dep[pj, pi_]
        ok &= (dz > 0) & ((z - dz).abs() < 0.015 * z + 0.002)
        col = torch.from_numpy(raw_train[n]).to(dev).float() / 255
        if FLAT[0] is not None:                 # camera-fixed lamp / vignetting divided out (--flat-iters)
            col = col / FLAT[0][..., None].clamp(min=0.05)
        de = torch.minimum(torch.minimum(px, w - 1 - px), torch.minimum(py, h - 1 - py))
        gx, gy = px / (w - 1) * 2 - 1, py / (h - 1) * 2 - 1
    else:
        px, py = ccx + fx * xn, ccy + fy * yn
        ok &= (px >= 0) & (px <= CW - 1) & (py >= 0) & (py <= CH - 1)
        pi_, pj = px.round().long().clamp(0, CW - 1), py.round().long().clamp(0, CH - 1)
        ok &= VALID[pj, pi_]
        col, dep, al = render_model(view)
        dz = dep[pj, pi_]
        ok &= ((z - dz).abs() < 0.015 * z + 0.002) & (al[pj, pi_] >= a.alpha_min)
        gx, gy = px / (CW - 1) * 2 - 1, py / (CH - 1) * 2 - 1
        de = torch.minimum(torch.minimum(px, CW - 1 - px), torch.minimum(py, CH - 1 - py))
    c = F.grid_sample(col.permute(2, 0, 1)[None], torch.stack([gx, gy], -1)[None, None], align_corners=True)[0, :, 0].T
    ray = Xc / Xc.norm(dim=1, keepdim=True)
    cosi = (ray * (nrm @ R.T)).sum(1).abs().clamp(min=0.05)
    stretch = 1.0 if k1 == 0 else 1.0 / (1 + 3 * k1 * (xn ** 2 + yn ** 2)).clamp(min=0.2)
    fp = z / fx / cosi * stretch
    we = (de / max(a.edge_px, 1e-6)).clamp(0, 1) ** 2      # samples near the frame border count less
    return c, fp, z, ok, we


# ------------------------------------------------------------------------------ bake (two passes) -----------
cams = [(int(n), cam_u(views_all[int(n)])) for n in train]
if not a.mesh_surface:                          # move the tube onto the model's own surface (docs/11 Y2)
    gu = torch.from_numpy((g_u0 + (np.arange(g_nu) + 0.5) * g_du).astype(np.float32)).to(dev)
    gt = torch.from_numpy((-math.pi + (np.arange(g_nt) + 0.5) * g_dth).astype(np.float32)).to(dev)
    s_w = torch.zeros(g_nu * g_nt, device=dev)
    s_d = torch.zeros(g_nu * g_nt, device=dev)
    t_m = time.time()
    with torch.no_grad():
        for n, uc in cams:
            lo, hi = int(max(0, (uc - 0.15 - g_u0) / g_du)), int(min(g_nu, (uc + a.far - g_u0) / g_du))
            if hi <= lo:
                continue
            U = gu[lo:hi][:, None].expand(-1, g_nt).reshape(-1)
            TH = gt[None, :].expand(hi - lo, -1).reshape(-1)
            r = RG_t[lo:hi].reshape(-1)
            C, N, Bn = cl_frame(U)
            rad = torch.cos(TH + th0)[:, None] * N + torch.sin(TH + th0)[:, None] * Bn
            view = views_all[n]
            R_, t_ = view[:3, :3], view[:3, 3]
            Xc = (C + r[:, None] * rad) @ R_.T + t_
            z = Xc[:, 2]
            xn, yn = Xc[:, 0] / z.clamp(min=1e-6), Xc[:, 1] / z.clamp(min=1e-6)
            px, py = ccx + fx * xn, ccy + fy * yn
            ok = (z > 0.02) & (px >= 0) & (px <= CW - 1) & (py >= 0) & (py <= CH - 1) & ~OPEN_t[lo:hi].reshape(-1)
            pi_, pj = px.round().long().clamp(0, CW - 1), py.round().long().clamp(0, CH - 1)
            ok &= VALID[pj, pi_]
            _, dep, al = render_model(view)
            dz, az = dep[pj, pi_], al[pj, pi_]
            ok &= (az > 0.99) & ((z - dz).abs() < 0.015 * z + 0.004)
            radc = rad @ R_.T
            ray = Xc / Xc.norm(dim=1, keepdim=True)
            cosi = (ray * radc).sum(1).abs()
            dr = (dz / z.clamp(min=1e-6) - 1) * (Xc * radc).sum(1)       # radial part of the move along the ray
            wv = torch.where(ok, cosi ** 2 / z ** 2, torch.zeros_like(z))
            idx = torch.arange(lo * g_nt, hi * g_nt, device=dev)
            s_w.index_add_(0, idx, wv)
            s_d.index_add_(0, idx, wv * dr)
    sw, sd = s_w.view(g_nu, g_nt).cpu().numpy(), s_d.view(g_nu, g_nt).cpu().numpy()
    sig = (a.disp_smooth_mm / 1000 / g_du, a.disp_smooth_mm / 1000 / (g_dth * R_med))
    mode = ("nearest", "wrap") if closed else "nearest"
    from scipy.ndimage import gaussian_filter
    off = gaussian_filter(sd, sig, mode=mode) / np.maximum(gaussian_filter(sw, sig, mode=mode), 1e-12)
    off = np.where(gaussian_filter((sw > 0).astype(float), sig, mode=mode) > 0.05, off, 0.0)
    off = np.clip(off, -0.005, 0.005).astype(np.float32)
    seen_c = sw > 0
    log(f"model surface: {seen_c.mean():.1%} of the cells seen opaque; radial offset of the model from the tube median "
        f"{np.median(off[seen_c]) * 1000:+.2f} mm, |offset| p50 / p95 {np.median(np.abs(off[seen_c])) * 1000:.2f} / "
        f"{np.percentile(np.abs(off[seen_c]), 95) * 1000:.2f} mm [{time.time() - t_m:.0f} s]")
    RG = RG + off
    RG_t = torch.from_numpy(RG).to(dev)
    vi_, vj_ = torch.from_numpy(mz["vi"].astype(np.int64)).to(dev), torch.from_numpy(mz["vj"].astype(np.int64)).to(dev)
    C, N, Bn = cl_frame(gu[vi_])
    rad = torch.cos(gt[vj_] + th0)[:, None] * N + torch.sin(gt[vj_] + th0)[:, None] * Bn
    xyz = (C + RG_t[vi_, vj_][:, None] * rad).double().cpu().numpy()
    MR = MeshRaster(xyz, tri, K, k1, w, h, np.nan_to_num(XU), np.nan_to_num(YU), dev)
    np.savez(os.path.join(a.out, "mesh_bake.npz"), xyz=xyz.astype(np.float32), tri=tri.astype(np.int32), r=RG,
             offset=off)
NT = NU * NTH
FLAT = [None]


def cols_for(uc):
    lo, hi = int(max(0, (uc - 0.15 - U0) / du)), int(min(NU, (uc + a.far - U0) / du))
    return COLS[lo:hi], lo


# ------------------------------------------------------------------------------ the tube as a viewer draws it --
TEXc = [None]                                           # the texture being drawn
RX, RY = (torch.from_numpy(np.nan_to_num(v).astype(np.float32)).to(dev) for v in (XU, YU))


def to_pipe(X):
    sub = CLt["C"][::10]
    i = torch.cdist(X, sub).argmin(1) * 10
    u = CLt["u"][i]
    for _ in range(4):
        i = ((u - cl_u0) / cl_ds).round().long().clamp(0, len(CLt["u"]) - 1)
        u = CLt["u"][i] + ((X - CLt["C"][i]) * CLt["T"][i]).sum(1)
    i = ((u - cl_u0) / cl_ds).round().long().clamp(0, len(CLt["u"]) - 1)
    q = X - CLt["C"][i]
    th = torch.atan2((q * CLt["Bn"][i]).sum(1), (q * CLt["N"][i]).sum(1)) - th0
    return u, torch.remainder(th + math.pi, 2 * math.pi) - math.pi


@torch.no_grad()
def render_mesh(view):
    D, face = MR(view)
    ok = (face >= 0) & VR
    Xc = torch.stack([RX * D, RY * D, D], -1)[ok]
    R, t = view[:3, :3], view[:3, 3]
    Xw = (Xc - t) @ R
    out = torch.zeros(h, w, 3, device=dev)
    for s in range(0, len(Xw), 400000):
        u, th = to_pipe(Xw[s:s + 400000])
        gx = (u - U0) / (UT1 - U0) * 2 - 1
        gy = (th + math.pi) / (2 * math.pi) * 2 - 1
        c = F.grid_sample(TEXc[0].permute(2, 0, 1)[None], torch.stack([gx, gy], -1)[None, None], align_corners=False,
                          padding_mode="border")[0, :, 0].T
        idx = torch.nonzero(ok.view(-1))[s:s + 400000, 0]
        out.view(-1, 3)[idx] = c
    return out.mul(255).cpu().numpy(), ok.cpu().numpy()


@torch.no_grad()
def render_mesh_two(view):
    """--two-light: the mesh with the model's headlight, L x albedo + ambient per pixel (same rasteriser as render_mesh)"""
    D, face = MR(view)
    ok = (face >= 0) & VR
    Xc = torch.stack([RX * D, RY * D, D], -1)[ok]
    R, t = view[:3, :3], view[:3, 3]
    Xw = (Xc - t) @ R
    out = torch.zeros(h, w, 3, device=dev)
    for s in range(0, len(Xw), 400000):
        u, th = to_pipe(Xw[s:s + 400000])
        g = torch.stack([(u - U0) / (UT1 - U0) * 2 - 1, (th + math.pi) / (2 * math.pi) * 2 - 1], -1)[None, None]
        A = F.grid_sample(TEXA[0].permute(2, 0, 1)[None], g, align_corners=False, padding_mode="border")[0, :, 0].T
        E = F.grid_sample(TEXE[0].permute(2, 0, 1)[None], g, align_corners=False, padding_mode="border")[0, :, 0].T
        C, N, Bn = cl_frame(u)
        rad = torch.cos(th + th0)[:, None] * N + torch.sin(th + th0)[:, None] * Bn
        L = lamp_L(Xc[s:s + 400000], rad @ R.T)
        idx = torch.nonzero(ok.view(-1))[s:s + 400000, 0]
        out.view(-1, 3)[idx] = L[:, None] * A + E
    return out.mul(255).cpu().numpy(), ok.cpu().numpy()


def learn_lamp(render=None):
    """camera-fixed gain map (lampmap.learn_lamp_map's recipe) of the current texture (or of render) against the raw
    training frames"""
    lm_acc, lm_cnt = np.zeros((h, w)), np.zeros((h, w))
    for n in train[::4]:
        G, ok = (render or render_mesh)(views_all[int(n)])
        real = raw_train[int(n)].astype(np.float32)
        m = ok & valid_raw
        g = im.linear_match(G, real, m)
        lm_acc += np.where(m, np.log(im.grey(real) + 1.0) - np.log(im.grey(g) + 1.0), 0)
        lm_cnt += m
    lm_mean = np.where(lm_cnt > 0, lm_acc / np.maximum(lm_cnt, 1), 0).astype(np.float32)
    return np.exp(cv2.GaussianBlur(lm_mean, (0, 0), 6) / (cv2.GaussianBlur((lm_cnt > 0).astype(np.float32), (0, 0), 6) + 1e-6))


def bake_once():
    """the two passes; returns near, ref (rows = theta) and the seen mask"""
    fp_best = torch.full((NT,), float("inf"), device=dev)
    t_b = time.time()
    for n, uc in cams:
        cols, lo = cols_for(uc)
        if len(cols) == 0:
            continue
        _, fp, _, ok, _ = project_view(views_all[n], cols, n)
        sl = slice(lo * NTH, lo * NTH + len(cols) * NTH)
        fp_best[sl] = torch.minimum(fp_best[sl], torch.where(ok, fp, torch.full_like(fp, float("inf"))))
    log(f"pass 1 (best footprint) over {len(cams)} training views: {torch.isfinite(fp_best).float().mean().item():.1%} of "
        f"the texels seen; footprint median {fp_best[torch.isfinite(fp_best)].median().item() * 1000:.2f} mm [{time.time() - t_b:.0f} s]")
    acc_n = torch.zeros(NT, 3, device=dev)
    wn = torch.zeros(NT, device=dev)
    acc_r = torch.zeros(NT, 3, device=dev)
    wr = torch.zeros(NT, device=dev)
    for n, uc in cams:
        cols, lo = cols_for(uc)
        if len(cols) == 0:
            continue
        c, fp, z, ok, we = project_view(views_all[n], cols, n)
        sl = slice(lo * NTH, lo * NTH + len(cols) * NTH)
        w_n = torch.where(ok, we * torch.exp(-(fp / fp_best[sl] - 1) / a.near_tau), torch.zeros_like(fp))
        w_r = torch.where(ok, we * (torch.exp(-((z - ref_z) / ref_sigma) ** 2) + 1e-4), torch.zeros_like(fp))
        acc_n[sl] += w_n[:, None] * c
        wn[sl] += w_n
        acc_r[sl] += w_r[:, None] * c
        wr[sl] += w_r
    log(f"pass 2 (weighted means) done [{time.time() - t_b:.0f} s]")
    seen = (wn > 0).view(NU, NTH).T.cpu().numpy()
    near = (acc_n / wn.clamp(min=1e-12)[:, None]).view(NU, NTH, 3).permute(1, 0, 2).cpu().numpy()
    ref = (acc_r / wr.clamp(min=1e-12)[:, None]).view(NU, NTH, 3).permute(1, 0, 2).cpu().numpy()
    return near, ref, seen


def bake_two():
    """--two-light: albedo (least squares over the near views, the model's lamp factor and exposure known) and ambient
    (the model's ambient render, mean over all views); returns albedo, ambient (rows = theta), seen, typical lamp factor"""
    fp_best = torch.full((NT,), float("inf"), device=dev)
    t_b = time.time()
    for n, uc in cams:
        cols, lo = cols_for(uc)
        if len(cols) == 0:
            continue
        _, fp, _, ok, _ = project_view(views_all[n], cols, n)
        sl = slice(lo * NTH, lo * NTH + len(cols) * NTH)
        fp_best[sl] = torch.minimum(fp_best[sl], torch.where(ok, fp, torch.full_like(fp, float("inf"))))
    acc_a, w_a = torch.zeros(NT, 3, device=dev), torch.zeros(NT, device=dev)
    acc_e, w_e = torch.zeros(NT, 3, device=dev), torch.zeros(NT, device=dev)
    acc_l, w_l = torch.zeros(NT, device=dev), torch.zeros(NT, device=dev)
    for n, uc in cams:
        cols, lo = cols_for(uc)
        if len(cols) == 0:
            continue
        view = views_all[n]
        c, fp, z, ok, we = project_view(view, cols, n)
        X, nrm, _ = texel_points(cols)
        R, t = view[:3, :3], view[:3, 3]
        Xc = X @ R.T + t
        L = lamp_L(Xc, nrm @ R.T)
        zc = Xc[:, 2].clamp(min=1e-6)
        g = torch.stack([(ccx + fx * Xc[:, 0] / zc) / (CW - 1) * 2 - 1, (ccy + fy * Xc[:, 1] / zc) / (CH - 1) * 2 - 1], -1)
        E = F.grid_sample(render_ambient(view).permute(2, 0, 1)[None], g[None, None], align_corners=True)[0, :, 0].T
        y = (c - LP["off"][n]) / torch.exp(LP["gain"][n])          # the frame without the model's exposure
        sl = slice(lo * NTH, lo * NTH + len(cols) * NTH)
        wn_ = torch.where(ok, we * torch.exp(-(fp / fp_best[sl] - 1) / a.near_tau), torch.zeros_like(fp))
        if a.source == "delit":                 # the model's albedo itself (its colour before the lamp factor)
            acc_a[sl] += wn_[:, None] * c
            w_a[sl] += wn_
        else:                                   # least squares against the frame
            acc_a[sl] += (wn_ * L)[:, None] * (y - E)
            w_a[sl] += wn_ * L ** 2
        we_ = torch.where(ok, we, torch.zeros_like(fp))
        acc_e[sl] += we_[:, None] * E
        w_e[sl] += we_
        acc_l[sl] += wn_ * L
        w_l[sl] += wn_
    seen_t = w_a > 0
    L_typ = float((acc_l[seen_t] / w_l[seen_t]).median())
    lay = lambda v: v.view(NU, NTH, -1).permute(1, 0, 2).cpu().numpy()  # noqa: E731
    alb = lay(acc_a / w_a.clamp(min=1e-12)[:, None])
    amb = lay(acc_e / w_e.clamp(min=1e-12)[:, None])
    log(f"two-light bake over {len(cams)} training views: {seen_t.float().mean().item():.1%} of the texels seen; "
        f"albedo p50 {np.median(alb[lay(seen_t[:, None])[..., 0]]):.3f}, ambient p50 {np.median(amb[lay(seen_t[:, None])[..., 0]]):.3f}, "
        f"typical lamp factor {L_typ:.2f} (r_ref {R_REF * 1000:.0f} mm) [{time.time() - t_b:.0f} s]")
    return alb, amb, lay(seen_t[:, None])[..., 0], L_typ


@torch.no_grad()
def unwrap_two():
    """--source unwrap: the model's albedo and ambient seen from the centerline (module docstring); returns albedo,
    ambient (rows = theta) and the seen mask"""
    t_b = time.time()
    b_in, b_out = a.unwrap_band_mm[0] / 1000, a.unwrap_band_mm[1] / 1000
    mu = GS["means"]
    keep = torch.zeros(len(mu), dtype=torch.bool, device=dev)
    ug = torch.zeros(len(mu), device=dev)
    for s in range(0, len(mu), 200000):         # every Gaussian's place relative to the (moved) tube
        X = mu[s:s + 200000]
        u, th = to_pipe(X)
        C, N, Bn = cl_frame(u)
        q = X - C
        dr = torch.hypot((q * N).sum(1), (q * Bn).sum(1)) - grid_r(u, th)
        keep[s:s + 200000] = (dr > -b_in) & (dr < b_out) & (u > U0 - 0.02) & (u < UT1 + 0.02)
        ug[s:s + 200000] = u
    uc = torch.tensor([c[1] for c in cams], device=dev)
    _, N, Bn = cl_frame(uc)
    fwd = views_all[torch.tensor([c[0] for c in cams], device=dev)][:, 2, :3]     # the cameras' optical axes (world)
    sgn = 1.0 if float(torch.median((fwd * torch.cross(N, Bn, dim=1)).sum(1))) > 0 else -1.0
    sh = torch.cat([GS["sh0"], GS["shN"]], 1)[keep]
    if a.unwrap_colour == "dc":
        col = (0.5 + 0.28209479177387814 * sh[:, 0]).clamp(min=0)
    else:                                       # toward the centerline point the robot saw the Gaussian from
        vz = a.unwrap_view_z or ref_z
        dirs = F.normalize(mu[keep] - cl_frame(ug[keep] - sgn * vz)[0], dim=1)      # camera -> Gaussian, as gsplat
        col = torch.cat([_spherical_harmonics(3, dirs[s:s + 200000], sh[s:s + 200000])  # gsplat's own reference
                         for s in range(0, len(dirs), 200000)]).add(0.5).clamp(min=0)
    feats = torch.cat([col, AMB[keep]], 1)
    m_, q_, s_, o_ = mu[keep], GS["quats"][keep], torch.exp(GS["scales"][keep]), torch.sigmoid(GS["opacities"][keep])
    del sh, col
    nd, step = a.unwrap_dirs, a.unwrap_step_mm / 1000
    sector = 2 * math.pi / nd
    f_ = R_med / (a.texel_mm / 1000 / a.unwrap_ss)
    r_lo = float(np.percentile(RG, 0.5))
    W_ = 2 * int(math.ceil(f_ * math.tan(sector + math.radians(2)))) + 2
    H_ = 2 * int(math.ceil(f_ * 1.05 * step / (r_lo * math.cos(sector)))) + 2
    Ku = torch.tensor([[f_, 0, W_ / 2], [0, f_, H_ / 2], [0, 0, 1]], device=dev, dtype=torch.float32).repeat(nd, 1, 1)
    us = U0 + np.arange(int(math.ceil((UT1 - U0) / step)) + 1) * step
    th_c = torch.tensor(-math.pi + np.arange(nd) * sector + th0, device=dev, dtype=torch.float32)
    log(f"unwrap: {int(keep.sum()):,} of {len(mu):,} Gaussians within -{a.unwrap_band_mm[0]:.0f}..+{a.unwrap_band_mm[1]:.0f} mm "
        f"of the tube; colour {a.unwrap_colour}" + (f" (robot looks toward {'+' if sgn > 0 else '-'}u)" if a.unwrap_colour == "view" else "")
        + f"; {len(us)} stations x {nd} cameras, {W_} x {H_} px, f {f_:.0f}")
    acc_a, acc_e = torch.zeros(NT, 3, device=dev), torch.zeros(NT, 3, device=dev)
    acc_w, acc_r = torch.zeros(NT, device=dev), torch.zeros(NT, device=dev)
    acc_al, acc_wt = torch.zeros(NT, device=dev), torch.zeros(NT, device=dev)
    n_alpha, n_depth = 0.0, 0.0
    for k, u_s in enumerate(us):
        lo, hi = max(0, int(math.floor((u_s - step - U0) / du))), min(NU, int(math.ceil((u_s + step - U0) / du)))
        if hi <= lo:
            continue
        C, N, Bn = cl_frame(torch.full((nd,), float(u_s), device=dev))
        rad = torch.cos(th_c)[:, None] * N + torch.sin(th_c)[:, None] * Bn
        tan = -torch.sin(th_c)[:, None] * N + torch.cos(th_c)[:, None] * Bn
        Rcv = torch.stack([tan, torch.cross(N, Bn, dim=1), rad], 1)                 # rows: x, y, z of each camera
        vm = torch.eye(4, device=dev).repeat(nd, 1, 1)
        vm[:, :3, :3], vm[:, :3, 3] = Rcv, -(Rcv @ C[:, :, None])[..., 0]
        out, al, _ = gsplat.rasterization(m_, q_, s_, o_, feats[:, :3], vm, Ku, W_, H_, packed=True,
                                          render_mode="RGB+ED", near_plane=0.01, far_plane=100.0)
        oute, _, _ = gsplat.rasterization(m_, q_, s_, o_, feats[:, 3:], vm, Ku, W_, H_, packed=True, render_mode="RGB",
                                          near_plane=0.01, far_plane=100.0)        # (gsplat here: 3 or 4 channels)
        img = torch.cat([out[..., :3] / al.clamp(min=1e-4), oute / al.clamp(min=1e-4), out[..., 3:4], al],
                        -1).permute(0, 3, 1, 2)
        cols = COLS[lo:hi]
        X, _, op = texel_points(cols)
        U = tu[cols][:, None].expand(-1, NTH).reshape(-1)
        TH = tt[None, :].expand(len(cols), -1).reshape(-1)
        wu = (1 - (U - float(u_s)).abs() / step).clamp(min=0)
        jf = (TH + math.pi) / sector
        j0, frac = jf.floor().long() % nd, jf - jf.floor()
        for j in range(nd):
            idx = torch.nonzero(((j0 == j) | ((j0 + 1) % nd == j)) & (wu > 0) & ~op)[:, 0]
            if not len(idx):
                continue
            wt = torch.where(j0[idx] == j, 1 - frac[idx], frac[idx]) * wu[idx]
            Xc = (X[idx] - C[j]) @ Rcv[j].T
            z = Xc[:, 2].clamp(min=1e-6)
            px, py = f_ * Xc[:, 0] / z + W_ / 2, f_ * Xc[:, 1] / z + H_ / 2
            smp = F.grid_sample(img[j:j + 1], torch.stack([px / W_ * 2 - 1, py / H_ * 2 - 1], -1)[None, None],
                                align_corners=False, padding_mode="border")[0, :, 0].T
            inside = (Xc[:, 2] > 0.01) & (px >= 0) & (px <= W_) & (py >= 0) & (py <= H_)
            ok_a = inside & (smp[:, 7] >= a.unwrap_alpha)
            ok = ok_a & ((smp[:, 6] - z).abs() < 0.015 * z + 0.003)
            n_alpha += float((wt * inside * ~ok_a).sum())
            n_depth += float((wt * ok_a * ~ok).sum())
            w_ = wt * ok
            Xh = C[j] + (smp[:, 6:7] * torch.stack([(px - W_ / 2) / f_, (py - H_ / 2) / f_, torch.ones_like(px)], -1)) @ Rcv[j]
            Cq, Nq, Bq = cl_frame(U[idx])
            r_hit = torch.hypot(((Xh - Cq) * Nq).sum(1), ((Xh - Cq) * Bq).sum(1))
            g_idx = lo * NTH + idx
            acc_a.index_add_(0, g_idx, w_[:, None] * smp[:, :3])
            acc_e.index_add_(0, g_idx, w_[:, None] * smp[:, 3:6])
            acc_w.index_add_(0, g_idx, w_)
            acc_r.index_add_(0, g_idx, w_ * r_hit)
            acc_al.index_add_(0, g_idx, wt * inside * smp[:, 7])
            acc_wt.index_add_(0, g_idx, wt * inside)
    seen_t = acc_w > 0
    lay = lambda v: v.view(NU, NTH, -1).permute(1, 0, 2).cpu().numpy()  # noqa: E731
    alb = lay(acc_a / acc_w.clamp(min=1e-12)[:, None])
    amb = lay(acc_e / acc_w.clamp(min=1e-12)[:, None])
    al_m = lay((acc_al / acc_wt.clamp(min=1e-12))[:, None])[..., 0]
    r_m = lay((acc_r / acc_w.clamp(min=1e-12))[:, None])[..., 0]
    seen = lay(seen_t[:, None])[..., 0]
    r_tube = torch.cat([grid_r(tu[COLS[s:s + 200]][:, None].expand(-1, NTH).reshape(-1),     # the tube per texel
                               tt[None, :].expand(len(COLS[s:s + 200]), -1).reshape(-1)) for s in range(0, NU, 200)])
    dr_ =(lay(r_tube[:, None])[..., 0] - r_m)[seen]
    np.savez_compressed(os.path.join(a.out, "unwrap.npz"), alpha=al_m.astype(np.float16), radius=r_m.astype(np.float32),
                        seen=seen, u0=U0, du=du, nth=NTH)
    tot = float(acc_wt.sum())
    log(f"unwrap over {len(us)} stations: {seen.mean():.1%} of the texels seen; weight lost to alpha < {a.unwrap_alpha} "
        f"{n_alpha / max(tot, 1e-9):.1%}, to depth off the tube {n_depth / max(tot, 1e-9):.1%}; alpha p5 / p50 "
        f"{np.percentile(al_m[seen], 5):.2f} / {np.median(al_m[seen]):.2f}; rendered surface inside the tube by median "
        f"{np.median(dr_) * 1000:+.2f} mm (|.| p95 {np.percentile(np.abs(dr_), 95) * 1000:.2f} mm); albedo p50 "
        f"{np.median(alb[seen]):.3f}, ambient p50 {np.median(amb[seen]):.3f} [{time.time() - t_b:.0f} s]")
    return alb, amb, seen


def fill(img, m):
    """push-pull fill of the texels not in m (rows = theta, wrapping if the tube is closed)"""
    out = img.copy()
    if m.all():
        return out
    levels, cur, cm = [], img * m[..., None], m.astype(np.float32)
    while min(cur.shape[:2]) > 4:
        levels.append((cur, cm))
        cur = cv2.resize(cur, (max(1, cur.shape[1] // 2), max(1, cur.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        cm = cv2.resize(cm, (max(1, cm.shape[1] // 2), max(1, cm.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    est = np.where(cm[..., None] > 0, cur / np.maximum(cm[..., None], 1e-9), cur.mean((0, 1)))
    for cur_l, cm_l in reversed(levels):
        up = cv2.resize(est, (cur_l.shape[1], cur_l.shape[0]), interpolation=cv2.INTER_LINEAR)
        val = cur_l / np.maximum(cm_l[..., None], 1e-9)
        wgt = np.clip(cm_l, 0, 1)[..., None]
        est = wgt * np.where(cm_l[..., None] > 0, val, up) + (1 - wgt) * up
    return np.where(m[..., None], img, est)


def blur(img, s):
    pad = int(3 * s) + 1
    if closed:
        p = np.concatenate([img[-pad:], img, img[:pad]], 0)
        return cv2.GaussianBlur(p, (0, 0), s, borderType=cv2.BORDER_REFLECT)[pad:-pad]
    return cv2.GaussianBlur(img, (0, 0), s, borderType=cv2.BORDER_REFLECT)


TEXA, TEXE = [None], [None]                             # --two-light: albedo and ambient as drawn (the PNGs)
extra = {}
if a.two_light:
    alb, amb, seen, L_typ = bake_two()
    if a.source == "unwrap":                    # the video bake above gives the typical lamp factor of texture.png
        alb, amb, seen = unwrap_two()
    alb_f, amb_f = np.clip(fill(alb, seen), 0, None), np.clip(fill(amb, seen), 0, None)
    sA = 250.0 / max(float(np.percentile(alb_f, 99.5)), 1e-6)
    sE = 250.0 / max(float(np.percentile(amb_f, 99.5)), 1e-6)
    pngA = np.clip(alb_f * sA, 0, 255).round().astype(np.uint8)
    pngE = np.clip(amb_f * sE, 0, 255).round().astype(np.uint8)
    TEXA[0] = torch.from_numpy(pngA.astype(np.float32) / sA).to(dev)
    TEXE[0] = torch.from_numpy(pngE.astype(np.float32) / sE).to(dev)
    tex = pngA.astype(np.float32) / sA * L_typ + pngE.astype(np.float32) / sE      # the static look
    scale = 250.0 / max(float(np.percentile(tex, 99.5)), 1e-6)
    png = np.clip(tex * scale, 0, 255).round().astype(np.uint8)
    TEXc[0] = torch.from_numpy(png.astype(np.float32) / 255).to(dev)
    cv2.imwrite(os.path.join(a.out, "obj", "albedo.png"), pngA[:, :, ::-1])
    cv2.imwrite(os.path.join(a.out, "obj", "ambient.png"), pngE[:, :, ::-1])
    np.savez(os.path.join(a.out, "obj", "lamp.npz"), logM=LP["logM"][0, 0].cpu().numpy(), t=LP["t"].cpu().numpy(),
             r_ref=R_REF, K=Kc.cpu().numpy(), size=np.array([CW, CH]))
    extra = dict(two_light=dict(albedo_scale=sA, ambient_scale=sE, L_typ=L_typ, r_ref=R_REF,
                                note="colour = L x albedo.png / albedo_scale + ambient.png / ambient_scale, L = M x cos / "
                                     "(r / r_ref)^2 with M (camera-fixed, logM on the canvas grid, bilinear with "
                                     "align_corners) and the lamp offset t (camera frame) of lamp.npz; texture.png = "
                                     "L_typ x albedo + ambient (scale)"))
else:
    s_lp = a.lp_mm / a.texel_mm
    for it_f in range(a.flat_iters + 1):
        near, ref, seen = bake_once()
        near_f, ref_f = fill(near, seen), fill(ref, seen)
        tex = near_f * blur(ref_f, s_lp) / np.maximum(blur(near_f, s_lp), 1e-4) if a.lp_mm > 0 else near_f
        tex = np.clip(tex, 0, None)
        scale = 250.0 / max(float(np.percentile(tex, 99.5)), 1e-6)
        png = np.clip(tex * scale, 0, 255).round().astype(np.uint8)
        TEXc[0] = torch.from_numpy(png.astype(np.float32) / 255).to(dev)    # what is drawn and scored = the PNG
        if it_f < a.flat_iters and a.source == "video":
            LMf = learn_lamp()                      # the frames' camera-fixed shading against this texture
            FLAT[0] = torch.from_numpy((LMf / np.median(LMf)).astype(np.float32)).to(dev)
            log(f"flat field {it_f + 1}: camera-fixed gain p5..p95 {np.percentile(LMf / np.median(LMf), 5):.2f}.."
                f"{np.percentile(LMf / np.median(LMf), 95):.2f}; the frames are divided by it and baked again")
        elif it_f < a.flat_iters:
            break
cv2.imwrite(os.path.join(a.out, "obj", "texture.png"), png[:, :, ::-1])
json.dump(dict(u0=U0, u1=UT1, du=du, nth=NTH, dth=dth, scale=scale, seen=float(seen.mean()), closed=bool(closed),
               note="texture.png: column c covers u = u0 + c du, row r covers theta = -pi + r dth (the centerline's "
                    "frame, theta as vision_surface.py)", run=a.run, **extra),
          open(os.path.join(a.out, "texture.json"), "w"), indent=1)
log(f"texture {png.shape[1]} x {png.shape[0]}: seen {seen.mean():.1%} of the texels (rest filled), scale {scale:.1f}")

# ------------------------------------------------------------------------------ OBJ ---------------------------
vu_ = (g_u0 + (mz["vi"] + 0.5) * g_du - U0) / (UT1 - U0)
vth = (-math.pi + (mz["vj"] + 0.5) * g_dth)
vv = (vth + math.pi) / (2 * math.pi)
vt = np.c_[vu_, 1 - vv]                                # OBJ: v = 0 at the bottom of the image
vt_idx = tri.copy()
wrap = (vth[tri].max(1) - vth[tri].min(1)) > math.pi   # triangles across the theta seam (closed tube)
if wrap.any():
    extra = []
    for k in np.flatnonzero(wrap):
        for c in range(3):
            if vth[tri[k, c]] < 0:
                extra.append(tri[k, c])
                vt_idx[k, c] = len(vt) + len(extra) - 1
    ex = np.array(extra)
    vt = np.r_[vt, np.c_[vu_[ex], 1 - (vv[ex] + 1)]]
with open(os.path.join(a.out, "obj", "model.mtl"), "w", newline="\n") as fh:
    fh.write("newmtl wall\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd texture.png\n")
with open(os.path.join(a.out, "obj", "model.obj"), "w", newline="\n") as fh:
    fh.write(f"# bake_vhf.py: {len(xyz)} vertices, {len(tri)} faces, metres (scene frame)\nmtllib model.mtl\n")
    fh.write("".join(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in xyz))
    fh.write("".join(f"vt {s:.6f} {t:.6f}\n" for s, t in vt))
    fh.write("usemtl wall\n")
    fh.write("".join(f"f {p + 1}/{q + 1} {r + 1}/{s + 1} {t + 1}/{v + 1}\n"
                     for (p, r, t), (q, s, v) in zip(tri, vt_idx)))
log(f"OBJ {len(xyz):,} vertices, {len(tri):,} faces ({wrap.sum():,} across the seam) -> {a.out}/obj")

# ------------------------------------------------------------------------------ scoring (ruler v2) ------------
LM = learn_lamp()
np.save(os.path.join(a.out, "lamp_map.npy"), np.log(LM).astype(np.float32))
log(f"lamp map from {len(train[::4])} training frames: gain p5..p95 {np.percentile(LM, 5):.2f}..{np.percentile(LM, 95):.2f}")


def score(tag, imgs, nxt, views, B, Bm, ids, pos, render, sub):
    rows = []
    for k in range(len(ids)):
        G, ok = render(torch.from_numpy(views[k]).to(dev))
        real = imgs[k].astype(np.float32)
        m = Bm[k] & valid_raw
        row = dict(tag=tag, frame=int(ids[k]), pos=float(pos[k]), coverage=float(m.mean()), cov_G=float(ok.mean()),
                   cov_B=float(Bm[k].mean()))
        nal = im.linear_match(im.align_to(nxt[k].astype(np.float32), real), real, m)
        den = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(imgs[k]), None, 5, 5, 7, 21).astype(np.float32)
        mats = {}
        for name, x0 in (("G", G), ("B", B[k].astype(np.float32)), ("Nal", nal), ("D", den)):
            x = im.linear_match(x0, real, m)
            mats[name] = x
            for key, val in im.all_metrics(x, real, m).items():
                row[f"{name}_{key}"] = val
            for key, val in im.ruler2(x, real, nal, m).items():
                row[f"{name}_{key}"] = val
        rows.append(row)
        if a.save and k % a.save == 0:
            trip = np.hstack([real, mats["B"], mats["G"]]).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(a.out, "renders", f"{tag}{sub}_{int(ids[k]):06d}.jpg"), trip[:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        log(f"  {tag} {k + 1}/{len(ids)} frame {int(ids[k])}: PSNR mesh {row['G_psnr']:.2f} B {row['B_psnr']:.2f} "
            f"Nal {row['Nal_psnr']:.2f} | fid fine mesh {row['G_fid_fine']:.3f} B {row['B_fid_fine']:.3f}")
    return rows


def evaluate(render, suffix, label):
    """held-out and block frames scored with render (view -> image, coverage): per_frame_r2<suffix>.csv, summary_r2<suffix>.json"""
    rng = d["block_ranges"]
    in_block = lambda p: bool(np.any([(p >= lo) & (p < hi) for lo, hi in rng]))  # noqa: E731
    rows = score("held", d["held_images"], d["held_next"], d["held_viewmats"], d["held_B"], d["held_mask"], d["held_ids"],
                 d["held_pos"], render, suffix)
    for r in rows:
        r["in_block"] = in_block(r["pos"])
    bi = d["block_idx"]
    rows_b = score("block", raw_train[bi], d["block_next"], d["block_viewmats"], d["block_B"], d["block_mask"],
                   d["block_ids"], d["train_pos"][bi], render, suffix)
    for r in rows_b:
        r["in_block"] = True
    keys = [k for k in rows[0] if k not in ("tag", "frame", "in_block")]
    with open(os.path.join(a.out, f"per_frame_r2{suffix}.csv"), "w", newline="", encoding="utf-8") as fh:
        wr_ = csv.DictWriter(fh, ["tag", "frame", "in_block"] + keys)
        wr_.writeheader()
        wr_.writerows(rows + rows_b)
    med = lambda rs: {k: float(np.nanmedian([r[k] for r in rs])) for k in keys} if rs else {}  # noqa: E731
    summary = dict(scene=a.scene, args=vars(a), model=f"{label} ({a.name}) from {a.run}", texture=[NTH, NU],
                   held_all=med(rows), held_outside_blocks=med([r for r in rows if not r["in_block"]]), block=med(rows_b))
    json.dump(summary, open(os.path.join(a.out, f"summary_r2{suffix}.json"), "w"), indent=1)
    M = ("psnr", "lpips", "shift_px", "fid_fine", "fid_mid", "fid_coarse", "en_fine", "mse_low")
    for title, s in (("in-sequence held-out, outside the blocks", summary["held_outside_blocks"]), ("block frames", summary["block"])):
        log(f"\n{label}: {title} (median, ruler v2):")
        for tag, name in (("G", "baked mesh"), ("B", "lampmap B"), ("Nal", "next frame aligned"), ("D", "frame denoised")):
            log(f"  {name:18s} " + "  ".join(f"{k} {s[f'{tag}_{k}']:.3f}" for k in M))


def render_static(view):
    G, ok = render_mesh(view)
    return G * LM[..., None], ok


if a.two_light:
    evaluate(render_mesh_two, "", "baked mesh with the model's headlight (L x albedo + ambient)")
    evaluate(render_static, "_static", "baked mesh, static texture x camera-fixed lamp map")
    LM2 = learn_lamp(render_mesh_two)           # what the headlight misses and stays on the screen: the lamp's highlight
    np.save(os.path.join(a.out, "lamp_map_headlight.npy"), np.log(LM2).astype(np.float32))
    log(f"camera-fixed map on top of the headlight: gain p5..p95 {np.percentile(LM2, 5):.2f}..{np.percentile(LM2, 95):.2f}")
    evaluate(lambda v: (lambda g: (g[0] * LM2[..., None], g[1]))(render_mesh_two(v)), "_lm",
             "baked mesh with the headlight x camera-fixed lamp map")
else:
    evaluate(render_static, "", "baked mesh")
log("done")
