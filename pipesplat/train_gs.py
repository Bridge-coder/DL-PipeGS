"""Gaussian model of a pipe-inspection video: training and scoring (runs in the pipe3d WSL distro, gsplat on the GPU).

The main method (docs/MODEL.md) is 3D Gaussian Splatting with two plug-ins:

  train_gs.py SCENE_DIR OUT_DIR --baseline 3dgs --baseline-depth 1.0 --lamp --lamp-smooth-mm 20 --lamp-ambient
          [--surface surface_vhf.npz] [--points-every 2]

  loss      L = 0.8 L1(I_hat, I) + 0.2 (1 - SSIM(I_hat, I)) + 1.0 mean_u |D_hat(u) - D_s(u)| / D_s(u)
  image     I_hat_t(u) = g_t * [ L_t(u) C_hat(u) + A_hat(u) ] + b_t
  lamp      L_t(u) = M(u) cos(theta_t(u)) / (r_t(u) / r_ref)^2
C_hat, D_hat = colour (3rd-order spherical harmonics) and depth rendered by 3DGS; D_s = depth of the measured wall
(SCENE_DIR/surface.npz from export_surface.py, or surface_vhf.npz from vision_surface.py) rendered at the same pose;
r_t, theta_t = distance and angle of incidence from the lamp to the wall (from D_s and wall normals averaged over
--lamp-smooth-mm); M = learned camera-fixed 48 x 36 map (vignetting x lamp pattern), started so that M cos / r^2 is
1 on average at every pixel; lamp offset from the camera centre learned from 0; A_hat = a second, view-independent
colour per Gaussian that the lamp does not scale (ambient term); g_t, b_t = learned gain and offset per frame and
colour channel (exposure). 30 000 iterations, one training frame per iteration, Adam, gsplat's DefaultStrategy
(clone / split / prune), start from the OpenMVS dense cloud, poses fixed (SfM).

Options (one change at a time; these are the comparison methods and ablations of docs/EVALUATION.md)
  method      --baseline 3dgs   3D Gaussian Splatting (Kerbl et al. 2023): adaptive density control (gsplat
                                DefaultStrategy), start scale = RMS distance to the 3 nearest points, opacity 0.1,
                                SH degree raised every 1000 iterations
              --baseline 2dgs   2D Gaussian Splatting (Huang et al. 2024): surfels, normal consistency 0.05 from
                                --normal-start, depth distortion 0.01 from --dist-start (gsplat simple_trainer_2dgs)
              --anchor          anchored surfels (B1'): starting points snapped onto the measured wall (dropped if
                                farther than --anchor-snap-mm), thin discs in the wall (rotation = wall normal,
                                thickness <= --thin-mm), centres within +-(--anchor-clamp-mm) along the normal, dense
                                depth term with weight --anchor-depth-w; --alpha-w adds mean(1 - alpha) on wall pixels
  B1          --baseline-depth W   dense depth supervision on a baseline: the relative L1 to D_s above with weight W
  B2 + B4     --lamp            the lamp factor L_t and the per-frame gain / offset (needs --anchor or --baseline-depth)
  B3          --lamp-ambient    the ambient colour A_hat (extends --lamp)
  B4 alone    --exposure        only the per-frame gain / offset, no lamp factor (not with --lamp)
  lamp model variants (identifiability analysis, docs/EVALUATION.md "V1-V5"; all extend --lamp)
              --sh-degree 0     view-independent colour (Lambertian albedo)
              --lamp-radial N   M as a radial polynomial log M = sum c_k rho^(2k) (Goldman 2010) instead of the grid
              --lamp-colocated  lamp offset fixed at 0 (LEDs ring the lens)
              --lamp-gamma / --lamp-gamma-fixed P   power-law camera response lit^p, learned / fixed
              --lamp-minnaert   Minnaert reflectance cos^(2k-1) instead of Lambert's cosine
  data        --points-every N  start from every Nth dense point (GPU memory)   --surface FILE  (default surface.npz)
              --no-blocks       also train on the block frames (the block test is then meaningless)
  scoring     --eval-only       load OUT_DIR/model.pt and only score it (with the model options it was trained with;
                                pipesplat/rescore.py reads them from the run's summary)
              --extra-test NAME=FILE ...   other test sets (export_xrun.py: frames of other inspection runs)
              --no-held         only the --extra-test sets    --fit-half  gain/offset fitted on the left half, scored
                                on the right half (NeRF-W protocol)   --test-pose-iters N  register each test frame's
                                pose to the frozen model first (optical flow; a check, never mixed with the strict score)
              --dump DIR        write every scored frame (render, video frame, B) as JPEGs
              --suffix S        output names summary<S>.json / per_frame<S>.csv (default _r2 after training, _eval
                                with --eval-only)
Scoring ("ruler v2", docs/EVALUATION.md): on every held-out frame the render, baseline B (OpenMVS textured mesh x lamp
map), the next video frame aligned by optical flow (Nal) and the frame itself denoised (D) get each its own gain and
offset fitted (imgmetrics.linear_match), then PSNR / SSIM / LPIPS / pixel shift / band fidelity on the same pixels
(B's mask) for every model; depth error of the render against the dense cloud (and the wall). Test groups: 44
in-sequence held-out frames (interpolated poses) and every frame in two 10 cm blocks left out of training.

usage: TORCH_HOME=<TORCH_CACHE of scripts/paths.py> /opt/gsenv/bin/python pipesplat/train_gs.py SCENE_DIR OUT_DIR [options]
       SCENE_DIR  export_g0.py output (scene.npz) + the measured wall (surface*.npz)
       OUT_DIR    train.log, model.pt, summary_r2.json (medians per test group), per_frame_r2.csv,
                  renders/*.jpg (video | B | model)
The k1 fold: with k1 < 0, x_d = x_u (1 + k1 r^2) turns back beyond r_u = 1/sqrt(-3 k1); the canvas stays inside 95 %.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gsplat
from gsplat.strategy import DefaultStrategy
from fused_ssim import fused_ssim
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lampmap"))
import imgmetrics as im  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("scene"), ap.add_argument("out")
ap.add_argument("--iters", type=int, default=30000)
ap.add_argument("--no-blocks", action="store_true", help="train on the block frames too (block test then meaningless)")
ap.add_argument("--save", type=int, default=4, help="scoring: every Nth frame as a triptych in renders/ (0 = none)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dssim", type=float, default=0.2)
ap.add_argument("--eval-only", action="store_true", help="load OUT_DIR/model.pt and only score it")
ap.add_argument("--alpha-min", type=float, default=0.5,
                help="scoring: opacity above which a pixel counts as covered by the model (coverage and depth-error "
                     "columns only; PSNR / LPIPS / bands are scored on B's mask)")
ap.add_argument("--suffix", default=None, help="summary<S>.json / per_frame<S>.csv (default _r2, _eval with --eval-only)")
ap.add_argument("--ruler2", action="store_true", help="accepted for old command lines: scoring is always ruler v2")
ap.add_argument("--smoke", action="store_true", help="code check: train, save, no scoring")
ap.add_argument("--test-pose-iters", type=int, default=0,
                help="scoring only: before scoring, register each scored frame's own pose (6 numbers, model frozen) "
                     "to that frame by optical flow, this many rounds (3) - separates 'the model is wrong' from 'the "
                     "test frame's pose is wrong'. Use with its own --suffix, never mixed with the strict score")
ap.add_argument("--anchor", action="store_true", help="hold the Gaussians on the measured surface (anchored surfels)")
ap.add_argument("--points-every", type=int, default=1,
                help="start from every N-th geometry point (memory: lab A's dense cloud has 4.1 M points)")
ap.add_argument("--surface", default="surface.npz", help="measured-surface file in SCENE_DIR (export_surface.py)")
ap.add_argument("--anchor-depth-w", type=float, default=1.0)
ap.add_argument("--anchor-clamp-mm", type=float, default=2.0)
ap.add_argument("--anchor-snap-mm", type=float, default=10.0)
ap.add_argument("--thin-mm", type=float, default=0.05)
ap.add_argument("--lamp", action="store_true", help="near-field lamp model (needs --anchor or --baseline-depth)")
ap.add_argument("--lamp-ambient", action="store_true",
                help="add light that does not come from the lamp (room light through a slot / the pipe ends, light "
                     "reflected between the walls): per Gaussian a view-independent colour added after the lamp factor")
ap.add_argument("--lamp-smooth-mm", type=float, default=20.0,
                help="surface normals for the lamp's incidence angle averaged over about this radius")
ap.add_argument("--baseline", choices=("3dgs", "2dgs"), default=None,
                help="published methods on the same frames, canvas and scoring: "
                     "3dgs = 3D Gaussian Splatting (Kerbl et al. 2023): ADC (gsplat DefaultStrategy), start scale = RMS "
                     "distance to the 3 nearest points, opacity 0.1, SH degree raised every 1000 iterations, no depth; "
                     "2dgs = 2D Gaussian Splatting (Huang et al. 2024): surfels (rasterization_2dgs), ADC on its own "
                     "gradient, normal consistency 0.05 from 7k, distortion 0.01 from 3k (gsplat's simple_trainer_2dgs "
                     "defaults). Both start from the same dense MVS cloud as the anchored model (a stronger start than "
                     "the sparse SfM points of the papers)")
ap.add_argument("--normal-start", type=int, default=7000, help="--baseline 2dgs: normal consistency from this iteration")
ap.add_argument("--dist-start", type=int, default=3000, help="--baseline 2dgs: depth distortion from this iteration")
ap.add_argument("--alpha-w", type=float, default=0.0,
                help="with --anchor: weight of a coverage term mean(1 - alpha) over the pixels where the measured surface "
                     "is. The pipe wall is opaque; with --lamp the near wall otherwise turns semi-transparent, i.e. the "
                     "model darkens it through transparency instead of colour")
ap.add_argument("--dump", default=None,
                help="scoring: write every scored frame as JPEGs into this folder - the model's render (gain/offset "
                     "matched to the frame, as scored), and once per frame the video frame and B (for comparison "
                     "videos, review/compare_video.py)")
ap.add_argument("--extra-test", nargs="+", default=[], metavar="NAME=FILE",
                help="scoring: these test sets too, e.g. xrun_r2=xrun_r2.npz (in SCENE_DIR, from export_xrun.py): "
                     "frames of another inspection run of the same pipe, posed by a joint SfM of both runs")
ap.add_argument("--fit-half", action="store_true",
                help="scoring: fit each frame's gain and offset on the left half of the scored pixels and measure on the "
                     "right half (the NeRF-W evaluation protocol, Martin-Brualla et al. 2021) instead of on the same pixels")
ap.add_argument("--no-held", action="store_true", help="scoring: only the --extra-test sets, not the in-sequence / block frames")
ap.add_argument("--baseline-depth", type=float, default=0.0,
                help="with --baseline: add the same dense depth supervision as --anchor (relative L1 between the "
                     "rendered depth and the measured surface's depth over every pixel it covers) with this weight "
                     "(1.0 = --anchor-depth-w): depth-supervised 3DGS / 2DGS")
ap.add_argument("--sh-degree", type=int, default=3,
                help="spherical-harmonic degree of the Gaussian colours (not --baseline); 0 = view-independent colour, "
                     "i.e. a Lambertian albedo under --lamp: the inverse-square and cosine falloff can then not be "
                     "absorbed by view-dependent colour terms")
ap.add_argument("--lamp-radial", type=int, default=0,
                help="with --lamp: the camera-fixed map M as a radially symmetric polynomial, log M = sum_k c_k rho^(2k), "
                     "k = 1..N, rho = distance from the principal point in normalised image coordinates (the polynomial "
                     "vignetting model of Goldman 2010; an LED ring round the lens has a radially symmetric intensity "
                     "pattern as well), instead of the free 48 x 36 grid. With the camera at a fixed pose in a straight pipe, "
                     "every image direction sees the wall at one angle only, so a free map can trade brightness with the "
                     "albedo of each angle; a radial map cannot")
ap.add_argument("--lamp-colocated", action="store_true",
                help="with --lamp: the light source is co-located with the camera centre (offset fixed at 0, not "
                     "learned) - the Minicam's LEDs ring the lens. A free offset ran off to 16 cm ahead of the camera when "
                     "the falloff did not fit")
ap.add_argument("--lamp-minnaert", action="store_true",
                help="with --lamp: Minnaert reflectance (Minnaert 1941) instead of Lambert's: with the light at the camera "
                     "the angles of incidence and of view are equal, I ~ cos^k(i) cos^(k-1)(e) = cos^(2k-1)(i), the exponent "
                     "2k-1 learned (start 1 = Lambert). Rough surfaces lit and seen from the same direction look brighter at "
                     "grazing angles than a Lambertian one")
ap.add_argument("--lamp-gamma-fixed", type=float, default=0.0,
                help="with --lamp: a fixed power-law response lit^P instead of a learned one (0.45 = the exponent of the "
                     "ITU-R BT.709 transfer characteristic)")
ap.add_argument("--lamp-gamma", action="store_true",
                help="with --lamp: a learned gamma (power-law camera response) on the lit colour, I = gain x lit^p + "
                     "offset with p learned (start 1): the video is gamma-encoded, so the physical inverse-square law in "
                     "radiance appears with another exponent in the recorded values")
ap.add_argument("--exposure", action="store_true",
                help="per-frame exposure compensation alone: a gain and an offset per frame and colour channel on the "
                     "render, I = exp(g_t) x C + b_t, learned with the model - the per-frame gain / offset of --lamp "
                     "(same parameters, start and learning rates) without the lamp factor, i.e. the ablation that "
                     "separates the two (B4). An affine brightness transfer per image as in DSO (Engel et al. "
                     "2018); Hierarchical 3DGS (Kerbl et al. 2024) learns a full 3 x 4 affine colour transform per image")
a = ap.parse_args()
assert a.baseline or a.anchor, "choose the representation: --baseline 3dgs|2dgs or --anchor (main method: see the docstring)"
assert not (a.baseline and a.anchor), "--baseline is a method of its own (not with --anchor)"
assert not a.lamp or not a.baseline or (a.baseline == "3dgs" and a.baseline_depth > 0), \
    "--lamp on a baseline: 3dgs with --baseline-depth"
if a.baseline == "2dgs":
    # this gsplat build (1.6.0 @512d366) hands the 2DGS projection backward a non-contiguous depth gradient
    # ("grad.depths must be contiguous") for every depth render mode; make its inputs contiguous
    import gsplat.cuda._wrapper as _gw
    _lazy = _gw._make_lazy_cuda_func

    def _lazy_contig(name):
        f = _lazy(name)
        if name != "projection_2dgs_fused_bwd":
            return f
        return lambda *args: f(*[x.contiguous() if torch.is_tensor(x) else x for x in args])
    _gw._make_lazy_cuda_func = _lazy_contig
assert not a.lamp or a.anchor or a.baseline_depth > 0, "--lamp uses the measured-surface depth (--anchor / --baseline-depth)"
assert not a.lamp_ambient or a.lamp, "--lamp-ambient extends --lamp"
assert not a.lamp_gamma or a.lamp, "--lamp-gamma extends --lamp"
assert not a.lamp_radial or a.lamp, "--lamp-radial extends --lamp"
assert not a.lamp_colocated or a.lamp, "--lamp-colocated extends --lamp"
assert not (a.lamp_minnaert or a.lamp_gamma_fixed) or a.lamp, "--lamp-minnaert / --lamp-gamma-fixed extend --lamp"
assert not (a.lamp_gamma and a.lamp_gamma_fixed), "--lamp-gamma learns the exponent, --lamp-gamma-fixed fixes it"
assert not (a.exposure and a.lamp), "--lamp has its own per-frame gain and offset"
assert a.sh_degree == 3 or not a.baseline, "--sh-degree is for the anchored models (the baselines raise it to 3)"
ADC = a.baseline is not None                     # the baselines densify (gsplat DefaultStrategy); --anchor does not
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = "cuda"
os.makedirs(os.path.join(a.out, "renders"), exist_ok=True)
T0 = time.time()
_logf = open(os.path.join(a.out, "train.log"), "a", encoding="utf-8")


def log(msg):
    line = f"{msg} [{time.time() - T0:.0f} s]"
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


log(f"args {vars(a)} | torch {torch.__version__} gsplat {gsplat.__version__}")
d = np.load(os.path.join(a.scene, "scene.npz"))
if "depth_scan" in d.files:
    sys.exit("this scene holds LiDAR scans (the early InPipe LiDAR export); it is no longer supported - see the tag "
             "v2.1-results")
K = d["K"].astype(np.float64)
k1 = float(d["k1"])
w, h = (int(v) for v in d["size"])
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]


# ------------------------------------------------------------------------------ camera canvas ---------------
def undistort_grid(xd, yd, k1):
    """inverse of x_d = x_u (1 + k1 r_u^2) by iteration (render_views.undistort_grid, copied: that module needs the
    Windows paths); NaN beyond 95 % of the radius the model reaches"""
    out = (xd ** 2 + yd ** 2 >= (0.95 * 2 / (3 * np.sqrt(-3 * k1))) ** 2) if k1 < 0 else np.zeros(xd.shape, bool)
    xd, yd = np.where(out, 0.0, xd), np.where(out, 0.0, yd)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(20):
        g = 1 + k1 * (xu ** 2 + yu ** 2)
        xu, yu = xd / g, yd / g
    xu[out], yu[out] = np.nan, np.nan
    return xu, yu


if k1 == 0.0:
    CW, CH, ccx, ccy = w, h, cx, cy
    valid_raw = np.ones((h, w), bool)
    to_canvas = to_raw = None
    valid_canvas = np.ones((h, w), bool)
    RXY = np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy)
else:                                           # an undistorted pinhole canvas holding the whole raw frame
    XU, YU = undistort_grid(*np.meshgrid((np.arange(w) - cx) / fx, (np.arange(h) - cy) / fy), k1)
    valid_raw = np.isfinite(XU)
    CW, CH = 2 * int(np.ceil(np.nanmax(np.abs(XU)) * fx)) + 4, 2 * int(np.ceil(np.nanmax(np.abs(YU)) * fy)) + 4
    ccx, ccy = CW / 2 - 0.5, CH / 2 - 0.5
    to_raw = (np.nan_to_num(ccx + XU * fx, nan=-10).astype(np.float32), np.nan_to_num(ccy + YU * fy, nan=-10).astype(np.float32))
    xu, yu = np.meshgrid((np.arange(CW) - ccx) / fx, (np.arange(CH) - ccy) / fy)
    g = 1 + k1 * (xu ** 2 + yu ** 2)
    ud, vd = (cx + fx * xu * g).astype(np.float32), (cy + fy * yu * g).astype(np.float32)
    valid_canvas = ((xu ** 2 + yu ** 2) < (0.95 / np.sqrt(-3 * k1)) ** 2) & (ud >= 0) & (ud <= w - 1) & (vd >= 0) & (vd <= h - 1)
    to_canvas = (ud, vd)
    RXY = (xu, yu)
Kc = torch.tensor([[fx, 0, ccx + 0.5], [0, fy, ccy + 0.5], [0, 0, 1]], device=dev, dtype=torch.float32)  # gsplat: centres at +0.5
VALID = torch.from_numpy(valid_canvas).to(dev)
log(f"raw {w}x{h}, k1 {k1}; training canvas {CW}x{CH} (pinhole), valid {valid_canvas.mean():.1%} of it")
RX, RY = (torch.from_numpy(np.ascontiguousarray(v, np.float32)).to(dev) for v in RXY)  # ray x/z, y/z per pixel


def project(c):
    """pixel (column, row) on the training canvas of camera-space points c, and whether they land inside"""
    z = c[:, 2].clamp(min=1e-6)
    xn, yn = c[:, 0] / z, c[:, 1] / z
    ok = c[:, 2] > 0.01
    return torch.round(ccx + fx * xn).long(), torch.round(ccy + fy * yn).long(), ok


def canvas(img):
    return img if to_canvas is None else cv2.remap(img, to_canvas[0], to_canvas[1], cv2.INTER_LINEAR, borderValue=0)


# ------------------------------------------------------------------------------ frames, depth ---------------
views_all = torch.from_numpy(d["train_viewmats"]).to(dev)
eyes = d["train_eyes"]
block_idx = d["block_idx"]
train = np.setdiff1d(np.arange(len(views_all)), [] if a.no_blocks else block_idx)
if "train_drop" in d.files:                     # export_multi.py --only-added: the reference run is test-only
    train = np.setdiff1d(train, d["train_drop"])
raw_train = d["train_images"]
IM = {int(i): torch.from_numpy(canvas(raw_train[i])) for i in train}
if sum(v.numel() for v in IM.values()) < 1.0e9:  # pinned for fast upload; WSL refuses ~1.2 GB of pinned memory
    IM = {k: v.pin_memory() for k, v in IM.items()}   # (Pipe1), and the failed CUDA call breaks the next one
else:
    log("training frames not pinned (more than 1 GB)")
log(f"{len(train)} training frames (left out: {0 if a.no_blocks else len(block_idx)} block frames in "
    f"{d['block_ranges'].round(3).tolist()} m)")

DP = torch.from_numpy(d["depth_points"]).to(dev)


def depth_map(view):
    """depth of the dense MVS cloud on the canvas (0 = no point), nearest point per pixel: the reference of the
    scored depth error and of the consistency check of the measured surface"""
    c = DP @ view[:3, :3].T + view[:3, 3]
    i, j, k = project(c)
    z = c[:, 2]
    k &= (i >= 0) & (i < CW) & (j >= 0) & (j < CH)
    D = torch.full((CH * CW,), float("inf"), device=dev)
    D.scatter_reduce_(0, j[k] * CW + i[k], z[k], reduce="amin")
    D[torch.isinf(D)] = 0
    return D.view(CH, CW)


def quat_z_to(n):
    """unit quaternions (wxyz) rotating the z axis onto the unit vectors n"""
    z = torch.tensor([0.0, 0.0, 1.0], device=n.device)
    axis = torch.cross(z.expand_as(n), n, dim=1)
    s = axis.norm(dim=1, keepdim=True)
    c = n[:, 2:3].clamp(-1, 1)
    ang = torch.atan2(s, c)
    axis = torch.where(s > 1e-8, axis / s.clamp(min=1e-12), torch.tensor([1.0, 0, 0], device=n.device).expand_as(n))
    return torch.cat([torch.cos(ang / 2), axis * torch.sin(ang / 2)], dim=1)


if a.anchor or a.baseline_depth > 0:            # the measured wall: one opaque disc per surface triangle
    sf = np.load(os.path.join(a.scene, a.surface))
    S_c = torch.from_numpy(sf["centres"]).to(dev)
    S_n = torch.from_numpy(sf["normals"]).to(dev)
    S_q = quat_z_to(S_n)
    S_s = torch.log(torch.stack([0.6 * torch.from_numpy(sf["size"]).to(dev)] * 2 + [torch.full((len(S_c),), 1e-5, device=dev)], 1))
    S_o = torch.full((len(S_c),), 0.999, device=dev)
    S_col = torch.zeros(len(S_c), 3, device=dev)

    @torch.no_grad()
    def surface_depth(view):
        """depth of the measured surface on the canvas (0 where it does not cover the pixel)"""
        out, al, _ = gsplat.rasterization(S_c, S_q, torch.exp(S_s), S_o, S_col, view[None], Kc[None], CW, CH,
                                          packed=False, render_mode="ED", near_plane=0.01)
        dep = out[0, ..., 0]
        return torch.where(al[0, ..., 0] > 0.5, dep, torch.zeros_like(dep))

    t_s = time.time()
    DA = {int(n): surface_depth(views_all[n]).half() for n in train}      # D_s of every training frame
    chk = []
    for n in list(train)[:: max(1, len(train) // 20)]:
        Da, Ds = DA[int(n)].float(), depth_map(views_all[n])
        m = (Da > 0) & (Ds > 0) & VALID
        chk.append(((Da - Ds).abs() / Ds)[m].median().item())
    log(f"measured surface: {len(S_c):,} triangles; dense depth for {len(DA)} training frames in {time.time() - t_s:.0f} s; "
        f"median relative difference to the dense-cloud depth {np.median(chk):.2%} (consistency check)")

if a.lamp:
    GY, GX = 36, 48
    r_ref = float(np.median([DA[int(n)][DA[int(n)] > 0].float().median().item() for n in list(train)[::10]]))
    lamp = torch.nn.ParameterDict({
        "t": torch.nn.Parameter(torch.zeros(3, device=dev)),                     # lamp in camera coordinates, m
        "gain": torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev)),  # log gain per frame and channel
        "off": torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev)),
    })
    if a.lamp_radial:                             # log M = sum_k c_k rho^(2k): radially symmetric (Goldman 2010)
        lamp["rad"] = torch.nn.Parameter(torch.zeros(a.lamp_radial, device=dev))
        RHO2 = RX ** 2 + RY ** 2                  # squared tangent of the angle from the optical axis, per pixel,
        RHO2 = RHO2 / RHO2[VALID].max()           # scaled to 1 at the edge of the picture (well-conditioned powers)
    else:                                         # free camera-fixed map on a 48 x 36 grid
        lamp["logM"] = torch.nn.Parameter(torch.zeros(1, 1, GY, GX, device=dev))
    if a.lamp_gamma:                              # log of the power-law exponent p (camera response)
        lamp["logp"] = torch.nn.Parameter(torch.zeros(1, device=dev))
    if a.lamp_minnaert:                           # log of the Minnaert exponent 2k - 1 on cos(incidence)
        lamp["logq"] = torch.nn.Parameter(torch.zeros(1, device=dev))

    def response(lit):
        """the camera response on the lit colour: identity, lit^p learned (--lamp-gamma) or fixed (--lamp-gamma-fixed)"""
        if a.lamp_gamma_fixed > 0:
            return lit.clamp(min=1e-6) ** a.lamp_gamma_fixed
        return lit.clamp(min=1e-6) ** torch.exp(lamp["logp"]) if a.lamp_gamma else lit

    def smooth_normals(c, n, radius):
        """structure-tensor average of the triangle normals within about radius (voxels of radius/2, 27 neighbours);
        sign kept as the triangle's own"""
        key = torch.floor(c / (radius / 2)).long()
        key = key - key.min(0).values + 1
        dims = key.max(0).values + 2
        lin = (key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]
        uniq, inv = torch.unique(lin, return_inverse=True)
        T = (n[:, :, None] * n[:, None, :]).reshape(-1, 9)
        V = torch.zeros(len(uniq), 9, device=c.device).index_add_(0, inv, T)
        acc = torch.zeros_like(T)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    q = lin + (ox * dims[1] + oy) * dims[2] + oz
                    pos = torch.searchsorted(uniq, q).clamp(max=len(uniq) - 1)
                    hit = uniq[pos] == q
                    acc[hit] += V[pos[hit]]
        A = acc.reshape(-1, 3, 3)
        m = n.clone()                     # power iteration from the triangle's own normal
        for _ in range(15):
            m = torch.bmm(A, m[:, :, None])[:, :, 0]
            m = m / m.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return torch.where((m * n).sum(1, keepdim=True) < 0, -m, m)

    S_ns = smooth_normals(S_c, S_n, a.lamp_smooth_mm / 1000)
    ang = torch.rad2deg(torch.acos((S_ns * S_n).sum(1).clamp(-1, 1)))
    log(f"lamp normals averaged over ~{a.lamp_smooth_mm:.0f} mm: change from the raw triangle normals median "
        f"{ang.median().item():.1f} deg, p95 {ang.quantile(0.95).item() if len(ang) < 16_000_000 else float('nan'):.1f} deg")

    @torch.no_grad()
    def surface_dn(view):
        """depth and smoothed normal (camera coordinates) of the measured surface on the canvas, and where it covers"""
        out, al, _ = gsplat.rasterization(S_c, S_q, torch.exp(S_s), S_o, S_ns @ view[:3, :3].T, view[None], Kc[None],
                                          CW, CH, packed=False, render_mode="RGB+ED", near_plane=0.01)
        a0 = al[0, ..., 0]
        ok = a0 > 0.5
        D = torch.where(ok, out[0, ..., 3], torch.zeros_like(a0))
        N = out[0, ..., :3] / a0[..., None].clamp(min=1e-6)
        return D, N / N.norm(dim=-1, keepdim=True).clamp(min=1e-12), ok

    def lamp_geometry(view):
        """cos(incidence) / (r / r_ref)^2 per canvas pixel; 1 where the measured surface does not cover the pixel"""
        D, N, ok = surface_dn(view)
        X = torch.stack([RX * D, RY * D, D], -1)
        Lv = lamp["t"] - X
        r = Lv.norm(dim=-1).clamp(min=1e-3)
        cos = ((N * Lv).sum(-1) / r).abs().clamp(min=0.05)
        if a.lamp_minnaert:
            cos = cos ** torch.exp(lamp["logq"])
        return torch.where(ok, cos / (r / r_ref) ** 2, torch.ones_like(D))

    def lamp_map():
        """M on the canvas: the free grid, or the radial polynomial in rho^2"""
        if a.lamp_radial:
            return torch.exp(sum(lamp["rad"][k] * RHO2 ** (k + 1) for k in range(a.lamp_radial)))
        return torch.exp(F.interpolate(lamp["logM"], size=(CH, CW), mode="bilinear", align_corners=True))[0, 0]

    def lamp_factor(view):
        return lamp_map() * lamp_geometry(view)

    with torch.no_grad():                         # start: M x cos / r^2 = 1 on average at every pixel
        lg = torch.stack([torch.log(lamp_geometry(views_all[int(n)])) for n in list(train)[::8]]).mean(0)
        if a.lamp_radial:                         # radial map starts flat (M = 1): the geometry factor is not radially
            pass                                  # symmetric (camera below the axis), a radial fit of it is meaningless
        else:
            lamp["logM"][:] = -F.adaptive_avg_pool2d(lg[None, None], (GY, GX))
    log(f"lamp model: r_ref {r_ref * 1000:.0f} mm, M " + (f"radial polynomial of degree {a.lamp_radial} in rho^2, "
        "started flat" if a.lamp_radial else f"on a {GX}x{GY} grid, started from the mean geometry factor"))
if a.exposure:                                  # the per-frame gain / offset of --lamp without the lamp factor
    expo = torch.nn.ParameterDict({
        "gain": torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev)),  # log gain per frame and channel
        "off": torch.nn.Parameter(torch.zeros(len(views_all), 3, device=dev)),
    })
    log("per-frame exposure compensation: gain and offset per frame and channel, no lamp factor")

# ------------------------------------------------------------------------------ Gaussians --------------------
pts = torch.from_numpy(d["points"][::a.points_every]).to(dev)
if a.anchor:                                    # snap the starting points onto the measured surface
    dist, idx = cKDTree(sf["centres"]).query(d["points"][::a.points_every], k=1)
    idx = torch.from_numpy(idx).to(dev)
    n0 = S_n[idx]
    off0 = ((pts - S_c[idx]) * n0).sum(1, keepdim=True)
    near = torch.from_numpy(dist <= a.anchor_snap_mm / 1000).to(dev)
    log(f"anchor: {int(near.sum()):,} of {len(pts):,} points within {a.anchor_snap_mm} mm of the surface; "
        f"offset along the normal median {off0.abs().median().item() * 1000:.2f} mm")
    pts, n0 = (pts - off0 * n0)[near], n0[near]
best_z = torch.full((len(pts),), float("inf"), device=dev)
rgb = torch.zeros(len(pts), 3, device=dev)
for n in train:                                 # colour = the training frame that sees the point closest
    v = views_all[n]
    c = pts @ v[:3, :3].T + v[:3, 3]
    z = c[:, 2]
    i, j, ok = project(c)
    ok &= (i >= 8) & (i < CW - 8) & (j >= 8) & (j < CH - 8)
    ok &= VALID[j.clamp(0, CH - 1), i.clamp(0, CW - 1)]
    better = ok & (z < best_z)
    if better.any():
        img = IM[int(n)].to(dev)
        rgb[better] = img[j[better], i[better]].float() / 255
        if a.lamp_radial:                       # start from albedo = frame value / lamp factor of that frame, so that
            with torch.no_grad():               # the first render already matches the frames (a flat M does not)
                lf = lamp_factor(v)
            lin = rgb[better] ** (1 / a.lamp_gamma_fixed) if a.lamp_gamma_fixed > 0 else rgb[better]  # undo response
            rgb[better] = lin / lf[j[better], i[better]].clamp(min=1e-3)[:, None]
        best_z[better] = z[better]
seen = torch.isfinite(best_z)
pts, rgb, best_z = pts[seen], rgb[seen], best_z[seen]
if a.anchor:
    n0 = n0[seen]
    mu0 = pts.clone()
nn_d, _ = cKDTree(pts.cpu().numpy()).query(pts.cpu().numpy(), k=2)
spacing = torch.from_numpy(nn_d[:, 1].astype(np.float32)).to(dev)
scale0 = torch.maximum(best_z / fx, 0.5 * spacing)   # anchored surfels: depth / focal, at least half the spacing
opa0 = 0.1
if a.baseline:                                  # the papers' start: RMS distance to the 3 nearest points
    nn3, _ = cKDTree(pts.cpu().numpy()).query(pts.cpu().numpy(), k=4)
    scale0 = torch.from_numpy(np.sqrt((nn3[:, 1:] ** 2).mean(1)).astype(np.float32)).to(dev).clamp(min=1e-5)
log(f"{int(seen.sum()):,} of {len(seen):,} points seen by a training frame; scale median {scale0.median().item() * 1000:.2f} mm "
    f"(depth/focal {(best_z / fx).median().item() * 1000:.2f} mm, half spacing {(0.5 * spacing).median().item() * 1000:.2f} mm)")

C0 = 0.28209479177387814
params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(pts.clone()),
    "scales": torch.nn.Parameter(torch.log(scale0).unsqueeze(1).repeat(1, 3)),
    "quats": torch.nn.Parameter(quat_z_to(n0) if a.anchor else torch.tensor([1.0, 0, 0, 0], device=dev).repeat(len(pts), 1),
                                requires_grad=not a.anchor),
    "opacities": torch.nn.Parameter(torch.logit(torch.full((len(pts),), opa0, device=dev))),
    "sh0": torch.nn.Parameter(((rgb - 0.5) / C0).unsqueeze(1)),
    "shN": torch.nn.Parameter(torch.zeros(len(pts), 15, 3, device=dev)),
})
if a.lamp_ambient:
    params["amb"] = torch.nn.Parameter(torch.full((len(pts), 3), -4.6, device=dev))   # sigmoid -> 0.01
THIN = math.log(a.thin_mm / 1000)
if a.anchor:                                    # thin discs in the surface
    with torch.no_grad():
        params["scales"][:, 2] = THIN
if a.eval_only:                                 # score an already trained model
    saved = torch.load(os.path.join(a.out, "model.pt"))
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(saved[k].to(dev)) for k in params.keys()})
    log(f"eval only: loaded {len(params['means']):,} Gaussians from {a.out}/model.pt")
scene_scale = 1.1 * float(np.max(np.linalg.norm(eyes - eyes.mean(0), axis=1)))
lrs = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
if a.lamp_ambient:
    lrs["amb"] = 5e-3
optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()
              if not (a.anchor and k == "quats")}
if a.lamp:
    if a.eval_only:
        with torch.no_grad():
            for k in lamp.keys():
                lamp[k][:] = saved["lamp_" + k].to(dev)
    for k, lr in dict(logM=3e-3, rad=1e-3, t=1e-4, gain=1e-3, off=1e-4, logp=1e-3, logq=1e-3).items():
        if k in lamp and not (k == "t" and a.lamp_colocated):
            optimizers["lamp_" + k] = torch.optim.Adam([lamp[k]], lr=lr, eps=1e-15)
if a.exposure:                                  # the learning rates of --lamp's gain / offset
    for k, lr in dict(gain=1e-3, off=1e-4).items():
        optimizers["expo_" + k] = torch.optim.Adam([expo[k]], lr=lr, eps=1e-15)

sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / a.iters))
strategy = state = None
if ADC:
    strategy = DefaultStrategy(refine_stop_iter=min(15000, a.iters // 2),
                               key_for_gradient="gradient_2dgs" if a.baseline == "2dgs" else "means2d")
    state = strategy.initialize_state(scene_scale=scene_scale)
    GSOPT = {k: v for k, v in optimizers.items() if k in params}   # the strategy handles the Gaussians' own optimizers
    strategy.check_sanity(params, GSOPT)
SH = [0 if a.baseline else a.sh_degree]         # SH degree in use (the baselines raise it every 1000 iterations)
log(f"scene scale {scene_scale:.2f} m; ADC {'on' if ADC else 'off'}; anchor {'on' if a.anchor else 'off'}; "
    f"{a.iters} iterations" + (f"; baseline {a.baseline}" if a.baseline else ""))


def rasterize(view, mode="RGB+ED"):
    p = params
    if a.baseline == "2dgs":                    # surfels; normals and distortion kept in info for the losses
        if torch.is_grad_enabled() and mode == "RGB+ED":
            mode = "RGB+D"                      # as gsplat's simple_trainer_2dgs (its ED backward fails here)
        out, alpha, nrm, snrm, dist, _, info = gsplat.rasterization_2dgs(
            p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
            torch.cat([p["sh0"], p["shN"]], 1), view[None], Kc[None], CW, CH, sh_degree=SH[0], packed=False,
            render_mode=mode, near_plane=0.01, distloss=True)
        info["n2d"], info["sn2d"], info["dist2d"] = nrm, snrm, dist
        return out, alpha, info
    return gsplat.rasterization(p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                                torch.cat([p["sh0"], p["shN"]], 1), view[None], Kc[None], CW, CH, sh_degree=SH[0],
                                packed=False, render_mode=mode, near_plane=0.01)


def ambient(view):
    """the per-Gaussian ambient colour rendered on the canvas (--lamp-ambient)"""
    p = params
    out, _, _ = gsplat.rasterization(p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                                     torch.sigmoid(p["amb"]), view[None], Kc[None], CW, CH, packed=False,
                                     render_mode="RGB", near_plane=0.01)
    return out[0]


# ------------------------------------------------------------------------------ training ---------------------
torch.cuda.reset_peak_memory_stats()
t_train = time.time()
seq, done = [], 0
nonfinite = [0, 0]                              # steps skipped (non-finite loss), steps with non-finite gradients
lam = a.dssim
try:
    for it in range(1, 0 if a.eval_only else a.iters + 1):
        if not seq:
            seq = list(np.random.permutation(train))
        n = int(seq.pop())
        gt = IM[n].to(dev, non_blocking=True).float() / 255
        view = views_all[n]
        if a.baseline:
            SH[0] = min(it // 1000, 3)
        out, alpha, info = rasterize(view)
        if ADC:
            strategy.step_pre_backward(params, GSOPT, state, it, info)
        ren, dep = out[0, ..., :3], out[0, ..., 3]
        if a.lamp:                              # I = g_t [L_t C + A] + b_t
            lit = lamp_factor(view.detach())[..., None] * ren + (ambient(view) if a.lamp_ambient else 0)
            ren = torch.exp(lamp["gain"][n]) * response(lit) + lamp["off"][n]
        elif a.exposure:
            ren = torch.exp(expo["gain"][n]) * ren + expo["off"][n]
        l1 = (ren - gt).abs()[VALID].mean()
        ssim = fused_ssim(torch.where(VALID[..., None], ren, gt).permute(2, 0, 1)[None].contiguous(),
                          gt.permute(2, 0, 1)[None].contiguous())
        if a.anchor:                            # dense relative depth to the measured surface
            Da = DA[n].float()
            md = (Da > 0) & VALID
            ld = ((dep - Da).abs() / Da.clamp(min=1e-3))[md].mean() if md.any() else torch.zeros((), device=dev)
            loss = (1 - lam) * l1 + lam * (1 - ssim) + a.anchor_depth_w * ld
            if a.alpha_w > 0 and md.any():      # the wall is opaque: no darkening through transparency
                loss = loss + a.alpha_w * (1 - alpha[0, ..., 0])[md].mean()
        elif a.baseline_depth > 0:              # dense depth supervision (B1): relative L1 to the measured wall
            Da = DA[n].float()
            md = (Da > 0) & VALID
            # 2DGS renders accumulated depth in training ("RGB+D"); divide by the opacity for the expected depth
            de = dep / alpha[0, ..., 0].clamp(min=1e-3) if a.baseline == "2dgs" else dep
            ld = ((de - Da).abs() / Da.clamp(min=1e-3))[md].mean() if md.any() else torch.zeros((), device=dev)
            loss = (1 - lam) * l1 + lam * (1 - ssim) + a.baseline_depth * ld
        else:                                   # baselines: no depth term (and no zero gradient into 2DGS depth)
            ld = torch.zeros((), device=dev)
            loss = (1 - lam) * l1 + lam * (1 - ssim)
        if a.baseline == "2dgs":                # normal consistency (from 7k) and depth distortion (from 3k)
            if it > a.normal_start:
                sn = info["sn2d"][0] * alpha[0].detach()
                loss = loss + 0.05 * (1 - (info["n2d"][0] * sn).sum(-1)).mean()
            if it > a.dist_start:
                loss = loss + 0.01 * info["dist2d"].mean()
        if not torch.isfinite(loss):            # a non-finite loss: skip the step (no update from it)
            nonfinite[0] += 1
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        loss.backward()
        bad = 0                                 # non-finite gradients (2DGS on lab A: one NaN step spread to 90 % of
        for t_ in list(params.values()) + [v for v in info.values() if torch.is_tensor(v) and v.grad is not None]:
            if t_.grad is not None and not torch.isfinite(t_.grad).all():      # the Gaussians) are set to zero
                bad += 1
                torch.nan_to_num_(t_.grad, nan=0.0, posinf=0.0, neginf=0.0)
        nonfinite[1] += bad > 0
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()
        if ADC:
            strategy.step_post_backward(params, GSOPT, state, it, info, packed=False)
        if a.anchor:
            with torch.no_grad():                  # stay within the clamp along the normal, stay thin
                off = ((params["means"] - mu0) * n0).sum(1, keepdim=True)
                lim = a.anchor_clamp_mm / 1000
                params["means"] -= (off - off.clamp(-lim, lim)) * n0
                params["scales"][:, 2].clamp_(max=THIN)
        done = it
        if it % 500 == 0 or it == 1:
            with torch.no_grad():
                ps = -10 * torch.log10(((ren - gt) ** 2)[VALID].mean()).item()
            dtxt = f"depth {ld.item():.2%}" if (a.anchor or a.baseline_depth > 0) else "no depth term"
            log(f"iter {it:6d} loss {loss.item():.4f} L1 {l1.item():.4f} SSIM {ssim.item():.3f} {dtxt} "
                f"PSNR {ps:.2f} | {len(params['means']):,} Gaussians | {it / (time.time() - t_train):.1f} it/s | "
                f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
except torch.cuda.OutOfMemoryError as e:
    log(f"OUT OF MEMORY at iteration {done + 1} with {len(params['means']):,} Gaussians: {e}")
torch.cuda.synchronize()
SH[0] = 3 if a.baseline else a.sh_degree        # scoring (and --eval-only) with the full SH degree
train_s = time.time() - t_train
peak = torch.cuda.max_memory_allocated() / 1e9
if not a.eval_only:
    extra = {"lamp_" + k: v.detach().cpu() for k, v in lamp.items()} if a.lamp else {}
    extra |= {"expo_" + k: v.detach().cpu() for k, v in expo.items()} if a.exposure else {}
    torch.save({k: v.detach().cpu() for k, v in params.items()} | extra | dict(K=Kc.cpu(), size=(CW, CH)),
               os.path.join(a.out, "model.pt"))
    if a.lamp:
        Mv = lamp_map()[VALID]
        log(f"lamp: offset from the camera centre {(lamp['t'] * 1000).round().tolist()} mm; M range "
            f"{Mv.min().item():.2f}..{Mv.max().item():.2f}"
            + (f" (radial coefficients {[round(c, 4) for c in lamp['rad'].tolist()]})" if a.lamp_radial else "")
            + (f"; response exponent p {torch.exp(lamp['logp']).item():.3f}" if a.lamp_gamma else "")
            + (f"; Minnaert exponent on cos {torch.exp(lamp['logq']).item():.3f}" if a.lamp_minnaert else ""))
    if a.exposure:                              # spread over the trained frames (the others keep their start values)
        tr = torch.from_numpy(train).to(dev)
        g, o = expo["gain"].detach()[tr], expo["off"].detach()[tr]
        log(f"exposure: log gain sd per channel {[round(x, 3) for x in g.std(0).tolist()]}, gain p5-p95 "
            f"{torch.exp(g).quantile(0.05).item():.3f}..{torch.exp(g).quantile(0.95).item():.3f}; offset median "
            f"{[round(x, 4) for x in o.median(0).values.tolist()]}")
    log(f"trained {done} iterations in {train_s / 60:.1f} min, {len(params['means']):,} Gaussians, peak {peak:.2f} GB; "
        f"steps skipped for a non-finite loss {nonfinite[0]}, steps with non-finite gradients zeroed {nonfinite[1]}")


# ------------------------------------------------------------------------------ evaluation -------------------
@torch.no_grad()
def render_raw(view):
    out, alpha, _ = rasterize(view)
    if a.lamp:                                  # the frame's own gain and offset are fitted by the scoring
        lit = out[..., :3] * lamp_factor(view)[None, ..., None] + (ambient(view)[None] if a.lamp_ambient else 0)
        out = torch.cat([response(lit), out[..., 3:]], -1)
    # with a per-frame gain the colours carry no fixed scale: not clipped at 1 (the scoring's gain / offset fit applies)
    ren = (out[0, ..., :3].clamp(min=0) if a.lamp or a.exposure else out[0, ..., :3].clamp(0, 1)).mul(255).cpu().numpy()
    al = alpha[0, ..., 0].cpu().numpy()
    dep_c = out[0, ..., 3]
    if to_raw is not None:
        ren = cv2.remap(ren, *to_raw, cv2.INTER_LINEAR, borderValue=0)
        al = cv2.remap(al, *to_raw, cv2.INTER_LINEAR, borderValue=0)
    return ren, al, dep_c, alpha[0, ..., 0]


def so3_t(wv):
    th = wv.norm().clamp(min=1e-12)
    k = wv / th
    Kx = torch.zeros(3, 3, device=wv.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 2] = -k[2], k[1], -k[0]
    Kx = Kx - Kx.T
    return torch.eye(3, device=wv.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def pose_step(view, wv, dc):
    """view moved by a rotation vector wv (camera axes) and a camera-centre shift dc (camera axes), as tensors"""
    R, t = view[:3, :3], view[:3, 3]
    C = -R.T @ t
    R2 = so3_t(wv + 1e-12) @ R
    return torch.cat([torch.cat([R2, (-R2 @ (C + R.T @ dc))[:, None]], 1), view[3:4]], 0)


def proj_raw(Xw, view):
    """raw-frame pixel coordinates (float) of world points (numpy), with the radial k1"""
    R, t = view[:3, :3].cpu().numpy().astype(np.float64), view[:3, 3].cpu().numpy().astype(np.float64)
    c = Xw @ R.T + t
    xn, yn = c[:, 0] / c[:, 2], c[:, 1] / c[:, 2]
    g = 1 + k1 * (xn ** 2 + yn ** 2)
    return np.stack([cx + fx * xn * g, cy + fy * yn * g], 1)


def test_pose(view, img, Bmask):
    """test-time registration of one frame (--test-pose-iters), model frozen: render, optical flow from the frame to
    the render on CLAHE-normalised images at its detailed pixels, back-project those pixels with the rendered depth and
    solve the 6 pose numbers whose image motion cancels the flow (numerical Jacobian, Huber-weighted least squares);
    a.test_pose_iters rounds. Detail-driven: smooth shading does not pull the pose."""
    real = img.astype(np.float32)
    v0 = view
    xu_raw = (np.arange(w) - cx) / fx
    yu_raw = (np.arange(h) - cy) / fy
    if k1 != 0:
        XUr, YUr = np.nan_to_num(XU), np.nan_to_num(YU)
    else:
        XUr, YUr = np.meshgrid(xu_raw, yu_raw)
    for _ in range(a.test_pose_iters):
        G, aG, dep_c, _ = render_raw(view)
        D = dep_c.cpu().numpy()
        if to_raw is not None:
            D = cv2.remap(D, *to_raw, cv2.INTER_NEAREST, borderValue=0)
        m = Bmask & valid_raw & (aG > 0.5) & (D > 0.01)
        Gm = im.linear_match(G, real, m)
        A = im._CLAHE.apply(np.clip(im.grey(real), 0, 255).astype(np.uint8))
        Bq = im._CLAHE.apply(np.clip(im.grey(Gm), 0, 255).astype(np.uint8))
        fl = im._DIS.calc(A, Bq, None)
        gs = cv2.GaussianBlur(A.astype(np.float32), (0, 0), 1.5)
        gm = np.hypot(cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1))
        inner = im.erode(m, 15)
        if inner.sum() < 500:
            break
        sel = inner & (gm >= np.percentile(gm[inner], 70)) & (np.hypot(fl[..., 0], fl[..., 1]) < 40)
        ys, xs = np.nonzero(sel)
        if len(ys) > 20000:
            k_ = np.random.default_rng(0).choice(len(ys), 20000, replace=False)
            ys, xs = ys[k_], xs[k_]
        z = D[ys, xs].astype(np.float64)
        Xc = np.stack([XUr[ys, xs] * z, YUr[ys, xs] * z, z], 1)
        R, t = view[:3, :3].cpu().numpy().astype(np.float64), view[:3, 3].cpu().numpy().astype(np.float64)
        Xw = (Xc - t) @ R
        p0 = proj_raw(Xw, view)
        J = np.zeros((len(ys), 2, 6))
        for k_ in range(6):
            e = np.zeros(6)
            e[k_] = 1e-4
            vk = pose_step(view, torch.tensor(e[:3], device=dev, dtype=torch.float32),
                           torch.tensor(e[3:], device=dev, dtype=torch.float32))
            J[:, :, k_] = (proj_raw(Xw, vk) - p0) / 1e-4
        f = fl[ys, xs].astype(np.float64)
        Jm, rhs = J.reshape(-1, 6), (-f).reshape(-1)
        wgt = np.ones(len(rhs))
        for _ in range(4):                         # Huber IRLS, 1 px
            Aw = Jm * wgt[:, None]
            xi = np.linalg.lstsq(Aw.T @ Jm + 1e-6 * np.eye(6), Aw.T @ rhs, rcond=None)[0]
            res = np.abs(Jm @ xi - rhs)
            wgt = np.where(res < 1.0, 1.0, 1.0 / np.maximum(res, 1e-9))
        view = pose_step(view, torch.tensor(xi[:3], device=dev, dtype=torch.float32),
                         torch.tensor(xi[3:], device=dev, dtype=torch.float32))
    R0, t0 = v0[:3, :3], v0[:3, 3]
    R1, t1 = view[:3, :3], view[:3, 3]
    dC = (-R1.T @ t1) - (-R0.T @ t0)
    ang = torch.acos(((torch.trace(R1 @ R0.T) - 1) / 2).clamp(-1, 1))
    return view.detach(), float(dC.norm().item() * 1000), float(ang.item() * 1000)


LEFT = np.zeros((h, w), bool)                   # --fit-half: gain/offset fitted here, metrics on the rest
LEFT[:, : w // 2] = True


def score(tag, imgs, nxt, views, B, Bm, ids, pos):
    """ruler v2 on one test set: every candidate (model G, baseline B, aligned next frame Nal, denoised frame D) gets
    its own per-frame gain / offset, then the metrics on the same pixels (B's mask)"""
    rows = []
    for k in range(len(ids)):
        view = torch.from_numpy(views[k]).to(dev)
        tp = None
        if a.test_pose_iters:
            view, tp_mm, tp_mrad = test_pose(view, imgs[k], Bm[k])
            tp = (tp_mm, tp_mrad)
        G, aG, dep_c, al_c = render_raw(view)
        mG = valid_raw & (aG > a.alpha_min)
        m = Bm[k] & valid_raw                   # the same pixels for every model: B's mask, no opacity threshold
        real = imgs[k].astype(np.float32)
        Bk = B[k].astype(np.float32)
        row = dict(tag=tag, frame=int(ids[k]), pos=float(pos[k]), coverage=float(m.mean()), cov_G=float(mG.mean()),
                   cov_B=float(Bm[k].mean()))
        if tp is not None:
            row["test_pose_mm"], row["test_pose_mrad"] = tp
        mats = {}
        nal = im.linear_match(im.align_to(nxt[k].astype(np.float32), real), real, m)
        den = cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(imgs[k]), None, 5, 5, 7, 21).astype(np.float32)
        for name, img in (("G", G), ("B", Bk), ("Nal", nal), ("D", den)):
            fm, em = (m & LEFT, m & ~LEFT) if a.fit_half else (m, m)
            x = im.linear_match(img, real, fm)
            mats[name] = x
            for key, val in im.all_metrics(x, real, em).items():
                row[f"{name}_{key}"] = val
            for key, val in im.ruler2(x, real, nal, em).items():
                row[f"{name}_{key}"] = val
        Ds = depth_map(view)
        md = (Ds > 0) & (al_c > a.alpha_min) & VALID
        rel = ((dep_c - Ds).abs() / Ds)[md]
        row["depth_rel_err"] = float(rel.median()) if len(rel) else np.nan
        if a.anchor:
            Da = surface_depth(view)
            ma = (Da > 0) & (al_c > a.alpha_min) & VALID
            rel = ((dep_c - Da).abs() / Da)[ma]
            row["depth_rel_err_surface"] = float(rel.median()) if len(rel) else np.nan
        rows.append(row)
        if a.dump:
            os.makedirs(a.dump, exist_ok=True)
            for name, img in (("G", mats["G"]), ("real", real), ("B", mats["B"])):
                fn = os.path.join(a.dump, f"{tag}_{int(ids[k]):06d}_{name}.jpg")
                if name == "G" or not os.path.exists(fn):
                    cv2.imwrite(fn, np.clip(img, 0, 255).astype(np.uint8)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        if a.save and k % a.save == 0 and not a.eval_only:
            trip = np.hstack([real, mats["B"], mats["G"]]).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(a.out, "renders", f"{tag}_{int(ids[k]):06d}.jpg"), trip[:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        log(f"  {tag} {k + 1}/{len(ids)} frame {int(ids[k])}: PSNR G {row['G_psnr']:.2f} B {row['B_psnr']:.2f} "
            f"Nal {row['Nal_psnr']:.2f} | LPIPS G {row['G_lpips']:.3f} B {row['B_lpips']:.3f} | depth err {row['depth_rel_err']:.1%}")
    return rows


def evaluate_all():
    rng = d["block_ranges"]
    in_block = lambda p: bool(np.any([(p >= lo) & (p < hi) for lo, hi in rng]))  # noqa: E731
    rows, rows_b = [], []
    if not a.no_held:
        rows = score("held", d["held_images"], d["held_next"], d["held_viewmats"], d["held_B"], d["held_mask"],
                     d["held_ids"], d["held_pos"])
        for r in rows:
            r["in_block"] = in_block(r["pos"])
        bi = block_idx
        rows_b = score("block", raw_train[bi], d["block_next"], d["block_viewmats"], d["block_B"], d["block_mask"],
                       d["block_ids"], d["train_pos"][bi])
        for r in rows_b:
            r["in_block"] = True
    extra = {}
    for spec in a.extra_test:                   # frames of other runs (export_xrun.py)
        name, fn = spec.split("=", 1)
        X = np.load(fn if os.path.isabs(fn) else os.path.join(a.scene, fn))
        rx = score(name, X["xrun_images"], X["xrun_next"], X["xrun_viewmats"], X["xrun_B"], X["xrun_mask"], X["xrun_ids"],
                   X["xrun_pos"])
        for r in rx:
            r["in_block"] = in_block(r["pos"])
        extra[name] = rx

    all_rows = rows + rows_b + [r for rx in extra.values() for r in rx]
    keys = [k for k in all_rows[0] if k not in ("tag", "frame", "in_block")]
    with open(os.path.join(a.out, f"per_frame{a.suffix}.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, ["tag", "frame", "in_block"] + keys)
        wr.writeheader()
        wr.writerows(all_rows)

    def med(rs):
        return {k: float(np.nanmedian([r[k] for r in rs])) for k in keys} if rs else {}

    summary = dict(scene=a.scene, args=vars(a), canvas=[CW, CH], train_frames=int(len(train)), iterations_done=done,
                   train_minutes=round(train_s / 60, 2), it_per_s=round(done / max(train_s, 1e-9), 1), peak_GB=round(peak, 2),
                   gaussians=int(len(params["means"])), block_ranges_m=rng.tolist(),
                   held_all=med(rows), held_outside_blocks=med([r for r in rows if not r["in_block"]]),
                   block=med(rows_b), n=dict(held=len(rows), held_outside_blocks=sum(not r["in_block"] for r in rows),
                                             block=len(rows_b)), fit_half=a.fit_half)
    parts = [] if a.no_held else [("in-sequence held-out, outside the blocks", "held_outside_blocks"), ("block frames", "block")]
    for name, rx in extra.items():
        summary[name], summary[name + "_in_blocks"] = med(rx), med([r for r in rx if r["in_block"]])
        summary["n"][name], summary["n"][name + "_in_blocks"] = len(rx), sum(r["in_block"] for r in rx)
        parts += [(f"{name}: frames of another run", name)]
    json.dump(summary, open(os.path.join(a.out, f"summary{a.suffix}.json"), "w"), indent=1)
    M = ("psnr", "lpips", "shift_px", "detail_corr", "hf_ratio", "mse_low", "mse_high")
    M2 = M + tuple(f"{q}_{b}" for q in ("fid", "en") for b, _, _ in im.BANDS)
    for title, s in ((t, summary[k]) for t, k in parts):
        log(f"\n{title} (median; coverage {s['coverage']:.1%}, depth error G {s['depth_rel_err']:.1%}):")
        for tag, name in (("G", "model"), ("B", "lampmap B"), ("Nal", "next frame aligned"), ("D", "frame denoised")):
            log(f"  {name:18s} " + "  ".join(f"{k} {s[f'{tag}_{k}']:.3f}" for k in M2))
    log("done")


if a.smoke:
    log("smoke run: no scoring")
    sys.exit(0)
a.suffix = a.suffix if a.suffix is not None else ("_eval" if a.eval_only else "_r2")
log(f"=== scoring (ruler v2: B's mask for every model; files *{a.suffix}) ===")
evaluate_all()
