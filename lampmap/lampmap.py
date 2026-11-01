"""Lamp-map model: the reconstructed pipe (OpenMVS textured mesh) seen from the camera, times one gain map that is fixed
in the image - the robot's lamp moves with the camera, so the shading it casts on the wall (and the lens vignetting)
stays nearly the same in every frame. The map is learned only from the frames the model was made from and checked
on held-out frames (frames halfway between them, which took no part in SfM, dense reconstruction or texturing).

usage: python -I lampmap.py WORK_DIR VIDEO OUT_DIR [--n 44] [--train 40] [--ahead 3.0] [--save 4]
                            [--plain [--crop X0 Y0 X1 Y1] [--stretch-x S]]
       WORK_DIR  scripts/vision/run_pipeline.py work folder after step 13 (mvs/scene_textured.obj, frames.csv, sfm/recon)
       VIDEO     the original video (cut and stretched as extract_frames.py does, but no CLAHE: the goal is the video)
       --plain/--crop/--stretch-x  the frame geometry given to run_pipeline.py for a non-Sheffield video
       OUT_DIR   lamp_map.npy (log gain, image size), lamp_map.png, summary.json, per_frame.csv,
                 renders/heldout_<frame>.jpg (--save of them: video frame | model | model x lamp map)
pipesplat/export_g0.py uses the same functions to put baseline B and its mask (the pixels every model is scored on)
into each training scene; --lamp auto there learns the map exactly as here.
Run with a Windows Python that has torch + lpips (on the author's machine the da3 conda env) and TORCH_HOME set to
get LPIPS as well; without them LPIPS is NaN (imgmetrics.py).

Learning the map (model frames only): each frame's render is drawn from its own registered pose (focal length,
principal point, radial k1), brightness-matched to the frame (gain + offset per channel), and the per-pixel
log(frame / render) is averaged over the frames and smoothed (Gaussian 6 px, filled where no frame saw the wall).
Using it: render x exp(map), then the frame's own gain + offset (its exposure) - nothing else is fitted on a
held-out frame. The comparison:
  A  model       render, gain + offset only (the usual "brightness-matched" comparison)
  B  model x lamp map
  N  next frame  the video's own next frame in place of a render: about what a perfect model could reach given the
                 video's noise (an optimistic reference for LPIPS: the next frame shares the video's noise texture)
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, os.path.join(SCRIPTS, "vision"))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
import paths  # noqa: E402
import render_views as rv  # noqa: E402
from video_compare import source_frame_reader  # noqa: E402
import imgmetrics as im  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
METRICS = ("psnr", "ssim", "lpips", "shift_px", "detail_corr", "hf_ratio", "mse_low", "mse_high")


# ------------------------------------------------------------------------------------------ rendering ---------
def textured(m, cam, sel):
    """z-buffered textured triangles sel of mesh m (render_views layout: xyz, tri, uv, tex, blank) at the oversized
    drawing resolution, averaged down: RGB (float, premultiplied) and the share of each pixel covered by a
    textured face (not by the faces no image covers)"""
    _, fbuf, b1, b2 = rv.rasterize(cam, m["xyz"], m["tri"][sel], 0.02)
    hit = fbuf >= 0
    f, b1, b2 = sel[fbuf[hit]], b1[hit][:, None], b2[hit][:, None]
    uv = (1 - b1 - b2) * m["uv"][f, 0] + b1 * m["uv"][f, 1] + b2 * m["uv"][f, 2]
    where = np.flatnonzero(hit)
    good = hit.copy()
    good.flat[where[m["blank"][f]]] = False
    tex = m["tex"][0]
    mx, my = np.full(hit.shape, -1, np.float32), np.full(hit.shape, -1, np.float32)
    mx.flat[where] = uv[:, 0] * tex.shape[1] - 0.5
    my.flat[where] = (1 - uv[:, 1]) * tex.shape[0] - 0.5
    rgb = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, :, ::-1].astype(np.float32)
    rgb[~good] = 0
    size = (hit.shape[1] // rv.SS, hit.shape[0] // rv.SS)
    return (cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
            cv2.resize(good.astype(np.float32), size, interpolation=cv2.INTER_AREA))


def finish(rgb, cov, valid=True):
    """un-premultiply and keep only the pixels fully covered by textured faces"""
    mask = (cov > 0.999) & valid
    return np.clip(np.where(mask[..., None], rgb / np.maximum(cov, 1e-6)[..., None], 0), 0, 255), mask


class Renderer:
    """the OpenMVS textured mesh through the registered camera model (pinhole drawn oversized, then remapped through
    radial k1); pose = (eye, ahead point, above point) in pipe coordinates as render_views.load_model gives them.
    Returns (RGB float, mask)."""

    def __init__(self, model, ahead):
        self.m, self.ahead = model, ahead
        intr = model["intrinsics"]
        f, (cx, cy), k1 = intr["focal_length"], intr["principal_point"], intr.get("disto_k1", [0.0])[0]
        self.w, self.h = int(intr["width"]), int(intr["height"])
        xu, yu = rv.undistort_grid(*np.meshgrid((np.arange(self.w) - cx) / f, (np.arange(self.h) - cy) / f), k1)
        self.valid = np.isfinite(xu)
        self.f = f
        self.cw = 2 * int(np.ceil(np.nanmax(np.abs(xu)) * f)) + 4
        self.ch = 2 * int(np.ceil(np.nanmax(np.abs(yu)) * f)) + 4
        self.mx = np.nan_to_num(self.cw / 2 + xu * f - 0.5, nan=-10).astype(np.float32)
        self.my = np.nan_to_num(self.ch / 2 + yu * f - 0.5, nan=-10).astype(np.float32)
        self.cx_face = model["centroid"][:, 0]

    def __call__(self, pose):
        eye, ahead, above = pose
        cam = rv.camera(eye, ahead, (self.cw, self.ch), focal=self.f, up=above - eye)
        sel = np.flatnonzero((self.cx_face > eye[0] - 0.05) & (self.cx_face < eye[0] + self.ahead))
        rgb, cov = textured(self.m, cam, sel)
        rgb = cv2.remap(rgb, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cov = cv2.remap(cov, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return finish(rgb, cov, self.valid)


# ------------------------------------------------------------------------------------------- lamp map ---------
def learn_lamp_map(render, read, train):
    """log gain map (image size) from model frames: train = [(pose, frame key)]"""
    acc, cnt = 0.0, 0.0
    for pose, key in train:
        ren, msk = render(pose)
        real = read(key)[:, :, ::-1]
        g = im.linear_match(ren, real, msk)                 # take out the frame's overall exposure first
        acc = acc + np.where(msk, np.log(im.grey(real) + 1.0) - np.log(im.grey(g) + 1.0), 0)
        cnt = cnt + msk
    mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0).astype(np.float32)
    wsum = cv2.GaussianBlur((cnt > 0).astype(np.float32), (0, 0), 6) + 1e-6
    return cv2.GaussianBlur(mean, (0, 0), 6) / wsum


def apply_lamp(ren, lamp):
    """the model under the lamp, before the frame's exposure (gain + offset) is applied"""
    return np.clip(ren * np.exp(lamp)[..., None], 0, 255)


def save_lamp_map(out, lamp):
    np.save(os.path.join(out, "lamp_map.npy"), lamp.astype(np.float32))
    g = np.exp(lamp)
    vis = cv2.applyColorMap((np.clip(g / np.percentile(g, 99.5), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imencode(".png", vis)[1].tofile(os.path.join(out, "lamp_map.png"))


def evaluate(render, read, train, heldout, out, info, t0, save=4):
    """learn the lamp map on train, then compare A (model), B (model x lamp map) and N (next video frame) on the
    held-out frames: heldout = [(pose, key, next key, extra columns)]. Writes the lamp map, per_frame.csv,
    summary.json and every save-th held-out triplet (video | A | B)."""
    os.makedirs(os.path.join(out, "renders"), exist_ok=True)
    lamp = learn_lamp_map(render, read, train)
    save_lamp_map(out, lamp)
    print(f"lamp map from {len(train)} model frames: gain {np.exp(np.percentile(lamp, 5)):.2f}..{np.exp(np.percentile(lamp, 95)):.2f} "
          f"[{time.time() - t0:.0f} s]")
    rows = []
    for n, (pose, key, key2, extra) in enumerate(heldout):
        ren, msk = render(pose)
        real, real2 = read(key)[:, :, ::-1], read(key2)[:, :, ::-1]
        v = dict(A=im.linear_match(ren, real, msk), B=im.linear_match(apply_lamp(ren, lamp), real, msk),
                 N=im.linear_match(real2.astype(np.float32), real, msk))
        row = dict(frame=key, **extra, coverage=float(msk.mean()))
        for tag, img in v.items():
            for k, val in im.all_metrics(img, real, msk).items():
                row[f"{tag}_{k}"] = val
        rows.append(row)
        if save and n % save == 0:
            trip = np.hstack([real, v["A"], v["B"]]).astype(np.uint8)
            cv2.imencode(".jpg", trip[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(
                os.path.join(out, "renders", f"heldout_{key:06d}.jpg"))
        print(f"  {n + 1}/{len(heldout)} frame {key} cover {row['coverage']:.0%}: PSNR model {row['A_psnr']:.1f}, "
              f"x lamp map {row['B_psnr']:.1f}, next frame {row['N_psnr']:.1f} [{time.time() - t0:.0f} s]")
    keys = list(rows[0])
    with open(os.path.join(out, "per_frame.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, keys)
        wr.writeheader()
        wr.writerows(rows)
    num = [k for k in keys if isinstance(rows[0][k], float)]
    summ = dict(info, held_out_frames=len(rows), model_frames_for_lamp_map=len(train),
                median={k: float(np.nanmedian([r[k] for r in rows])) for k in num})
    json.dump(summ, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    md = summ["median"]
    print(f"\nmedian over {len(rows)} held-out frames (coverage {md['coverage']:.0%}):")
    for tag, name in (("A", "model"), ("B", "model x lamp map"), ("N", "next video frame")):
        print(f"  {name:17s} " + "  ".join(f"{k} {md[f'{tag}_{k}']:.3f}" for k in METRICS))
    print(f"[{time.time() - t0:.0f} s]")
    return summ


# ------------------------------------------------------------------------------------- vision work dir ---------
def lerp_pose(model, k0, k1, w):
    g = lambda a: (1 - w) * model[a][k0] + w * model[a][k1]  # noqa: E731
    return g("cam_xyz"), g("cam_ahead"), g("cam_above")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("work"), ap.add_argument("video"), ap.add_argument("out")
    ap.add_argument("--n", type=int, default=44), ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--ahead", type=float, default=3.0), ap.add_argument("--save", type=int, default=4)
    ap.add_argument("--plain", action="store_true"), ap.add_argument("--crop", nargs=4, type=int)
    ap.add_argument("--stretch-x", type=float, default=1.0)
    a = ap.parse_args()
    t0 = time.time()
    model = rv.load_model(paths.CONVERTER, a.work, [None], straighten=False)
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(a.work, "frames.csv"), encoding="utf-8"))}
    src = np.array([frames[n] for n in model["cam_file"]])
    order = np.argsort(src)
    step = int(np.median(np.diff(src[order])))
    if step < 2:
        sys.exit("every video frame was used for the model (--every 1): there are no held-out frames in between")
    render = Renderer(model, a.ahead)
    print(f"{len(src)} registered frames, step {step}; render {render.w}x{render.h} [{time.time() - t0:.0f} s]")
    pose = lambda k: (model["cam_xyz"][k], model["cam_ahead"][k], model["cam_above"][k])  # noqa: E731
    train = [(pose(order[j]), int(src[order[j]])) for j in np.linspace(0, len(order) - 1, a.train).round().astype(int)]
    pairs = [(order[i], order[i + 1]) for i in range(len(order) - 1) if src[order[i + 1]] - src[order[i]] == step]
    heldout = []
    for k0, k1 in (pairs[i] for i in np.linspace(0, len(pairs) - 1, a.n).round().astype(int)):
        s = int((src[k0] + src[k1]) // 2)
        p = lerp_pose(model, k0, k1, (s - src[k0]) / (src[k1] - src[k0]))
        heldout.append((p, s, s + 1, dict(x=float(p[0][0]))))
    evaluate(render, source_frame_reader(a.video, a.plain, a.crop, a.stretch_x), train, heldout, a.out,
             dict(model="OpenMVS textured mesh", work=a.work, video=a.video, faces_ahead_m=a.ahead), t0, a.save)


if __name__ == "__main__":
    main()
