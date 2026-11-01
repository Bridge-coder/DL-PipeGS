"""Export one data set for training and scoring (runs on Windows): everything the WSL trainer train_gs.py needs, in
one npz. (The name is historical: the first Gaussian baseline of the project was called G0.)

usage: python -I pipesplat/export_g0.py vision WORK_DIR VIDEO OUT_DIR --lamp LAMP_MAP.npy|auto [--plain --crop --stretch-x]
       WORK_DIR  scripts/vision/run_pipeline.py work folder after step 13 (sfm/recon, mvs/scene_dense.ply,
                 mvs/scene_textured.obj, frames.csv; counter.csv if there is travel ground truth)
       VIDEO     the original video: frames are read as recorded (no CLAHE), with the same --plain/--crop/--stretch-x
                 as the front end
       --lamp    the camera-fixed lamp map of baseline B (lampmap/lampmap.py); auto = learn it here from 40 training
                 frames. A video used with --every 1 gets 44 registered frames held out of training instead of
                 in-between frames.
The metric scale is the vision model's with the bore = PIPE3D_PIPE_DIAMETER_M (scripts/vision/evaluate.py, default
0.30 m); the axial origin follows WORK_DIR/counter.csv when it exists.

Contents of OUT_DIR/scene.npz (positions in metres in pipe coordinates, cameras as OpenCV world-to-camera matrices,
intrinsics with pixel centres at integer coordinates, images RGB uint8 as recorded):
  train_*   every SfM-registered frame: image, viewmat, camera centre, frame id, axial position
  held_*    44 in-sequence held-out frames: video frames halfway between two model frames (interpolated pose) - or,
            with --every 1, registered frames held out of training (own SfM pose); per frame the next video frame
            (reference N), the pose, and baseline B (OpenMVS textured mesh x lamp map) with its mask = the pixels
            every model is scored on
  block_*   the "block" test: the model frames inside two 10 cm stretches (at 35 % and 65 % of the run) that
            train_gs.py leaves out of training; per frame its next video frame and B / mask
  points    initial Gaussian centres = the OpenMVS dense cloud (also depth_points: the depth-error reference)
  K, k1, size   raw-frame camera (one focal length, one radial distortion term k1)
Next to it: scene.json (counts, block ranges, camera) and, with --lamp auto, lamp_map.npy.
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
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (os.path.join(SCRIPTS, "vision"), SCRIPTS, os.path.join(ROOT, "lampmap"), HERE):
    sys.path.insert(0, p)
import paths  # noqa: E402
import render_views as rv  # noqa: E402
import evaluate as ev  # noqa: E402
import lampmap as lm  # noqa: E402
from ply_io import read_ply_vertices  # noqa: E402
from video_compare import source_frame_reader  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
T0 = time.time()
BLOCK_AT, BLOCK_LEN = (0.35, 0.65), 0.10


def log(msg):
    print(f"{msg} [{time.time() - T0:.0f} s]")


def cv_viewmat(eye, ahead, above):
    """world-to-camera 4x4 in the OpenCV convention for a camera at eye looking at ahead, with above-eye as up
    (the same right/up/forward axes as render_views.camera)"""
    f = (ahead - eye) / np.linalg.norm(ahead - eye)
    r = np.cross(f, above - eye)
    r /= np.linalg.norm(r)
    up = np.cross(r, f)
    R = np.stack([r, -up, f])
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, -R @ eye
    return m


def blocks(pos):
    """two stretches of BLOCK_LEN metres starting at 35 % and 65 % of the camera positions"""
    lo, hi = float(np.min(pos)), float(np.max(pos))
    return np.array([[lo + f * (hi - lo), lo + f * (hi - lo) + BLOCK_LEN] for f in BLOCK_AT])


def in_blocks(pos, rng):
    pos = np.asarray(pos)
    return np.any([(pos >= a) & (pos < b) for a, b in rng], axis=0)


def b_render(render, lamp, pose):
    """lampmap's B before the per-frame gain/offset: model render x lamp map, and its mask"""
    ren, msk = render(pose)
    return lm.apply_lamp(ren, lamp).astype(np.float16), msk


# ----------------------------------------------------------------------------------------- vision --------
def export_vision(a):
    model = rv.load_model(paths.CONVERTER, a.work, [None], straighten=False)
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(a.work, "frames.csv"), encoding="utf-8"))}
    src = np.array([frames[n] for n in model["cam_file"]])
    order = np.argsort(src)
    step = int(np.median(np.diff(src[order])))
    intr = model["intrinsics"]
    f, (cx, cy), k1 = intr["focal_length"], intr["principal_point"], intr.get("disto_k1", [0.0])[0]
    w, h = int(intr["width"]), int(intr["height"])
    read = source_frame_reader(a.video, a.plain, a.crop, a.stretch_x)

    def grab(key):
        fr = read(int(key))
        if fr is None or fr.shape[:2] != (h, w):
            sys.exit(f"video frame {key}: {None if fr is None else fr.shape} - expected {h}x{w}")
        return fr[:, :, ::-1].copy()

    pose = lambda k: (model["cam_xyz"][k], model["cam_ahead"][k], model["cam_above"][k])  # noqa: E731
    out = dict(K=np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], np.float32), k1=np.float32(k1), size=np.array([w, h], np.int32))
    in_model = step < 2                          # every video frame is in the model (Pipe1, --every 1)
    if in_model:
        # no frames in between: the usual novel-view protocol instead - 44 registered frames spread over the run
        # (outside the blocks) are held out of the Gaussian training and scored from their own SfM pose. The SfM,
        # the dense cloud, the mesh and B's texture did see them (B is advantaged; the Gaussians are not)
        rng_pre = blocks(model["cam_xyz"][order, 0])
        cand = np.flatnonzero(~in_blocks(model["cam_xyz"][order, 0], rng_pre))[1:-1]
        hj = cand[np.linspace(0, len(cand) - 1, 44).round().astype(int)]
        held_k = order[hj]
        order = np.delete(order, hj)
    out["train_ids"] = src[order].astype(np.int32)
    out["train_viewmats"] = np.stack([cv_viewmat(*pose(k)) for k in order]).astype(np.float32)
    out["train_eyes"] = model["cam_xyz"][order].astype(np.float32)
    out["train_pos"] = model["cam_xyz"][order, 0].astype(np.float32)
    out["train_t"] = np.zeros(len(order), np.int64)
    log(f"vision: {len(order)} registered frames, step {step}; reading them from the video")
    out["train_images"] = np.stack([grab(s) for s in src[order]])

    # the 44 held-out frames exactly as lampmap.main picks them
    render = lm.Renderer(model, 3.0)
    if a.lamp == "auto":                         # no lampmap run for this video: learn B's lamp map here, as lampmap
        lt = [(pose(order[j]), int(src[order[j]])) for j in np.linspace(0, len(order) - 1, 40).round().astype(int)]
        lamp = lm.learn_lamp_map(render, read, lt)
        lm.save_lamp_map(a.out, lamp)
        log(f"lamp map learnt from {len(lt)} training frames -> {a.out}/lamp_map.npy")
    else:
        lamp = np.load(a.lamp)
    if in_model:
        held = [(pose(k), int(src[k])) for k in held_k]
    else:
        pairs = [(order[i], order[i + 1]) for i in range(len(order) - 1) if src[order[i + 1]] - src[order[i]] == step]
        held = []
        for k0, k1_ in (pairs[i] for i in np.linspace(0, len(pairs) - 1, 44).round().astype(int)):
            s = int((src[k0] + src[k1_]) // 2)
            held.append((lm.lerp_pose(model, k0, k1_, (s - src[k0]) / (src[k1_] - src[k0])), s))
        ref = os.path.join(os.path.dirname(a.lamp), "per_frame.csv")
        if os.path.exists(ref):
            same = [int(r["frame"]) for r in csv.DictReader(open(ref, encoding="utf-8"))] == [s for _, s in held]
            log(f"held-out frames identical to lampmap's per_frame.csv: {same}")
    fill_eval(out, "held", held, render, lamp, grab)
    out["held_pos"] = np.array([p[0][0] for p, _ in held], np.float32)

    rng = rng_pre if in_model else blocks(out["train_pos"])
    bidx = np.flatnonzero(in_blocks(out["train_pos"], rng))
    out["block_ranges"], out["block_idx"] = rng.astype(np.float32), bidx.astype(np.int32)
    fill_eval(out, "block", [(pose(order[i]), int(src[order[i]])) for i in bidx], render, lamp, grab, own=False)

    # dense cloud -> pipe coordinates with the same transform as the mesh (render_views.load_model)
    counter_csv = os.path.join(a.work, "counter.csv")
    counter = (np.array([float(r["counter_m"]) for r in csv.DictReader(open(counter_csv, encoding="utf-8"))])
               if os.path.exists(counter_csv) else np.zeros(max(frames.values()) + 1))
    pf = ev.pipe_frame(paths.CONVERTER, a.work, None, frames, counter, straighten=False)
    v = read_ply_vertices(os.path.join(a.work, "mvs", "scene_dense.ply"))
    x, th, r = pf["to_pipe"](np.c_[v["x"], v["y"], v["z"]].astype(np.float64))
    keep = r < 1.0
    pts = np.c_[rv.pipe_xyz(x[keep], th[keep], r[keep])].astype(np.float32)
    log(f"dense cloud: {len(v):,} points, {keep.sum():,} within 1 m of the axis")
    out["points"] = out["depth_points"] = pts
    return out, dict(kind="vision", work=a.work, video=a.video, step=step,
                     held_protocol="registered frames held out of training (own SfM pose)" if in_model else
                     "video frames halfway between model frames (interpolated pose), as lampmap")


def fill_eval(out, tag, items, render, lamp, grab, own=True):
    """frames to score: image, next frame (N), pose, B render + mask; items = [(pose, frame id)]"""
    imgs, nxt, views, B, M, ids = [], [], [], [], [], []
    for p, key in items:
        b, m = b_render(render, lamp, p)
        imgs.append(grab(key)), nxt.append(grab(key + 1)), views.append(cv_viewmat(*p)), B.append(b), M.append(m), ids.append(key)
    if own:
        out[f"{tag}_images"] = np.stack(imgs)
    out[f"{tag}_next"] = np.stack(nxt)
    out[f"{tag}_viewmats"] = np.stack(views).astype(np.float32)
    out[f"{tag}_B"], out[f"{tag}_mask"] = np.stack(B), np.stack(M)
    out[f"{tag}_ids"] = np.array(ids, np.int32)
    log(f"{tag}: {len(items)} frames with B renders")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    v = sub.add_parser("vision")
    v.add_argument("work"), v.add_argument("video"), v.add_argument("out"), v.add_argument("--lamp", required=True)
    v.add_argument("--plain", action="store_true"), v.add_argument("--crop", nargs=4, type=int)
    v.add_argument("--stretch-x", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out, meta = export_vision(a)
    path = os.path.join(a.out, "scene.npz")
    np.savez(path, **out)
    meta.update(lamp=a.lamp, train=int(len(out["train_ids"])), held=int(len(out["held_ids"])),
                block_ranges_m=out["block_ranges"].tolist(), block_frames=int(len(out["block_idx"])),
                points=int(len(out["points"])), depth_points=int(len(out["depth_points"])),
                K=out["K"].tolist(), k1=float(out["k1"]), size=out["size"].tolist())
    json.dump(meta, open(os.path.join(a.out, "scene.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"wrote {path} ({os.path.getsize(path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
