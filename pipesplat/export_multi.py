"""Multi-session training scene: the reference run's scene (export_g0.py) plus the frames of other inspection runs of
the same pipe as additional training frames, all in the reference scene's pipe coordinates (runs on Windows).

usage: python -I pipesplat/export_multi.py REF_SCENE_DIR OUT_SCENE_DIR --add NAME ALIGN_DIR QUERY_CSV VIDEO
                                           [--add ...] [--only-added] [--every K]
       REF_SCENE_DIR  scene of the reference run (scene.npz, scene.json naming its work folder, surface.npz)
       --add          one more run: its frames' poses from ALIGN_DIR/sfm_data_expanded.json (align_runs.py: joint SfM
                      of both runs, moved into the reference reconstruction by local similarities), QUERY_CSV
                      (query_file,source_frame) and its VIDEO (frames read as recorded, reference frame geometry)
       --only-added   train on the added runs only (the reference run's frames are dropped from training; its
                      in-sequence and block frames stay as test frames - then of another run)
       --every K      use every K-th frame of an added run (default 1 = all registered)
Frames of an added run inside the reference scene's two block stretches are not added (the blocks stay unseen).
OUT_SCENE_DIR gets scene.npz with train_run (0 = reference, 1.. = added runs in --add order), scene.json, and a copy
of surface.npz; the measured surface and the starting points stay the reference run's.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (os.path.join(SCRIPTS, "vision"), SCRIPTS, os.path.join(ROOT, "lampmap"), HERE):
    sys.path.insert(0, p)
import paths  # noqa: E402
import render_views as rv  # noqa: E402
import evaluate as ev  # noqa: E402
from video_compare import source_frame_reader  # noqa: E402
from export_g0 import cv_viewmat, in_blocks  # noqa: E402
from export_xrun import localised_poses  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
T0 = time.time()


def log(msg):
    print(f"{msg} [{time.time() - T0:.0f} s]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ref"), ap.add_argument("out")
    ap.add_argument("--add", nargs=4, action="append", required=True, metavar=("NAME", "ALIGN_DIR", "QUERY_CSV", "VIDEO"))
    ap.add_argument("--only-added", action="store_true"), ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--ref-every", type=int, default=1,
                    help="train on every K-th reference frame only (the others go to train_drop): with --every K as well, "
                         "a multi-session scene with about as many training frames as the reference run alone")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = dict(np.load(os.path.join(a.ref, "scene.npz")))
    meta = json.load(open(os.path.join(a.ref, "scene.json"), encoding="utf-8"))
    work = meta["work"]
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(work, "frames.csv"), encoding="utf-8"))}
    counter = np.array([float(r["counter_m"]) for r in csv.DictReader(open(os.path.join(work, "counter.csv"), encoding="utf-8"))])
    pf = ev.pipe_frame(paths.CONVERTER, work, None, frames, counter, straighten=False)
    to_xyz = lambda p: np.stack(rv.pipe_xyz(*pf["to_pipe"](np.atleast_2d(p))), 1)  # noqa: E731
    w, h = (int(v) for v in d["size"])
    n_ref = len(d["train_ids"])
    d["train_run"] = np.zeros(n_ref, np.int16)
    added = []
    for k, (name, align_dir, qcsv, video) in enumerate(a.add, 1):
        poses = localised_poses(align_dir)
        q = {r["query_file"]: int(r["source_frame"]) for r in csv.DictReader(open(qcsv, encoding="utf-8"))}
        items = sorted((q[f], f) for f in poses if f in q)[::a.every]
        step = 0.01 * float(np.ptp(np.array([c for _, c in poses.values()]), axis=0).max())
        read = source_frame_reader(video)
        ids, views, eyes, imgs = [], [], [], []
        skipped = 0
        for s, f in items:
            R, C = poses[f]
            eye, ahead, above = to_xyz(C)[0], to_xyz(C + step * R[2])[0], to_xyz(C - step * R[1])[0]
            if in_blocks([eye[0]], d["block_ranges"])[0]:
                skipped += 1
                continue
            fr = read(s)
            if fr is None or fr.shape[:2] != (h, w):
                sys.exit(f"{video} frame {s}: {None if fr is None else fr.shape}, expected {h}x{w}")
            ids.append(s), views.append(cv_viewmat(eye, ahead, above)), eyes.append(eye), imgs.append(fr[:, :, ::-1])
        log(f"run {name}: {len(ids)} frames added ({skipped} inside the block stretches left out)")
        added.append(dict(name=name, n=len(ids), video=video, align=align_dir))
        d["train_ids"] = np.r_[d["train_ids"], np.array(ids, np.int32)]
        d["train_viewmats"] = np.r_[d["train_viewmats"], np.array(views, np.float32)]
        d["train_eyes"] = np.r_[d["train_eyes"], np.array(eyes, np.float32)]
        d["train_pos"] = np.r_[d["train_pos"], np.array([e[0] for e in eyes], np.float32)]
        d["train_t"] = np.r_[d["train_t"], np.zeros(len(ids), np.int64)]
        d["train_images"] = np.concatenate([d["train_images"], np.stack(imgs)])
        d["train_run"] = np.r_[d["train_run"], np.full(len(ids), k, np.int16)]
    if a.only_added:
        d["train_drop"] = np.arange(n_ref, dtype=np.int32)      # train_gs.py leaves these out of training
    elif a.ref_every > 1:                                         # every K-th reference frame by frame number
        order = np.argsort(d["train_ids"][:n_ref])
        drop = np.setdiff1d(np.arange(n_ref), order[::a.ref_every])
        d["train_drop"] = np.setdiff1d(drop, d["block_idx"]).astype(np.int32)  # block frames are left out anyway
    np.savez(os.path.join(a.out, "scene.npz"), **d)
    shutil.copyfile(os.path.join(a.ref, "surface.npz"), os.path.join(a.out, "surface.npz"))
    n_drop = len(d["train_drop"]) if "train_drop" in d else 0
    meta.update(multi=dict(reference=a.ref, added=added, only_added=a.only_added, every=a.every, ref_every=a.ref_every),
                train=int(len(d["train_ids"]) - n_drop))
    json.dump(meta, open(os.path.join(a.out, "scene.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"wrote {a.out}/scene.npz: {len(d['train_ids'])} frames ({n_ref} reference"
        f"{', not trained on' if a.only_added else ''} + {len(d['train_ids']) - n_ref} added)")


if __name__ == "__main__":
    main()
