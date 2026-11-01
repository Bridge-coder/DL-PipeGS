"""Cross-run test set: frames of a second inspection run of the same pipe, localised in the reference run's SfM model,
put into the reference scene's pipe coordinates, for scoring models trained on the reference run (runs on Windows).

usage: python -I pipesplat/export_xrun.py REF_WORK LOC_DIR QUERY_CSV QUERY_VIDEO OUT_NPZ --lamp LAMP_MAP.npy
                                          [--n 60] [--plain --crop X0 Y0 X1 Y1 --stretch-x S]
       REF_WORK    run_pipeline.py work folder of the reference run (the scene the models were trained on)
       LOC_DIR     output of openMVG_main_SfM_Localization (sfm_data_expanded.json): the query frames localised in
                   REF_WORK/sfm/recon/sfm_data.bin by 2D-3D matching and P3P (Nordberg et al. 2018) inside AC-RANSAC
                   (Moisan et al. 2012), using the reference run's intrinsics (-s)
       QUERY_CSV   query_file,source_frame of the localised images (names differ from the reference images)
       QUERY_VIDEO the second run's video; frames are read as recorded (no CLAHE), with the reference geometry
       OUT_NPZ     xrun_* arrays as export_g0.py writes for its held-out frames: images, next frames (noise
                   reference), OpenCV world-to-camera matrices in the reference scene's pipe coordinates, axial
                   positions, frame ids, and the reference run's B (OpenMVS textured mesh x flat-field lamp map) with
                   its mask, so that train_gs.py --extra-test scores every model on the same pixels
Frames whose localised pose is implausible (camera outside the pipe, viewing direction off the axis, or axial
position far from what the on-screen counter says) are left out and listed in OUT_NPZ's json.
"""
import argparse
import csv
import json
import os
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
import lampmap as lm  # noqa: E402
from video_compare import source_frame_reader  # noqa: E402
from export_g0 import cv_viewmat, fill_eval  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
T0 = time.time()


def log(msg):
    print(f"{msg} [{time.time() - T0:.0f} s]")


def localised_poses(loc_dir):
    """file name -> (rotation world-to-camera 3x3, camera centre) of every view with a pose"""
    d = json.load(open(os.path.join(loc_dir, "sfm_data_expanded.json"), encoding="utf-8"))
    poses = {e["key"]: (np.array(e["value"]["rotation"], float), np.array(e["value"]["center"], float))
             for e in d["extrinsics"]}
    out = {}
    for v in d["views"]:
        x = v["value"]["ptr_wrapper"]["data"]
        if x["id_pose"] in poses:
            out[x["filename"]] = poses[x["id_pose"]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ref_work"), ap.add_argument("loc_dir"), ap.add_argument("query_csv"), ap.add_argument("video")
    ap.add_argument("out"), ap.add_argument("--lamp", required=True), ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--plain", action="store_true"), ap.add_argument("--crop", nargs=4, type=int)
    ap.add_argument("--stretch-x", type=float, default=1.0)
    ap.add_argument("--query-counter", default=None, help="counter.csv of the query run (on-screen distance)")
    ap.add_argument("--blocks-from", default=None,
                    help="scene.npz of the reference run: take every plausible frame inside its block stretches (the "
                         "places with no training frame of the reference run) instead of --n frames over the whole run")
    a = ap.parse_args()

    model = rv.load_model(paths.CONVERTER, a.ref_work, [None], straighten=False)
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(a.ref_work, "frames.csv"), encoding="utf-8"))}
    counter = np.array([float(r["counter_m"]) for r in csv.DictReader(open(os.path.join(a.ref_work, "counter.csv"), encoding="utf-8"))])
    pf = ev.pipe_frame(paths.CONVERTER, a.ref_work, None, frames, counter, straighten=False)
    to_xyz = lambda p: np.stack(rv.pipe_xyz(*pf["to_pipe"](np.atleast_2d(p))), 1)  # noqa: E731

    poses = localised_poses(a.loc_dir)
    qrows = list(csv.DictReader(open(a.query_csv, encoding="utf-8")))
    qsrc = {r["query_file"]: int(r["source_frame"]) for r in qrows}
    found = sorted((qsrc[f], f) for f in poses if f in qsrc)
    log(f"{len(found)} of {len(qrows)} query frames localised in {a.ref_work}")
    step = 0.01 * float(np.ptp(np.array([c for _, c in poses.values()]), axis=0).max())   # model units
    qcount = None
    if a.query_counter:
        qcount = np.array([float(r["counter_m"]) for r in csv.DictReader(open(a.query_counter, encoding="utf-8"))])
    cand = []
    for s, f in found:
        R, C = poses[f]
        eye = to_xyz(C)[0]
        ahead = to_xyz(C + step * R[2])[0]
        above = to_xyz(C - step * R[1])[0]
        fwd = (ahead - eye) / np.linalg.norm(ahead - eye)
        lateral = float(np.hypot(eye[1], eye[2]))
        why = []
        if lateral > 0.10:
            why.append(f"camera {lateral * 1000:.0f} mm off the axis")
        if fwd[0] < np.cos(np.radians(25)):
            why.append(f"view {np.degrees(np.arccos(np.clip(fwd[0], -1, 1))):.0f} deg off the axis")
        if qcount is not None and qcount[s] >= 0.25 and abs(eye[0] - qcount[s]) > 0.25:
            why.append(f"axial {eye[0]:.2f} m vs counter {qcount[s]:.2f} m")
        cand.append(dict(src=s, file=f, eye=eye, ahead=ahead, above=above, lateral=lateral, why=why))
    import cv2
    cap = cv2.VideoCapture(a.video)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    for c in cand:                               # the scoring needs the next video frame (noise reference)
        if c["src"] + 1 >= n_video:
            c["why"].append("last frame of the video (no next frame)")
    good = [c for c in cand if not c["why"]]
    bad = [c for c in cand if c["why"]]
    for c in bad[:20]:
        log(f"  left out frame {c['src']}: {'; '.join(c['why'])}")
    lat = np.array([c["lateral"] for c in good])
    log(f"{len(good)} plausible poses ({len(bad)} left out); camera off the axis median {np.median(lat) * 1000:.1f} mm")
    if qcount is not None:
        dx = np.array([c["eye"][0] - qcount[c["src"]] for c in good if qcount[c["src"]] >= 0.25])
        log(f"axial position minus the query run's own counter: median {np.median(dx) * 100:+.1f} cm, "
            f"MAD {np.median(np.abs(dx - np.median(dx))) * 100:.1f} cm (the two counters need not agree exactly)")
    if a.blocks_from:
        rng = np.load(a.blocks_from)["block_ranges"]
        pick = [c for c in good if any(lo <= c["eye"][0] < hi for lo, hi in rng)]
        log(f"{len(pick)} plausible frames inside the reference blocks {np.round(rng, 3).tolist()} m")
    else:
        pick = [good[i] for i in np.unique(np.linspace(0, len(good) - 1, min(a.n, len(good))).round().astype(int))]

    read = source_frame_reader(a.video, a.plain, a.crop, a.stretch_x)
    intr = model["intrinsics"]
    w, h = int(intr["width"]), int(intr["height"])

    def grab(key):
        fr = read(int(key))
        if fr is None or fr.shape[:2] != (h, w):
            sys.exit(f"video frame {key}: {None if fr is None else fr.shape} - expected {h}x{w}")
        return fr[:, :, ::-1].copy()

    render = lm.Renderer(model, 3.0)
    lamp = np.load(a.lamp)
    out = {}
    fill_eval(out, "xrun", [((c["eye"], c["ahead"], c["above"]), c["src"]) for c in pick], render, lamp, grab)
    out["xrun_pos"] = np.array([c["eye"][0] for c in pick], np.float32)
    out["xrun_lateral"] = np.array([c["lateral"] for c in pick], np.float32)
    np.savez(a.out, **out)
    meta = dict(ref_work=a.ref_work, loc_dir=a.loc_dir, video=a.video, lamp=a.lamp, localised=len(found),
                query=len(qrows), plausible=len(good), scored=len(pick),
                left_out=[dict(frame=c["src"], why=c["why"]) for c in bad],
                axial_range_m=[float(out["xrun_pos"].min()), float(out["xrun_pos"].max())])
    json.dump(meta, open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"wrote {a.out} ({os.path.getsize(a.out) / 1e9:.2f} GB): {len(pick)} frames, "
        f"x {out['xrun_pos'].min():.2f}..{out['xrun_pos'].max():.2f} m")


if __name__ == "__main__":
    main()
