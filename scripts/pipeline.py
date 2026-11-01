"""From one pipe-inspection video to the 3-D model: every stage of the method in order (docs/全流程运行.md).

usage: python -I scripts/pipeline.py NAME VIDEO [options]            run the stages that are not done yet
       python -I scripts/pipeline.py NAME VIDEO [options] --from train --to export
       python -I scripts/pipeline.py --stages                         list the stages
NAME   name of the data set: work folder WORK_NEW/NAME, scene folder GS_ROOT/g0/NAME (scripts/paths.py)
VIDEO  the inspection video: a forward-looking camera moving along the pipe, lamp next to the camera

Stages (a stage whose output exists is skipped, so an interrupted run resumes when started again;
--from STAGE redoes that stage and everything after it)
  frontend  Windows  scripts/vision/run_pipeline.py steps 1-13: frames + CLAHE, OpenMVG SfM with the calibrated
                     intrinsics held fixed, OpenMVS dense cloud, mesh, textured mesh   -> WORK/{frames.csv,sfm,mvs}
  gt        Windows  only with --gt: per-frame travel ground truth -> WORK/counter.csv, then
                     scripts/vision/eval_gt_scale.py (localisation error, scale drift, implied bore) -> WORK/gt_eval*
  scene     Windows  pipesplat/export_g0.py: training frames, held-out frames (44 in-sequence + every frame in two
                     10 cm blocks), poses, dense cloud, baseline B and its scoring mask    -> SCENE/scene.npz
  surface   Windows  pipesplat/export_surface.py, the OpenMVS mesh as triangles (always)   -> SCENE/surface.npz
            WSL      --surface vhf: pipesplat/vision_surface.py, a height field r(u, theta) fitted to the dense cloud
                     (dark, smooth or wet pipes whose mesh has holes), plus the mesh triangles outside the fitted
                     tube (e.g. beyond the pipe's end)                             -> SCENE/surface_vhf.npz
  train     WSL GPU  pipesplat/train_gs.py, the main method: 3DGS + dense depth supervision + near-field lamp model
                     + ambient term + per-frame exposure; scored at the end           -> SCENE/RUN/{model.pt,summary_r2.json}
  export    WSL      pipesplat/export_ply.py                                         -> SCENE/RUN/gaussians.ply
                     pipesplat/vision_surface.py (if not done) + pipesplat/bake_vhf.py --two-light: textured mesh
                                                                                     -> SCENE/RUN_mesh/obj/model.obj
  figures   Windows  pipesplat/review/pipe_figures.py: whole pipe, three views, cut-away, unwrapped wall, held-out
                     frames (video | model | mesh)                                    -> SCENE/RUN_mesh/figures/

Options
  camera   --camera-json FILE (from scripts/vision/calibrate_checkerboard.py: focal, cx, cy, k1, stretch-x; the
           principal point is moved by --crop's X0 Y0)  |  --focal F --cx CX --cy CY --k1 K1 [--stretch-x S]
           |  --camera sheffield (the Sheffield Minicam, the default without the options above)
  frames   --every N (about 1 cm of travel between kept frames)  --first F --last F  (source frame numbers)
           --crop X0 Y0 X1 Y1 (cut burnt-in text)
  SfM      --loose (feature-poor pipes: initial pairs with 40 instead of 60 matches, scripts/vision/pick_pair.py)
           --frontend-steps 7 8 ... (run_pipeline.py steps to run, e.g. to resume after step 5; default 1-13)
  --diameter-m D           inner diameter that sets the metric scale of the vision model (default 0.30)
  --gt CSV [--gt-row ROW]  travel ground truth: a counter.csv (source_frame,counter_m), or with --gt-row the wide table
                           of the Sheffield sewer data set ("Ground truth distances.csv", one row per pipe, e.g. Pipe2)
  --surface mesh|vhf       measured wall for the depth supervision and the lamp model (default mesh)
  --points-every N         start from every Nth dense point (default: 1, or 2 above 2.5 M points; 8 GB GPU)
  --run NAME --iters N --seed S        (run_main, 30000, 0)
  --work DIR --scene DIR   instead of WORK_NEW/NAME and GS_ROOT/g0/NAME
WSL distro and Python: PIPE3D_WSL_DISTRO (default pipe3d), PIPE3D_WSL_PYTHON (default /opt/gsenv/bin/python).
The PC is kept awake while this runs (SetThreadExecutionState); a sleep of the PC still stops a WSL GPU job.
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paths  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
STAGES = ("frontend", "gt", "scene", "surface", "train", "export", "figures")
# the main method (docs/模型说明.md): 3DGS + dense depth supervision (B1) + lamp model (B2) + ambient term (B3);
# --lamp also learns the per-frame exposure (B4)
MAIN = ["--baseline", "3dgs", "--baseline-depth", "1.0", "--lamp", "--lamp-smooth-mm", "20", "--lamp-ambient"]
DISTRO = os.environ.get("PIPE3D_WSL_DISTRO", "pipe3d")
WPY = os.environ.get("PIPE3D_WSL_PYTHON", "/opt/gsenv/bin/python")
PY = [sys.executable, "-I"]


def log(msg):
    print(f"[pipeline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, env=None):
    log("$ " + subprocess.list2cmdline(cmd))
    code = subprocess.call(cmd, env=env)
    if code:
        sys.exit(f"[pipeline] failed with exit code {code}")


def wsl_path(p):
    """a Windows path as the WSL distro sees it (D:/x -> /mnt/d/x)"""
    p = os.path.abspath(p).replace("\\", "/")
    return f"/mnt/{p[0].lower()}{p[2:]}" if len(p) > 1 and p[1] == ":" else p


def wsl(script, *args):
    """a pipesplat script in the WSL distro, started without a shell (no $-expansion of the arguments)"""
    for k in range(6):                            # WSL can refuse to start while memory is short (HCS 0x800705aa)
        if subprocess.call(["wsl", "-d", DISTRO, "-u", "root", "-e", "true"]) == 0:
            break
        log(f"WSL distro {DISTRO} did not start (try {k + 1} of 6); waiting 30 s")
        time.sleep(30)
    else:
        sys.exit(f"[pipeline] WSL distro {DISTRO} does not start (wsl -l -v; close programs to free memory)")
    run(["wsl", "-d", DISTRO, "-u", "root", "-e", "env", f"TORCH_HOME={wsl_path(paths.TORCH_CACHE)}", WPY,
         wsl_path(os.path.join(REPO, script))] + [str(x) for x in args])


def keep_awake():
    if os.name == "nt":                           # ES_CONTINUOUS | ES_SYSTEM_REQUIRED, until this process ends
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)


def camera_options(a):
    """run_pipeline.py options of the camera and the frame geometry (also needed to read the video frames again)"""
    if a.camera_json:
        cam = json.load(open(a.camera_json, encoding="utf-8"))
        s = a.stretch_x or float(cam.get("stretch_x", 1.0))
        x0, y0 = (a.crop[0], a.crop[1]) if a.crop else (0, 0)
        f, cx, cy, k1 = cam["focal"], cam["cx"] - x0 * s, cam["cy"] - y0, cam["k1"]
    elif a.focal:
        f, cx, cy, k1, s = a.focal, a.cx, a.cy, a.k1, a.stretch_x or 1.0
    else:
        return ["--camera", "sheffield"], []
    geo = ["--plain"] + (["--crop"] + [str(v) for v in a.crop] if a.crop else []) + ["--stretch-x", f"{s:.6f}"]
    return geo + ["--focal", f"{f:.4f}", "--cx", f"{cx:.4f}", "--cy", f"{cy:.4f}", "--k1", f"{k1:.6f}"], geo


def done(path, a, stage):
    """skip a stage whose output exists, unless --from asked for it to be redone"""
    if a.redo is not None and STAGES.index(stage) >= STAGES.index(a.redo):
        return False
    if os.path.exists(path):
        log(f"{stage}: {path} exists - skipped (--from {stage} redoes it)")
        return True
    return False


# ------------------------------------------------------------------------------------------- stages --------
def frontend(a, env):
    if done(os.path.join(a.work, "mvs", "scene_textured.obj"), a, "frontend"):
        return
    opts, _ = camera_options(a)
    frames = ["--every", str(a.every)] + (["--first", str(a.first)] if a.first is not None else []) \
        + (["--last", str(a.last)] if a.last is not None else [])
    run(PY + [os.path.join(HERE, "vision", "run_pipeline.py"), a.work, "--video", a.video] + frames + opts
        + (["--loose"] if a.loose else []) + [str(s) for s in a.frontend_steps], env)


def gt(a, env):
    if not a.gt:
        log("gt: no --gt given - skipped (the scene's axial origin is then the first camera, not the counter)")
        return
    out = os.path.join(a.work, "counter.csv")
    if not done(os.path.join(a.work, "gt_eval.json"), a, "gt"):
        if a.gt_row:                              # wide table: header ",Frame,1,2,..."; row "<text>,<ROW>,v1,v2,..."
            rows = list(csv.reader(open(a.gt, encoding="utf-8-sig")))
            row = next((r for r in rows[1:] if len(r) > 1 and r[1] == a.gt_row), None)
            if row is None:
                sys.exit(f"[pipeline] no row {a.gt_row} in {a.gt}; rows: {[r[1] for r in rows[1:] if len(r) > 1]}")
            vals = [v for v in row[2:] if v.strip() != ""]
            with open(out, "w", newline="", encoding="utf-8") as fh:   # column k (frame k + 1) = source frame k
                fh.write("source_frame,counter_m\n")
                fh.writelines(f"{k},{float(v):.2f}\n" for k, v in enumerate(vals))
            log(f"gt: {len(vals)} frames of {a.gt_row} -> {out}")
        elif os.path.abspath(a.gt) != os.path.abspath(out):
            shutil.copyfile(a.gt, out)
        run(PY + [os.path.join(HERE, "vision", "eval_gt_scale.py"), a.work, os.path.join(a.work, "gt_eval")], env)


def scene(a, env):
    if done(os.path.join(a.scene, "scene.npz"), a, "scene"):
        return
    _, geo = camera_options(a)
    run(PY + [os.path.join(REPO, "pipesplat", "export_g0.py"), "vision", a.work, a.video, a.scene, "--lamp", "auto"]
        + geo, env)


def surface(a, env):
    if not done(os.path.join(a.scene, "surface.npz"), a, "surface"):      # vision_surface.py builds on it too
        run(PY + [os.path.join(REPO, "pipesplat", "export_surface.py"), "vision", a.work, a.scene], env)
    if a.surface == "vhf" and not done(os.path.join(a.scene, "surface_vhf.npz"), a, "surface"):
        wsl("pipesplat/vision_surface.py", wsl_path(a.scene), *a.vhf_args)


def train(a, env):
    if done(os.path.join(a.run_dir, "summary_r2.json"), a, "train"):
        return
    every = a.points_every
    if every is None:                             # the dense cloud's size decides (8 GB GPU memory)
        n = json.load(open(os.path.join(a.scene, "scene.json"), encoding="utf-8"))["points"]
        every = max(1, math.ceil(n / 2.5e6))
        log(f"train: {n:,} dense points -> --points-every {every}")
    surf = [] if a.surface == "mesh" else ["--surface", "surface_vhf.npz"]
    wsl("pipesplat/train_gs.py", wsl_path(a.scene), wsl_path(a.run_dir), *MAIN, *surf, "--points-every", every,
        "--iters", a.iters, "--seed", a.seed)
    s = json.load(open(os.path.join(a.run_dir, "summary_r2.json"), encoding="utf-8"))
    for key, name in (("held_outside_blocks", "in-sequence held-out"), ("block", "10 cm blocks left out")):
        m = s[key]
        log(f"train: {name}: PSNR model {m['G_psnr']:.2f} dB (baseline B {m['B_psnr']:.2f}, next frame aligned "
            f"{m['Nal_psnr']:.2f}), LPIPS {m['G_lpips']:.3f}, depth error {m['depth_rel_err']:.1%}")


def export(a, env):
    if not done(os.path.join(a.run_dir, "gaussians.ply"), a, "export"):
        wsl("pipesplat/export_ply.py", wsl_path(a.run_dir))
    if not os.path.exists(os.path.join(a.scene, "mesh_vhf.npz")):
        wsl("pipesplat/vision_surface.py", wsl_path(a.scene), *a.vhf_args)
    if not done(os.path.join(a.mesh_dir, "obj", "model.obj"), a, "export"):
        wsl("pipesplat/bake_vhf.py", wsl_path(a.scene), wsl_path(a.run_dir), wsl_path(a.mesh_dir), "--two-light",
            "--source", "video", *a.bake_args)


def figures(a, env):
    if done(os.path.join(a.mesh_dir, "figures", "fig1_whole.jpg"), a, "figures"):
        return
    run(PY + [os.path.join(REPO, "pipesplat", "review", "pipe_figures.py"), a.scene, os.path.join(a.mesh_dir, "obj"),
              os.path.join(a.mesh_dir, "figures"), "--title", a.name, "--model-run", a.run_dir], env)


def main():
    if "--stages" in sys.argv:
        print(__doc__[__doc__.index("Stages"):__doc__.index("Options")])
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name"), ap.add_argument("video")
    ap.add_argument("--camera-json"), ap.add_argument("--camera", default=None, choices=("sheffield",))
    ap.add_argument("--focal", type=float), ap.add_argument("--cx", type=float), ap.add_argument("--cy", type=float)
    ap.add_argument("--k1", type=float), ap.add_argument("--stretch-x", type=float, default=None)
    ap.add_argument("--every", type=int, default=5), ap.add_argument("--first", type=int), ap.add_argument("--last", type=int)
    ap.add_argument("--crop", type=int, nargs=4)
    ap.add_argument("--loose", action="store_true")
    ap.add_argument("--frontend-steps", type=int, nargs="+", default=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13])
    ap.add_argument("--diameter-m", type=float, default=0.30)
    ap.add_argument("--gt"), ap.add_argument("--gt-row")
    ap.add_argument("--surface", choices=("mesh", "vhf"), default="mesh")
    ap.add_argument("--vhf-args", nargs="*", default=[], help="extra vision_surface.py options, e.g. --rounds 6")
    ap.add_argument("--bake-args", nargs="*", default=[], help="extra bake_vhf.py options, e.g. --texel-mm 1.0")
    ap.add_argument("--points-every", type=int, default=None)
    ap.add_argument("--run", default="run_main"), ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work"), ap.add_argument("--scene")
    ap.add_argument("--from", dest="redo", choices=STAGES), ap.add_argument("--to", choices=STAGES, default=STAGES[-1])
    a = ap.parse_args()
    if a.camera_json and a.focal:
        sys.exit("[pipeline] give either --camera-json or --focal/--cx/--cy/--k1")
    if a.focal and None in (a.cx, a.cy, a.k1):
        sys.exit("[pipeline] --focal needs --cx, --cy and --k1 (pixels of the cropped and stretched frames)")
    a.video = os.path.abspath(a.video)
    a.work = os.path.abspath(a.work or os.path.join(paths.WORK_NEW, a.name, "work"))
    a.scene = os.path.abspath(a.scene or os.path.join(paths.GS_ROOT, "g0", a.name))
    a.run_dir = os.path.join(a.scene, a.run)
    a.mesh_dir = os.path.join(a.scene, a.run + "_mesh")
    env = dict(os.environ, PIPE3D_PIPE_DIAMETER_M=str(a.diameter_m))
    keep_awake()
    log(f"{a.name}: video {a.video}\n  work {a.work}\n  scene {a.scene}\n  run {a.run_dir}")
    start = STAGES.index(a.redo) if a.redo else 0
    for stage in STAGES[start:STAGES.index(a.to) + 1]:
        t0 = time.time()
        log(f"=== {stage}")
        globals()[stage](a, env)
        log(f"=== {stage} done ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
