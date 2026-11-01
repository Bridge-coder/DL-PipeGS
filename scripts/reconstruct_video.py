"""Front end only (no Gaussian model): reconstruction, 3-D pictures of the OpenMVS mesh and the side-by-side video.
The whole method (front end + Gaussian model) is scripts/pipeline.py.

usage: python -I scripts/reconstruct_video.py VIDEO WORK_DIR [--every N] [--first F] [--last F]
                                              [--camera sheffield | --focal F --cx CX --cy CY --k1 K1 --plain]
                                              [--crop X0 Y0 X1 Y1] [--stretch-x S]
                                              [--skip-recon] [--video-every N] [--straighten]
  1. vision/run_pipeline.py WORK_DIR --video VIDEO ... 1 2 3 4 5 7 8 9 10 11 12 13
  2. vision/render_views.py  -> WORK_DIR/views/ (outside, cut-away, side elevation, views from inside + frames)
  3. vision/video_compare.py -> WORK_DIR/compare.mp4 (input frame | model from the same camera | deviation | position)
--plain/--crop/--stretch-x also go to step 3 (same frame geometry); --straighten goes to steps 2 and 3 only
(slice-by-slice pipe frame for long real sewers that bend, see render_views.py).
Ground truth (optional): put WORK_DIR/counter.csv (vision/evaluate.py counter ...) before step 2/3 for travel numbers,
and run vision/evaluate.py sparse / dense for the full evaluation (docs/PIPELINE.md).
If step 8 registers less than 90 % of the frames, the result covers only part of the video: see docs/PIPELINE.md 2.6.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

args = sys.argv[1:]
skip = "--skip-recon" in args
if skip:
    args.remove("--skip-recon")
video_every = "1"
if "--video-every" in args:
    i = args.index("--video-every")
    video_every = args[i + 1]
    del args[i:i + 2]
straighten = ["--straighten"] if "--straighten" in args else []
if straighten:
    args.remove("--straighten")
video, work, extra = os.path.abspath(args[0]), os.path.abspath(args[1]), args[2:]
geometry = ["--plain"] if "--plain" in extra else []          # frame geometry of step 2, for the left video panel
for flag, n in (("--crop", 4), ("--stretch-x", 1)):
    if flag in extra:
        i = extra.index(flag)
        geometry += extra[i:i + 1 + n]
PY = [sys.executable, "-I"]


def run(cmd):
    print("$ " + subprocess.list2cmdline(cmd), flush=True)
    code = subprocess.call(cmd)
    if code:
        sys.exit(f"failed with exit code {code}")


if not skip:
    run(PY + [os.path.join(HERE, "vision", "run_pipeline.py"), work, "--video", video] + extra
        + [str(s) for s in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13)])
run(PY + [os.path.join(HERE, "vision", "render_views.py"), paths.CONVERTER, work, os.path.join(work, "views")]
    + straighten)
run(PY + [os.path.join(HERE, "vision", "video_compare.py"), work, os.path.join(work, "compare.mp4"),
          "--video", video, "--every", video_every] + geometry + straighten)
print(f"done: model {os.path.join(work, 'mvs', 'scene_textured.obj')}, views {os.path.join(work, 'views')}, "
      f"video {os.path.join(work, 'compare.mp4')}")
