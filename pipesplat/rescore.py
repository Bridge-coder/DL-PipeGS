"""Re-score trained runs with ruler v2 (train_gs.py --eval-only), each with the model flags it was trained with (read
from its own summary), one after another, and print the old and the new numbers side by side: the check after a
restore from GitHub or after a code change (docs/PIPELINE.md, section 5). Runs in WSL.

usage: /opt/gsenv/bin/python pipesplat/rescore.py DATA/RUN [DATA/RUN ...] [--suffix _rescore] [--dump]
       DATA/RUN under GS_ROOT/g0 (scripts/paths.py); writes RUN/summary<suffix>.json and per_frame<suffix>.csv;
       --dump also writes every scored frame as JPEGs to GS_ROOT/review/video/DATA/RUN (review/compare_video.py)
"""
import json
import os
import subprocess
import sys

PS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(PS), "scripts"))
import paths  # noqa: E402

G = os.path.join(paths.GS_ROOT, "g0")
FLAGS = ("anchor", "lamp", "lamp_ambient", "lamp_gamma", "lamp_colocated", "lamp_minnaert", "exposure")  # renderer switches
# baseline_depth does not change the render, but train_gs.py needs it with --baseline 3dgs --lamp (the main method)
VALUES = ("lamp_smooth_mm", "surface", "points_every", "baseline", "baseline_depth", "thin_mm", "sh_degree", "lamp_radial",
          "lamp_gamma_fixed")
PARTS = (("held_outside_blocks", "in-seq"), ("block", "block"))


def model_flags(run_dir):
    """the train_gs.py options of a trained run that change what it renders, and the summary they came from"""
    summ = next(os.path.join(run_dir, f) for f in ("summary_r2.json", "summary.json") if os.path.exists(os.path.join(run_dir, f)))
    args = json.load(open(summ))["args"]
    out = [f"--{k.replace('_', '-')}" for k in FLAGS if args.get(k)]
    for k in VALUES:
        if args.get(k) is not None and not (k == "points_every" and args[k] == 1) and not (k == "baseline_depth" and not args[k]):
            out += [f"--{k.replace('_', '-')}", str(args[k])]
    return out


def score(ds, run, suffix, dump=None):
    """train_gs.py --eval-only --ruler2 of G/ds/run; returns (exit code, stderr tail)"""
    rd = os.path.join(G, ds, run)
    # --alpha-min 0.5: the opacity threshold of the ruler-v2 scoring at the end of training (train_gs.py); it only
    # changes the coverage and depth-error columns, PSNR / LPIPS / bands are scored on B's mask
    cmd = [sys.executable, os.path.join(PS, "train_gs.py"), os.path.join(G, ds), rd, "--eval-only", "--ruler2",
           "--alpha-min", "0.5", "--suffix", suffix] + (["--dump", dump] if dump else []) + model_flags(rd)
    print(" ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                       env=dict(os.environ, TORCH_HOME=os.environ.get("TORCH_HOME", paths.TORCH_CACHE)))
    return r.returncode, r.stderr[-1500:]


if __name__ == "__main__":
    argv = sys.argv[1:]
    suffix = argv[argv.index("--suffix") + 1] if "--suffix" in argv else "_rescore"
    items = [x for i, x in enumerate(argv) if not x.startswith("--") and (i == 0 or argv[i - 1] != "--suffix")]
    for item in items:
        ds, run = item.split("/")
        rd = os.path.join(G, ds, run)
        code, err = score(ds, run, suffix, os.path.join(paths.GS_ROOT, "review", "video", ds, run) if "--dump" in argv else None)
        if code:
            print(f"  {item}: exit {code}\n{err}", flush=True)
            continue
        old, new = (json.load(open(os.path.join(rd, f))) for f in ("summary_r2.json", f"summary{suffix}.json"))
        for part, label in PARTS:
            print(f"  {item} {label:6s} PSNR model {old[part]['G_psnr']:.3f} -> {new[part]['G_psnr']:.3f} "
                  f"({new[part]['G_psnr'] - old[part]['G_psnr']:+.3f})   B {old[part]['B_psnr']:.3f} -> {new[part]['B_psnr']:.3f}  "
                  f"LPIPS {old[part]['G_lpips']:.3f} -> {new[part]['G_lpips']:.3f}", flush=True)
