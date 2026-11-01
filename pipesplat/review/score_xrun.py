"""Score trained runs on frames of other inspection runs (export_xrun.py), each with the model flags it was trained
with, and print one table per test set. Runs in WSL.

usage: $PY pipesplat/review/score_xrun.py DATA RUN [RUN ...] --test NAME=FILE [--test NAME=FILE ...] [--suffix _xrun]
                                          [--fit-half] [--test-pose-iters N] [--with-held] [--table-only]
       DATA/RUN under GS_ROOT/g0; FILE relative to GS_ROOT/g0/DATA (or absolute). Writes RUN/summary<suffix>.json.
       --fit-half         gain/offset fitted on the left half, metrics on the right half (NeRF-W protocol)
       --test-pose-iters  register each test frame to the model first (optical-flow pose refinement, model frozen)
       --with-held        also re-score the in-sequence and block frames (default: only the other runs' frames)
       --table-only       do not score, print the tables from existing summary<suffix>.json files
"""
import json
import os
import subprocess
import sys

PS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PS)
sys.path.insert(0, os.path.join(os.path.dirname(PS), "scripts"))
import paths  # noqa: E402
from rescore import model_flags  # noqa: E402

G = os.path.join(paths.GS_ROOT, "g0")
COLS = ("psnr", "ssim", "lpips", "shift_px", "fid_fine", "fid_mid", "fid_coarse")


def main():
    argv = sys.argv[1:]
    tests = []
    while "--test" in argv:
        i = argv.index("--test")
        tests.append(argv[i + 1])
        del argv[i:i + 2]

    def opt(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default
    suffix, tpi = opt("--suffix", "_xrun"), opt("--test-pose-iters")
    switches = {s: s in argv for s in ("--fit-half", "--with-held", "--table-only")}
    argv = [x for x in argv if x not in switches]
    ds, runs = argv[0], argv[1:]
    done = []
    for run in runs:
        rd = os.path.join(G, ds, run)
        if not switches["--table-only"]:
            cmd = [sys.executable, os.path.join(PS, "train_gs.py"), os.path.join(G, ds), rd, "--eval-only", "--ruler2",
                   "--alpha-min", "0.5", "--suffix", suffix] + model_flags(rd)
            cmd += ([] if switches["--with-held"] else ["--no-held"]) + (["--fit-half"] if switches["--fit-half"] else [])
            cmd += ["--test-pose-iters", tpi] if tpi else []
            cmd += ["--extra-test"] + tests
            print(" ".join(cmd[2:]), flush=True)
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                               env=dict(os.environ, TORCH_HOME=os.environ.get("TORCH_HOME", paths.TORCH_CACHE)))
            if r.returncode:
                print(f"  {run}: exit {r.returncode}\n{r.stderr[-2000:]}", flush=True)
                continue
        f = os.path.join(rd, f"summary{suffix}.json")
        if os.path.exists(f):
            done.append((run, json.load(open(f))))
    for t in tests:
        name = t.split("=", 1)[0]
        print(f"\n{ds}, frames of {name} (median; G = model, B = OpenMVS mesh x flat-field, Nal = aligned next frame)")
        print("run".ljust(22) + "".join(f"{c:>10s}" for c in COLS) + "   depth  | in blocks: PSNR")
        for k, (run, s) in enumerate(done):
            if name not in s:
                continue
            x, xb, n = s[name], s.get(name + "_in_blocks", {}), s["n"]
            for tag in ("G", "B", "Nal") if k == 0 else ("G",):
                label = run if tag == "G" else f"  ({tag})"
                line = label.ljust(22) + "".join(f"{x.get(f'{tag}_{c}', float('nan')):10.3f}" for c in COLS)
                line += f"  {x['depth_rel_err']:6.1%}" if tag == "G" else "         "
                if xb:
                    line += f" | {xb.get(f'{tag}_psnr', float('nan')):.2f} ({n.get(name + '_in_blocks', 0)})"
                print(line, flush=True)


if __name__ == "__main__":
    main()
