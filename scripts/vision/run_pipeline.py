"""Vision reconstruction of a pipe-inspection VIDEO with OpenMVG + OpenMVS, in numbered steps.

Usage: run_pipeline.py WORK_DIR --video VIDEO [--every N] [--first FRAME] [--last FRAME]
                       [--camera sheffield | --focal F --cx CX --cy CY --k1 K1 --plain] STEP [STEP ...]
       --video  the inspection video (mp4/avi)
       --every  keep one video frame out of N in step 2 (default 5; choose N so that the robot moves about
                1 cm between kept frames: N = 0.01 m / (speed in m/s / fps))
       --first/--last  source frame range to use in step 2 (e.g. drop the frames after the robot stops)
       --camera sheffield (default): the Minicam camera of the Sheffield ORDA videos, calibrated on them
                (768x548 frames after extract_frames.py; f 806, principal point 383/288, k1 -0.33)
       --focal/--cx/--cy/--k1: any other camera, in pixels of the frames written by step 2 (calibrate it first,
                see docs/PIPELINE.md 2.3); --plain: frames are not PAL-stretched or cropped (any non-Sheffield video)
       --crop X0 Y0 X1 Y1 / --stretch-x S   (with --plain) passed to extract_frames.py: cut burnt-in text, make pixels
                square; the intrinsics must be given in the pixels after both (calibrate_checkerboard.py prints them)
       --part NAME --views FIRST LAST
                reconstruct one stretch of the video on its own (when SfM cannot carry on across a jolt):
                step 7 only ranks initial pairs inside that range of view ids, and steps 8-13 write to
                sfm/recon_NAME and mvs_NAME. Steps 2-5 are shared by all parts.
       --loose  step 7: accept initial pairs with fewer matches (feature-poor pipes, see pick_pair.py)

Steps
   1 pre-flight: start each program once with -h
   2 extract frames from the video (extract_frames.py, contrast enhancement on)
   3 openMVG_main_SfMInit_ImageListing, then set_distortion.py (known radial distortion)
   4 openMVG_main_ComputeFeatures
   5 openMVG_main_ComputeMatches   (pairs + matching + geometric filter)
   7 pick_pair.py: rank candidate initial image pairs
   8 openMVG_main_IncrementalSfM, from the candidates of step 7 in turn until most frames are registered
   9 openMVG_main_ComputeSfM_DataColor
  10 clean_sfm.py (drop degenerate 3-D points), openMVG_main_openMVG2openMVS, then fix_mvs_k.py
     (principal point in the form OpenMVS 2.4.0 expects)
  11 DensifyPointCloud   12 ReconstructMesh   13 TextureMesh

Program locations come from scripts/paths.py (config/paths.json). Full output of every step is written to
LOGS/<name of WORK_DIR>/ (paths.py); only the tail is echoed.

OpenMVG 1.6 is used instead of 2.0: the 2.0 Windows binaries were built with AVX-512 and crash with an
illegal-instruction error on CPUs without it (e.g. Ryzen 5 5600, AVX2 only). The algorithms are the same; 1.6
does pair selection, matching and geometric filtering in one program and names the SfM program differently.
"""
import os
import re
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

args = sys.argv[1:]


def take_option(flag, default=None):
    """Remove `flag VALUE` from the argument list and return VALUE."""
    if flag not in args:
        return default
    i = args.index(flag)
    value = args[i + 1]
    del args[i:i + 2]
    return value


video = take_option("--video")
video = os.path.abspath(video) if video else None
every = take_option("--every", "5")     # keep one video frame out of this many
first = take_option("--first")          # first source frame to use
last = take_option("--last")            # last source frame to use (e.g. where the robot stops)
camera = take_option("--camera", "sheffield")
focal_opt, cx_opt, cy_opt, k1_opt = (take_option(f) for f in ("--focal", "--cx", "--cy", "--k1"))
plain = "--plain" in args
if plain:
    args.remove("--plain")
crop = None
if "--crop" in args:
    k = args.index("--crop")
    crop = args[k + 1:k + 5]
    del args[k:k + 5]
stretch_x = take_option("--stretch-x")
loose = "--loose" in args
if loose:
    args.remove("--loose")
part = take_option("--part")            # name of a separately reconstructed stretch of the video
views = None                            # view-id range of that stretch
if "--views" in args:
    k = args.index("--views")
    views = (int(args[k + 1]), int(args[k + 2]))
    del args[k:k + 3]
work = os.path.abspath(args[0])
steps = [int(s) for s in args[1:]]
suffix = f"_{part}" if part else ""

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPTS))
import paths  # noqa: E402

MVG, MVS = paths.MVG_BIN, paths.MVS_BIN
VIDEO = video
if VIDEO is None and 2 in steps:
    sys.exit("step 2 needs --video VIDEO")
LOGS = os.path.join(paths.LOGS, os.path.basename(work), *([part] if part else []))

# Camera of the Sheffield Minicam videos after extract_frames.py (768x548 frames). The dataset has no
# calibration; these values come from vision/calibrate.py on video CLOSED_1 (docs/history/标定与检验.md):
# K1 is the radial distortion that makes the reconstructed pipe keep a constant diameter, FOCAL the focal
# length that makes the travelled distance agree with the on-screen counter of that video. The principal
# point is assumed (image centre before the bottom strip was cropped).
CAMERAS = {"sheffield": (806.0, 383.0, 288.0, -0.33)}
if camera not in CAMERAS:
    sys.exit(f"unknown --camera {camera}; known: {', '.join(CAMERAS)}, or give --focal --cx --cy --k1")
FOCAL, CX, CY, K1 = CAMERAS[camera]
if focal_opt:                                   # a camera of its own: all four values must be given
    if None in (cx_opt, cy_opt, k1_opt):
        sys.exit("--focal needs --cx, --cy and --k1 as well (in pixels of the extracted frames)")
    FOCAL, CX, CY, K1 = float(focal_opt), float(cx_opt), float(cy_opt), float(k1_opt)

images = os.path.join(work, "images")
matches = os.path.join(work, "sfm", "matches")
recon = os.path.join(work, "sfm", "recon" + suffix)
mvs = os.path.join(work, "mvs" + suffix)
sfm_json = os.path.join(matches, "sfm_data.json")
pair_file = os.path.join(work, "sfm", f"init_pairs{suffix}.txt")


def mvg(name):
    return os.path.join(MVG, f"openMVG_main_{name}.exe")


def mvs_exe(name):
    return os.path.join(MVS, f"{name}.exe")


# Settings found in the parameter exploration (variant E8, see results/exploration/variants.csv):
#  - every 5th frame (about 5 per second) with contrast enhancement, SIFT preset ULTRA
#  - each frame matched with the 25 frames that follow it (-v 25); -g f: fundamental-matrix RANSAC filter
#  - camera intrinsics held fixed (-f NONE): the camera only moves forward, so the images cannot determine
#    the focal length and letting it float makes SfM diverge
#  - camera model 2 (pinhole + one radial distortion coefficient), the coefficient written by set_distortion.py
COMMANDS = {
    2: ("extract_frames", [sys.executable, "-I", os.path.join(SCRIPTS, "extract_frames.py"), str(VIDEO), images,
                           "--every", every, "--clahe"] + (["--first", first] if first else [])
        + (["--last", last] if last else []) + (["--plain"] if plain else [])
        + (["--crop"] + crop if crop else []) + (["--stretch-x", stretch_x] if stretch_x else [])),
    3: ("ImageListing", [mvg("SfMInit_ImageListing"), "-i", images, "-o", matches, "-c", "2",
                         "-k", f"{FOCAL:g};0;{CX:g};0;{FOCAL:g};{CY:g};0;0;1"]),
    4: ("ComputeFeatures", [mvg("ComputeFeatures"), "-i", sfm_json, "-o", matches, "-m", "SIFT", "-p", "ULTRA"]),
    5: ("ComputeMatches", [mvg("ComputeMatches"), "-i", sfm_json, "-o", matches, "-g", "f", "-v", "25"]),
    # Left to choose its initial pair by itself, OpenMVG sometimes starts at the flat end cap and then fails,
    # depending on small random differences in the matching step; step 8 passes the pairs ranked here (-a/-b).
    7: ("pick_pair", [sys.executable, "-I", os.path.join(SCRIPTS, "pick_pair.py"), matches, pair_file]
        + (["--views", str(views[0]), str(views[1])] if views else []) + (["--loose"] if loose else [])),
    9: ("ComputeSfM_DataColor", [mvg("ComputeSfM_DataColor"), "-i", os.path.join(recon, "sfm_data.bin"),
                                 "-o", os.path.join(recon, "sparse_colorized.ply")]),
    10: ("openMVG2openMVS", [mvg("openMVG2openMVS"), "-i", os.path.join(recon, "sfm_data_clean.json"),
                             "-o", os.path.join(mvs, "scene_openmvg.mvs"), "-d", os.path.join(mvs, "images")]),
    # --dense-config-file: scripts/Densify.ini is OpenMVS's default dense configuration with three changes
    # (Optim Angle 12 -> 3, Min Angle 3 -> 0.5, View Min Score 2 -> 0). With the defaults 388 frames were
    # "paired with 0 views" and only 13 depth maps came out: a camera driving straight ahead sees its
    # neighbours under angles of only 1-3 degrees, which the default view-selection scores reject.
    # Everything else is left at the OpenMVS defaults (ROI estimate, tower mode, dense-fuse filter).
    # --remove-dmaps 1: delete the per-image depth maps (8.4 MB each) once they are fused into the dense
    # cloud; mesh and texture do not need them. Without it a 477-frame video leaves 4 GB behind.
    11: ("DensifyPointCloud", [mvs_exe("DensifyPointCloud"), "-w", mvs, "-i", "scene.mvs",
                               "--dense-config-file", "Densify.ini", "--resolution-level", "0",
                               "--remove-dmaps", "1"]),
    12: ("ReconstructMesh", [mvs_exe("ReconstructMesh"), "-w", mvs, "-i", "scene_dense.mvs", "-p", "scene_dense.ply"]),
    # Both seam-levelling passes are switched off: with them (the defaults) most faces of the texture atlas
    # come out black or in saturated colours on these videos - 57-82% of the faces black on the models checked;
    # on part b2 of the 15 m video: global only 98% black, local only 76%, both off 0.5%. The faces then keep
    # the colours of the image they were taken from, so brightness steps between patches remain visible.
    13: ("TextureMesh", [mvs_exe("TextureMesh"), "-w", mvs, "-i", "scene_dense.mvs", "-m", "scene_dense_mesh.ply",
                         "-o", "scene_textured.mvs", "--export-type", "obj",
                         "--global-seam-leveling", "0", "--local-seam-leveling", "0"]),
}

# Run right before the step of the same number. A 3-D point behind or almost inside a camera that observes
# it makes DensifyPointCloud crash (exit code 0xC0000409); clean_sfm.py drops those points (well under 1%).
BEFORE = {
    10: ("clean_sfm", [sys.executable, "-I", os.path.join(SCRIPTS, "clean_sfm.py"), mvg("ConvertSfM_DataFormat"),
                       os.path.join(recon, "sfm_data.bin"), os.path.join(recon, "sfm_data_clean.json")]),
}

# Run right after the step of the same number. OpenMVG 1.6 writes the principal point in a form that
# OpenMVS 2.4.0 reads half an image off (see fix_mvs_k.py); scene.mvs is the corrected copy.
AFTER = {
    3: ("set_distortion", [sys.executable, "-I", os.path.join(SCRIPTS, "set_distortion.py"), sfm_json, f"{K1:g}"]),
    10: ("fix_mvs_k", [sys.executable, "-I", os.path.join(SCRIPTS, "fix_mvs_k.py"),
                       os.path.join(mvs, "scene_openmvg.mvs"), os.path.join(mvs, "scene.mvs"), "WIDTH", "HEIGHT"]),
}

PREFLIGHT = [mvg(n) for n in ("SfMInit_ImageListing", "ComputeFeatures", "ComputeMatches", "IncrementalSfM",
                              "ComputeSfM_DataColor", "openMVG2openMVS",
                              "ConvertSfM_DataFormat")] + [mvs_exe(n) for n in ("DensifyPointCloud", "ReconstructMesh",
                                                                                 "TextureMesh")]


def newest_own_log(program, since):
    """Path of the newest <program>-*.log written in the work or mvs folder after `since`, or None."""
    found = []
    for d in (work, mvs):
        if os.path.isdir(d):
            for n in os.listdir(d):
                p = os.path.join(d, n)
                if n.startswith(program + "-") and n.endswith(".log") and os.path.getmtime(p) >= since - 1:
                    found.append(p)
    return max(found, key=os.path.getmtime) if found else None


def run(cmd, log_path, tail=25):
    """Run cmd, write everything to log_path, echo the last `tail` lines. Returns (exit code, seconds)."""
    t0 = time.time()
    lines = []
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n\n")
        p = subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        for raw in p.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            # progress bars rewrite the same line with \r; keep only the final state
            line = line.split("\r")[-1]
            log.write(line + "\n")
            lines.append(line)
        code = p.wait()
        dt = time.time() - t0
        # OpenMVS programs print nothing to stdout; they write <Program>-<stamp>.log into their working folder
        own = newest_own_log(os.path.splitext(os.path.basename(cmd[0]))[0], t0)
        if own:
            with open(own, encoding="utf-8", errors="replace") as f:
                own_lines = f.read().splitlines()
            log.write(f"--- content of {own} ---\n" + "\n".join(own_lines) + "\n")
            lines += own_lines
        log.write(f"\nEXIT CODE: {code}   TIME: {dt:.1f} s\n")
    for line in [ln for ln in lines if ln.strip()][-tail:]:
        print("   | " + line[:220])
    return code, dt


def run_sfm():
    """Step 8: incremental SfM from the candidate pairs of step 7, best first, until one registers at least
    90% of the frames. The attempt with the most registered frames ends up in sfm/recon."""
    with open(pair_file, encoding="utf-8") as f:
        pairs = [ln.split()[:2] for ln in f if ln.strip()]
    best = None
    for k, (a, b) in enumerate(pairs, 1):
        out = f"{recon}_try{k}"
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        cmd = [mvg("IncrementalSfM"), "-i", sfm_json, "-m", matches, "-o", out, "-f", "NONE", "-a", a, "-b", b]
        log = os.path.join(LOGS, f"step08_IncrementalSfM_try{k}.log")
        print(f"[8] IncrementalSfM, attempt {k} of {len(pairs)}: initial pair {a} + {b}")
        print("   $ " + subprocess.list2cmdline(cmd))
        code, dt = run(cmd, log, tail=4)
        with open(log, encoding="utf-8", errors="replace") as f:
            m = re.search(r"#Camera calibrated:\s*(\d+)\s*from\s*(\d+)", f.read())
        reg, total = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        print(f"   -> exit code {code}, {dt:.1f} s, registered {reg} of {total} frames")
        if reg and (best is None or reg > best[0]):
            best = (reg, k, out, log)
        wanted = views[1] - views[0] + 1 if views else total     # --views: only that stretch is expected
        if total and reg >= 0.9 * wanted:
            break
    if best is None:
        sys.exit("step 8 (IncrementalSfM): no candidate pair gave a reconstruction; stopping")
    shutil.rmtree(recon, ignore_errors=True)
    os.replace(best[2], recon)
    shutil.copyfile(best[3], os.path.join(LOGS, "step08_IncrementalSfM.log"))
    for k in range(1, len(pairs) + 1):
        shutil.rmtree(f"{recon}_try{k}", ignore_errors=True)
    print(f"   kept attempt {best[1]} ({best[0]} frames) in {recon}")


os.makedirs(LOGS, exist_ok=True)
for d in (work, matches, recon, mvs):
    os.makedirs(d, exist_ok=True)


def frame_size():
    """width, height of the extracted frames as strings (fix_mvs_k.py needs them; 768x548 for Sheffield)"""
    import cv2
    name = min(n for n in os.listdir(images) if n.endswith(".png"))
    h, w = cv2.imread(os.path.join(images, name), cv2.IMREAD_UNCHANGED).shape[:2]
    return str(w), str(h)

for step in steps:
    if step == 8:
        run_sfm()
        continue
    if step == 1:
        print("[1] pre-flight: can each program start?")
        bad = 0
        for exe in PREFLIGHT:
            name = os.path.basename(exe)
            try:
                t0 = time.time()
                p = subprocess.run([exe, "-h"], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, timeout=60)
                text = p.stdout.decode("utf-8", errors="replace")
                own = newest_own_log(os.path.splitext(name)[0], t0)
                if own:  # OpenMVS: help text lands in its own log file, not on stdout
                    with open(own, encoding="utf-8", errors="replace") as f:
                        text += f.read()
                    os.remove(own)
                first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "(no output)")
                started = len(text.strip()) > 0 and p.returncode in (0, 1)
                print(f"   {'ok  ' if started else 'FAIL'} {name:<45} exit={p.returncode:<11} {first[:90]}")
                with open(os.path.join(LOGS, f"step01_{name}.log"), "w", encoding="utf-8", errors="replace") as f:
                    f.write(text)
                bad += not started
            except Exception as e:  # noqa: BLE001
                print(f"   FAIL {name}: {e!r}")
                bad += 1
        if bad:
            sys.exit(f"pre-flight: {bad} program(s) did not start")
        continue
    if step in BEFORE:
        name, cmd = BEFORE[step]
        print(f"[{step}a] {name}")
        print("   $ " + subprocess.list2cmdline(cmd))
        code, dt = run(cmd, os.path.join(LOGS, f"step{step:02d}a_{name}.log"))
        print(f"   -> exit code {code}, {dt:.1f} s")
        if code != 0:
            sys.exit(f"step {step}a ({name}) failed with exit code {code}; stopping")
    name, cmd = COMMANDS[step]
    if "--dense-config-file" in cmd:   # the dense configuration lives with the scripts; OpenMVS reads it from its working folder
        shutil.copyfile(os.path.join(SCRIPTS, "Densify.ini"), os.path.join(mvs, "Densify.ini"))
    print(f"[{step}] {name}")
    print("   $ " + subprocess.list2cmdline(cmd))
    code, dt = run(cmd, os.path.join(LOGS, f"step{step:02d}_{name}.log"))
    print(f"   -> exit code {code}, {dt:.1f} s")
    if code != 0:
        sys.exit(f"step {step} ({name}) failed with exit code {code}; stopping")
    if step in AFTER:
        name, cmd = AFTER[step]
        if "WIDTH" in cmd:
            w, h = frame_size()
            cmd = [w if c == "WIDTH" else h if c == "HEIGHT" else c for c in cmd]
        print(f"[{step}b] {name}")
        print("   $ " + subprocess.list2cmdline(cmd))
        code, dt = run(cmd, os.path.join(LOGS, f"step{step:02d}b_{name}.log"))
        print(f"   -> exit code {code}, {dt:.1f} s")
        if code != 0:
            sys.exit(f"step {step}b ({name}) failed with exit code {code}; stopping")
