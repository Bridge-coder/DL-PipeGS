"""Joint (multi-session) SfM of several inspection runs of the same pipe with OpenMVG, so that all their frames get
poses in one frame by one bundle adjustment.

Usage: joint_sfm.py OUT_WORK --run NAME WORK FIRST LAST [--run NAME WORK FIRST LAST ...]
                    [--window 25] [--cross-cm 8] [--init FILE_A FILE_B] [--steps 1 2 3 4]
       --run      a run_pipeline.py work folder whose steps 2-5 are done (CLAHE frames in WORK/images, SIFT features
                  in WORK/sfm/matches, counter.csv); FIRST..LAST is the source-frame range to use. The first run is
                  the reference: the initial pair (--init, default: the first candidate of its sfm/init_pairs.txt)
                  is taken from it.
       --window   each frame is matched with this many following frames of the same run (run_pipeline.py -v 25)
       --cross-cm frames of two runs are matched when their on-screen counters differ by at most this much (the
                  runs' counters must agree to a few cm: lab A and mix15_2 differ by about 1.4 cm); frames before the
                  counter starts (< 0.25 m) get no cross-run pairs (they are tied in through their own run)
Steps: 1 images, features and image listing (copied, renamed NAME_frame_XXXXXX.png; features are reused - all runs
         must have the same image_describer.json), 2 pair list + openMVG_main_ComputeMatches -l (fundamental-matrix
         filter, as run_pipeline.py), 3 openMVG_main_IncrementalSfM with fixed intrinsics, 4 registration report.
OUT_WORK/frames.csv: file,run,source_frame,counter_m.
"""
import argparse
import csv
import json
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
T0 = time.time()
FOCAL, CX, CY, K1 = 806.0, 383.0, 288.0, -0.33      # run_pipeline.py CAMERAS["sheffield"]


def log(msg):
    print(f"{msg} [{time.time() - T0:.0f} s]")


def mvg(name):
    return os.path.join(paths.MVG_BIN, f"openMVG_main_{name}.exe")


def run(cmd, log_path):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as fh:
        fh.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n\n")
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    log(f"   {os.path.basename(cmd[0])}: exit {p.returncode}, {time.time() - t0:.0f} s (log {log_path})")
    if p.returncode != 0:
        sys.exit(f"{cmd[0]} failed")


def read_matches(path):
    """OpenMVG binary match file: {(i, j): number of matches}"""
    data = open(path, "rb").read()
    (n,) = struct.unpack_from("<Q", data, 1)
    off, out = 9, {}
    for _ in range(n):
        i, j, m = struct.unpack_from("<IIQ", data, off)
        off += 16 + 8 * m
        out[(i, j)] = m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--run", nargs=4, action="append", required=True, metavar=("NAME", "WORK", "FIRST", "LAST"))
    ap.add_argument("--window", type=int, default=25), ap.add_argument("--cross-cm", type=float, default=8.0)
    ap.add_argument("--init", nargs=2, default=None), ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4])
    a = ap.parse_args()
    images, matches, recon = (os.path.join(a.out, *p) for p in (("images",), ("sfm", "matches"), ("sfm", "recon")))
    logs = os.path.join(a.out, "logs")
    for d_ in (images, matches, logs):
        os.makedirs(d_, exist_ok=True)
    sfm_json = os.path.join(matches, "sfm_data.json")
    runs = [dict(name=n, work=w, first=int(f), last=int(la)) for n, w, f, la in a.run]

    rows = []
    for r in runs:
        fr = [x for x in csv.DictReader(open(os.path.join(r["work"], "frames.csv"), encoding="utf-8"))
              if r["first"] <= int(x["source_frame"]) <= r["last"]]
        cnt = np.array([float(x["counter_m"]) for x in csv.DictReader(open(os.path.join(r["work"], "counter.csv"), encoding="utf-8"))])
        for x in fr:
            s = int(x["source_frame"])
            rows.append(dict(file=f"{r['name']}_{x['file']}", run=r["name"], source_frame=s, counter_m=float(cnt[s]),
                             src_file=x["file"], work=r["work"]))
    if 1 in a.steps:
        desc0 = open(os.path.join(runs[0]["work"], "sfm", "matches", "image_describer.json"), encoding="utf-8").read()
        for r in runs[1:]:
            if open(os.path.join(r["work"], "sfm", "matches", "image_describer.json"), encoding="utf-8").read() != desc0:
                sys.exit(f"{r['work']}: features made with other settings than {runs[0]['work']}")
        shutil.copyfile(os.path.join(runs[0]["work"], "sfm", "matches", "image_describer.json"),
                        os.path.join(matches, "image_describer.json"))
        for x in rows:
            shutil.copyfile(os.path.join(x["work"], "images", x["src_file"]), os.path.join(images, x["file"]))
            stem, nstem = os.path.splitext(x["src_file"])[0], os.path.splitext(x["file"])[0]
            for ext in (".feat", ".desc"):
                shutil.copyfile(os.path.join(x["work"], "sfm", "matches", stem + ext), os.path.join(matches, nstem + ext))
        with open(os.path.join(a.out, "frames.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["file", "run", "source_frame", "counter_m"])
            wr.writerows([(x["file"], x["run"], x["source_frame"], f"{x['counter_m']:.3f}") for x in rows])
        log(f"{len(rows)} frames from {len(runs)} runs copied (images + features)")
        run([mvg("SfMInit_ImageListing"), "-i", images, "-o", matches, "-c", "2",
             "-k", f"{FOCAL:g};0;{CX:g};0;{FOCAL:g};{CY:g};0;0;1"], os.path.join(logs, "listing.log"))
        sfm = json.load(open(sfm_json, encoding="utf-8"))
        for e in sfm["intrinsics"]:
            e["value"]["ptr_wrapper"]["data"]["disto_k1"] = [K1]
        json.dump(sfm, open(sfm_json, "w", encoding="utf-8"), indent=1)
        log(f"image listing: {len(sfm['views'])} views, {len(sfm['intrinsics'])} intrinsic(s), k1 set to {K1}")

    sfm = json.load(open(sfm_json, encoding="utf-8"))
    vid = {v["value"]["ptr_wrapper"]["data"]["filename"]: v["value"]["ptr_wrapper"]["data"]["id_view"] for v in sfm["views"]}
    by_run = {r["name"]: sorted((x for x in rows if x["file"] in vid and x["run"] == r["name"]),
                                key=lambda x: x["source_frame"]) for r in runs}
    pair_txt = os.path.join(a.out, "sfm", "pairs.txt")
    if 2 in a.steps:
        pairs = {}
        for name, xs in by_run.items():                   # within a run: the next --window frames
            ids = [vid[x["file"]] for x in xs]
            for k, i in enumerate(ids):
                pairs.setdefault(i, set()).update(ids[k + 1:k + 1 + a.window])
        names = [r["name"] for r in runs]
        for p_, n1 in enumerate(names):                   # across runs: frames at the same counter reading
            for n2 in names[p_ + 1:]:
                c1 = np.array([x["counter_m"] for x in by_run[n1]])
                c2 = np.array([x["counter_m"] for x in by_run[n2]])
                ok1, ok2 = c1 >= 0.25, c2 >= 0.25
                n_cross = 0
                for k, x in enumerate(by_run[n1]):
                    if not ok1[k]:
                        continue
                    near = np.flatnonzero(ok2 & (np.abs(c2 - c1[k]) <= a.cross_cm / 100))
                    if len(near):
                        pairs.setdefault(vid[x["file"]], set()).update(vid[by_run[n2][j]["file"]] for j in near)
                        n_cross += len(near)
                log(f"cross-run pairs {n1}-{n2}: {n_cross} (counter difference <= {a.cross_cm:g} cm)")
        with open(pair_txt, "w", encoding="utf-8") as fh:
            for i in sorted(pairs):
                js = sorted(j for j in pairs[i] if j != i)
                if js:
                    fh.write(" ".join(str(v) for v in [i] + js) + "\n")
        log(f"pair list: {sum(len(v) for v in pairs.values())} pairs -> {pair_txt}")
        run([mvg("ComputeMatches"), "-i", sfm_json, "-o", matches, "-g", "f", "-l", pair_txt],
            os.path.join(logs, "matches.log"))
        m = read_matches(os.path.join(matches, "matches.f.bin"))
        run_of = {vid[x["file"]]: x["run"] for x in rows if x["file"] in vid}
        cross = [v for (i, j), v in m.items() if run_of[i] != run_of[j]]
        within = [v for (i, j), v in m.items() if run_of[i] == run_of[j]]
        log(f"verified pairs: within runs {len(within)} (median {np.median(within):.0f} matches), across runs "
            f"{len(cross)} (median {np.median(cross) if cross else 0:.0f} matches)")
    if 3 in a.steps:
        if a.init:
            ia, ib = a.init
        else:
            first = open(os.path.join(runs[0]["work"], "sfm", "init_pairs.txt"), encoding="utf-8").readline().split()
            ia, ib = f"{runs[0]['name']}_{first[0]}", f"{runs[0]['name']}_{first[1]}"
        shutil.rmtree(recon, ignore_errors=True)
        os.makedirs(recon)
        log(f"incremental SfM from the initial pair {ia} + {ib}, intrinsics fixed")
        run([mvg("IncrementalSfM"), "-i", sfm_json, "-m", matches, "-o", recon, "-f", "NONE", "-a", ia, "-b", ib],
            os.path.join(logs, "sfm.log"))
    if 4 in a.steps:
        out_json = os.path.join(recon, "sfm_data_full.json")
        subprocess.run([mvg("ConvertSfM_DataFormat"), "-i", os.path.join(recon, "sfm_data.bin"), "-o", out_json,
                        "-V", "-I", "-E", "-S"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        d = json.load(open(out_json, encoding="utf-8"))
        posed = {e["key"] for e in d["extrinsics"]}
        reg = {v["value"]["ptr_wrapper"]["data"]["filename"] for v in d["views"]
               if v["value"]["ptr_wrapper"]["data"]["id_pose"] in posed}
        for name, xs in by_run.items():
            log(f"run {name}: {sum(x['file'] in reg for x in xs)} of {len(xs)} frames registered")
        log(f"3-D points: {len(d['structure']):,}")


if __name__ == "__main__":
    main()
