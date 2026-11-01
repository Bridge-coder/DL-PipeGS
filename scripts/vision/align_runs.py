"""Bring the other runs of a joint SfM (joint_sfm.py) into the reference run's own, earlier reconstruction.

Usage: align_runs.py JOINT_WORK REF_NAME REF_WORK OUT_DIR

The reference run's frames have a pose in both reconstructions. The similarity transform (scale, rotation,
translation) between the two is estimated from them - the rotation from the camera orientations by orthogonal
Procrustes, then scale and translation from the camera centres by least squares (refitted three times without the
cameras whose residual exceeds 3x the median) - and applied to the poses of the other runs.
Everything trained on REF_WORK can then be scored on frames of the other runs without being rebuilt.
Writes OUT_DIR/sfm_data_expanded.json (views + extrinsics, the layout openMVG_main_SfM_Localization writes, so
export_xrun.py reads either) with the other runs' frames as q_<run>_frame_XXXXXX.png, OUT_DIR/query_<run>.csv
(query_file,source_frame), OUT_DIR/align.json (residuals: camera centres in mm, rotations in mrad) and OUT_DIR/loo.csv
(per reference camera: residual of the one global similarity, leave-one-out error of the local similarity).
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402
import evaluate as ev  # noqa: E402
import render_views as rv  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
LOCAL_SIGMA_M = float(os.environ.get("ALIGN_SIGMA_M", "0.15"))   # width of the local similarity fit


def poses_of(sfm_bin):
    """file name -> (world-to-camera rotation, centre) of every posed view"""
    out_json = os.path.splitext(sfm_bin)[0] + "_full.json"
    subprocess.run([paths.CONVERTER, "-i", sfm_bin, "-o", out_json, "-V", "-I", "-E", "-S"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = json.load(open(out_json, encoding="utf-8"))
    ext = {e["key"]: (np.array(e["value"]["rotation"], float), np.array(e["value"]["center"], float)) for e in d["extrinsics"]}
    return {v["value"]["ptr_wrapper"]["data"]["filename"]: ext[v["value"]["ptr_wrapper"]["data"]["id_pose"]]
            for v in d["views"] if v["value"]["ptr_wrapper"]["data"]["id_pose"] in ext}


def procrustes_rotation(M):
    """the rotation closest to M in the Frobenius norm (orthogonal Procrustes, SVD with the determinant fixed)"""
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def similarity(Rsrc, Csrc, Rdst, Cdst, wt=None):
    """s, R, t with Cdst ~ s R Csrc + t and Rdst ~ Rsrc R^T (weighted least squares). The camera centres of a pipe run
    lie almost on one line, so they leave the roll about that line undetermined (Umeyama's centre-only fit was
    52 mrad off): R comes from the camera orientations (orthogonal Procrustes on sum w_k Rsrc_k^T Rdst_k), then s and
    t from the centres"""
    wt = np.ones(len(Csrc)) if wt is None else wt
    R = procrustes_rotation(np.einsum("k,kji,kjl->il", wt, Rsrc, Rdst)).T
    ms, md = (wt[:, None] * Csrc).sum(0) / wt.sum(), (wt[:, None] * Cdst).sum(0) / wt.sum()
    xs, xd = (Csrc - ms) @ R.T, Cdst - md
    s = float((wt[:, None] * xs * xd).sum() / (wt[:, None] * xs ** 2).sum())
    return s, R, md - s * R @ ms


def local_fit(C, Rsrc, Csrc, Rdst, Cdst, sigma, skip=None):
    """similarity fitted to the reference cameras near C (Gaussian weights of their distance, sigma in model units);
    skip = index of a reference camera to leave out (leave-one-out check)"""
    wt = np.exp(-0.5 * (np.linalg.norm(Csrc - C, axis=1) / sigma) ** 2)
    if skip is not None:
        wt[skip] = 0.0
    return similarity(Rsrc, Csrc, Rdst, Cdst, wt)


def main():
    joint, ref_name, ref_work, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    J = poses_of(os.path.join(joint, "sfm", "recon", "sfm_data.bin"))
    Rf = poses_of(os.path.join(ref_work, "sfm", "recon", "sfm_data.bin"))
    common = sorted(n for n in Rf if f"{ref_name}_{n}" in J)
    src = np.array([J[f"{ref_name}_{n}"][1] for n in common])
    dst = np.array([Rf[n][1] for n in common])
    Rs_ = np.array([J[f"{ref_name}_{n}"][0] for n in common])
    Rd_ = np.array([Rf[n][0] for n in common])
    keep = np.ones(len(common), bool)
    for _ in range(4):
        s, R, t = similarity(Rs_[keep], src[keep], Rd_[keep], dst[keep])
        res = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        keep = res <= 3 * np.median(res)
    # model units -> mm, from the reference's pipe coordinates (one scale from the 300 mm bore)
    frames = {r["file"]: int(r["source_frame"]) for r in csv.DictReader(open(os.path.join(ref_work, "frames.csv"), encoding="utf-8"))}
    counter = np.array([float(r["counter_m"]) for r in csv.DictReader(open(os.path.join(ref_work, "counter.csv"), encoding="utf-8"))])
    pf = ev.pipe_frame(paths.CONVERTER, ref_work, None, frames, counter, straighten=False)
    xyz = np.stack(rv.pipe_xyz(*pf["to_pipe"](dst)), 1)
    k = float(np.median(np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.maximum(np.linalg.norm(np.diff(dst, axis=0), axis=1), 1e-12)))
    rot = np.array([np.degrees(np.arccos(np.clip((np.trace(Rf[n][0] @ (J[f"{ref_name}_{n}"][0] @ R.T).T) - 1) / 2, -1, 1)))
                    for n in common]) * np.pi / 180 * 1000
    print(f"{len(common)} reference cameras in both reconstructions, {keep.sum()} kept for the fit; scale {s:.5f}")
    print(f"one similarity for the whole run - camera centres: median {np.median(res) * k * 1000:.2f} mm, p90 "
          f"{np.percentile(res, 90) * k * 1000:.2f} mm, max {res.max() * k * 1000:.2f} mm; rotations median "
          f"{np.median(rot):.2f} mrad, p90 {np.percentile(rot, 90):.2f} mrad")
    # the two reconstructions drift differently along the pipe (millimetres over metres), so each frame of another run
    # is moved by a similarity fitted to the reference cameras around it; leave-one-out on the reference cameras
    # tells how well that transfers a pose
    sigma = LOCAL_SIGMA_M * 1000 / (k * 1000) / s                # metres -> joint-model units
    loo_c, loo_r = [], []
    for i in range(len(common)):
        si, Ri, ti = local_fit(src[i], Rs_, src, Rd_, dst, sigma, skip=i)
        loo_c.append(np.linalg.norm(dst[i] - (si * Ri @ src[i] + ti)) * k * 1000)
        Ro = Rs_[i] @ Ri.T
        loo_r.append(np.arccos(np.clip((np.trace(Rd_[i] @ Ro.T) - 1) / 2, -1, 1)) * 1000)
    loo_c, loo_r = np.array(loo_c), np.array(loo_r)
    with open(os.path.join(out, "loo.csv"), "w", newline="", encoding="utf-8") as fh:   # per reference camera
        wr = csv.writer(fh)
        wr.writerow(["file", "global_centre_mm", "loo_centre_mm", "loo_rot_mrad"])
        wr.writerows((n, f"{g * k * 1000:.4f}", f"{c:.4f}", f"{r:.4f}") for n, g, c, r in zip(common, res, loo_c, loo_r))
    print(f"local similarity (Gaussian weights, sigma {LOCAL_SIGMA_M:g} m), leave-one-out on the reference cameras - "
          f"centres: median {np.median(loo_c):.2f} mm, p90 {np.percentile(loo_c, 90):.2f} mm, max {loo_c.max():.2f} mm; "
          f"rotations median {np.median(loo_r):.2f} mrad, p90 {np.percentile(loo_r, 90):.2f} mrad")
    views, ext, runs = [], [], {}
    for n, (Rj, Cj) in sorted(J.items()):
        run_name = n.split("_", 1)[0]
        if run_name == ref_name:
            continue
        q = f"q_{n}"
        k_ = len(ext)
        sl, Rl, tl = local_fit(Cj, Rs_, src, Rd_, dst, sigma)
        views.append({"key": k_, "value": {"ptr_wrapper": {"data": {"filename": q, "id_view": k_, "id_pose": k_}}}})
        ext.append({"key": k_, "value": {"rotation": (Rj @ Rl.T).tolist(), "center": (sl * Rl @ Cj + tl).tolist()}})
        runs.setdefault(run_name, []).append((q, int(n.split("_frame_")[1].split(".")[0])))
    json.dump({"views": views, "extrinsics": ext}, open(os.path.join(out, "sfm_data_expanded.json"), "w", encoding="utf-8"))
    for run_name, items in runs.items():
        with open(os.path.join(out, f"query_{run_name}.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["query_file", "source_frame"])
            wr.writerows(items)
        print(f"run {run_name}: {len(items)} frames written")
    json.dump(dict(joint=joint, ref=ref_work, scale=float(s), mm_per_unit=k * 1000, n_common=len(common),
                   n_fit=int(keep.sum()), centre_res_mm=dict(median=float(np.median(res) * k * 1000),
                                                             p90=float(np.percentile(res, 90) * k * 1000),
                                                             max=float(res.max() * k * 1000)),
                   rot_res_mrad=dict(median=float(np.median(rot)), p90=float(np.percentile(rot, 90))),
                   local_sigma_m=LOCAL_SIGMA_M,
                   local_loo_centre_mm=dict(median=float(np.median(loo_c)), p90=float(np.percentile(loo_c, 90)),
                                            max=float(loo_c.max())),
                   local_loo_rot_mrad=dict(median=float(np.median(loo_r)), p90=float(np.percentile(loo_r, 90)))),
              open(os.path.join(out, "align.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
