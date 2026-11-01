"""11-01: figures of a textured pipe model in the style of the photogrammetry papers (e.g. Fang et al. 2022, Automation
in Construction 142, 104494, figs. 20-21): the whole pipe from outside, front / side / top views, the inside through a
cut, close-ups, a view from inside and the unwrapped wall (管壁展开图). The OBJ of bake_vhf.py (any --source) drawn
by render_views.py (z-buffer, texture, both sides lit). Runs on Windows.

Frame of the pictures: x along the pipe (start of the texture = 0), z = the inspection camera's mean "up", y to the
left; the cut sides are taken against the local centerline point (centerline_<name>.npz), so a bent pipe (InPipe) is
cut along its own axis. Seen from outside, the wall shows its inside texture from the back (as any textured pipe mesh).

usage: python -I pipesplat/review/pipe_figures.py SCENE_DIR OBJ_DIR OUT_DIR [--name vhf] [--title T] [--details X ..]
           [--gain 1.1] [--row-m 6]
SCENE_DIR  the Gaussian scene: scene.npz (training cameras -> up), centerline_<name>.npz
OBJ_DIR    model.obj + texture.png of bake_vhf.py, texture.json one folder up (u0, u1, theta layout)
writes OUT_DIR/<view>.png and the sheets fig1_whole.jpg, fig2_views.jpg, fig3_inside.jpg, fig4_unwrapped.jpg
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "scripts", "vision"), os.path.join(ROOT, "scripts"), os.path.join(ROOT, "lampmap"), HERE):
    sys.path.insert(0, p)
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
import render_views as rv  # noqa: E402



def read_tex_obj(path):
    """the OBJ of bake_vhf.py: vertices, triangles, per-corner UVs (m, 3, 2) into texture.png, and the texture (BGR)"""
    data = open(path, "rb").read()                   # layout: v.., vt.., usemtl wall, f v/vt ..
    i_v, i_vt, i_w = data.find(b"\nv "), data.find(b"\nvt "), data.find(b"\nusemtl wall")
    v = np.fromstring(data[i_v:i_vt].replace(b"\nv ", b" "), dtype=np.float64, sep=" ").reshape(-1, 3)
    vt = np.fromstring(data[i_vt:i_w].replace(b"\nvt ", b" "), dtype=np.float64, sep=" ").reshape(-1, 2)
    block = data[i_w:]
    block = block[block.find(b"\nf "):]
    f = np.fromstring(block.replace(b"\nf ", b" ").replace(b"/", b" "), dtype=np.int64, sep=" ").reshape(-1, 6) - 1
    tri, tuv = f[:, 0::2], f[:, 1::2]
    wall = cv2.imdecode(np.fromfile(os.path.join(os.path.dirname(path), "texture.png"), np.uint8), cv2.IMREAD_COLOR)
    Hw, Ww = wall.shape[:2]
    pad = 16                                         # seam faces reach just past the last row: wrap, as a viewer does
    atlas = np.zeros((Hw + pad, Ww, 3), np.uint8)
    atlas[:Hw], atlas[Hw:] = wall, wall[:pad]
    H = atlas.shape[0]
    uv = vt[tuv].astype(np.float64)                  # (m, 3, 2), OBJ convention (v = 0 at the bottom)
    uv = np.stack([uv[..., 0], 1 - (1 - uv[..., 1]) * Hw / H], -1).astype(np.float32)
    return v, tri.astype(np.int32), uv, atlas


def vertex_normals(xyz, tri):
    n = np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]])
    vn = np.zeros_like(xyz)
    for k in range(3):
        np.add.at(vn, tri[:, k], n)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("scene"), ap.add_argument("obj"), ap.add_argument("out")
ap.add_argument("--name", default="vhf")
ap.add_argument("--title", default="")
ap.add_argument("--details", type=float, nargs="*", default=None, help="close-up centres, m along the pipe (default 3)")
ap.add_argument("--detail-m", type=float, default=0.8, help="length of a close-up section")
ap.add_argument("--gain", type=float, default=1.1)
ap.add_argument("--light", type=float, default=0.35, help="share of the brightness that follows the surface normal")
ap.add_argument("--row-m", type=float, default=6.0, help="side / top views: rows of at most this length")
ap.add_argument("--only", nargs="*", default=None, help="draw only these views (names below)")
ap.add_argument("--model-run", default=None,
                help="the Gaussian model run that was baked: fig5 = held-out frames, video | model | textured mesh "
                     "(their renders/ triplets and per_frame_r2.csv, the mesh's one folder above OBJ_DIR)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
FONT = "C:/Windows/Fonts/msyh.ttc"

# ------------------------------------------------------------------------------ model in the picture frame -----
sc = np.load(os.path.join(a.scene, "scene.npz"))
cl = np.load(os.path.join(a.scene, f"centerline_{a.name}.npz"))
tj = json.load(open(os.path.join(os.path.dirname(os.path.normpath(a.obj)), "texture.json"), encoding="utf-8"))
v, tri, uv, atlas = read_tex_obj(os.path.join(a.obj, "model.obj"))
ci = (cl["u"] >= tj["u0"]) & (cl["u"] <= tj["u1"])
C = cl["C"][ci].astype(np.float64)
x_ax = C[-1] - C[0]
x_ax /= np.linalg.norm(x_ax)
trav = 1 if sc["train_viewmats"][:, 2, :3].astype(np.float64).mean(0) @ x_ax > 0 else -1   # robot moves along +-u
x_ax *= trav                                                              # x = the robot's direction of travel
up = -sc["train_viewmats"][:, 1, :3].astype(np.float64).mean(0)          # image up of the cameras, in the world
z_ax = up - (up @ x_ax) * x_ax
z_ax /= np.linalg.norm(z_ax)
Rp = np.stack([x_ax, np.cross(z_ax, x_ax), z_ax])                         # world -> picture frame (x, y left, z up)
o = C[0] if trav > 0 else C[-1]                                           # where the robot entered the textured part
xyz = (v - o) @ Rp.T
Cp = (C - o) @ Rp.T                                                      # centerline in the picture frame
Cp = Cp[np.argsort(Cp[:, 0])]                                            # from the robot's entry on
L = float(Cp[-1, 0] - Cp[0, 0])
model = dict(xyz=xyz, tri=tri, uv=uv, tex=[atlas], tid=np.zeros(len(tri), int), blank=np.zeros(len(tri), bool),
             centroid=xyz[tri].mean(1), vn=vertex_normals(xyz, tri))
order = np.argsort(Cp[:, 0])
cy = np.interp(model["centroid"][:, 0], Cp[order, 0], Cp[order, 1])
cz = np.interp(model["centroid"][:, 0], Cp[order, 0], Cp[order, 2])
dy, dz = model["centroid"][:, 1] - cy, model["centroid"][:, 2] - cz          # face offset from the local axis
R_med = float(np.median(np.hypot(dy, dz)))
print(f"{a.obj}: {len(v):,} vertices, {len(tri):,} faces, atlas {atlas.shape[1]}x{atlas.shape[0]}; pipe {L:.2f} m, "
      f"radius median {R_med * 1000:.0f} mm")


def want(name):
    return a.only is None or name in a.only


def draw(name, cam, faces=None, light=None):
    path = os.path.join(a.out, f"{name}.png")
    if not want(name) and os.path.exists(path):
        return cv2.cvtColor(cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = rv.draw(model, cam, faces, light=a.light if light is None else light, gain=a.gain)
    rv.save(path, img)
    return img


def axis_point(x):
    return np.array([x, np.interp(x, Cp[order, 0], Cp[order, 1]), np.interp(x, Cp[order, 0], Cp[order, 2])])


# ------------------------------------------------------------------------------ views -------------------------
views = {}
# (1) the whole pipe from outside, above and beside its start (as the photogrammetry figures)
cam = rv.outside_camera(model, 28, 20, (3200, 1500), fov=30)
views["whole"] = draw("whole", cam)
cam = rv.outside_camera(model, 152, 18, (3200, 1500), fov=30)        # from beside its far end
views["whole_back"] = draw("whole_back", cam)
# (2) front view: from outside the start, looking into the pipe along its axis
p0 = axis_point(Cp[0, 0])
cam = rv.camera(p0 + [-2.2 * R_med, 0.0, 0.5 * R_med], axis_point(Cp[0, 0] + 6 * R_med), (1400, 1400), fov=70)
views["front"] = draw("front", cam)
# (3) side (from the left of the robot, -y looking +y... the picture's y is left) and top, orthographic, in rows
nrow = int(math.ceil(L / a.row_m - 1e-9))
scale = 3200 / min(L / nrow + 0.1, a.row_m + 0.1)
for k in range(nrow):
    x0, x1 = Cp[0, 0] + k * L / nrow - 0.05, Cp[0, 0] + (k + 1) * L / nrow + 0.05
    sel = (model["centroid"][:, 0] >= x0 - 0.02) & (model["centroid"][:, 0] <= x1 + 0.02)
    xm = 0.5 * (x0 + x1)
    zc = np.percentile(model["centroid"][sel, 2], [0.5, 99.5])
    yc = np.percentile(model["centroid"][sel, 1], [0.5, 99.5])
    h_side = int((zc[1] - zc[0] + 0.1) * scale)
    h_top = int((yc[1] - yc[0] + 0.1) * scale)
    c_side = np.array([xm, 0.0, 0.5 * (zc[0] + zc[1])])
    cam = rv.camera(c_side + [0, -50.0, 0], c_side, (int((x1 - x0) * scale), h_side), scale=scale)
    views[f"side{k}"] = draw(f"side{k}", cam, sel)
    c_top = np.array([xm, 0.5 * (yc[0] + yc[1]), 0.0])
    cam = rv.camera(c_top + [0, 0, 50.0], c_top, (int((x1 - x0) * scale), h_top), scale=scale, up=(0.0, 1.0, 0.0))
    views[f"top{k}"] = draw(f"top{k}", cam, sel)
    # (4) the inside: the near half cut away (side elevation of the far wall) and the upper half cut away (from above)
    cam = rv.camera(c_side + [0, -50.0, 0], c_side, (int((x1 - x0) * scale), h_side), scale=scale)
    views[f"side{k}_cut"] = draw(f"side{k}_cut", cam, sel & (dy > 0), light=0.15)
    cam = rv.camera(c_top + [0, 0, 50.0], c_top, (int((x1 - x0) * scale), h_top), scale=scale, up=(0.0, 1.0, 0.0))
    views[f"top{k}_cut"] = draw(f"top{k}_cut", cam, sel & (dz < 0), light=0.15)
# (5) the whole pipe with its upper half cut away, from above at an angle
cam = rv.outside_camera(model, 28, 48, (3200, 1500), fov=30)
views["whole_cut"] = draw("whole_cut", cam, dz < 0, light=0.15)
# (6) close-ups: sections of --detail-m with the upper half cut away, and a view from inside along the axis
det = list(Cp[0, 0] + (np.array(a.details) if a.details else L * np.array([0.2, 0.5, 0.8])))
for k, xd in enumerate(det):
    sel = (np.abs(model["centroid"][:, 0] - xd) < a.detail_m / 2) & (dz < 0)
    tgt = axis_point(xd) - [0, 0, 0.3 * R_med]
    el, az = math.radians(55), math.radians(25)
    back = np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), math.sin(el)])
    cam = rv.camera(tgt + 2.2 * a.detail_m * back, tgt, (1800, 1200), fov=30)
    views[f"detail{k}"] = draw(f"detail{k}", cam, sel, light=0.15)
    p = axis_point(xd - 1.2 * R_med)
    cam = rv.camera(p, axis_point(xd + 3 * R_med), (1500, 1100), fov=85)
    views[f"inside{k}"] = draw(f"inside{k}", cam, light=0.0)


# ------------------------------------------------------------------------------ sheets ------------------------
def text_img(s, width, size=40, pad=14, bold=False):
    font = ImageFont.truetype(FONT, size, index=1 if bold else 0)
    im = Image.new("RGB", (width, size + 2 * pad), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    tw = dr.textlength(s, font=font)
    dr.text(((width - tw) / 2, pad - size * 0.1), s, font=font, fill=(0, 0, 0))
    return np.asarray(im)


def panel(img, caption, width=None, pad=30):
    img = rv.crop(img, pad)
    if width:
        img = cv2.resize(img, (width, max(1, int(img.shape[0] * width / img.shape[1]))), interpolation=cv2.INTER_AREA)
    return np.vstack([img, text_img(caption, img.shape[1], 34)]) if caption else img


def hstack(ims, gap=40):
    h = max(i.shape[0] for i in ims)
    out = []
    for k, i in enumerate(ims):
        out.append(np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0)), constant_values=255))
        if k < len(ims) - 1:
            out.append(np.full((h, gap, 3), 255, np.uint8))
    return np.hstack(out)


def vstack(ims, gap=30, width=None):
    W = width or max(i.shape[1] for i in ims)
    out = []
    for k, i in enumerate(ims):
        if i.shape[1] > W:
            i = cv2.resize(i, (W, int(i.shape[0] * W / i.shape[1])), interpolation=cv2.INTER_AREA)
        l_ = (W - i.shape[1]) // 2
        out.append(np.pad(i, ((0, 0), (l_, W - i.shape[1] - l_), (0, 0)), constant_values=255))
        if k < len(ims) - 1:
            out.append(np.full((gap, W, 3), 255, np.uint8))
    return np.vstack(out)


def save_sheet(name, body, caption):
    sheet = vstack([body, text_img(caption, body.shape[1], 44, bold=True)], gap=10)
    sheet = np.pad(sheet, ((40, 40), (40, 40), (0, 0)), constant_values=255)
    if sheet.shape[1] > 3000:
        sheet = cv2.resize(sheet, (3000, int(sheet.shape[0] * 3000 / sheet.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imencode(".jpg", sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(os.path.join(a.out, name))
    print("sheet", os.path.join(a.out, name), f"{sheet.shape[1]}x{sheet.shape[0]}")


T = a.title + "：" if a.title else ""
W0 = 2800
side = [panel(views[f"side{k}"], "") for k in range(nrow)]
save_sheet("fig1_whole.jpg", vstack([panel(views["whole"], "(a) 从起点一侧看", W0),
                                     panel(views["whole_back"], "(b) 从终点一侧看", W0),
                                     vstack(side, gap=12, width=W0), text_img("(c) 侧视（正投影，按比例）", W0, 34)], gap=24),
           f"图 1. {T}带纹理的管道三维模型（整体）")
tops = vstack([panel(views[f"top{k}"], "") for k in range(nrow)], gap=12, width=W0)
sides = vstack(side, gap=12, width=W0)
front = panel(views["front"], "(a) 正视图（从管口看进管内）", 1100)
save_sheet("fig2_views.jpg", vstack([front, vstack([sides, text_img("(b) 侧视图", W0, 34)], gap=0),
                                      vstack([tops, text_img("(c) 俯视图", W0, 34)], gap=0)], gap=30),
           f"图 2. {T}带纹理三维模型的三个视图")
cuts = vstack([panel(views[f"side{k}_cut"], "") for k in range(nrow)], gap=12, width=W0)
tcut = vstack([panel(views[f"top{k}_cut"], "") for k in range(nrow)], gap=12, width=W0)
dets = hstack([panel(views[f"detail{k}"], f"({chr(100 + k)}) 局部 {xd:.1f} m 处", 900) for k, xd in enumerate(det)])
ins = hstack([panel(views[f"inside{k}"], f"({chr(100 + len(det) + k)}) 管内视角 {xd:.1f} m 处", 900, pad=0)
              for k, xd in enumerate(det)])
save_sheet("fig3_inside.jpg", vstack([panel(views["whole_cut"], "(a) 去掉上半管，从斜上方看", W0),
                                       vstack([cuts, text_img("(b) 去掉近侧半管的侧视（看到远侧内壁）", W0, 34)], gap=0),
                                       vstack([tcut, text_img("(c) 去掉上半管的俯视（看到管底内壁）", W0, 34)], gap=0),
                                       dets, ins], gap=30),
           f"图 3. {T}管内纹理")

# the unwrapped wall: texture.png, rows = theta (centerline frame), turned so that the crown is at the top edge
tex = cv2.cvtColor(cv2.imdecode(np.fromfile(os.path.join(a.obj, "texture.png"), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
nth = tex.shape[0]
mid = len(cl["u"]) // 2
Nm, Bm = cl["N"][mid], cl["Bn"][mid]
th_up = math.atan2(up @ Bm, up @ Nm) - math.radians(float(cl["th0"]))   # theta of the crown
r_up = int(round(((th_up + math.pi) % (2 * math.pi)) / (2 * math.pi) * nth))
tex = np.roll(tex, -r_up, axis=0)                                         # crown at row 0, invert in the middle
# theta grows from N towards Bn = T x N, i.e. crown -> right -> invert -> left looking along +u
sides = ("右", "左")
if trav < 0:                                                              # columns in the robot's direction of travel
    tex, sides = tex[:, ::-1], ("左", "右")
du_px = tex.shape[1] / (tj["u1"] - tj["u0"])
seg = int(math.ceil((tj["u1"] - tj["u0"]) / 2.5 - 1e-9))
f_ = ImageFont.truetype(FONT, 30)
rows = []
for k in range(seg):
    c0, c1 = int(k * tex.shape[1] / seg), int((k + 1) * tex.shape[1] / seg)
    part = cv2.resize(tex[:, c0:c1], (2700, int(2700 * nth / (c1 - c0))), interpolation=cv2.INTER_AREA)
    im = Image.fromarray(np.pad(part, ((10, 60), (150, 30), (0, 0)), constant_values=255))
    dr = ImageDraw.Draw(im)
    hpx = part.shape[0]
    for frac, lab in ((0, "顶"), (0.25, sides[0]), (0.5, "底"), (0.75, sides[1]), (1.0, "顶")):
        y = 10 + frac * hpx
        dr.line([(140, y), (150, y)], fill=(0, 0, 0), width=2)
        dr.text((100, y - 18), lab, font=f_, fill=(0, 0, 0))
    d0, d1 = c0 / du_px, c1 / du_px                                     # m from the start of the texture
    for t in np.arange(math.ceil(d0 * 2) / 2, d1 + 1e-9, 0.5):
        xp = 150 + (t - d0) * du_px * 2700 / (c1 - c0)
        dr.line([(xp, 10 + hpx), (xp, 22 + hpx)], fill=(0, 0, 0), width=2)
        dr.text((xp - 30, 24 + hpx), f"{t:.1f} m", font=f_, fill=(0, 0, 0))
    rows.append(np.asarray(im))
un = vstack(rows, gap=20)
cv2.imencode(".png", un[:, :, ::-1])[1].tofile(os.path.join(a.out, "unwrapped.png"))
save_sheet("fig4_unwrapped.jpg", un, f"图 4. {T}管壁展开图（横轴：沿机器人前进方向的距离；纵轴：环向，朝前进方向看）")

# held-out frames: the video, the Gaussian model, the textured mesh (each matched to the frame by gain + offset, as
# scored; the renders/ triplets of train_gs.py and bake_vhf.py are video | lampmap B | model)
if a.model_run:
    import csv
    bake = os.path.dirname(os.path.normpath(a.obj))
    psnr = [{(r["tag"], int(r["frame"])): (float(r["G_psnr"]), r["in_block"] == "True")
             for r in csv.DictReader(open(os.path.join(d_, "per_frame_r2.csv"), encoding="utf-8"))} for d_ in (a.model_run, bake)]
    names = sorted(f for f in os.listdir(os.path.join(a.model_run, "renders")) if os.path.exists(os.path.join(bake, "renders", f)))
    key = lambda f: (f.split("_")[0], int(f.split("_")[1][:6]))  # noqa: E731
    held = [f for f in names if f.startswith("held") and not psnr[0].get(key(f), (0, True))[1]]
    blk = [f for f in names if f.startswith("block")]
    pick = [held[len(held) // 4], held[3 * len(held) // 4]] if len(held) > 1 else held
    pick += [blk[len(blk) // 3], blk[2 * len(blk) // 3]] if len(blk) > 1 else blk
    rows5 = []
    for f in pick:
        tm = cv2.cvtColor(cv2.imdecode(np.fromfile(os.path.join(a.model_run, "renders", f), np.uint8), 1), cv2.COLOR_BGR2RGB)
        tb = cv2.cvtColor(cv2.imdecode(np.fromfile(os.path.join(bake, "renders", f), np.uint8), 1), cv2.COLOR_BGR2RGB)
        w3 = tm.shape[1] // 3
        k_ = key(f)
        what = "整段留出帧" if k_[0] == "block" else "序列内留出帧"
        cap = [f"视频原帧（{what} {k_[1]}）", f"主方法高斯模型 {psnr[0][k_][0]:.1f} dB", f"带纹理网格 {psnr[1][k_][0]:.1f} dB"]
        rows5.append(hstack([vstack([im, text_img(c_, im.shape[1], 26)], gap=0)
                             for im, c_ in zip((tm[:, :w3], tm[:, 2 * w3:], tb[:, 2 * w3:]), cap)], gap=12))
    save_sheet("fig5_frames.jpg", vstack(rows5, gap=24),
               f"图 5. {T}留出帧上的管内纹理：视频 / 主方法高斯模型 / 带纹理网格（PSNR，按帧增益 + 偏移匹配后）")
print("done")
