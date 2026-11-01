"""README figure docs/images/models_outside.jpg: the textured meshes baked from the main model (bake_vhf.py --two-light
--source video), one row per data set: the whole pipe from outside (whole.png), a cut-open close-up (detail1.png) and a
view from inside (inside0.png) - the per-view pictures written by pipe_figures.py.

usage: python -I pipesplat/review/fig_models_outside.py OUT.jpg --fig-dir DATA=DIR [--fig-dir DATA=DIR ...]
       DATA one of concrete_lab, concrete_2, inpipe, sewer_pipe1, sewer_pipe2 (rows in this order, missing ones left out);
       DIR  the pipe_figures.py output folder of that data set
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROWS = [("concrete_lab", "15 m 管网（混凝土段，约 4.3 m）"),
        ("concrete_2", "5 m 混凝土管（约 4.2 m）"),
        ("inpipe", "InPipe（两节管 + 弯头，有开缝）"),
        ("sewer_pipe1", "Pipe1 真实污水管（约 10.6 m）"),
        ("sewer_pipe2", "Pipe2 真实污水管（新数据，约 3.9 m；喇叭形 = 单目尺度漂移）")]
FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
W1, W2, W3, H = 1000, 640, 640, 470


def crop_white(img, pad=20):
    """trim the white margin around the rendered pipe"""
    ys, xs = np.nonzero(np.any(img < 245, axis=2))
    return img if len(ys) == 0 else img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def fit(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w, 3), 255, np.uint8)
    y, x = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
    canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
    return canvas


def read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--fig-dir", nargs="+", required=True, metavar="DATA=DIR")
    a = ap.parse_args()
    dirs = dict(s.split("=", 1) for s in a.fig_dir)
    rows = []
    for data, label in ROWS:
        if data not in dirs:
            continue
        d = dirs[data]
        gap = np.full((H, 20, 3), 255, np.uint8)
        row = np.hstack([fit(crop_white(read(os.path.join(d, "whole.png"))), W1, H), gap,
                         fit(crop_white(read(os.path.join(d, "detail1.png"))), W2, H), gap,
                         fit(read(os.path.join(d, "inside0.png")), W3, H)])
        img = Image.new("RGB", (row.shape[1], H + 44), "white")
        ImageDraw.Draw(img).text((16, 6), label, font=FONT, fill="black")
        img.paste(Image.fromarray(cv2.cvtColor(row, cv2.COLOR_BGR2RGB)), (0, 44))
        rows.append(img)
    head = Image.new("RGB", (rows[0].width, 44), "white")
    dh = ImageDraw.Draw(head)
    for x, t in ((W1 // 2, "带纹理网格：从管外看"), (W1 + 20 + W2 // 2, "剖开的局部（放大）"), (W1 + W2 + 40 + W3 // 2, "管内视角")):
        dh.text((x - dh.textlength(t, font=FONT) / 2, 6), t, font=FONT, fill="black")
    fig = Image.new("RGB", (rows[0].width, 44 + sum(r.height for r in rows)), "white")
    fig.paste(head, (0, 0))
    y = 44
    for r in rows:
        fig.paste(r, (0, y))
        y += r.height
    fig = fig.resize((fig.width * 3 // 4, fig.height * 3 // 4), Image.LANCZOS)
    fig.save(a.out, quality=84, optimize=True)
    print(a.out, fig.size, os.path.getsize(a.out) // 1024, "KB")


if __name__ == "__main__":
    main()
