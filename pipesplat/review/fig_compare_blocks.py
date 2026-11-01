"""README figure docs/images/compare_blocks.jpg: for each data set one block frame (inside a 10 cm stretch left out of
training, so no model saw a frame within 10 cm), shown as video | baseline B | 3DGS | depth-supervised 3DGS | main
method, each labelled with its PSNR.

The pictures are the scored images themselves: the dumps of a ruler-v2 scoring run (train_gs.py --eval-only --dump DIR
--suffix _dump1102, every render with the gain / offset the scoring fitted), and the labels are the PSNR of that same
run (DIR/per_frame_dump1102.csv). The frame of a data set is the block frame whose main - 3DGS PSNR gain is closest to
the median gain over all block frames of that data set - a rule, not a choice. The script also reports how far the
dumped scores are from the stored training-end scores (per_frame_r2.csv) and writes the selection to OUT_selection.txt.

usage: python -I pipesplat/review/fig_compare_blocks.py OUT.jpg [--dump-root DIR] [--scene-root DATA=DIR ...]
       --dump-root   DIR/<data>/<run>/ holds the dumps and per_frame_dump1102.csv (default GS_ROOT/review/figdump)
       --scene-root  where <data>/<run>/per_frame_r2.csv is, per data set (default GS_ROOT/g0)
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
import paths  # noqa: E402

ROWS = [("15 m 管网|（混凝土段）", "concrete_lab", "run_3dgs", "run_3dgs_depth", "run_3dgs_depth_lamp_amb"),
        ("5 m 混凝土管", "concrete_2", "run_3dgs", "run_3dgs_depth", "run_3dgs_depth_lamp_amb"),
        ("InPipe", "inpipe_vision", "run_3dgs", "run_u_3dgs_depth", "run_u_3dgs_depth_lamp_amb"),
        ("Pipe1|真实污水管", "sewer_pipe1", "run_3dgs", "run_u_3dgs_depth", "run_u_3dgs_depth_lamp_amb"),
        ("Pipe2|真实污水管|（新数据）", "sewer_pipe2", "run_3dgs", "run_3dgs_depth", "run_main")]
COLS = ["视频原帧（没参加训练）", "传统网格贴图 B", "3DGS", "深度监督 3DGS", "主方法"]
PW, LW = 380, 230
FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
FONT_S = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 19)


def table(path):
    return {(r["tag"], int(r["frame"])): r for r in csv.DictReader(open(path, encoding="utf-8"))}


def read(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--dump-root", default=os.path.join(paths.GS_ROOT, "review", "figdump"))
    ap.add_argument("--scene-root", nargs="*", default=[], metavar="DATA=DIR")
    a = ap.parse_args()
    roots = dict(s.split("=", 1) for s in a.scene_root)
    rows_img, report = [], []
    for label, data, *runs in ROWS:
        dumps = [os.path.join(a.dump_root, data, r) for r in runs]
        tabs = [table(os.path.join(d, "per_frame_dump1102.csv")) for d in dumps]
        for r, t in zip(runs, tabs):                 # the re-score must equal the stored training-end scores
            stored = table(os.path.join(roots.get(data, os.path.join(paths.GS_ROOT, "g0")), data, r, "per_frame_r2.csv"))
            diff = max(abs(float(t[k]["G_psnr"]) - float(stored[k]["G_psnr"])) for k in t if k in stored)
            report.append(f"{data}/{r}: max |PSNR dump - stored| = {diff:.2e} dB over {len(t)} frames")
        frames = sorted(f for (tag, f) in tabs[0] if tag == "block")
        gains = np.array([float(tabs[2][("block", f)]["G_psnr"]) - float(tabs[0][("block", f)]["G_psnr"]) for f in frames])
        fr = frames[int(np.argmin(np.abs(gains - np.median(gains))))]
        panels = [read(os.path.join(dumps[0], f"block_{fr:06d}_real.jpg")), read(os.path.join(dumps[0], f"block_{fr:06d}_B.jpg"))]
        panels += [read(os.path.join(d, f"block_{fr:06d}_G.jpg")) for d in dumps]
        h = int(round(panels[0].shape[0] * PW / panels[0].shape[1]))
        panels = [cv2.resize(p, (PW, h), interpolation=cv2.INTER_AREA) for p in panels]
        texts = [f"第 {fr} 帧", f"{float(tabs[0][('block', fr)]['B_psnr']):.1f} dB"] + \
                [f"{float(t[('block', fr)]['G_psnr']):.1f} dB" for t in tabs]
        canvas = Image.new("RGB", (5 * PW + 4 * 6 + LW, h + 30), "white")
        d = ImageDraw.Draw(canvas)
        lines = label.split("|")                    # "|" separates the lines of a row label
        for k_, ln in enumerate(lines):
            d.text((10, h // 2 - 16 * len(lines) + 32 * k_), ln, font=FONT, fill="black")
        for k, (p, t) in enumerate(zip(panels, texts)):
            x = LW + k * (PW + 6)
            canvas.paste(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)), (x, 0))
            d.text((x + PW // 2 - d.textlength(t, font=FONT_S) / 2, h + 3), t, font=FONT_S, fill="black")
        rows_img.append(canvas)
        report.append(f"{data}: {len(frames)} block frames, chose frame {fr}: main - 3DGS {gains[frames.index(fr)]:+.2f} dB "
                      f"(median over the block frames {np.median(gains):+.2f}); B / 3DGS / depth / main = {', '.join(texts[1:])}")
    W = rows_img[0].width
    head = Image.new("RGB", (W, 40), "white")
    dh = ImageDraw.Draw(head)
    for k, c in enumerate(COLS):
        x = LW + k * (PW + 6)
        dh.text((x + PW // 2 - dh.textlength(c, font=FONT) / 2, 6), c, font=FONT, fill="black")
    fig = Image.new("RGB", (W, 40 + sum(r.height + 10 for r in rows_img)), "white")
    fig.paste(head, (0, 0))
    y = 40
    for r in rows_img:
        fig.paste(r, (0, y))
        y += r.height + 10
    fig.save(a.out, quality=86, optimize=True)
    report.append(f"{os.path.basename(a.out)}: {fig.size[0]} x {fig.size[1]} px, {os.path.getsize(a.out) // 1024} KB")
    open(os.path.splitext(a.out)[0] + "_selection.txt", "w", encoding="utf-8").write("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
