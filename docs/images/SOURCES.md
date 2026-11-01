# 图的来源

每张图用了哪些模型、哪次评分、怎么选的帧、用什么脚本生成。模型目录指作者机器上的 `GS_ROOT/g0/<数据>/<训练名>/`
（主方法的模型没有上传，见 [../DATA.md](../DATA.md) 第 6 节）。图由数据集画面生成，沿用数据集的许可：
Sheffield 管道视频、InPipe-SLAM 为 CC BY 4.0，真实污水管 Pipe1、Pipe2 为 CC BY-NC-SA 4.0（仅非商业用途）。

| 图 | 内容 | 来源 | 生成 |
|---|---|---|---|
| `compare_blocks.jpg` | 每段数据一帧整段留出帧：视频原帧 / 对照 B / 3DGS / 深度监督 3DGS / 主方法，标 PSNR | 15 个模型（每段数据 3 个，种子 0）：15 m 管网、5 m 管 `run_3dgs`、`run_3dgs_depth`、`run_3dgs_depth_lamp_amb`；InPipe、Pipe1 `run_3dgs`、`run_u_3dgs_depth`、`run_u_3dgs_depth_lamp_amb`；Pipe2 `run_3dgs`、`run_3dgs_depth`、`run_main`。图片和数字取自同一次量尺 v2 评分（2026-11-02，`train_gs.py --eval-only --dump DIR --suffix _dump1102`），图片就是被评分的那张（含评分时拟合的每帧增益 + 偏移） | `pipesplat/review/fig_compare_blocks.py`；选帧规则：该段数据全部整段留出帧里，"主方法 − 3DGS"的 PSNR 差最接近中位数的那一帧。选出的帧、每个模型的重评分与原存档的差别记录在 `compare_blocks_selection.txt` |
| `models_outside.jpg` | 主方法烘焙的带纹理网格：从管外看 / 剖开的局部 / 管内视角 | `bake_vhf.py --two-light --source video`（模型 = 各段数据的主方法），每个视角的图由 `pipesplat/review/pipe_figures.py` 画出（`whole.png`、`detail1.png`、`inside0.png`） | `pipesplat/review/fig_models_outside.py OUT --fig-dir 数据=pipe_figures 输出目录 …` |
| `unwrapped_concrete_2.jpg` | 5 m 混凝土管的管壁展开图 | 同上的烘焙网格（5 m 管主方法），`pipe_figures.py` 的 `fig4_unwrapped.jpg` | 缩放到 1600 px 宽 |
| `pipe2_heldout_frames.jpg` | Pipe2 没参加训练的帧：视频原帧 / 主方法高斯模型 / 带纹理网格，标 PSNR | `pipe_figures.py --model-run run_main` 的 `fig5_frames.jpg`：图片是主方法和网格各自评分时写出的三联图（量尺 v2），数字取自同一次评分的逐帧表 | 缩放到 1400 px 宽，去掉底部标题行 |
