# 数据、目录与文件格式

本文件说明：公开数据集从哪里下载；一次重建在磁盘上产生哪些文件夹和文件；每个中间文件（`scene.npz`、`surface*.npz`、`model.pt`、`summary_r2.json` ……）里有什么。
模型见 [MODEL.md](MODEL.md)，运行见 [PIPELINE.md](PIPELINE.md)。

---

## 1. 公开数据集

`scripts/download_data.py` 从 figshare 的官方接口下载并核对 MD5（`--list` 列出每个数据集的全部文件、大小和 MD5）：

```bash
python -I scripts/download_data.py --list sheffield-sewer
python -I scripts/download_data.py sheffield-sewer D:/data/sewer Pipe2.avi "Calibration Checkerboard.mp4" "Ground truth distances.csv"
```

| 键 | 数据集 | DOI / 许可 | 内容 | 本项目的用途 |
|---|---|---|---|---|
| `sheffield-concrete5m` | Sheffield Minicam，300 mm 混凝土管 5 m | 10.15131/shef.data.28046066，CC BY 4.0 | 5 段视频（同一段管 5 次巡检，720 × 576 PAL，25 fps），画面底部烧录线缆计数器 | concrete_1 标定 f、k1；concrete_2 训练，其余为跨巡检测试 |
| `sheffield-mix15m` | Sheffield Minicam，15 m 混合管网（混凝土、陶土、塑料、两个检查井） | 10.15131/shef.data.28047074，CC BY 4.0 | 15 段视频（两档灯光 × 两档速度 × 重复），文件名写着灯光（lm）和速度（u） | 开头 0–4.6 m：r1 训练，r2–r5 跨巡检测试；另外 10 段没用过 |
| `sheffield-sewer` | Sheffield 真实地下污水管（Edwards 等，Frontiers in Robotics and AI 2023，10.3389/frobt.2023.1150508） | 10.15131/shef.data.21198070，**CC BY-NC-SA 4.0（仅非商业）** | Pipe1、Pipe2、Pipe3 视频；棋盘格标定视频；逐帧里程真值（宽表，一行一条管）；检查井 GPS | Pipe1 前 10.6 m；Pipe2 第 450–1500 帧（端到端测试）；Pipe3 没用过 |
| — | InPipe-SLAM 序列 172011（Zenodo 18615583，CC BY 4.0，10 GB 压缩包） | — | ROS 2 bag（激光、RealSense、编码器……） | 只用彩色相机；场景文件直接用 Release v2.0 里的 `inpipe_172011_scene.zip`，不需要原始数据 |

Sheffield 视频的计数器是视觉分支的里程真值；`scripts/vision/evaluate.py counter`（人工给字形簇标数字）或 `scripts/vision/counter_ocr.py`（模板匹配）读出 `counter.csv`。
污水管数据集的 `Ground truth distances.csv` 由 `scripts/pipeline.py --gt 文件 --gt-row Pipe2` 转成同样格式（第 k 列 = 源视频第 k − 1 帧）。

**名字对照**（文档里的代号、视频、场景文件夹 `GS_ROOT/g0/<文件夹>`、前端工作目录的来源）：

| 文档里的代号 | 视频 | 场景文件夹 | 前端工作目录 |
|---|---|---|---|
| 15 m 管网（混凝土段，r1；旧记录里叫"实验室管"、lab） | `300mm_mix_15meters_ON_3lm_3u_CLOSED_1_stop_at_14meters.mp4` | `concrete_lab` | Release v1.0 `concrete_lab.zip` |
| 5 m 混凝土管 | `300mm_concrete_5meters_ON_2lm_2u_CLOSED_2.mp4`（concrete_2） | `concrete_2` | Release v1.0 `sheffield_concrete_2.zip` |
| InPipe | InPipe-SLAM 序列 172011 的彩色相机 | `inpipe_vision` | 无（直接用 Release v2.0 的 `inpipe_172011_scene.zip`） |
| Pipe1 | `Pipe1.avi` | `sewer_pipe1` | Release v1.0 `sewer_pipe1.zip` |
| Pipe2 | `Pipe2.avi` | `sewer_pipe2` | 端到端测试时新建（PIPELINE.md 第 4 节） |

训练名（场景文件夹下的 `run_<训练名>/`）：`run_3dgs`（3DGS）、`run_3dgs_depth`（深度监督 3DGS）、`run_3dgs_depth_lamp`（+ 车灯模型，不含环境光项）、
`run_3dgs_depth_lamp_amb`（主方法）、`run_3dgs_depth_expo`（只有每帧曝光），后缀 `_s1`、`_s2` 是种子 1、2。InPipe、Pipe1 上同样的配置名字前多一个 `u_`
（如 `run_u_3dgs_depth_lamp_amb`，选项相同，只多 `--surface surface_vhf.npz`）。Pipe2 和 `pipeline.py` 默认的主方法叫 `run_main`。

---

## 2. 目录布局

位置由 `scripts/paths.py` 决定（`config/paths.json` 或环境变量 `PIPE3D_<键>`）：

```
WORK_NEW/<名字>/work/                前端工作目录（scripts/pipeline.py 默认；也可 --work 指定）
  frames.csv                        抽出的帧：file, source_frame, time
  images/frame_<源帧号>.png          CLAHE 后的帧（SfM 用）；osd/：裁掉的烧录文字条
  counter.csv                       可选：source_frame, counter_m（计数器 / 里程真值；场景的 x 原点与它对齐）
  sfm/matches/                      特征、匹配、sfm_data.json（内参）、init_pairs.txt
  sfm/recon/sfm_data.bin            SfM 结果（位姿 + 稀疏点）；sparse_colorized.ply
  mvs/scene_dense.ply               OpenMVS 稠密点云
  mvs/scene_dense_mesh.ply          网格；mvs/scene_textured.obj + 贴图：带贴图网格
  gt_eval.json / .png / _track.csv  有真值时：eval_gt_scale.py 的定位误差、尺度漂移、推出的管径
LOGS/<work 文件夹名>/                每一步的完整日志（step05_ComputeMatches.log ……）

GS_ROOT/g0/<名字>/                   高斯部分的场景目录（--scene 指定；"g0" 是历史名字）
  scene.npz, scene.json             训练与评分需要的一切（第 3 节）
  lamp_map.npy                      对照 B 的灯光图（export_g0.py --lamp auto 学的）
  surface.npz                       实测管壁：OpenMVS 网格三角形
  surface_vhf.npz, mesh_vhf.npz, centerline_vhf.npz, vhf.json   管道先验拟合的高度场管壁、网格、中心线
  xrun_<巡检>.npz, xblk_<巡检>.npz  跨巡检测试集（export_xrun.py）
  run_<训练名>/                     一次训练（第 4 节）
  run_<训练名>_mesh/                烘焙网格：obj/model.obj、model.mtl、texture.png、albedo.png、ambient.png、lamp.npz；
                                    texture.json（纹理的 u、θ 布局）；summary_r2.json（网格的评分）；figures/（效果图）
```

---

## 3. `scene.npz`（`pipesplat/export_g0.py` 写，Windows）

位置单位米，在管道坐标里（x 沿管轴；见 MODEL.md 第 4 节"管道坐标和尺度"）；相机矩阵是 OpenCV 约定的世界→相机 4 × 4；内参像素中心在整数坐标；图像是原始视频帧（RGB uint8，未做 CLAHE），按前端同样的裁剪 / 拉伸。

| 键 | 形状 | 内容 |
|---|---|---|
| `K`、`k1`、`size` | 3 × 3、标量、(w, h) | 原始画面的相机：单焦距、主点、径向畸变 k1 |
| `train_images` | N × h × w × 3 | 所有 SfM 注册帧（训练帧 + 整段留出帧） |
| `train_viewmats`、`train_eyes`、`train_pos`、`train_ids`、`train_t` | N × 4 × 4、N × 3、N、N、N | 位姿、相机中心、沿管位置、源帧号、时间（视频为 0） |
| `block_ranges`、`block_idx` | 2 × 2、M | 两段 10 cm 整段留出的范围（米）和落在里面的训练帧下标（不参加训练） |
| `block_next`、`block_viewmats`、`block_B`、`block_mask`、`block_ids` | | 整段留出帧的下一视频帧、位姿、对照 B 的渲染（float16）、B 的掩码（= 评分像素）、帧号 |
| `held_images`、`held_next`、`held_viewmats`、`held_B`、`held_mask`、`held_ids`、`held_pos` | 44 × … | 序列内留出帧（两帧之间的视频帧，插值位姿；`--every 1` 时是不参加训练的注册帧） |
| `points` | P × 3 | 稠密点（高斯的起始位置） |
| `depth_points` | P × 3 | 同一稠密点（评分时"渲染深度误差"的参照） |
| `train_run`（可选） | N | `export_multi.py` 写：每个训练帧来自哪次巡检（0 = 参考巡检，1、2… = 按 `--add` 顺序加进来的巡检） |
| `train_drop`（可选） | | `export_multi.py --only-added`：参考巡检只作测试 |

`scene.json`：数据来源（work、video）、抽帧步长、帧数、整段留出范围、点数、相机。

## 4. 实测管壁

| 文件 | 写它的脚本 | 键 |
|---|---|---|
| `surface.npz` | `pipesplat/export_surface.py vision WORK SCENE`（Windows） | `centres`、`normals`（单位）、`size`（√(2 × 面积)）：每个带贴图的 OpenMVS 三角形 |
| `surface_vhf.npz` | `pipesplat/vision_surface.py SCENE`（WSL；要先有 `surface.npz`） | 同上：高度场网格的三角形 + `surface.npz` 里落在拟合管道之外的三角形 |
| `mesh_vhf.npz` | 同上 | `xyz`、`tri`、`r`（半径场）、`u0`、`du`、`dth`、`vi`、`vj`、`open`：高度场网格（u 沿中心线、θ 绕中心线） |
| `centerline_vhf.npz` | 同上 | `u`、`C`（中心点）、`T`、`N`、`Bn`（平行移动的标架）、`th0` |

`train_gs.py` 把管壁每个三角形当作一个不透明薄圆片（直径约 1.2 × size）渲染，得到实测深度 D_s 和平滑法向。

## 5. 一次训练的输出（`GS_ROOT/g0/<名字>/run_<训练名>/`）

| 文件 | 内容 |
|---|---|
| `train.log` | 全部参数、每 500 次迭代的损失 / PSNR / 高斯数 / 显存、车灯模型学到的值、评分的逐帧行和汇总 |
| `model.pt` | `torch.save` 的字典：`means`、`scales`（log）、`quats`、`opacities`（logit）、`sh0`、`shN`、可选 `amb`（环境光，sigmoid 前）；车灯模型 `lamp_t`、`lamp_gain`、`lamp_off`、`lamp_logM`（或 `lamp_rad`、`lamp_logp`、`lamp_logq`）；曝光 `expo_gain`、`expo_off`；`K`、`size`（训练画布） |
| `summary_r2.json` | `args`（训练选项，`rescore.py` 从这里读）、`canvas`、`train_frames`、`iterations_done`、`train_minutes`、`peak_GB`、`gaussians`、`block_ranges_m`；每个测试组的中位数：`held_all`、`held_outside_blocks`、`block`、`xrun_*`…；每组帧数 `n` |
| `per_frame_r2.csv` | 每个测试帧一行：`tag`（held / block / xrun_…）、`frame`、`in_block`、`pos`、覆盖率，以及 G / B / Nal / D 四个候选的全部指标（`G_psnr`、`G_ssim`、`G_lpips`、`G_shift_px`、`G_fid_fine` ……）、`depth_rel_err` |
| `renders/<组>_<帧>.jpg` | 每 4 帧一张：视频帧 \| 对照 B \| 模型 |
| `gaussians.ply` | `export_ply.py`：通用 3DGS 格式（INRIA 布局），SuperSplat 等查看器可打开 |
| `summary_<后缀>.json`、`per_frame_<后缀>.csv` | `--eval-only` 重新评分的结果（`rescore.py` 默认后缀 `_rescore`，`score_xrun.py` 默认 `_xrun`） |

2026-11-02 之前训练的模型目录里还有 `summary.json`、`summary_alpha050.json`（旧量尺 v1 的评分，已不再写）。

---

## 6. GitHub Releases（早期阶段的结果，供查阅）

| Release | 内容 | 现在的用处 |
|---|---|---|
| `v1.0-results` | 前端（OpenMVG + OpenMVS）在各数据上的工作目录、带贴图网格、稠密点云、对比视频 | concrete_2、实验室管（15 m 管网 r1）、Pipe1 的前端工作目录：重新导出这三段的场景就从这里开始（PIPELINE.md 第 5 节） |
| `v2.0-results` | 锚定高斯模型（早期最好模型）、烘焙网格、对比视频、中间文件（管壁、中心线、灯光图）、**InPipe 的场景文件** | InPipe 的场景（`inpipe_172011_scene.zip`）和中间文件；灯光图 `lamp_maps.zip` |
| `v2.1-results` | 第二轮的 Z1 模型（锚定 + 车灯）、InPipe 两种光源烘焙网格（上传未完成：缺 8 个分卷） | 历史记录 |

主方法的模型没有上传（本机训练一次 24–63 分钟，按 PIPELINE.md 重新训练）。
