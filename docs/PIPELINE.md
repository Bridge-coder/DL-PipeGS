# 全流程运行：从下载一段管道视频到三维模型

本文件是操作手册：装环境、处理一段新的管道视频（下载 → 标定 → 前端 → 场景 → 训练 → 评分 → 导出），以及复现四段实验数据。
第 4 节是 2026-11-02 在一段从未用过的视频（Sheffield 真实污水管 Pipe2）上从下载开始完整跑通的记录。
模型本身见 [MODEL.md](MODEL.md)，文件格式见 [DATA.md](DATA.md)。

---

## 0. 流程一览

| 阶段 | 在哪运行 | 做什么 | 输出 | 用时（约 500 帧） |
|---|---|---|---|---|
| 下载 | Windows | `scripts/download_data.py` | 视频 + 标定视频 + 真值 | 1 分钟 |
| 标定 | Windows | `scripts/vision/calibrate_checkerboard.py` | `camera.json`（f、cx、cy、k1、拉伸） | 2 分钟 |
| frontend | Windows（CPU） | `scripts/vision/run_pipeline.py` 步骤 1–13：抽帧、SIFT、匹配、SfM、OpenMVS 稠密 / 网格 / 贴图 | `WORK/` | 15–25 分钟 SfM + 30–60 分钟 MVS |
| gt | Windows | 真值 → `counter.csv`；`eval_gt_scale.py` | 定位误差、推出的管径 | 1 分钟 |
| scene | Windows | `pipesplat/export_g0.py` | `SCENE/scene.npz` | 3–5 分钟 |
| surface | Windows + WSL | `export_surface.py`（OpenMVS 网格，总是做）；选 `vhf` 时再 `vision_surface.py`（管道先验，建在网格之上） | `surface.npz`、`surface_vhf.npz` | 1–5 分钟 |
| train | WSL（GPU） | `pipesplat/train_gs.py` 主方法 + 评分 | `SCENE/run_main/` | 30–80 分钟 |
| export | WSL | `export_ply.py`、`bake_vhf.py` | `.ply`、带纹理 OBJ | 10–20 分钟 |
| figures | Windows | `pipesplat/review/pipe_figures.py` | 效果图 | 5 分钟 |

`scripts/pipeline.py` 把 frontend 到 figures 串成一条命令；每个阶段输出已存在就跳过，中断后再运行同一条命令即可接着做。
**一次只跑一个重任务**：16 GB 内存、6 核的机器上同时跑 OpenMVS 稠密 + SfM + GPU 训练曾导致蓝屏。**训练期间不要让电脑睡眠**（睡眠会停掉 WSL 虚拟机，训练白跑）。

---

## 1. 环境（一次性）

**硬件**：Windows 10/11 x64；NVIDIA 显卡 ≥ 8 GB 显存（作者用 RTX 5060 Ti 8 GB）；16 GB 内存；前端不需要 GPU。

### 1.1 Windows 侧

```bash
git clone https://github.com/Bridge-coder/PipeRecon3D.git
cd PipeRecon3D
pip install -r requirements.txt            # Python 3.13：numpy、opencv-python、matplotlib、pillow
python tools/install_third_party.py        # OpenMVG 1.6 + OpenMVS 2.4.0（官方发布页，校验 SHA-256，解压到 third_party/）
python scripts/paths.py                    # 检查每个路径
```

- 不要换成 OpenMVG 2.0：它的 Windows 程序用了 AVX-512，在没有该指令集的 CPU（如 Ryzen 5000）上直接崩溃；脚本按 1.6 的程序名和主点约定写的。
- 数据和大文件不放在仓库里：复制 `config/paths.example.json` 为 `config/paths.json`（git 忽略）后改 `WORK_NEW`、`GS_ROOT` 等，或用环境变量 `PIPE3D_<键>` 临时覆盖。
- 中文路径：`cv2.imwrite` 不能写非 ASCII 路径，脚本里统一用 `cv2.imencode(...).tofile(...)`。

### 1.2 WSL 侧（高斯训练）

1. `C:\Users\<你>\.wslconfig` 给 WSL 足够内存：`[wsl2]` 下 `memory=8GB`、`swap=8GB`。
2. 建发行版（名字 `pipe3d`；放在空间大的盘上）：

   ```bash
   wsl --install Ubuntu-24.04 --name pipe3d --location D:\wsl\pipe3d --no-launch
   wsl -d pipe3d
   ```

3. 依次运行 4 个安装脚本（以 root；第 4 个编译 gsplat，是 CPU 重任务，单独跑）：

   ```bash
   wsl -d pipe3d -u root -- bash /mnt/<盘>/<路径>/PipeRecon3D/pipesplat/setup/01a_system_packages.sh
   wsl -d pipe3d -u root -- bash /mnt/<盘>/<路径>/PipeRecon3D/pipesplat/setup/01b_cuda_toolkit.sh
   wsl -d pipe3d -u root -- bash /mnt/<盘>/<路径>/PipeRecon3D/pipesplat/setup/02_python_env.sh
   wsl -d pipe3d -u root -- bash /mnt/<盘>/<路径>/PipeRecon3D/pipesplat/setup/03_build_gsplat.sh
   ```

   得到 `/opt/gsenv`（Python 3.12、torch 2.9.1 + CUDA 12.8、gsplat 1.6.0 固定在 commit `512d366`、fused-ssim、lpips、scipy）。
   **换了显卡**：`03_build_gsplat.sh` 里的 `TORCH_CUDA_ARCH_LIST="12.0"` 只适用于 RTX 50 系列，按你的显卡改（RTX 40 系列 `8.9`，RTX 30 系列 `8.6`）。
4. 检查：`wsl -d pipe3d -u root -- /opt/gsenv/bin/python /mnt/<…>/pipesplat/setup/check_gsplat.py`。
5. LPIPS 的 AlexNet 权重第一次评分时自动下载到 `TORCH_CACHE`（需要联网一次）。

`scripts/pipeline.py` 从 Windows 调用 WSL（`wsl -d pipe3d -u root -e …`，不经过 shell）；发行版名和 Python 位置可用环境变量 `PIPE3D_WSL_DISTRO`、`PIPE3D_WSL_PYTHON` 改。

---

## 2. 一段新视频：步骤

### 2.1 下载

公开数据集（DOI、许可、MD5 见 [DATA.md](DATA.md) 第 1 节）：

```bash
python -I scripts/download_data.py --list sheffield-sewer
python -I scripts/download_data.py sheffield-sewer D:/data/sewer Pipe2.avi "Calibration Checkerboard.mp4" "Ground truth distances.csv"
```

自己的视频直接用。要求：前视相机沿管轴前进；灯装在相机旁；原始分辨率、尽量少压缩；至少 4–5 m 连续管道；同一相机、同一变焦的标定视频（或一段等径直管 + 已知行程）。

### 2.2 看视频，定参数

先拼一张预览图（每隔几百帧取一帧），确认：

| 参数 | 怎么定 | 例（Pipe2） |
|---|---|---|
| `--crop X0 Y0 X1 Y1` | 裁掉烧录在画面上的文字（距离、日期）；静止的文字会变成"不动的特征点"，把 SfM 拉坏 | `0 0 720 490`（计数器在第 495–520 行） |
| `--every N` | 让相邻抽帧之间机器人前进约 1 cm：N ≈ 0.01 m ÷ (速度 m/s ÷ 帧率) | 13 cm/s、25 fps → 每帧 5.2 mm → `--every 2` |
| `--first` / `--last` | 只用机器人在管道里匀速前进的一段（去掉起步前静止、检查井、云台转动、倒车） | `450 1500`（计数器 0.2 → 5.1 m） |
| `--diameter-m` | 管道内径（只决定模型的公制尺度）；不知道时先用 0.30，有真值时看 gt 阶段推出的管径 | 0.30 |
| `--surface` | `mesh`（OpenMVS 网格，纹理足的混凝土管）或 `vhf`（管道先验拟合；深色、光滑、有水的管） | `vhf`（真实污水管） |

### 2.3 标定相机

- 有棋盘格视频（推荐）：

  ```bash
  python -I scripts/vision/calibrate_checkerboard.py "D:/data/sewer/Calibration Checkerboard.mp4" D:/work/camera.json --pattern 9 6 --crop 190 0 1090 720 --resize 720 576
  ```

  `--crop/--resize` 把标定视频的画面换算到检测视频的像素（这个数据集的标定视频是同一 PAL 画面放大 1.25 倍嵌在 1280 × 720 里）。
  输出 f、cx、cy、k1 和横向拉伸（模拟视频的像素不是方的）。`pipeline.py --camera-json` 直接读它，并按 `--crop` 的左上角平移主点。
- 没有棋盘格、但有一段等径直管和行程真值：`scripts/vision/calibrate.py` 先扫 k1（让管径沿程不变），再调 f（让行程等于真值）。
- Sheffield Minicam 的视频直接用 `--camera sheffield`（f 806、k1 −0.33，作者已标定）。
- **不要让 SfM 自己估焦距**：沿管轴前进时焦距几乎不可观测，COLMAP 默认自标定把管轴拉长了 42 %。
- 怀疑标定不属于这台相机：`scripts/analysis/focal_check_rotation.py`（视频里有云台转动、机器人不动的一段时直接测 f）、`focal_check_rings.py`（用真值里程测 f × 管半径）。

### 2.4 一条命令

```bash
python -I scripts/pipeline.py sewer_pipe2 D:/data/sewer/Pipe2.avi --camera-json D:/work/camera.json --every 2 --first 450 --last 1500 --crop 0 0 720 490 --loose --gt "D:/data/sewer/Ground truth distances.csv" --gt-row Pipe2 --to gt
python -I scripts/pipeline.py sewer_pipe2 D:/data/sewer/Pipe2.avi --camera-json D:/work/camera.json --every 2 --first 450 --last 1500 --crop 0 0 720 490 --loose --gt "D:/data/sewer/Ground truth distances.csv" --gt-row Pipe2 --diameter-m 0.194 --surface vhf
```

第一条跑到真值检验为止，看 `WORK/gt_eval.json` 里推出的管径（Pipe2：194 mm）；第二条用它定尺度，从场景导出接着跑完（前端和真值检验已完成，自动跳过）。
不知道管径、也没有真值时直接一条命令跑完，`--diameter-m` 用估计值（默认 0.30）：它决定模型的公制尺度；几个以毫米计的参数（车灯法向的 20 mm 平滑、高度场管壁的网格）随之按比例变化，估计值差几十个百分点以内时影响很小。`--work`、`--scene` 不给时用 `WORK_NEW/<名字>/work` 和 `GS_ROOT/g0/<名字>`。
`--to scene` 只跑到某个阶段，`--from train` 从某个阶段重做；`python -I scripts/pipeline.py --stages` 列出全部阶段。

### 2.5 检查结果

| 看什么 | 在哪 | 正常的样子 |
|---|---|---|
| SfM 注册率 | 日志 `[8] … registered N of M frames` | ≥ 90 %；低于 90 % 时只有一部分视频被重建（见 2.6） |
| 定位误差 / 推出的管径 | `WORK/gt_eval.json`（有真值时） | 行程误差几个百分点以内；推出的管径接近实际 |
| 稠密点 | `WORK/mvs/scene_dense.ply`（MeshLab 打开） | 点落在管壁上、整圈覆盖 |
| 训练 | `SCENE/run_main/train.log` 末尾和 pipeline 打印的两行 | 序列内 PSNR 明显高于对照 B，接近"对齐后的下一帧"；整段留出不崩；深度误差 ~1–2 % |
| 模型 | `SCENE/run_main/gaussians.ply` | 用 SuperSplat（https://superspl.at/editor）等 3DGS 查看器从管内看 |
| 网格 / 效果图 | `SCENE/run_main_mesh/obj/model.obj`、`figures/fig*.jpg` | MeshLab / Blender 打开；从管外看整条管道 |

### 2.6 常见问题

| 现象 | 原因 | 办法 |
|---|---|---|
| 第 7 步 `no usable initial pair found` | 管壁纹理太少，相隔 6 帧以上的帧对凑不出 60 个匹配 | 加 `--loose`（门槛 40 个匹配），用 `--frontend-steps 7 8 9 10 11 12 13` 只重跑后面的步骤 |
| 注册率 < 90 % | SfM 在颠簸、无纹理处断开 | 结果只覆盖一部分；或用 `run_pipeline.py --part b --views FIRST LAST` 分段重建，再 `merge_parts.py` 拼接 |
| 管子越走越细 / 行程差 10–20 % | 焦距或 k1 不对 | 重新标定；用 `focal_check_*.py` 核实 |
| DensifyPointCloud 崩溃（0xC0000409） | 稀疏模型里有落在相机背后的点 | 第 10 步的 `clean_sfm.py` 已处理；仍崩溃看 `LOGS/<工作目录>/step11_*.log` |
| 稠密重建中断 | — | 删掉 `WORK/mvs/depth*.dmap`，从步骤 11 重跑（`--frontend-steps 11 12 13`） |
| `WSL distro pipe3d did not start`（HCS 0x800705aa） | 可用内存不够（`.wslconfig` 要 8 GB） | 关掉占内存的程序；`pipeline.py` 自己会重试 6 次 |
| 训练 `OUT OF MEMORY` | 稠密点太多 | `--points-every 2`（pipeline.py 在 > 2.5 M 点时自动设） |
| 训练停在中途、WSL 停了 | 电脑睡眠了 | `pipeline.py` 运行时阻止空闲睡眠，但手动睡眠、合盖仍会打断；删掉半途的 `run_main/` 重跑 train |
| 对比视频 / 场景里画面对不上 | 非 Sheffield 视频没给 `--plain --crop --stretch-x` | 用 `pipeline.py`（自动传）；单独运行脚本时三个参数和前端一致 |

---

## 3. 不用 pipeline.py：每一步的命令

```bash
# Windows，仓库根目录。W = 前端工作目录，S = 场景目录，V = 视频；相机参数来自 camera.json
python -I scripts/vision/run_pipeline.py W --video V --every 2 --first 450 --last 1500 --plain --crop 0 0 720 490 --stretch-x 1.0155 --focal 494.88 --cx 335.99 --cy 265.30 --k1 -0.2526 --loose 1 2 3 4 5 7 8 9 10 11 12 13
python -I scripts/vision/eval_gt_scale.py W W/gt_eval                         # 先把真值写成 W/counter.csv（source_frame,counter_m）
set PIPE3D_PIPE_DIAMETER_M=0.30                                               # PowerShell：$env:PIPE3D_PIPE_DIAMETER_M='0.30'
python -I pipesplat/export_g0.py vision W V S --lamp auto --plain --crop 0 0 720 490 --stretch-x 1.0155
python -I pipesplat/export_surface.py vision W S                               # 管壁 = OpenMVS 网格
```

```bash
# WSL：wsl -d pipe3d -u root，然后
source /mnt/<盘>/<路径>/PipeRecon3D/pipesplat/env.sh      # P = pipesplat/、GS = GS_ROOT、PY = /opt/gsenv/bin/python、TORCH_HOME
$PY $P/vision_surface.py S                                 # 管壁 = 管道先验高度场（--surface vhf 时；也是烘焙网格的几何；要先有 S/surface.npz）
$PY $P/train_gs.py S S/run_main --baseline 3dgs --baseline-depth 1.0 --lamp --lamp-smooth-mm 20 --lamp-ambient [--surface surface_vhf.npz] [--points-every 2]
$PY $P/export_ply.py S/run_main                            # -> S/run_main/gaussians.ply
$PY $P/bake_vhf.py S S/run_main S/run_main_mesh --two-light --source video
```

```bash
# Windows：效果图
python -I pipesplat/review/pipe_figures.py S S/run_main_mesh/obj S/run_main_mesh/figures --title 名字 --model-run S/run_main
```

几个训练一个接一个跑：`pipesplat/review/queue.sh "数据 训练名 选项" "数据 训练名 选项" …`（已有 `summary_r2.json` 的跳过）。

---

## 4. 实例：Pipe2 端到端测试（2026-11-02）

目的：在一段**从未用过**的视频上，从下载开始，只用本仓库的脚本，检验整条流程能否跑通、得到三维模型。
数据：Sheffield 真实污水管数据集的 Pipe2（CC BY-NC-SA 4.0），工作区 `D:/pipe3d_e2e`（与作者原来的数据完全分开）。

**命令**（实际运行的，Git Bash；PowerShell 同样）：

```bash
python -I scripts/download_data.py sheffield-sewer D:/pipe3d_e2e/data Pipe2.avi "Calibration Checkerboard.mp4" "Ground truth distances.csv"
python -I scripts/vision/calibrate_checkerboard.py "D:/pipe3d_e2e/data/Calibration Checkerboard.mp4" D:/pipe3d_e2e/camera.json --pattern 9 6 --crop 190 0 1090 720 --resize 720 576
python -I scripts/pipeline.py sewer_pipe2 D:/pipe3d_e2e/data/Pipe2.avi --camera-json D:/pipe3d_e2e/camera.json --every 2 --first 450 --last 1500 --crop 0 0 720 490 --loose --gt "D:/pipe3d_e2e/data/Ground truth distances.csv" --gt-row Pipe2 --diameter-m 0.194 --surface vhf --work D:/pipe3d_e2e/work/sewer_pipe2 --scene D:/pipe3d_e2e/gs/g0/sewer_pipe2
```

**每一步**（RTX 5060 Ti 8 GB、Ryzen 5 5600、16 GB 内存）：

| 步骤 | 用时 | 结果 |
|---|---|---|
| 下载 | 约 1 分钟（132 MB） | 3 个文件，MD5 与 figshare 一致 |
| 看视频、定参数 | — | 720 × 576、25 fps、15 174 帧。计数器烧录在第 495–520 行 → `--crop 0 0 720 490`；第 450 帧起前进、约 13 cm/s → `--every 2`（每帧约 1 cm）；第 2758 帧进检查井，第 8276 帧后在管端转镜头、倒车 → 只用第 450–1500 帧（计数器 0.2–5.1 m） |
| 标定 | 约 2 分钟 | 60 张棋盘格，像素宽高比 1.0155，f 494.9 px、k1 −0.253、重投影 0.33 px（10-17 给 Pipe1 用同一视频标定的是 f 489.7、k1 −0.248，相差约 1 %） |
| frontend，第一次 | 7 分钟后停下 | 第 7 步 `no usable initial pair found`：管壁光滑、纹理少，可靠匹配数随帧间隔下降很快（中位数：相邻 1 帧 147、2 帧 78、3 帧 58、6 帧 39），没有相隔 ≥ 6 帧且 ≥ 60 个匹配的帧对 |
| frontend，加 `--loose` 从第 7 步重跑 | 52.8 分钟 | 8 个候选起始帧对，每次 SfM 都注册 412–413 / 526 帧（各约 2.6 分钟），保留 413 帧的一次：一段连续，对应真值 1.0–5.1 m（开头 0.2–1.0 m 没接上）。OpenMVS 稠密 25 分钟（243 万点）、网格 1 分钟、贴图 5.5 分钟 |
| gt | 1 秒 | 定位误差 MAE 15.3 cm（这段长度的 3.7 %），最大 43 cm；两个 2 m 窗口的局部尺度相差 31 %（单目尺度漂移）；按真值尺度推出的管径约 194 mm → 后面用 `--diameter-m 0.194` |
| scene | 4.9 分钟 | 413 帧（20 帧在两段 10 cm 整段留出里，训练 393 帧），44 张序列内留出帧；对照 B 的灯光图自动学习 |
| surface | 2.2 分钟 | 网格管壁 + 管道先验高度场；高度场半径中位 97 mm，但 p5–p95 为 45–151 mm（见下面"几何"） |
| train | 42.6 分钟 | 98 万个高斯，显存峰值 4.9 GB，没有非有限的损失或梯度；学到的光源偏移 (−2, −6, −21) mm |
| export | 4.7 分钟 | `gaussians.ply`（244 MB）；带纹理网格 `model.obj` + `texture.png`（另有可重新打光的 `albedo.png`、`ambient.png`、`lamp.npz`） |
| figures | 0.9 分钟 | 整体、三视图、剖切、展开图、留出帧对照共 5 张 |

**测试中发现并修好的问题**：
1. 低纹理管道 SfM 起步失败 → 流程里本来就有的 `--loose`（`pipeline.py` 新增了这个选项和 `--frontend-steps`，只重跑第 7 步以后）。
2. `--surface vhf` 时 `vision_surface.py` 要用 `surface.npz`（把网格里落在拟合管道之外的三角形并进来），新流程原来没先导出它 → surface 阶段改为总是先导出网格管壁。
3. WSL 有一次没起来（HCS 0x800705aa，可用内存 7.8 GB 而 `.wslconfig` 要 8 GB）→ 重试一次就好；`pipeline.py` 自己会重试 6 次。

**结果**（量尺 v2，PSNR dB；对照 B 和"对齐后的下一帧"在同一批像素上）：

| | 序列内留出（43 帧） | 整段留出（20 帧） |
|---|---|---|
| 主方法（高斯模型） | **36.59**（LPIPS 0.333，错位 2.1 px） | **34.35**（LPIPS 0.362，错位 1.2 px） |
| 主方法烘焙的带纹理网格 | 33.69 | 32.55 |
| 对照 B：OpenMVS 网格贴图 × 平场灯光图 | 22.74（错位 8.8 px） | 22.58（错位 11.9 px） |
| 参照：视频自己的下一帧（对齐后） | 38.17 | 38.22 |

渲染深度对稠密点的误差 1.2 % / 1.3 %。没参加训练的帧上，主方法离"视频自身噪声决定的上限"只差 1.6–3.9 dB，比传统的网格贴图交付物高约 12–14 dB。
同一场景上的对照（种子 0）：3DGS 26.91 / 19.36 dB（渲染深度错约 90 %），深度监督 3DGS 36.30 / 32.40 dB；主方法比深度监督 3DGS 在整段留出上高 1.83 dB（全部帧），详见 [EVALUATION.md](EVALUATION.md) 第 8 节。

**几何**：从管外看，模型是一根起点细、终点粗的喇叭形管（高度场半径沿程约 45 → 150 mm），与真值检验的 31 % 尺度漂移一致。
用视频末尾镜头原地转动的一段直接测焦距（`scripts/analysis/focal_check_rotation.py`，第 8320–8940 帧）没有结论：在任何焦距下残差都在 7–9 px（镜头离管壁只有几厘米，云台转轴不在光心，视差盖过了纯转动），
所以分不清是"低纹理管里单目 SfM 的尺度漂移"还是"标定视频的变焦与这段视频不同"。画面预测不受影响（位姿、几何、图像彼此一致），但**这段的管径和沿程尺寸不能直接当测量值**。
对策（[EXTENDING.md](EXTENDING.md) 第 5 节第 1、4 条）：学习型特征提高匹配、按计数器分段定尺度、光束法平差里加等径约束；或在同一相机、同一变焦下拍一段已知管径的直管，用 `scripts/vision/calibrate.py` 核实内参。

---

## 5. 复现四段实验数据

主要实验在四段数据上（[EVALUATION.md](EVALUATION.md)）。场景文件带图像（每段 0.4–1 GB），没有上传；按下面重新导出。

**下载**（GitHub Releases + 原始视频）：

```bash
gh release download v1.0-results --repo Bridge-coder/PipeRecon3D --dir D:/dl/v1.0 -p sheffield_concrete_2.zip -p concrete_lab.zip -p sewer_pipe1.zip
gh release download v2.0-results --repo Bridge-coder/PipeRecon3D --dir D:/dl/v2.0 -p "*_intermediates.zip" -p inpipe_172011_scene.zip -p lamp_maps.zip
python -I scripts/download_data.py sheffield-concrete5m D:/data/concrete5m 300mm_concrete_5meters_ON_2lm_2u_CLOSED_2.mp4
python -I scripts/download_data.py sheffield-mix15m D:/data/mix15m 300mm_mix_15meters_ON_3lm_3u_CLOSED_1_stop_at_14meters.mp4
python -I scripts/download_data.py sheffield-sewer D:/data/sewer Pipe1.avi
```

**放置**：`sheffield_concrete_2.zip` 解压到 `WORK_SHEFFIELD`，把 `sheffield_concrete_2/` 改名为 `concrete_2/`；`concrete_lab.zip` 解压到 `WORK_NEW`；
`sewer_pipe1.zip` 解压到 `WORK_NEW/sewer_pipe1/` 并把 `sewer_pipe1/` 改名为 `work/`；`*_intermediates.zip`、`inpipe_172011_scene.zip` 解压到 `GS_ROOT`（得到 `g0/<数据>/` 下的管壁、中心线、灯光图和 InPipe 的 scene.npz）；
`lamp_maps.zip` 解压到仓库根目录（`lampmap/results/<数据>/lamp_map.npy`）。下载后对照各 Release 的 `SHA256SUMS.txt` 校验。

**导出场景**（Windows；用作者当时的灯光图，B 的渲染和评分像素才与记录一致）：

```bash
python -I pipesplat/export_g0.py vision <WORK_SHEFFIELD>/concrete_2 <视频>/300mm_concrete_5meters_ON_2lm_2u_CLOSED_2.mp4 <GS_ROOT>/g0/concrete_2 --lamp lampmap/results/concrete_2/lamp_map.npy
python -I pipesplat/export_surface.py vision <WORK_SHEFFIELD>/concrete_2 <GS_ROOT>/g0/concrete_2
powershell -ExecutionPolicy Bypass -File pipesplat/review/export_lab.ps1 -Video <视频>/300mm_mix_15meters_ON_3lm_3u_CLOSED_1_stop_at_14meters.mp4   # 15 m 管网 r1：场景 + 管壁
$env:PIPE3D_PIPE_DIAMETER_M='0.435'    # Pipe1（PowerShell）
python -I pipesplat/export_g0.py vision <WORK_NEW>/sewer_pipe1/work <视频>/Pipe1.avi <GS_ROOT>/g0/sewer_pipe1 --lamp <GS_ROOT>/g0/sewer_pipe1/lamp_map.npy --plain --crop 0 0 720 490 --stretch-x 1.0159
# InPipe：g0/inpipe_vision/scene.npz 已在 inpipe_172011_scene.zip 里
```

2026-10-26 的恢复测试：只用 GitHub 上的文件重新导出的 Pipe1 场景与作者的逐数组逐位相同，重新评分的 258 个汇总数、逐帧表全部相同。

**训练**（WSL；数据名 = `GS_ROOT/g0/` 下的文件夹）：

| 模型 | 选项 |
|---|---|
| 主方法 | `--baseline 3dgs --baseline-depth 1.0 --lamp --lamp-smooth-mm 20 --lamp-ambient` |
| 深度监督 3DGS | `--baseline 3dgs --baseline-depth 1.0` |
| 3DGS / 2DGS | `--baseline 3dgs` / `--baseline 2dgs` |
| 深度监督 3DGS + 车灯（不含环境光项） | `--baseline 3dgs --baseline-depth 1.0 --lamp --lamp-smooth-mm 20` |
| 只有每帧曝光 | `--baseline 3dgs --baseline-depth 1.0 --exposure` |
| 锚定面元 + 车灯 | `--anchor --lamp --lamp-smooth-mm 20 --alpha-w 0.1` |
| 数据相关的选项 | 15 m 管网（`concrete_lab`）加 `--points-every 2`；InPipe（`inpipe_vision`）、Pipe1（`sewer_pipe1`）加 `--surface surface_vhf.npz`；种子 `--seed 1` / `--seed 2` |

例：`bash pipesplat/review/queue.sh "concrete_2 run_main --baseline 3dgs --baseline-depth 1.0 --lamp --lamp-smooth-mm 20 --lamp-ambient"`。
重新训练的模型与记录里的不会逐位相同（GPU 上的求和顺序），种子之间的差 0.01–0.14 dB。已训练的模型重新评分是逐位（带车灯模型的到 10⁻⁶ 的浮点噪声）相同的：

```bash
wsl -d pipe3d -u root -- /opt/gsenv/bin/python /mnt/<…>/pipesplat/rescore.py concrete_2/run_main sewer_pipe1/run_main
```

---

## 6. 多次巡检评测（跨巡检测试集、联合训练）

需要同一段管道的 ≥ 2 次巡检（同一相机、各自有计数器）。15 m 管网用了 r1–r5，5 m 混凝土管用了 concrete_1–5。

1. 每次巡检单独跑前端步骤 2–5（CLAHE 帧、SIFT，`run_pipeline.py … 1 2 3 4 5`），读出计数器：`scripts/vision/counter_ocr.py`（模板取自人工核对过的视频）。
2. 联合 SfM：`scripts/vision/joint_sfm.py OUT --run r1 W1 FIRST LAST --run r2 W2 FIRST LAST …`（第一个是参考巡检）。
3. 位姿迁移到参考巡检的原重建：`ALIGN_SIGMA_M=0.10 python -I scripts/vision/align_runs.py OUT r1 W1 ALIGN_DIR`（输出留一法误差）。
4. 测试集：`pipesplat/export_xrun.py W1 ALIGN_DIR ALIGN_DIR/query_r2.csv 视频_r2 S/xrun_r2.npz --lamp S/lamp_map.npy --query-counter 计数器_r2 --n 60`；
   加 `--blocks-from S/scene.npz` 得到落在整段留出里的帧（`xblk_r2.npz`）。
5. 训练时加 `--extra-test xrun_r2=xrun_r2.npz xblk_r2=xblk_r2.npz …`，或对已训练模型 `pipesplat/review/score_xrun.py 数据 训练名 --test xrun_r2=xrun_r2.npz …`。
6. 联合训练的场景：`pipesplat/export_multi.py S S_r12 --add r2 ALIGN_DIR ALIGN_DIR/query_r2.csv 视频_r2 [--every 2 --ref-every 2]`。
7. 表和统计：`pipesplat/review/table_x5.py`、`seeds_table.py`、`paired_stats.py`。
