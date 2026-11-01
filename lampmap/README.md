# lampmap：对照方法 B 与图像指标

这个文件夹有两样东西，都被主方法的评测直接用到：

| 文件 | 作用 | 谁用它 |
|---|---|---|
| `lampmap.py` | **对照方法 B**：前端的 OpenMVS 带贴图网格从相机位姿渲染，再乘一张"固定在画面上的灯光图"。机器人的灯跟着相机走，它在管壁上投下的明暗（连同镜头暗角）每一帧几乎相同，所以一张画面固定的增益图就能补掉大部分明暗。B 的有效像素（掩码）就是所有模型的评分像素 | `pipesplat/export_g0.py`（把 B 的渲染和掩码写进每个场景）、`pipesplat/export_xrun.py` |
| `imgmetrics.py` | 全部图像指标：每帧增益 + 偏移匹配（`linear_match`）、PSNR、SSIM、LPIPS（AlexNet）、DIS 光流错位、细 / 中 / 粗三个频段的保真度（`ruler2`） | `pipesplat/train_gs.py`、`bake_vhf.py` |

## B 是什么、为什么要它

- **做法**：每张建模帧从自己的注册位姿渲染网格，先按每通道一个增益 + 一个偏移匹配亮度（去掉这一帧的曝光），再逐像素算 log(视频帧 / 渲染)，40 帧平均、6 px 高斯平滑，得到灯光图。用时：渲染 × exp(灯光图)，再拟合这一帧自己的增益 + 偏移。
- **作用**：它是"传统摄影测量交付物（网格 + 贴图）+ 最简单的灯光补偿"的上限，作为对照；同时它的掩码固定了"在哪些像素上比较"，所有模型（主方法、3DGS、2DGS……）都在同一批像素上评分（docs/EVALUATION.md 第 1 节）。
- **效果**（44 张序列内留出帧的中位数）：concrete_2 只用网格 25.88 dB，乘灯光图 29.82 dB；实验室管 A 20.87 → 25.26 dB。主方法在同样的帧上是 35.16 / 30.49 dB（docs/EVALUATION.md）。

## 单独运行（一般不需要）

`pipesplat/export_g0.py --lamp auto` 已经在导出场景时学好了灯光图（`<场景>/lamp_map.npy`）。只想看 B 本身的分数时：

```bash
python -I lampmap/lampmap.py <前端工作目录> <原视频> <输出目录> [--plain --crop X0 Y0 X1 Y1 --stretch-x S]
```

输出 `lamp_map.npy`（log 增益，画面大小）、`lamp_map.png`、`summary.json`、`per_frame.csv`、`renders/`（视频帧 | 网格 | 网格 × 灯光图）。
LPIPS 需要装了 torch + lpips 的 Python，并设 `TORCH_HOME`；没有时 LPIPS 记为 NaN，其余指标不受影响。
