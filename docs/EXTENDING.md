# 创新入口：在这个模型上做改进

给接下来要改模型的人（或 AI）：代码怎么组织、加一个新模块要动哪几处、怎样评测才算数、哪些方向已经试过、还有哪些问题没解决。
模型见 [MODEL.md](MODEL.md)，评测协议见 [EVALUATION.md](EVALUATION.md)。

---

## 1. 代码是怎样组织的

- **训练和评分在一个文件里**：`pipesplat/train_gs.py`。所有方法（主方法、3DGS、2DGS、锚定面元、各种消融）都是它的选项，同一套数据加载、画布、评分代码，
  所以任何两个模型的分数可以直接比较。改模型 = 给它加选项，不要复制出第二个训练脚本。
- **一次只改一处**：每个新选项默认关闭，打开时只改变一件事；选项之间的依赖写成 `assert`（例如 `--lamp-ambient` 必须和 `--lamp` 一起）。
- **场景与模型分开**：Windows 侧导出的 `scene.npz` 固定了训练帧、留出帧、位姿和评分像素；模型改动不需要重新导出场景，换场景也不需要改模型。
- **前端与模型分开**：`scripts/vision/`（SfM / MVS）只通过 `WORK/` 目录交给 `export_g0.py` / `export_surface.py`。

## 2. 加一个新模块的步骤（以"成像模型里加一项"为例）

1. **选项**：在 argparse 里加 `--my-term`（help 写清楚是什么、出处、为什么），在断言里写清它依赖什么。
2. **参数**：像 `lamp` / `expo` 那样建一个 `torch.nn.ParameterDict`，给每个参数单独的 Adam 优化器和学习率（加进 `optimizers`）。
   每帧参数（如 g_t）只对训练帧有意义；留出帧上没有，评分时要么不依赖每帧参数，要么像增益 / 偏移那样由评分拟合。
3. **训练路径**：在训练循环里"成像模型"那几行（`if a.lamp: … elif a.exposure: …`）或损失那几行加上它。渲染深度 `dep`、实测深度 `DA[n]`、车灯因子 `lamp_factor(view)` 都可以直接用。
4. **评分路径**：`render_raw()` 决定评分时渲染什么；新项如果改变画面，这里也要加。
5. **保存 / 读回**：训练结束保存 `model.pt` 时把新参数以前缀放进 `extra`；`--eval-only` 时按前缀读回（照 `lamp_*` 的写法）。
6. **重新评分**：如果新选项改变渲染，把它加进 `pipesplat/rescore.py` 的 `FLAGS` 或 `VALUES`，`rescore.py` / `score_xrun.py` 才会带着它评分。
7. **冒烟测试**：`train_gs.py SCENE OUT … --my-term --iters 600 --smoke`（训练、保存，不评分），再 `--eval-only --suffix _smoke` 跑一次评分路径。
8. **正式检验**：
   - 先写下判定规则（例如"主方法 + 新项 − 主方法在 ≥ 3 个测试组中位差 > 种子间差且 ≥ 80 % 的帧更好才采用"），再跑；
   - 种子 0，四段数据（15 m 管网、5 m 混凝土管、InPipe、Pipe1），与主方法同样的选项只多 `--my-term`；
   - `pipesplat/review/paired_stats.py 新/per_frame_r2.csv 主方法/per_frame_r2.csv` 看逐帧配对；
   - 差距与种子间差（0.01–0.14 dB）同一量级时补种子 1、2，用 `seeds_table.py` 报均值 ± 标准差；
   - 结果（含失败的）写进实验记录，方法和公式用文献里的通用名称。

## 3. 评测规则（改模型时不要改这些）

- 所有模型在**同一批像素**（对照 B 的掩码）上评分，每帧的增益 + 偏移由评分拟合；不要换评分像素、不要换测试帧。
- 四组测试都要报：序列内、整段留出、跨巡检、跨巡检 + 外推。只在序列内好的改动往往在离开轨迹时变差（3DGS 就是例子）。
- 模拟视频上 LPIPS 主要在比噪声颗粒，读它要结合 PSNR 和频段保真度；不要只凭 LPIPS 下结论。
- 渲染深度误差是相对实测管壁量的，被它监督的模型当然低，不能当几何精度。
- 逐帧 p 值只说明两个训练好的模型不同；方法之间的差别看多个种子。
- 一次只跑一个重任务（GPU 训练、OpenMVS 稠密、SfM 不要并行）；训练期间不要让电脑睡眠。

## 4. 已经试过的（不用再试，或要换思路才值得再试）

| 试过什么 | 结果 | 记录 |
|---|---|---|
| Gaussian-LIC2 式的稀疏深度项 + 不做密度控制（"G0"） | 序列内好，离开训练视角就崩 | 标签 `v2.1-results` 的 docs/06–08 |
| 激光雷达位姿 + 激光管壁（InPipe） | 激光管壁半径小 13–15 %，与图像毫米级错位，模型离开训练视角就崩；改用视觉位姿 + 视觉几何 | 同上 docs/10 X6–X7 |
| 锚定面元代替深度监督 | 序列内 / 整段留出低约 0.1 dB，跨巡检相同 | EVALUATION.md 第 4 节 |
| 物理车灯模型（径向渐晕 + 同位光源 + gamma + Minnaert，V1–V5） | 可辨识但序列内低约 1 dB；不加 gamma 不收敛 | EVALUATION.md 第 5 节 |
| 每帧曝光单独 | 没有增益，离开轨迹反而低 | EVALUATION.md 第 5 节 |
| 3DGS-MCMC（gsplat 默认参数） | 在 concrete_2 上发散 | 标签 `v2.1-results` 的 docs/11 |
| 学习的位姿修正、3DGUT 直接在原始畸变画面上训练、不透明面元、尺度上限 | 没有改善，代码已删除 | 标签 `v2.1-results` 的 docs/10 和 train_gs.py |
| 纹理模型（固定网格只学高分辨率贴图）、网格烘焙的多种纹理来源 | 网格比高斯模型低 5–10 dB，瓶颈是网格本身（平滑管壁 + 静态贴图，错位 3–5 px） | 标签 `v2.1-results` 的 docs/10–13；`bake_vhf.py --source video\|model\|delit\|unwrap` |
| 时域平均 + 去噪救低纹理管道的 SfM | 无效 | 标签 `v2.1-results` 的 docs/01 |
| 学习型外观（双边网格，NeRF-W 式外观嵌入） | **没做**（决定暂不做），是公平对照的缺口 | — |

## 5. 还没解决的问题（按影响排序）

1. **低纹理管道的前端**：SIFT 在光滑、深色管壁上匹配太少（Pipe2：相隔 1 cm 的帧只有约 78 个可靠匹配），SfM 断开或注册率低。
   方向：学习型特征和匹配（SuperPoint + LightGlue，Sarlin 等 hloc）；前馈式位姿 / 几何（DUSt3R、MASt3R、VGGT）；圆柱约束的光束法平差（Kagami 等 2020）。
   改 `scripts/vision/run_pipeline.py` 步骤 4–5（特征和匹配）或在步骤 8 之后换位姿来源；输出仍是 `WORK/sfm/recon/sfm_data.bin` 的话，后面全部不用改。
2. **管底的水和沉积物**：MVS 在水面没有一致的深度，模型把水面当成管壁的一部分。方向：水面掩码（OpenMVS DensifyPointCloud 支持 `--mask-path`；训练损失现在只有画布有效区 `VALID`，要加每帧掩码）；反光表面的成像项。
3. **反照率的可辨识性**：自由 M 分数高但圆周分布不可辨识（展开图管底亮带）；物理 M 可辨识但低约 1 dB。方向：多次巡检、多档灯光联合约束 M；在径向 M 上加小的自由残差；
   近场光度立体（Logothetis 等 2017；Quéau 等 2018）、内窥镜的同位光源方法（LightNeuS 2023、PR-ENDO 2024）。
4. **单目尺度漂移与绝对尺度**：Pipe1 尺度沿程 ±15 %；管径绝对值依赖标定。方向：计数器分段定尺度进 SfM；管径恒定作为光束法平差约束；已知尺寸的标记。
5. **只沿轴前进的外推范围很小**（跨巡检相机横向差 2–9 mm）：离开管轴的新视角（例如从管外、或大的横向偏移）没有约束。方向：管道先验（圆柱 / 高度场）作为表示的一部分；对称性约束。
6. **网格交付物**：比高斯模型低 5–10 dB。方向：从高斯模型直接提取网格（GOF、PGSR、2DGS 的网格提取）；可重光照的贴图（albedo + 环境光 + 车灯参数，`bake_vhf.py --two-light` 已输出）。
7. **多次巡检的利用**：灯光不同时联合训练有益；每次巡检单独的车灯参数、跨巡检的变化检测（缺陷随时间的变化）还没做。
8. **下游**：展开图上的缺陷检测（EN 13508-2 / MSCC 编码）、几何测量的独立真值都还没有数据。

## 6. 每个模块可以替换的方法（作用相近）

| 模块 | 现在用的 | 可替换 / 可借鉴 |
|---|---|---|
| A1 标定 | 棋盘格（Zhang 2000）；管径 + 计数器 | GeoCalib（ECCV 2024）；圆柱约束 BA（Kagami 2020） |
| A2 SfM | OpenMVG 增量式 | COLMAP、GLOMAP（ECCV 2024）、ORB-SLAM3、DROID-SLAM、DUSt3R / MASt3R / VGGT |
| A3 MVS | OpenMVS + 管道先验 | COLMAP PatchMatch、MVSNet / CasMVSNet；单目深度基础模型 Depth Anything V2、Metric3D v2、Depth Pro（Depth Anything 3 在管壁近处深度约错一倍，10-27 检查过） |
| 主体框架 | 3DGS（gsplat） | 2DGS、Gaussian surfels、PGSR、GOF、Mip-Splatting、3DGS-MCMC、Scaffold-GS、3DGUT（直接处理畸变相机） |
| B1 深度监督 | 稠密相对 L1 | DN-Splatter（深度 + 法向）、DS-NeRF、Dense Depth Priors、Hierarchical 3DGS 的深度正则 |
| B2 车灯 | 平方反比 + Lambert + 平场图 | LightNeuS、PR-ENDO、PPSNet、IRON、近场光度立体；渐晕与响应：Goldman 2010、DSO 光度标定 |
| B3 环境光 | Phong 环境光项 | GS-IR（间接光）、Relightable 3D Gaussians、GaNI、GLOW |
| B4 曝光 | 每帧仿射 | NeRF-W 外观嵌入、VastGaussian、WildGaussians、双边网格（Wang 等 2024）、Luminance-GS |
| C1 联合 SfM | OpenMVG 多会话 | hloc + SuperPoint / LightGlue（光照变化下的跨会话匹配）、ORB-SLAM3 Atlas |
| D2 展开图 | 多视角纹理烘焙 | MVS-Texturing、NeRF2Mesh |

## 7. 机器条件（作者的机器）

Ryzen 5 5600（6 核）、16 GB 内存、RTX 5060 Ti 8 GB；WSL 分 8 GB 内存。单个模型训练 30–80 分钟、显存峰值约 4–4.5 GB；
三种子 × 四段数据 × 一个方法约 6–8 小时 GPU。OpenMVS 稠密 500 帧约 30–60 分钟 CPU。
