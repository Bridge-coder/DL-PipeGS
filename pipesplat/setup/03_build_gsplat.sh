#!/bin/bash
# 第 0 步 / docs/07 第 4.5 节：从源码编译 gsplat（固定 commit）和 fused-ssim，再跑 check_gsplat.py。
# CPU 重任务：单独运行，不和 OpenMVS/COLMAP 等同时跑。
# 在 pipe3d 发行版里以 root 运行：wsl -d pipe3d -u root -- bash <本文件>
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"   # REPO, P (pipesplat), PY, GS, TORCH_HOME from scripts/paths.py
set -euo pipefail
LOG=$GS/logs/setup_03.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
# 只为本机显卡编译：计算能力取自 nvidia-smi（RTX 50 系 12.0、40 系 8.9、30 系 8.6）；要别的架构就先设 TORCH_CUDA_ARCH_LIST
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$CAP}"
# 一次只编一个文件：3 个 RasterizeToPixelsFromWorld3DGS*.cu 编译时每个要好几 GB 内存（最多约 7 GB）；WSL 8 GB 时 4 个并行会把
# 内存和交换区用光，编译器被杀（2026-11-09 实测）。WSL 内存 16 GB 以上可先设 MAX_JOBS=2
export MAX_JOBS="${MAX_JOBS:-1}"
VENV=/opt/gsenv
SRC=/opt/src/gsplat
GSPLAT_COMMIT=512d366b6707           # 2026-09-19 的 main（gsplat 1.6.0 开发版）
FUSED_SSIM="git+https://github.com/rahul-goel/fused-ssim@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== $(date '+%F %T') clone gsplat @ $GSPLAT_COMMIT"
mkdir -p /opt/src
[ -d "$SRC/.git" ] || git clone https://github.com/nerfstudio-project/gsplat.git "$SRC"
git -C "$SRC" checkout "$GSPLAT_COMMIT"
git -C "$SRC" submodule update --init gsplat/cuda/csrc/third_party/glm   # 只要 glm；googletest 只用于测试
git -C "$SRC" log -1 --format='%H %cd %s'

# 换了显卡再跑：先卸掉旧的编译结果、删掉源码里的 build/，否则 setuptools 认为扩展"已是最新"，把旧架构的 .so 原样装回去
"$VENV/bin/pip" uninstall -y gsplat fused-ssim
rm -rf "$SRC/build"

echo "== $(date '+%F %T') build + install gsplat (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST MAX_JOBS=$MAX_JOBS)"
"$VENV/bin/pip" install --no-cache-dir --no-build-isolation -v "$SRC"

echo "== $(date '+%F %T') build + install fused-ssim"
"$VENV/bin/pip" install --no-cache-dir --no-build-isolation "$FUSED_SSIM"

echo "== $(date '+%F %T') check"
"$VENV/bin/python" "$HERE/check_gsplat.py"

"$VENV/bin/pip" list > $GS/logs/pip_list_03.txt
du -sh "$VENV" /opt/src
df -h / | tail -n 1
echo "== $(date '+%F %T') done"
