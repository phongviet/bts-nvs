#!/usr/bin/env bash
# Provision a rented 4090 (SECURE) for the splatfacto-big + RaDe-GS-depth DIBR
# refiner pipeline (the "airace" env). Image assumed: runpod/pytorch 2.4 cu121 -devel
# (has nvcc, which gsplat needs to JIT its kernels at first run).
#
#   usage: bash provision_pod_nerfstudio.sh
# Idempotent-ish: re-running rebuilds the env from scratch only if missing.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
ARCH="${TORCH_CUDA_ARCH_LIST:-8.9}"   # 4090 = Ada sm_89
export TORCH_CUDA_ARCH_LIST="$ARCH"

MC=/opt/miniconda
if [ ! -d "$MC" ]; then
  echo "== installing miniconda =="
  wget -qO /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/mc.sh -b -p "$MC"
fi
export PATH=$MC/bin:$PATH
source $MC/etc/profile.d/conda.sh

# Anaconda ToS must be accepted BEFORE any `conda create`, or a fresh pod aborts
# instantly with CondaToSNonInteractiveError and the GPU sits there billing.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

if ! conda env list | grep -q "/airace$"; then
  echo "== creating airace env (py3.10) =="
  conda create -y -n airace python=3.10
fi
conda activate airace

echo "== installing CUDA toolkit 12.1 + gcc-12 into env (for gsplat JIT) =="
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-toolkit || true
conda install -y -c conda-forge gxx=12 gcc=12 || true
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

echo "== torch 2.4.1 cu121 =="
pip install --quiet torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo "== nerfstudio 1.1.5 + gsplat (pulls the DIBR/refiner deps too) =="
pip install --quiet nerfstudio==1.1.5
pip install --quiet gsplat==1.4.0
# refiner extras (10_refiner_pilot / 04 warper): most come with nerfstudio; ensure these
pip install --quiet opencv-python-headless imageio tqdm lpips

echo "== sanity =="
python - <<'PY'
import torch, nerfstudio, gsplat, cv2
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("nerfstudio OK; gsplat", gsplat.__version__)
PY
which ns-train && echo ">>> PROVISION COMPLETE (airace)"
