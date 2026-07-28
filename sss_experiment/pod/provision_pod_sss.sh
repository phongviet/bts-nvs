#!/usr/bin/env bash
# Provision a fresh RunPod box (RTX 4090, CUDA 12.1) for the SSS bonsai gate-1 run.
# Creates conda env `sss` (torch 2.4.1+cu121), builds the SSS CUDA extensions with
# the RESULTS.md gotchas handled, and installs the bts-nvs metrics deps.
#
# Assumes this repo tree has been scp'd to $HOME/sss_gate1/ (see pack_gate1.sh).
# Idempotent-ish: re-running skips the env create if `sss` already exists.
set -euo pipefail

ROOT=${ROOT:-$HOME/sss_gate1}
SSS=$ROOT/3D-student-splatting-and-scooping

# --- miniconda (idempotent: reuse existing install; conda is not on PATH in a fresh shell) ---
if [ ! -d /opt/miniconda ]; then
  echo "== installing miniconda =="
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p /opt/miniconda
fi
export PATH=/opt/miniconda/bin:$PATH
source /opt/miniconda/etc/profile.d/conda.sh

# Accept Anaconda channel ToS (newer conda blocks `conda create` non-interactively
# on the default channels until accepted).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

# --- env: sss (torch 2.4.1+cu121, matches the local `sss` env) ---
if ! conda env list | grep -q "^sss "; then
  echo "== creating conda env sss (python 3.10) =="
  conda create -y -n sss python=3.10
fi
conda activate sss

python - <<'PY' 2>/dev/null || NEED_TORCH=1
import torch, sys
sys.exit(0 if torch.__version__.startswith("2.4.1") and torch.cuda.is_available() else 1)
PY
if [ "${NEED_TORCH:-0}" = 1 ]; then
  echo "== installing torch 2.4.1+cu121 =="
  pip install --quiet torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
fi

echo "== python deps (SSS + metrics) =="
pip install --quiet plyfile tqdm opencv-python-headless pillow numpy \
    scikit-image lpips

# Install a matching CUDA 12.1 toolkit INTO the env so nvcc == torch's cu121,
# regardless of the base image's system CUDA. This kills the extension-build
# skew RESULTS.md hit on the cu128 base. CUDA_HOME points at the env prefix.
if [ ! -x "$CONDA_PREFIX/bin/nvcc" ]; then
  echo "== installing cuda-toolkit 12.1 into env (nvcc parity) =="
  conda install -y -c "nvidia/label/cuda-12.1.1" cuda-toolkit
fi
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version | tail -2 || { echo "FATAL: nvcc missing after install"; exit 1; }

# CUDA 12.1's nvcc rejects gcc > 12, but Ubuntu 24.04 ships gcc-13/14. Install
# gcc-12 and make nvcc use it as the host compiler.
if ! command -v g++-12 >/dev/null 2>&1; then
  echo "== installing gcc-12/g++-12 (nvcc host-compiler cap) =="
  apt-get update -qq && apt-get install -y -qq gcc-12 g++-12
fi
export CC=gcc-12 CXX=g++-12
export NVCC_PREPEND_FLAGS="-ccbin g++-12"

# --- build SSS CUDA extensions (--no-build-isolation so they see the env torch) ---
# simple_knn.cu already has the <cfloat> include patched locally before scp.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.9"}   # 4090 = Ada = sm_89
echo "== building diff-t-rasterization (arch $TORCH_CUDA_ARCH_LIST) =="
pip install --quiet --no-build-isolation "$SSS/submodules/diff-t-rasterization"
echo "== building simple-knn =="
pip install --quiet --no-build-isolation "$SSS/submodules/simple-knn"

echo "== sanity: import SSS renderer + torch cuda =="
cd "$SSS"
python - <<'PY'
import torch
from t_renderer import render  # noqa
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("SSS extensions import OK")
PY
echo "== PROVISION OK =="
