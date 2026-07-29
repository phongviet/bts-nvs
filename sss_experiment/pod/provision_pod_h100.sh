#!/usr/bin/env bash
# Provision a rented H100 80GB (Hopper, sm_90, CUDA 12.1) for the full-quality
# pipeline: builds TWO conda envs and their CUDA extensions --
#   sss : SSS backbone (torch 2.4.1+cu121) + diff_t_rasterization + simple_knn.
#         ALSO runs the COLMAP-native refiner (torch + cv2 + lpips + skimage;
#         no nerfstudio -- 04 vendors its own COLMAP reader).
#   gs  : RaDe-GS depth (torch 2.4.1+cu121) + diff_gaussian_rasterization +
#         simple_knn + warp_patch_ncc + fused_ssim.
#
# Assumes the bundle has been scp'd + untarred to $ROOT (see pack_h100.sh).
# THE ONE H100 GOTCHA: TORCH_CUDA_ARCH_LIST=9.0 (Hopper). The 4090 provision
# used 8.9 (Ada); building the extensions for 8.9 on an H100 yields "no kernel
# image is available for execution on the device" at runtime.
set -euo pipefail

ROOT=${ROOT:-$HOME/h100_run}
SSS=$ROOT/3D-student-splatting-and-scooping
RADEGS=$ROOT/radegs
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"9.0"}   # H100 = Hopper = sm_90

# --- miniconda ---
if [ ! -d /opt/miniconda ]; then
  echo "== installing miniconda =="
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p /opt/miniconda
fi
export PATH=/opt/miniconda/bin:$PATH
source /opt/miniconda/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

# --- host toolchain: cuda-12.1 nvcc + gcc-12 (nvcc rejects gcc>12) ---
ensure_toolchain() {
  if [ ! -x "$CONDA_PREFIX/bin/nvcc" ]; then
    echo "== installing cuda-toolkit 12.1 into $CONDA_DEFAULT_ENV =="
    conda install -y -c "nvidia/label/cuda-12.1.1" cuda-toolkit
  fi
  export CUDA_HOME="$CONDA_PREFIX"; export PATH="$CUDA_HOME/bin:$PATH"
  if ! command -v g++-12 >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq gcc-12 g++-12
  fi
  export CC=gcc-12 CXX=g++-12 NVCC_PREPEND_FLAGS="-ccbin g++-12"
  nvcc --version | tail -1
}

install_torch() {
  python - <<'PY' 2>/dev/null || pip install --quiet torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
import torch, sys
sys.exit(0 if torch.__version__.startswith("2.4.1") and torch.cuda.is_available() else 1)
PY
}

# ============================ env: sss ================================
if ! conda env list | grep -q "^sss "; then conda create -y -n sss python=3.10; fi
conda activate sss
install_torch
echo "== sss deps (SSS + refiner: cv2, lpips, skimage) =="
pip install --quiet plyfile tqdm opencv-python-headless pillow numpy scikit-image lpips torchvision
ensure_toolchain
echo "== build SSS exts (arch $TORCH_CUDA_ARCH_LIST) =="
pip install --quiet --no-build-isolation "$SSS/submodules/diff-t-rasterization"
pip install --quiet --no-build-isolation "$SSS/submodules/simple-knn"
python - <<'PY'
import torch; from t_renderer import render  # noqa
print("  sss OK:", torch.__version__, torch.cuda.get_device_name(0))
PY

# ============================ env: gs (RaDe-GS) =======================
if ! conda env list | grep -q "^gs "; then conda create -y -n gs python=3.10; fi
conda activate gs
install_torch
echo "== gs deps (RaDe-GS) =="
pip install --quiet plyfile tqdm opencv-python-headless pillow numpy scikit-image \
    scipy trimesh open3d lmdb matplotlib
ensure_toolchain
echo "== build RaDe-GS exts (arch $TORCH_CUDA_ARCH_LIST) =="
pip install --quiet --no-build-isolation "$RADEGS/submodules/diff-gaussian-rasterization"
pip install --quiet --no-build-isolation "$RADEGS/submodules/simple-knn"
pip install --quiet --no-build-isolation "$RADEGS/submodules/warp-patch-ncc"
echo "== fused_ssim from git (RaDe-GS SSIM loss, not vendored) =="
pip install --quiet --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim.git"
python - <<'PY'
import torch, diff_gaussian_rasterization, simple_knn, warp_patch_ncc, fused_ssim  # noqa
print("  gs OK:", torch.__version__, torch.cuda.get_device_name(0))
PY

echo "== PROVISION OK (sss + gs, sm_90) =="
