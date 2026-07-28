#!/usr/bin/env bash
# Provision the torch 2.5.1 / cu121 devel image for the capacity gate.
# The base image already carries torch 2.5.1+cu121 and nvcc, so only the airace-specific
# packages are added, with the same post-nerfstudio numpy re-pin the Kaggle notebooks use.
set -euo pipefail

echo "== base torch =="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
command -v nvcc >/dev/null && nvcc --version | tail -1 || { echo "FATAL: no nvcc, gsplat cannot build"; exit 1; }

echo "== gsplat 1.4.0 (source; kernels JIT-compile at first use) =="
pip install -q ninja
pip install -q --no-build-isolation gsplat==1.4.0

echo "== nerfstudio 1.1.5 + scorer deps =="
pip install -q nerfstudio==1.1.5
# nerfstudio's resolver bumps these; re-pin to the locally-validated versions
pip install -q numpy==1.26.4 opencv-python-headless==4.10.0.84 lpips==0.1.4 scikit-image==0.25.2

echo "== force gsplat CUDA compile + cache lpips VGG now (not mid-experiment) =="
python - <<'PY'
import torch
from gsplat import rasterization
m=torch.zeros(1,3,device='cuda'); q=torch.tensor([[1.,0,0,0]],device='cuda')
s=torch.ones(1,3,device='cuda')*.1; o=torch.ones(1,device='cuda')
c=torch.ones(1,3,device='cuda'); v=torch.eye(4,device='cuda')[None]
K=torch.tensor([[[100.,0,32],[0,100.,32],[0,0,1.]]],device='cuda')
rasterization(m,q,s,o,c,v,K,64,64)
print("gsplat kernels OK")
import lpips; lpips.LPIPS(net="vgg"); print("lpips vgg cached")
from importlib.metadata import version
print("nerfstudio", version("nerfstudio"), "| gsplat", version("gsplat"))
PY
echo "== PROVISION DONE =="
