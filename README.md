# bts-nvs — Viettel AI Race 2026, BTS Digital Twin (Novel View Synthesis)

3D Gaussian Splatting pipeline for the BTS tower NVS competition (Phase 1).
See `docs/strategy.md` (full method plan) and `docs/reproducibility_checklist.md`.

## Setup

```bash
conda create -n airace python=3.10 -y
conda activate airace
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nerfstudio gsplat
pip install lpips torchmetrics pillow numpy pandas tqdm pyyaml opencv-python plyfile
conda env export > environment.yml
```

## Data

Extract the organizers' archive into `data/raw/` (read-only, never hand-edited):

```
data/raw/phase1/public_set/<scene>/train/{images,sparse/0}/
data/raw/phase1/public_set/<scene>/test/{images,test_poses.csv}
data/raw/phase1/private_set1/<scene>/...   # test/images GT withheld or treated as hidden
```

`public_set` ships test GT images — use it to compute the real competition Score locally.

## Pipeline

```bash
ns-train splatfacto --data data/raw/phase1/public_set/<scene>/train \
  --output-dir runs/phase1/exp001_baseline_splatfacto

python src/render.py --checkpoint <ckpt> --poses <scene>/test/test_poses.csv --out runs/.../renders_test

python src/metrics.py --renders runs/.../renders_test --gt <scene>/test/images --out runs/.../metrics_val.json

python src/package_submission.py --runs-dir runs/phase1/exp001_baseline_splatfacto --out submissions/phase1/submission_exp001.zip
```

See `Documents/plan_week1.md` (outside this repo) for the day-by-day Week 1 plan.
