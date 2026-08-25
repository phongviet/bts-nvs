# bts-nvs — Viettel AI Race 2026, BTS Digital Twin (Novel View Synthesis) - Top 23

<img width="1198" height="75" alt="image" src="https://github.com/user-attachments/assets/509edacc-5179-4ef6-a302-73702e31704a" />

3D Gaussian Splatting + real-pixel reprojection pipeline for the BTS tower NVS
competition. Scoring: `Score = 0.4·(1−LPIPS) + 0.3·SSIM + 0.3·(PSNR/50)`
(LPIPS-VGG, `psnr_max=50` — both solved from leaderboard breakdowns; averaged
per scene). **All compute runs on Kaggle** (free T4×2 GPU sessions).

> ## ▶ Round 2 is the only submission target (deadline 2026-07-30)
>
> `data/raw/VAI_NVS_DATA_ROUND2/` — 7 scenes, 386 test frames, no GT. Phase-1
> `public_set` remains the local GT bench.
>
> **Best submission: LB 75.3793** (`submissions/round2/round2_v7a_all_drones_adv/`,
> md5 `9375ced6`). All lever families are measured and closed; **v7a stands**.
>
> **The one hard regime split:** the 5 drone `HCM*` scenes are `SIMPLE_RADIAL`
> (k ≈ +0.008 → the F1 remap applies), but `bonsai`/`chair` are indoor handheld
> video, `SIMPLE_PINHOLE`, **k = 0 → the F1 remap must be bypassed for them.**

## Documentation

Everything — competition rules, the method, the full experiment log, results,
and the reproduction path — lives in **[`docs/`](docs/README.md)**:

| doc | contents |
|---|---|
| [`docs/01_competition.md`](docs/01_competition.md) | Task, grader math, data (R1 & R2), submission format, compliance & provenance |
| [`docs/02_pipeline.md`](docs/02_pipeline.md) | The shipped stack: backbone → F1 remap → DIBR → refiner → encode |
| [`docs/03_experiments.md`](docs/03_experiments.md) | Full experiment log & lever graveyard |
| [`docs/04_results.md`](docs/04_results.md) | Leaderboard history, submissions, calibration |
| [`docs/05_reproducibility.md`](docs/05_reproducibility.md) | Env setup, exact reproduce commands, Kaggle workflow, gotchas |
| [`docs/06_sss_backbone.md`](docs/06_sss_backbone.md) | SSS backbone ablation (`sss_experiment/`) |

## The pipeline in one table

| stage | lever | measured LB gain |
|---|---|---|
| Backbone | dense-COLMAP-init `splatfacto-big`, anti-aliased | baseline 57.43 |
| **F1** | distortion remap (pinhole → true `SIMPLE_RADIAL`, drones only) | **+16.4** → 70.45 |
| **F2** | DIBR — warp real train pixels via 3DGS depth + occlusion/photometric guard | **+1.8** → 72.22 |
| **P2** | per-scene neural refiner (U-Net on [render, DIBR, mask], grader objective; naf+evidence+adv 0.003) | **+3.2** → 75.38 |
| encode | single-encode q98 4:4:4, pooled knapsack, budget-auto | — |

This is a view-**interpolation** problem: the highest levers reuse real train
pixels and correct the camera model, not invent detail. See
[`docs/02_pipeline.md`](docs/02_pipeline.md) for the full rationale and
[`docs/03_experiments.md`](docs/03_experiments.md) for what was refuted.

## Setup

```bash
conda env create -f environment.yml   # creates the `airace` env
conda activate airace
```

From scratch:

```bash
conda create -n airace python=3.10 -y && conda activate airace
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nerfstudio gsplat
pip install lpips torchmetrics pillow numpy pandas tqdm pyyaml opencv-python plyfile
```

## Reproduce the best result (v7a)

Full commands in [`docs/05_reproducibility.md`](docs/05_reproducibility.md). In
short, per scene (drone example):

```bash
python src/data_prep/filter_colmap_train.py --scene <scene> ...   # avoid test-leak
python src/data_prep/build_dense_colmap.py  --scene <scene> ...   # dense MVS init

python Analysis/kaggle_exp034_fleet.py \
    --phase round2 --scenes <scene> \
    --suffix _v2 --blocks naf --evidence --adv 0.003   # backbone→F1→DIBR→refiner

python Analysis/24_build_round2_submission.py --suffix _v2   # → submission_round1.zip
```

The 7-scene fleet runs on Kaggle T4×2 (2 scenes/session, `PARALLEL=2`) via
`Analysis/kaggle_exp034_fleet.py` (idempotent/resumable) and
`scripts/build_kaggle_round2_upload.py`. `bonsai` uses an SSS render-channel
override ([`docs/06_sss_backbone.md`](docs/06_sss_backbone.md)). The packager
falls back per scene (refiner → DIBR → F1) so a partial fleet still ships valid.

## Repository layout

```
src/            model methods, render, metrics, submission packaging
configs/        experiment configs
Analysis/       winning-stack scripts (F1/F2/P2, refiner, fleet driver, packager)
scripts/        Kaggle upload builders, sweep runners, diagnostics
sss_experiment/ SSS backbone ablation (heavy artifacts gitignored)
docs/           all documentation (see the table above)
results/        ablation CSVs, per-run metadata
submissions/    submission zips (versioned; heavy staging gitignored)
tests/          loss / method unit tests
```

Heavy artifacts (checkpoints, renders, dense point clouds, submission zips, raw
data) are gitignored and regenerable from the pipeline above.
