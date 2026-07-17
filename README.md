# bts-nvs — Viettel AI Race 2026, BTS Digital Twin (Novel View Synthesis)

3D Gaussian Splatting + real-pixel reprojection pipeline for the BTS tower NVS
competition. Scoring: `Score = 0.4·(1−LPIPS) + 0.3·SSIM + 0.3·PSNR/50`
(LPIPS is VGG-backbone, `psnr_max=50` — both solved from leaderboard breakdowns).

> ## ▶ ROUND 2 is the only submission target (since 2026-07-16)
>
> `data/raw/VAI_NVS_DATA_ROUND2/` — 7 scenes, 386 test frames, no GT. Phase-1 public
> scenes remain the **local GT bench**; phase-1 private scenes are retired.
>
> - **Plan of record:** `Analysis/PLAN_round2_2026-07-17.md`
> - **Test-set facts:** `Analysis/ROUND2_test_set_analysis_2026-07-16.md`
> - **Live status:** `results/PROGRESS.md` (Table 1b = per-scene backbone runs)
>
> **The one hard regime split:** the 5 drone `HCM*` scenes behave like phase 1
> (`SIMPLE_RADIAL`, k ≈ +0.0081…+0.0090 → the F1 remap applies), but `bonsai` and
> `chair` are indoor handheld video with `SIMPLE_PINHOLE`, **k = 0 → the F1 remap must
> be bypassed for them**. Applying it anyway warps correct geometry. Superseded phase-1
> plans live in `docs/archive_phase1/`.

## Current strategy

**This is a view-*interpolation* problem, not view-extrapolation.** 86–98 % of test
frames sit sequentially adjacent to a train frame (0.2–0.5× spacing), so the highest
lever is **reusing real train pixels and correcting the camera model**, not inventing
detail. Two independently-measured findings overturned the original plan:

- A generative post-processing/supervision stage (Difix3D+/SD-Turbo), the v1/v2
  thesis, was a **decisive net loss** on real test GT (exp015: −0.034/−0.037) — our
  renders are too artifact-light for a diffusion "fixer" to help.
- The test GT is raw `SIMPLE_RADIAL` while the pose CSV omits the distortion `k`; our
  pinhole renders were geometrically misaligned at the frame periphery. Correcting this
  was the single biggest score jump of the whole competition.

The current winning pipeline is a per-scene, embarrassingly-parallel stack, each rung
measured on public-scene GT before scaling to the private fleet:

| stage | lever | mechanism | measured LB gain |
|---|---|---|---|
| **Backbone** | dense-COLMAP-init `splatfacto-big` (anti-aliased / Mip) | careful init + capacity; MCMC, scale-reg, sky-mask, camera-opt all tested and dropped | baseline 57.43 |
| **F1** | distortion remap | remap pinhole renders into the true `SIMPLE_RADIAL` geometry (no retraining) | **+16.4** → 70.45 |
| **F2** | DIBR hybrid | warp real train pixels via 3DGS depth, occlusion z-test + photometric guard, 3DGS fills holes | **+1.8** → 72.22 |
| **P2** | per-scene neural refiner | small U-Net, 7-ch input `[F1 render, DIBR blend, visibility mask]` → residual on DIBR, trained per-scene on held-out train views against the grader objective | **+3.2** → 75.38 (rank 9) |
| **exp034** | full stack | single-encode JPEG q95 4:4:4, hflip TTA, ss=2 supersampled + cubic resample, `splatfacto-big` backbone, refiner v2 | **+1.26** → **76.639** |

Best submission to date: **LB 76.63890** (PSNR 25.34 / SSIM 85.25 / LPIPS 10.35) — but
that was scored on the retired phase-1 private set, so it is now a *method* validation,
not a standing score. The stack above carries over to round 2 unchanged except for the
conditional F1 remap noted at the top.

**Compliance:** the pipeline uses only provided train images + train poses + our own
3DGS model + provided test poses/intrinsics. No test images in training or inference, no
external data, no pretrained enhancement nets (LPIPS-VGG is used only inside the training
loss). Rasterizer is gsplat (allowed). Submissions are ≤ 350 MB, JPEG, exact test
filenames — enforced by the packager.

See `results/PROGRESS.md` for the full experiment ↔ leaderboard log, and
`docs/archive_phase1/REPORT_winning_strategy.md` for the method write-up
(`docs/archive_phase1/FINAL_PLAN_top1.md` holds the phase-1 top-1 ladder — its rails
still apply, its scene list does not). `docs/strategy.md` preserves the
original SOTA survey (Sections 1–6 still valid); its v2 action plan is superseded.

## Setup

```bash
conda env create -f environment.yml   # creates the `airace` env
conda activate airace
```

Or from scratch:

```bash
conda create -n airace python=3.10 -y && conda activate airace
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nerfstudio gsplat
pip install lpips torchmetrics pillow numpy pandas tqdm pyyaml opencv-python plyfile
```

## Data

Extract the organizers' archive into `data/raw/` (read-only, never hand-edited):

```
data/raw/phase1/public_set/<scene>/train/{images,sparse/0}/
data/raw/phase1/public_set/<scene>/test/{images,test_poses.csv}
data/raw/phase1/private_set1/<scene>/...   # test GT withheld
```

`public_set` ships test GT — use it to compute the real competition Score locally
(the leaderboard grades private scenes directly, so public-5 mean is the calibration
signal: it tracks private LB within ±0.004).

## Pipeline

**Baseline 3DGS backbone** (per scene):

```bash
ns-train splatfacto-big --data data/raw/phase1/public_set/<scene>/train \
  --output-dir runs/phase1/<exp>_<scene>          # dense-COLMAP init, anti-aliased

python src/render.py  --checkpoint <ckpt> --poses <scene>/test/test_poses.csv --out <renders>
python src/metrics.py --renders <renders> --gt <scene>/test/images --lpips-net vgg --psnr-max 50
```

**Winning stack** (F1 remap → F2 DIBR → P2 refiner → package), driven per scene:

```bash
python Analysis/03_x4_distortion_remap.py    # F1: pinhole -> SIMPLE_RADIAL geometry
python Analysis/04_x3_dibr_pilot.py          # F2: depth-warp real train pixels + guard
python Analysis/10_refiner_pilot.py --config <big_ckpt> --ss 2 --sample cubic \
  --base 48 --iters 6000 --ema 0.999 --tta   # P2: per-scene U-Net refiner (v2)
python Analysis/14_build_v2_submission.py    # single-encode q95 4:4:4, budget-auto, per-scene fallback
```

The full 8-scene private fleet runs on Kaggle T4×2 / a rented 4090 via
`Analysis/kaggle_exp034_fleet.py` (idempotent/resumable) and
`scripts/build_kaggle_exp034_upload.py`. The
builder falls back per scene (refiner v2 → v1 → DIBR → F1 remap) so partial fleets
still ship a valid submission.

## Repository layout

```
src/         model methods, render, metrics, submission packaging
configs/     experiment configs (exp004/022/023/024/029/034 …)
Analysis/    winning-stack scripts + reports (F1/F2/P2, refiner, JPEG-budget studies)
scripts/     sweep runners, Kaggle upload builders, diagnostics
docs/         strategy survey, runbook, reproducibility checklist, rules
results/      PROGRESS.md (experiment ↔ leaderboard truth), ablation CSVs
runs/         per-scene run metadata (heavy binaries gitignored)
tests/        loss / method unit tests
```

Heavy artifacts (checkpoints, renders, dense point clouds, submission zips, raw data)
are gitignored and regenerable from the pipeline above.
