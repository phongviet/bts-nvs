# 01 — Competition, grader, data, rules

## The task

Novel View Synthesis for the "BTS Digital Twin" track of the Viettel AI Race
2026. Per scene the organizers provide posed training images + a COLMAP sparse
model; we must render a fixed set of **test poses** (given as a CSV of
extrinsics + per-pose intrinsics + output resolution) and submit the rendered
images. There is **no test-time optimization**: the test cameras are fixed and
we never see test ground truth.

This is a **view-interpolation** problem, not view-extrapolation. 86–98 % of
test frames sit sequentially adjacent to a train frame (0.2–0.5× spacing), so
the highest-value levers reuse real train pixels and correct the camera model
rather than inventing detail. (Confirmed: three zero-parameter 2D
interpolation baselines collapse on the indoor scenes — indoor is parallax-heavy,
not frame-interp — so RIFE/FILM-family methods are ruled out.)

## Grader (fully solved)

```
Score = 0.4·(1 − LPIPS) + 0.3·SSIM + 0.3·(PSNR / 50)
LB    = 100 × Score,  averaged PER SCENE (mean over the scenes in the set)
```

- **LPIPS backbone = VGG** — CONFIRMED (re-scoring exp011 under VGG reproduced
  the leaderboard delta exactly). **All decision scoring uses `--lpips-net vgg`.**
  Distrust any local win that lives mostly in the AlexNet-LPIPS column.
- **`psnr_max = 50.0`** — SOLVED from per-metric leaderboard breakdowns.
- Aggregation is **per scene**, SETTLED. In Round 2 the 2 indoor scenes carry
  **2/7 = 28.6 % of the score on 22 % of the frames**.

### Grader sensitivities (for triaging levers)

| move | LB effect |
|---|---|
| −0.01 LPIPS | **+0.057** (the dominant term, weight 0.4) |
| +0.01 SSIM | +0.043 |
| +1 dB PSNR | +0.086 (but clamped at 50 → diminishing) |
| scoring noise floor | ±0.002 Score |

**Optimize preferentially for LPIPS / perceptual quality.**

## Data

### Round 1 (retired as a submission target since 2026-07-16; still the local GT bench)

- Phase 1: 13 scenes. `public_set/` (5: hcm0031, hcm0034, HCM0181, HCM0193,
  HCM0204) ships **test GT** and is the local calibration bench; `private_set1/`
  (8) had withheld GT and was graded directly by the leaderboard.
- Per scene: 240 train / 60 test (80/20), pre-scaled to 1/4 original size.
- `test_poses.csv`: `image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height`.
- **public-5 mean tracks the private LB within ±0.004** — it is the calibration
  signal for every local A/B.

### Round 2 (`data/raw/VAI_NVS_DATA_ROUND2/`, the ONLY submission target)

7 scenes, 386 test frames, no GT. **One hard regime split:**

- **5 drone `HCM*` scenes** behave like Phase 1: `SIMPLE_RADIAL`, distortion
  `k ≈ +0.0081…+0.0090` → **the F1 remap applies.**
- **`bonsai` + `chair`** are indoor handheld video: `SIMPLE_PINHOLE`, **`k = 0`
  → the F1 remap MUST be bypassed** (applying it warps correct geometry).
  Ultra-dense pose spacing (3–4° gaps). `bonsai` = 1920×1080, motion blur +
  glossy reflective table; `chair` = portrait 720×1280, late-capture motion blur.

## ⚠️ Critical data gotcha — test-leak in the COLMAP model

The provided `train/sparse/0/images.bin` holds poses for the scene's **full
capture**, not just the train images (e.g. hcm0034: 337 entries = 240 train + 60
test + 37 registration-only). Nerfstudio's default `ColmapDataParser`
(`eval_mode="interval"`) splits across ALL of `images.bin`, **leaking held-out
test images into training**. Always run `src/data_prep/filter_colmap_train.py`
first (writes a train-only filtered sparse model + a staging dir), and train with
`--pipeline.datamanager.dataparser.eval-mode=all` on that staging dir.

## Submission format

- ZIP literally named `submission_round1.zip` (competition requirement, unchanged
  in Round 2). One subfolder per scene, one image per required test pose.
- **Filename = the `image_name` column** from that scene's `test_poses.csv`
  (real files e.g. `DJI_..._V.JPG`); **image size = exact `width,height`** from
  that CSV row. Extension case is MIXED: drone `.JPG`, indoor `.jpg`.
- **All scenes in one ZIP** — a missing pose for any scene affects scoring.
- ≤ **350 MiB** (2²⁰, not decimal MB), JPEG.
- Limits: 5 uploads/day, 600 s cooldown; the system keeps the **last** upload
  before the deadline, so late re-submits are safe.

The packager (`Analysis/24_build_round2_submission.py`) validates names / sizes /
counts / decodability pre- and post-zip.

## Compliance / anti-cheating

**Allowed:** training 3DGS/NeRF only on provided images + provided COLMAP;
generically pretrained, scene-agnostic foundation models (depth / segmentation /
geometry / diffusion priors) so long as they are **not** fine-tuned on
competition/BTS imagery, and any fine-tuning uses **only our own renders of the
provided train images** (no external-scene pooling).

**Banned:** scraping BTS/telecom imagery or external 3D/telecom datasets of
similar objects; per-test-pose manual editing/compositing; test-time
optimization using test images (impossible here — we have no test GT).

**Non-commercial determination (2026-07-16):** participation counts as
non-commercial research/evaluation use, which unblocks the NC-licensed research
components (INRIA 3DGS lineage, NVIDIA non-commercial, CC BY-NC). Re-examine if
prize terms ever assign rights to submitted work.

### Pretrained-model provenance

Every generically-pretrained model used, with an explicit "not trained on
competition/BTS imagery" statement:

| Model | Role | Source | License |
|---|---|---|---|
| Depth Anything V2 (Small) | mono-depth prior (E2 depth-supervision arm; disparity-space, scale-shift-invariant) | HF | Apache-2.0 |
| SegFormer-B0 (ADE20K) | sky & transient masks (both dropped as levers) | `nvidia/segformer-b0-finetuned-ade-512-512` | Apache-2.0 |
| VGGT-1B | pseudo-point-cloud init arm (exp003, dropped) | `facebook/VGGT-1B`, vendored read-only in `third_party/vggt/` | CC BY-NC 4.0 |
| Difix3D+ / SD-Turbo | diffusion "fixer" (exp015/016 dropped; indoor LoRA revival closed) | `nvidia/difix` + vendored `DifixPipeline` | NVIDIA NC + Stability Community |
| LPIPS (AlexNet, VGG-16) | perceptual metric (local eval, VGG) and perceptual loss (E1) | `lpips` pkg / torchmetrics, ImageNet backbones | BSD |

None of these ships in the final submission except LPIPS-VGG (used only inside a
training loss). The submission uses only provided train images + provided
poses/intrinsics + our own 3DGS model. Rasterizer is gsplat (allowed).
