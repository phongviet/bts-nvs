# 02 — The shipped pipeline

A per-scene, embarrassingly-parallel stack. Each rung was measured on public-GT
before scaling to the fleet. Source of truth in code:
`Analysis/kaggle_exp034_fleet.py` (fleet driver), `Analysis/10_refiner_pilot.py`
(refiner), `Analysis/04_x3_dibr_pilot.py` (DIBR + render channel),
`Analysis/24_build_round2_submission.py` (encode/package).

## Stage ladder and measured leaderboard gains

| stage | lever | mechanism | measured LB gain |
|---|---|---|---|
| **Backbone** | dense-COLMAP-init `splatfacto-big`, anti-aliased | careful dense-MVS init + capacity | baseline 57.43 |
| **F1** | distortion remap | remap pinhole renders into the true `SIMPLE_RADIAL` geometry (no retraining) | **+16.4** → 70.45 |
| **F2** | DIBR hybrid | warp real train pixels via 3DGS depth, occlusion z-test + photometric guard; 3DGS fills holes | **+1.8** → 72.22 |
| **P2** | per-scene neural refiner | small U-Net, `[F1 render, DIBR blend, visibility mask]` → residual, trained per-scene against the grader objective | **+3.2** → 75.38 |
| **exp034** | full stack | single-encode JPEG q95 4:4:4, hflip TTA, ss=2 supersample + cubic resample, `splatfacto-big`, refiner v2 | **+1.26** → **76.639** (Round 1) |

## Stage detail

### Backbone — `splatfacto-big`, dense COLMAP init

- gsplat/Nerfstudio `splatfacto-big`, **30k iters**, `--rasterize-mode
  antialiased`, **`--downscale-factor 1`** (all 7 scenes).
- Init = **dense MVS COLMAP** (`image_undistorter → patch_match_stereo →
  stereo_fusion`), ~2 M points/scene. This is the single highest-ROI backbone
  choice; MCMC, scale-reg, sky-mask, camera-optimizer, bilateral grid, transient
  masks were all tested and **dropped** (see [03_experiments](03_experiments.md)).
- `bonsai`'s render channel is overridden with an **SSS backbone** (Student
  Splatting and Scooping) — the one scene whose splatfacto baseline was
  capacity-starved (0.25 gauss/px). See [06_sss_backbone](06_sss_backbone.md).
- **`--downscale-factor 1` is mandatory:** nerfstudio auto-downscales any image
  with long side > 1600 px and then *interactively* prompts, which throws
  EOFError in a non-interactive session and dies in seconds (looks like OOM; it
  is not). `bonsai` at 1920×1080 hits this.

### F1 — distortion remap (drones only)

The test GT is raw `SIMPLE_RADIAL` but the pose CSV omits the distortion `k`; our
pinhole renders were geometrically misaligned at the frame periphery. F1 remaps
each pinhole render into the true `SIMPLE_RADIAL` geometry per-scene
(`k ≈ +0.0081…+0.0090`), with **no retraining**. This was the single biggest
score jump of the whole competition (+16.4 LB). **Bypassed for `bonsai`/`chair`**
(`SIMPLE_PINHOLE`, `k = 0`).

### F2 — DIBR hybrid (real-pixel reprojection)

Depth-image-based rendering: warp real train pixels into each test view using
the 3DGS depth, with an occlusion z-test and a photometric guard (fixes
thin-structure ghosting); the 3DGS render fills disocclusion holes.

- **SSAA is already shipped here** (`04_x3_dibr_pilot.py:488`): the render channel
  is rendered at `ss×` resolution and cubic-downsampled (pixel-center-correct),
  so "render at 2× and downsample" is not an available lever.
- Guard-reject rate on drones is 0.1–0.4 % → flow-residual pre-alignment
  (exp039) found nothing to rescue and was dropped.

### P2 — per-scene neural refiner

A small U-Net taking a 7-channel input `[F1 render, DIBR blend, visibility mask]`
and predicting a residual on the DIBR blend. Trained **per scene** on held-out
train views against the grader objective. This is the top-8 mechanism.

Shipped config: `base=48`, **6k iters**, EMA 0.999, hflip TTA, memmap `PairPool`,
`--max-pairs 90` → lossless PNG intermediate. **All 7 v7a scenes** additionally
carry `--blocks naf --evidence --adv 0.003`:

- **`--blocks naf --evidence`** (W1): a NAFNet deblurring-family block +
  evidence channel. +0.0032–0.0035 on both indoor scenes (LPIPS −0.005). naf
  works indoors because both indoor scenes are blur-limited.
- **`--adv 0.003`** (W3): adversarial refiner — a spectral-norm PatchGAN critic
  **conditioned on the DIBR channel** (so it can't win by memorizing scene
  content), hinge loss, 0.67 M params. **Two-stage warm-started** (a critic on a
  random net just trains the critic), so `--adv > 0` runs stage 1 for the warm
  ckpt then stage 2. **`--adv 0` is bit-identical to the regression path.**
  0.003 is optimal on both indoor scenes (W9 ladder: monotone-decreasing above
  0.003 on both Score and LPIPS). On drones the adv rollout gained **+0.0593 LB,
  100 % LPIPS.**

**Do not judge an adversarial arm by `val_loss`** — it is a regression loss and
the critic's whole job is to move off the conditional mean, so a better
adversarial model scores *worse* on it. On the 4 drone scenes val_loss predicted
−0.093 LB; the actual result was +0.0593. Best-checkpoint selection stays on the
pure-regression `eval_val()`.

### Encode + package

`Analysis/24_build_round2_submission.py`:

- Floor **JPEG q98 4:4:4**, then a **pooled knapsack** over all 386 frames spends
  the remaining byte budget upward (`BUDGET_MIB = 348`, cap 350 MiB).
- Per-scene fallback chain (refiner v2 → v1 → DIBR → F1 remap) so a partial fleet
  still ships a valid submission.
- **Scene-weighted knapsack was tried and is worse** (−0.00039): bonsai's
  rate-distortion curve is saturated; the unweighted greedy already allocates by
  marginal gain per byte. Encode as a whole family is bounded ~0.02 LB.

## Compliance recap

Only provided train images + train poses + our own 3DGS model + provided test
poses/intrinsics are used. No test images in training or inference, no external
data, no pretrained enhancement net ships (LPIPS-VGG is used only inside the
training loss). Rasterizer is gsplat. See [01_competition](01_competition.md).
