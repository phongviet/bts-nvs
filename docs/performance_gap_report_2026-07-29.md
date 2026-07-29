# Round-2 gap report: measured path from 75.3793 toward 77

Date: 2026-07-29
Hardware used: RTX 4070 Ti SUPER, 16 GB; CUDA 12.1; Python 3.10

## Executive decision

The remaining error is not a global capacity problem. The five drone scenes are
already around 77.2–77.7; almost all useful headroom is in `bonsai` and
`chair`. Raising the seven-scene leaderboard score from 75.3793 to 77 requires
about +0.11345 total Score across those two indoor scenes if the drones stay
fixed, or roughly +0.0567 per indoor scene. Parameter polishing alone is very
unlikely to close that.

The two dominant gaps are:

1. `bonsai`: source evidence is spatially non-stationary. Repeated camera
   orbits make center-distance source selection alias views hundreds of frames
   apart, but replacing it with one global temporal/pose ranking damages
   occlusion and LPIPS. The needed change is **per-pixel learned source
   confidence/attention**, with the renderer as an explicit fallback.
2. `chair`: the capture becomes substantially blurrier through time. This is a
   **blur-aware reconstruction** problem, not a neighbor-count or refiner-data
   problem. The next backbone experiment should integrate a short, regularized
   exposure trajectory while keeping the supplied midpoint camera fixed.

Three tempting changes were tested here and rejected on a clean outer holdout:
global temporal source bracketing, global pose-aware source ranking, and direct
replacement of the refiner's L1 term by normalized PSNR. They must not be put
into the submission on proxy evidence alone.

## Repository and reproducibility confidence

- Active branch: `hai_dev`.
- Local `hai_dev` and fetched `origin/hai_dev` were identical at
  `a5bf966590fdb1fc9ece854d0702bfcbc5c6b1eb` on 2026-07-29.
- Fetched `origin/master` was `8e16cb6` and is contained in `hai_dev`.
- Local `master` is stale (`f72a9e7`, 21 commits behind `origin/master`), so it
  should not be used for experiments.
- A live GitHub authentication check was unavailable for this private remote;
  the statement above is against the fetched remote-tracking refs.
- The repository does not contain the production 30k checkpoints or dense
  point clouds. A leak-free 10k sparse-init bonsai checkpoint was therefore
  trained for the matched experiments below. Absolute pilot scores are not
  production scores; matched deltas are valid.

The supplied phase-2 layout is supported directly and through the documented
`data/raw/phase2/round2 -> .` compatibility link.

## What the existing evidence says

### Score concentration

The official metric is:

```text
Score = 0.4 * (1 - LPIPS_VGG) + 0.3 * SSIM + 0.3 * clip(PSNR / 50, 0, 1)
```

LPIPS has the largest coefficient, and the v7a drone gain was almost entirely
LPIPS. That makes texture preservation and view-consistent high-frequency
evidence more valuable than small PSNR-only changes.

Existing leak-free indoor holdouts:

| Stage | bonsai | chair |
|---|---:|---:|
| Backbone only | 0.6759 | 0.6506 |
| Earlier full stack | 0.6913 | 0.6650 |
| Best recorded branch | 0.7066 (SSS gate) | 0.6724 |

The strong bonsai SSS gain and chair SSS loss show that one indoor backbone is
not appropriate for both scenes.

### Bonsai: the actual source-selection failure

The zero-training diagnostic leaves every supplied train image out and predicts
its brightness/sharpness from K=3 selected sources. It never reads test GT.

| K=3 policy | brightness MAE | sharpness MAE | sharpness rho | train time-gap p90 | hidden-pose time-gap p90 | bracketed test poses |
|---|---:|---:|---:|---:|---:|---:|
| spatial (shipped) | 2.491 | 0.203 | 0.761 | 785 | 1001 | 18/28 |
| pose-aware | 1.689 | 0.192 | 0.802 | 481 | 1027 | 20/28 |
| temporal bracket | 1.606 | 0.193 | 0.800 | 50 | 75 | 28/28 |

This proved the temporal alias, but the full renderer A/B disproved the naive
solution:

| Policy | Score | PSNR | SSIM | LPIPS | delta vs spatial |
|---|---:|---:|---:|---:|---:|
| spatial | **0.658455** | 25.2599 | 0.810170 | **0.340389** | — |
| pose-aware | 0.655768 | 25.2095 | 0.808225 | 0.344893 | **-0.002687** |
| temporal bracket | 0.655311 | 25.2446 | 0.807965 | 0.346366 | **-0.003144** |

The temporal arm won only 6 frames and lost 14; the pose arm won 11 and lost
14. A few frames improved strongly (`frame_001440`, `frame_001500`), but
`frame_001040`, `frame_001860`, `frame_001900`, `frame_002580`, and
`frame_002610` suffered large LPIPS regressions. This is the signature of a
selector that is right at the image level but wrong at individual surfaces.

The existing DIBR also reports 38–56% fallback on difficult early bonsai views,
while its photometric guard rejects only about 1% of depth-consistent samples.
The limiting mechanism is source visibility/depth coverage, not the guard
threshold.

### Chair: capture-dependent blur

Existing diagnostics show a strong capture-time sharpness decline and much
worse late-frame LPIPS. Changing refiner pair count or selecting sharp,
stratified, or capture-strided pairs leaves the full score near 0.664–0.666.
That closes "more/better refiner pairs" as the main lever.

RIFE v4.25 midpoint interpolation was also decisively negative:

| Scene | strict RIFE | renderer control |
|---|---:|---:|
| chair | 0.4243 | 0.6521 |
| bonsai | 0.5313 | 0.6787 |

The required model is 3D exposure integration constrained by the scene and
camera trajectory, not 2D interpolation between source frames.

### Refiner objective and checkpoint selection

The shipped "grader" loss used L1 in the PSNR slot and legacy padded/Gaussian
SSIM, then selected a checkpoint on a small set of crops. The code now supports
exact normalized PSNR, evaluator-aligned SSIM, and fixed-iteration full-frame
VGG/SSIM/PSNR checkpoint selection.

The bounded outer-holdout result was:

| 60-pair / 1k arm | external Score | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| raw spatial DIBR | **0.658455** | 25.2599 | 0.810170 | **0.340389** |
| legacy L1 + legacy SSIM + crop selection | 0.657712 | 25.2381 | **0.810473** | 0.342146 |
| exact PSNR + metric SSIM + full-score selection | 0.657448 | **25.2415** | 0.810110 | 0.342586 |
| legacy loss + full-score selection only | 0.657429 | 25.2191 | 0.810506 | 0.342592 |

The exact arm's internal full-frame score peaked at step 500, while its crop
loss continued improving. That confirms the selection mismatch exists.
However, the selected model did not generalize to the separate 25-frame outer
holdout. These controls remain useful for a production-checkpoint A/B, but are
not approved submission defaults.

## Implemented changes

All behavior that could change the shipped v7a path is opt-in.

- Phase-2/round-2 raw, processed, and run-path compatibility in
  `Analysis/04_x3_dibr_pilot.py`.
- Pure and tested source policies (`spatial`, `pose`, `temporal`); `spatial`
  remains the default.
- Source-policy cache tags, so incompatible evidence cannot silently reuse a
  cache.
- A no-test-GT source diagnostic and a leak-free full-resolution holdout
  benchmark.
- Exact differentiable `PSNR/50`, evaluator-aligned valid-window SSIM, and
  fixed full-frame Score checkpoint selection in the refiner.
- Sequential full-pair loading, so full-frame selection stays bounded in RAM.
- CUDA 12.1 `nvcc` was restored in the active Conda environment; gsplat JIT
  compilation and rasterization now work on compute capability 8.9.

Unit tests verify that metric SSIM matches `skimage` and that normalized PSNR
matches the evaluator. The original spatial selection is also regression
tested.

## Recommended deep changes

### P0: restore the production-grade validation anchor

Before accepting any model change, rebuild or restore the exact 30k
dense-COLMAP holdout checkpoints and the bonsai SSS render override. The sparse
10k pilot is good for rejecting mechanisms, but it is not a faithful adoption
gate for a production refiner.

Use two validation layers:

1. internal fixed validation images for checkpoint/hyperparameter selection;
2. the existing 25-frame match-test holdout once for adoption.

Require global Score `>= +0.003`, no LPIPS regression, and a positive difficult
slice. Do not select variants from leaderboard feedback.

### P1: pixel-wise source attention for bonsai

Do not replace the entire K=3 list globally. Build a candidate pool of K=6–8
spatial/pose/temporal views and let a lightweight network assign a softmax
weight per pixel plus an explicit renderer/fallback weight.

Suggested channels per candidate:

- warped RGB;
- relative depth disagreement and valid/occluded flags;
- reprojection confidence;
- normalized camera-center distance;
- viewing-angle difference;
- signed/absolute frame-time gap;
- source sharpness and exposure difference.

Keep the fallback image in the softmax. Entropy/total-variation regularization
on weights and source dropout prevent a selector from memorizing view IDs.
Train leave-one-view-out; never expose hidden test pixels.

Start with K=5 if storage is tight. The present evidence tensor would have
`7 + 4K + 1 = 28` channels, still safe on 16 GB with 256 crops, base 32–48, and
batch size 2. The candidate pool may be K=8 while retaining only the best five
warps per pixel.

Relevant starting points are
[IBRNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_IBRNet_Learning_Multi-View_Image-Based_Rendering_CVPR_2021_paper.pdf)
and
[SIBRNet](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_Learning_Robust_Image-Based_Rendering_on_Sparse_Scene_Geometry_via_Depth_CVPR_2022_paper.html).

Search terms:

```text
per-pixel source view selection neural image based rendering
occlusion-aware multi-view feature aggregation IBR
pose-aware source attention sparse view synthesis
uncertainty-aware depth-guided image based rendering
```

### P2: fixed-midpoint blur-aware Gaussian training for chair

Represent each supplied camera pose as the fixed midpoint of an exposure.
Predict or optimize a small start/end SE(3) offset per training frame, render
M=5 samples along the exposure, average them, and supervise against the
observed blurred image.

Anti-overfit constraints:

- supplied midpoint pose is immutable;
- zero-mean start/end offsets around that midpoint;
- shared smoothness prior over neighboring capture times;
- small translation/rotation bounds;
- blur magnitude prior informed by measured train-image sharpness;
- evaluate M=9 only after M=5 passes the outer holdout.

This separates a sharp latent scene from the image-formation blur instead of
asking the refiner to reproduce a time-varying blur distribution.

Primary references:

- [DeblurGS](https://arxiv.org/abs/2404.11358)
- [Gaussian Splatting on the Move](https://arxiv.org/abs/2403.13327)
- [BARD-GS](https://openaccess.thecvf.com/content/CVPR2025/papers/Lu_BARD-GS_Blur-Aware_Reconstruction_of_Dynamic_Scenes_via_Gaussian_Splatting_CVPR_2025_paper.pdf)
- [Extreme motion-blur splat reconstruction](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Jang_Splat-based_3D_Scene_Reconstruction_with_Extreme_Motion-blur_ICCV_2025_paper.pdf)

Search terms:

```text
blur-aware 3D Gaussian splatting camera motion exposure integration
SE3 exposure trajectory novel view synthesis motion blur
latent sharp radiance field blurred multi-view supervision
event-free deblurring Gaussian splatting
```

### P3: geometry confidence, then reflective appearance

For bonsai's high-fallback areas, improve depth only if the change is evaluated
through DIBR coverage and final Score. Candidate directions are learned
multi-view depth priors and pixel-aligned depth refinement:

- [DepthSplat](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_DepthSplat_Connecting_Gaussian_Splatting_and_Depth_CVPR_2025_paper.html)
- [PAGaS](https://openaccess.thecvf.com/content/CVPR2026W/3DMV/html/Recasens_PAGaS_Pixel-Aligned_1DoF_Gaussian_Splatting_for_Depth_Refinement_CVPRW_2026_paper.html)

For the glossy tabletop only after coverage improves:

- [SpecTRe-GS](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_SpecTRe-GS_Modeling_Highly_Specular_Surfaces_with_Reflected_Nearby_Objects_by_CVPR_2025_paper.html)
- [EnvGS](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_EnvGS_Modeling_View-Dependent_Appearance_with_Environment_Gaussian_CVPR_2025_paper.pdf)

Any pretrained monocular/depth model must pass the competition's external-data
and provenance rules before use.

## GPU and run recommendations

Observed peak allocation on the RTX 4070 Ti SUPER:

| Workload | observed VRAM |
|---|---:|
| 10k antialiased splatfacto training | about 4.0 GB |
| full-resolution DIBR/refiner apply | about 7.1 GB |
| exact-metric refiner, 256 crops, batch 4, base 32 | about 7.3 GB |

The current 16 GB GPU is sufficient for all proposed pilots. Use batch size 2
and base 32–48 for K=5 evidence; keep images on CPU and use gsplat. A 24–48 GB
GPU is useful for speed, larger evidence batches, or parallel scenes, but is
not required to validate the next mechanism.

The local environment needed a real CUDA compiler for gsplat JIT. Launch
through Conda so `<env>/bin/nvcc` is on `PATH`:

```bash
conda run --no-capture-output -n airace python ...
```

The repository `environment.yml` already pins the CUDA 12.1 compiler/toolkit.
The remaining local production prerequisite is the `colmap` executable for
dense MVS; the lightweight pilots reused supplied sparse geometry.

## Adoption order

1. Restore/rebuild the 30k dense/SSS holdout anchor.
2. Implement K=5 per-pixel source attention for bonsai; gate at +0.003 global
   and +0.008 on the early/high-fallback slice.
3. Implement chair fixed-midpoint exposure integration with M=5; confirm with
   M=9 only after a positive gate.
4. Try depth confidence/refinement on bonsai.
5. Try reflective appearance only after the coverage error is reduced.
6. Re-run the opt-in exact loss/checkpoint selector on the production anchor;
   do not infer adoption from the sparse pilot.

The practical path to 77 is therefore two scene-specific model changes, not a
fleet-wide parameter sweep: learned surface-wise evidence selection for
`bonsai`, and physically constrained blur formation for `chair`.
