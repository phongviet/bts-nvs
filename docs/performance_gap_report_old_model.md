# Round-2 performance-gap audit: 75.3793 → 77

Date: 2026-07-28  
Target machine: RTX 4070 Ti SUPER, 16 GB VRAM  
Target environment: Python 3.10, `airace`, PyTorch 2.5.1+cu121

## Executive conclusion

The current model does not have one general performance problem. It has two
different indoor-scene problems:

1. **`chair` is a motion-blurred reconstruction problem.** Error and LPIPS get
   markedly worse later in the capture, while training-image sharpness falls
   strongly over the same interval. More 3DGS capacity, more refiner pairs,
   monodepth supervision, SSS, and ordinary appearance changes do not address
   this image-formation error.
2. **`bonsai` is primarily a sparse-coverage/source-selection problem, with a
   secondary reflective-appearance problem.** Its early capture is the weak
   portion. The current DIBR chooses source images by camera-center distance
   alone; on the repeated orbit this sometimes chooses images over 1,000 video
   frames away even though every target is temporally bracketed.

The five drone scenes already score approximately 77.2–77.7. They should be
frozen. Because the leaderboard averages scenes rather than frames, all useful
development effort belongs on `bonsai` and `chair`.

The best next sequence is:

1. Correct refiner loss/checkpoint selection and measure it on fixed full-frame
   holdouts.
2. Add **learned pose/temporal-aware source selection** for `bonsai`; do not
   hand-average more neighbors.
3. Add **fixed-world-frame exposure integration** for `chair`; do not optimize
   the supplied midpoint camera poses.
4. If those pass their gates, try dense learned geometry initialization and a
   reflective appearance model.

Pure video interpolation is not the answer. I ran a real pretrained RIFE v4.25
leave-one-out pilot. It lost to the existing raw renderer on every evaluated
frame in both indoor scenes.

Reaching 77 is ambitious. It needs +1.6207 leaderboard points from only two of
seven scenes, equivalent to an average **+0.05672 Score per indoor scene**.
Parameter polishing alone is very unlikely to supply that. At least one
structural fix—and probably one fix for each indoor scene—is required.

## 1. Scope, rules, and repository state

The competition objective in `docs/de_bai.md` is:

```text
Score = 0.4 * (1 - LPIPS_VGG)
      + 0.3 * SSIM
      + 0.3 * (PSNR / 50)
LB    = 100 * mean(scene Score)
```

The supplied training images, COLMAP cameras, sparse reconstruction, and test
poses may be used. Test ground truth is unavailable. The competition forbids
external data intended to reconstruct the test ground truth and manual
per-test-pose intervention. All diagnostics in this report use supplied
training images and leave-one-out validation targets only.

One requested document is missing:

- `docs/de_bai.md` is present and was read.
- `docs/de_bai_submission_1.md` is absent from the working tree, all Git refs,
  and Git history. Its requirements therefore could not be audited. Add it
  before the final packaging audit if it contains rules beyond `de_bai.md`.

The worktree was already in the middle of a cherry-pick. I did not alter,
discard, or commit existing user changes.

### Local runtime audit

The machine is suitable for the immediate experiments:

| Item | Observed |
|---|---:|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| VRAM | 16,376 MiB |
| Free during audit | about 15,630 MiB |
| Driver | 580.105.08 |
| Conda environment | `airace` |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 |
| CUDA available in PyTorch | yes |

The host driver/CUDA 13.1 capability is compatible with the environment's
CUDA 12.1 runtime. Do not rebuild the environment merely to make its CUDA
version string equal the driver capability. The environment is internally
consistent and already contains Nerfstudio, gsplat, LPIPS-VGG, OpenCV, SciPy,
and scikit-image.

`python -m compileall -q src Analysis scripts tests` succeeds. The conda
environment does not include `pytest`; the lightweight loss tests run directly,
but `pytest -q` cannot be used until pytest is installed.

### Reproduction path defect

The extracted phase-2 data are at:

```text
data/raw/phase2/<scene>
data/raw/phase2/round2 -> .
```

But the current DIBR/refiner code hard-codes:

```text
data/raw/VAI_NVS_DATA_ROUND2
runs/round2/phase_locked
```

while:

```bash
scripts/phase_run.sh <scene> round2 phase2
```

writes:

```text
data/processed/phase2
runs/phase2/phase_locked
```

Therefore the README's local phase-2 run and the DIBR/refiner stage do not join
without manual symlinks or path edits. This is a reproducibility defect, not an
explanation for the existing leaderboard score, but it will waste an expensive
training run if left unfixed.

Recommended repair:

- add `--raw-root`, `--processed-root`, and `--runs-root` arguments to
  `Analysis/04_x3_dibr_pilot.py`;
- pass those roots through `Analysis/10_refiner_pilot.py`;
- make the README use one canonical naming convention (`phase2` is preferable);
- add a preflight that resolves the scene directory and checkpoint before
  launching any GPU work.

## 2. Score accounting: what 77 actually requires

Current leaderboard:

```text
77.0000 - 75.3793 = 1.6207 LB points
```

The five drone scenes are already near the goal. If only the two indoor scenes
change:

```text
required sum of indoor Score gains = 1.6207 * 7 / 100
                                    = 0.113449

required average indoor Score gain = 0.0567245
```

Useful conversion:

| Change | Global LB effect |
|---|---:|
| +0.010 Score on one scene | +0.1429 |
| +0.010 Score on both indoor scenes | +0.2857 |
| +0.020 Score on both indoor scenes | +0.5714 |
| +0.040 Score on both indoor scenes | +1.1429 |
| +0.05672 on both indoor scenes | +1.6207 |

This is why another small JPEG, pair-count, or learning-rate sweep cannot be the
main strategy.

Existing score decomposition:

| Stage | `bonsai` | `chair` |
|---|---:|---:|
| Raw backbone holdout | 0.6759 | 0.6506 |
| DIBR + refiner v2 | 0.6913 | 0.6650 |
| Gain | +0.0154 | +0.0144 |
| Best measured NAF/evidence/adversarial arm | 0.7016 | 0.6724 |

The refiner helps both scenes by nearly the same amount. The gap is already in
the reconstructed signal that the refiner receives.

## 3. Current pipeline and the unmodeled variables

The shipped path is:

```text
dense COLMAP
  → splatfacto-big, or 4M/60k SSS for bonsai
  → render RGB/depth
  → DIBR real-pixel warp from K=3 views
  → NAF residual refiner with evidence + PatchGAN
  → q98 4:4:4 pooled encoding
```

The important omission is upstream of the final CNN. In
`Analysis/04_x3_dibr_pilot.py`, candidate views are ranked by:

```python
cand = np.argsort(np.linalg.norm(self.centers - center_T, axis=1))
```

This ignores:

- relative camera rotation;
- signed or absolute video-frame distance;
- whether a source is before or after the target;
- source-frame sharpness/exposure;
- target/source motion;
- source depth reliability;
- per-pixel consistency across views.

The evidence refiner receives distance-ordered source slots, but no explicit
metadata that lets it distinguish “near in 3D but a different pass around the
orbit” from “temporally adjacent and visually consistent.”

Wider DIBR blending was already negative because inverse-distance averaging
dilutes good pixels with bad ones. That result refutes **hand-averaging K=5**;
it does not refute **a K=5 candidate pool with learned per-pixel selection**.

## 4. Data diagnosis

### 4.1 Pose and temporal structure

| Property | `chair` | `bonsai` |
|---|---:|---:|
| Training images | 205 | 248 |
| Test poses | 58 | 28 |
| Registered poses | 263 | 276 |
| Test poses bracketed in video time | 58/58 | 28/28 |
| Typical train-to-test temporal half-gap | 5 frames | 10 frames |
| Current K=3 source temporal offset, median | 10 | 20 |
| Current K=3 source temporal offset, p90 | 10 | 1,038 |
| Current K=3 source temporal offset, max | 355 | 1,370 |

The enormous `bonsai` tail is caused by the camera revisiting nearby positions.
Euclidean center distance aliases different moments/passes of the orbit.

Adding a modest temporal penalty to a normalized pose cost was tested
leave-one-out over all training images:

```text
cost = center_distance / scene_extent
     + rotation_degrees / 45
     + lambda_t * log(1 + frame_delta / nominal_step)
```

For `bonsai`, `lambda_t = 0.03–0.10`:

- reduces source temporal p90 from about 70 to 20 frames in the training
  leave-one-out proxy;
- reduces brightness MAE from 2.49 to about 1.68 intensity values;
- reduces log-sharpness MAE from 0.413 to about 0.372;
- raises sharpness rank correlation from 0.712 to about 0.77;
- increases median geometric center distance only from about 0.0196 to
  0.022–0.023 in scene-normalized units.

That is the strongest low-cost evidence for a real unused signal: temporal
proximity materially improves source appearance matching while barely
sacrificing pose proximity.

### 4.2 Sharpness/capture-time behavior

Variance of the Laplacian was computed for every training image.

| Statistic | `chair` | `bonsai` |
|---|---:|---:|
| Sharpness p10 / p50 / p90 | 164 / 792 / 3,147 | 48 / 127 / 372 |
| Early-tertile median | 2,382 | 286 |
| Middle-tertile median | 639 | 117 |
| Late-tertile median | 369 | 93 |
| Spearman(sharpness, frame index) | **−0.701** | −0.451 |
| Spearman(sharpness, inferred camera motion) | **−0.355** | not dominant |

For `chair`, JPEG byte size and sharpness correlate at about 0.932, independently
confirming that the late frames contain much less high-frequency information.

### 4.3 Holdout error by time and sharpness

The following correlations use the existing raw-backbone leave-one-out
holdout—not the hidden test images.

#### `chair`

| Relation | Spearman ρ |
|---|---:|
| Score vs frame index | **−0.544** |
| Score vs source-image sharpness | **+0.456** |
| LPIPS vs frame index | **+0.658** |
| LPIPS vs sharpness | **−0.655** |
| PSNR vs sharpness | −0.404 |

| Capture third | Score | LPIPS |
|---|---:|---:|
| Early | 0.6672 | 0.2695 |
| Middle | 0.6435 | 0.3165 |
| Late | 0.6422 | 0.3305 |

PSNR can prefer a smooth/blurred conditional mean, which explains its negative
correlation with sharpness. LPIPS exposes the actual perceptual damage.

**Diagnosis:** `chair` is trained from a time-varying mixture of sharp and
motion-blurred observations. Ordinary 3DGS explains that mixture by smearing
geometry/appearance. The final refiner has no shutter or camera-velocity input
from which to invert the blur.

#### `bonsai`

| Relation | Spearman ρ |
|---|---:|
| Score vs frame index | **+0.702** |
| Score vs sharpness | −0.390 |

| Capture third | Score | LPIPS |
|---|---:|---:|
| Early | 0.6144 | 0.3781 |
| Middle | 0.6862 | 0.3134 |
| Late | 0.7213 | 0.2831 |

If blur were the main cause, the low-sharpness late portion should be worse. It
is much better. The early portion instead has sparse geometry/coverage. Existing
notes report that roughly 9/28 target-like poses have only about two useful
neighbors and a 38.7% fallback rate.

**Diagnosis:** solve `bonsai` coverage/warping first. Treat the glossy tabletop
as a second, appearance-specific limitation.

### 4.4 Sparse reconstruction evidence

| Scene | Sparse points | Median track length | Track length p90 | Two-view points |
|---|---:|---:|---:|---:|
| `chair` | 80,491 | 4 | 16 | 8.47% |
| `bonsai` | 54,422 | 5 | 16 | 4.70% |

The `chair` dense MVS build stopped before its configured point cap because its
short baseline and blurred imagery provided insufficient reliable matches. This
supports a geometry/blur explanation, but not a generic “increase the Gaussian
count” explanation: `splatfacto-big` tied ordinary splatfacto and SSS lost on
chair.

## 5. Bounded experiment performed in this audit: real RIFE

The previous experiment log rejected frame interpolation using copy/mean/weighted
mean baselines. That did not by itself establish that a learned VFI system would
fail, so I tested the official Practical-RIFE v4.25 lite pretrained model.

Initial optimistic protocol:

- each holdout target is removed from its own source pair;
- other holdout frames remain eligible as sources, giving RIFE smaller,
  production-like temporal gaps but more source information than the strict
  3DGS holdout;
- nearest preceding and following video frames are used;
- the exact fractional target timestamp is supplied to RIFE;
- full-resolution output is evaluated with the repository's VGG-LPIPS grader;
- only temporally bracketed targets are scored;
- no test GT or leaderboard feedback is used.

| Scene | Policy | N | RIFE Score | Existing raw renderer on same subset | RIFE wins |
|---|---|---:|---:|---:|---:|
| `chair` | optimistic leave-one-out | 24 | **0.4293** | 0.6521 | 0/24 |
| `chair` | strict all-holdout exclusion | 24 | **0.4243** | 0.6521 | 0/24 |
| `bonsai` | optimistic leave-one-out | 24 | **0.5454** | 0.6787 | 0/24 |
| `bonsai` | strict all-holdout exclusion | 24 | **0.5313** | 0.6787 | 0/24 |

The oracle that chooses RIFE only when it wins gains exactly zero on both
subsets under both policies. Pure 2D VFI is decisively refuted for this data.

Full strict metrics:

| Scene | PSNR | SSIM | LPIPS | Score |
|---|---:|---:|---:|---:|
| `chair` | 16.251 | 0.3197 | 0.4227 | 0.4243 |
| `bonsai` | 18.646 | 0.6010 | 0.4022 | 0.5313 |

This does **not** invalidate temporal information. It shows that video time must
be used to choose/weight 3D-warped evidence, not to interpolate unaligned 2D
frames through parallax and occlusion.

Reproduction script:

```bash
conda run -n airace python Analysis/27_rife_temporal_pilot.py \
  --scene chair \
  --rife-repo /path/to/official/Practical-RIFE \
  --model-dir /path/to/official/Practical-RIFE/train_log \
  --source-policy strict \
  --out-dir results/rife_temporal_pilot/chair/holdout_strict
```

The saved metrics and per-image predictions are in
`results/rife_temporal_pilot/`.

Diagnostic dependency provenance: official Practical-RIFE commit
`17d8c7a1005b37f4c97bfee04e316aaec7fdc536`, v4.25 lite weights, MIT code
license. It was used only for this negative diagnostic and does not ship in the
submission.

## 6. Code-level gaps worth fixing before another large model

### 6.1 The “grader loss” is not the grader loss

`Analysis/10_refiner_pilot.py` labels this as the grader objective:

```python
0.4 * LPIPS + 0.3 * (1 - SSIM) + 0.3 * L1
```

The real third term is `-0.3 * PSNR/50`, not L1. L1 is a reasonable generic
surrogate, but it has different sample weighting and gradient behavior. It can
prefer the wrong balance between difficult blurred regions and easy regions.

For images in `[0,1]`:

```text
PSNR = -10 log10(MSE)

-0.3 * PSNR / 50
    = 0.06 * log10(MSE)
    = 0.0260577 * ln(MSE)
```

An exact metric-shaped minimization term is therefore:

```python
mse = (pred - target).square().flatten(1).mean(1)
mse = mse.clamp_min(1e-5)  # PSNR cap at 50 dB
psnr_term = 0.0260577 * torch.log(mse).mean()

loss = 0.4 * lpips + 0.3 * (1 - ssim) + psnr_term
```

Use per-image MSE before averaging. Test three arms with every other setting
fixed:

1. current L1 control;
2. exact metric-shaped term;
3. 50/50 hybrid between current L1 and exact PSNR term.

The current differentiable SSIM also uses zero-padded borders, unlike the usual
valid/skimage evaluation behavior. Make it valid-window or reflect-padded for
the A/B. This is probably a small effect, but it is nearly free to correct.

### 6.2 Adversarial checkpoints are selected with the wrong ruler

The code evaluates EMA checkpoints every 250 iterations and retains the one
with the smallest regression `val_loss`. The project has already observed that:

- regression `val_loss` predicted the wrong sign for the adversarial drone
  rollout;
- it predicted improvement for two backbone swaps that lost on the leaderboard;
- the critic intentionally moves the result away from the conditional mean.

Even though adversarial weight 0.003 was selected by a final rendered A/B, the
checkpoint inside each arm is still chosen with the invalid regression ruler.

Fix:

- save EMA snapshots at 500, 1k, 2k, 4k, and 6k iterations;
- render the same fixed 25-frame full-image holdout for each;
- choose the iteration by the actual VGG-LPIPS/SSIM/PSNR Score;
- use that iteration once for final all-train fitting;
- never pick an iteration using hidden-test leaderboard probes.

This is higher priority than another adversarial-weight sweep; weight 0.003 is
already well measured.

### 6.3 The refiner lacks variables that explain the error

Recommended per-source conditioning:

- relative camera-center vector and normalized distance;
- relative rotation, not only a scalar rank;
- signed frame delta and absolute frame delta;
- temporal interpolation fraction;
- source sharpness and brightness;
- source-to-target depth residual;
- valid/occlusion/photometric-guard confidence;
- local disagreement among warped source colors.

The network should predict per-pixel softmax weights over source candidates,
plus a residual. A source dimension with shared weights is preferable to
hard-coded channel slots because it is permutation-aware and scales from K=3 to
K=5/6.

## 7. Prioritized experiment program

### P0 — repair validation and optimize the real metric

**Cost:** low; 4070 Ti SUPER is sufficient.  
**Why first:** no new backbone is needed, and the present selection criterion is
known to be unreliable for the shipped adversarial model.

Keep fixed:

```text
NAF blocks
evidence enabled
adv = 0.003
EMA = 0.999
base = 48
crop = 256
iterations = 6,000
same fit/validation names and RNG seeds
```

Arms:

```text
A: current L1 surrogate + current checkpoint rule (control)
B: current L1 surrogate + full-frame Score checkpoint rule
C: exact PSNR-shaped loss + full-frame Score checkpoint rule
D: hybrid L1/PSNR loss + full-frame Score checkpoint rule
```

Adoption gate:

- mean full-frame Score gain ≥ 0.002;
- no scene loses more than 0.001;
- LPIPS must not pay for a small PSNR-only gain;
- repeat the winner with a second seed.

This will not produce the whole 1.62 LB gap, but it makes every later refiner
experiment trustworthy.

### P1 — `bonsai`: temporal/pose-aware learned source attention

**Cost:** low-to-medium; 16 GB is sufficient with crop training.  
**Confidence:** highest structural hypothesis.

Initial candidate set:

```text
1 nearest earlier temporal frame
1 nearest later temporal frame
3 nearest pose-aware spatial frames after de-duplication
```

Pose-aware spatial cost:

```text
center_distance / scene_extent + rotation_deg / 45
```

Do not average all five. Warp each separately and expose its confidence and
metadata to a lightweight attention head. Predict a per-pixel masked softmax
over views at 1/4 resolution, upsample the weights, blend the full-resolution
warps, then let the existing NAF residual head refine the result.

Suggested first configuration:

```text
K = 5
attention feature width = 32
attention resolution = 1/4
temperature = learned, initialized to 1
crop = 256
base = 48
max pairs = 90 for pilot
iterations = 3,000 pilot, 6,000 confirm
adv = 0 for architecture gate, then 0.003 for the winner
```

Why gate without GAN first: it separates better evidence aggregation from
critic variance. Re-enable the already-proven critic only after the input path
wins.

Validation slices:

- all fixed `bonsai` holdout frames;
- early capture third;
- sparse-coverage subset defined from train-pose KNN density, not by inspecting
  test outputs;
- middle/late subset as a regression guard.

Adopt if:

- all-frame gain ≥ 0.003;
- early/sparse slice gain ≥ 0.008;
- no late-slice regression > 0.002;
- gain survives the full refiner + adversarial stack.

Do not tune `lambda_t` from test images. The direct `[previous,next]` guarantee
already avoids a fragile scalar sweep. The temporal-penalty proxy can be used
only for ordering additional spatial candidates.

### P2 — `chair`: fixed-midpoint exposure integration

**Cost:** medium-to-high; 16 GB is enough for a careful implementation.  
**Confidence:** high mechanism confidence, moderate implementation risk.

The failed `3dgs-deblur` experiment does not close this hypothesis. That method
optimized training camera poses/velocities and drifted the Gaussian world frame
away from the fixed raw COLMAP/test-pose frame. Its reported test gains depend
on optimizing evaluation cameras, which is inappropriate here.

The competition-safe version must preserve the supplied midpoint pose exactly:

1. Estimate a camera path derivative from adjacent registered COLMAP poses.
2. For training frame `i`, create 5 symmetric subposes along the local SE(3)
   path around the supplied pose.
3. Render all subposes and average them to predict the observed blurred frame.
4. Optimize the canonical Gaussians and, initially, one non-negative shutter
   scale per frame.
5. Strongly smooth shutter scales over video time and regularize them toward a
   scene median.
6. Never optimize the midpoint pose or scene-wide world transform.
7. At a target pose, infer the shutter scale from neighboring training
   sharpness/motion and integrate along the derivative of the supplied target
   camera sequence.

Parameter ladder:

```text
M subposes:            5 pilot → 9 confirmation
shutter scale:         softplus parameter
temporal smoothness:   1e-2, then one 3x sensitivity check
shutter prior:         1e-3
midpoint pose:         frozen
rolling shutter:       disabled in first arm
```

Only add row-time rolling shutter after global exposure integration wins.

Validation:

- report early/middle/late thirds separately;
- require a substantial improvement on the late third;
- check that inferred shutter length correlates negatively with observed
  sharpness and positively with estimated path speed;
- reject any solution whose camera midpoints differ from the input poses.

Adopt if:

- all-frame raw-backbone Score gain ≥ 0.005;
- late-third gain ≥ 0.010;
- improvement is fidelity-positive or at least survives the existing refiner;
- the win repeats with 5 and 9 samples or is stable to the integration count.

### P3 — learned dense geometry initialization, fixed to COLMAP

**Cost:** medium; preprocessing fits 16 GB if chunked.  
**Use for:** early `bonsai` coverage and blurred/short-baseline `chair`.

The earlier phase-1 VGGT pseudo-cloud result does not establish that all learned
geometry is ineffective. It tested a different data regime and integration.
Likewise, injecting Depth Anything as a scale/shift-invariant training loss is
not the same as supplying multi-view-consistent dense points.

Recommended gate:

1. Run VGGT or MASt3R/MVSFormer++ on overlapping chunks of 8–16 images at about
   518 px.
2. Keep the organizer's COLMAP cameras fixed.
3. Align predicted depth/point maps to COLMAP using robust scale and depth
   fitting on known sparse tracks.
4. Remove low-confidence, inconsistent, behind-camera, and duplicate points.
5. Fuse 1–2M points and use them only as 3DGS initialization.
6. Compare raw render Score before attaching the refiner.

Do not replace or bundle-adjust the official camera coordinate system. The
world-frame drift lesson is strong and repeated.

Adopt if:

- the early `bonsai` slice gains ≥ 0.008 and global `bonsai` ≥ 0.003;
- `chair` improves all three metrics or survives the refiner;
- training-view reconstruction does not improve while novel-pose holdout gets
  worse—a signature of memorization already seen with SSS.

### P4 — `bonsai` reflective appearance

**Cost:** low for SH sweep, high for a new renderer.

The glossy table is poorly handled by raw real-pixel DIBR and SH degree 3.
Start with the smallest test:

```text
SH degree 3 control
SH degree 4
SH degree 5
```

Run the complete refiner after each raw-render gate because a backbone LPIPS
gain can be absorbed or inverted downstream.

If higher SH is positive but insufficient, investigate GaussianShader or a
view-adaptive Gaussian method. Condition on viewing direction and camera
distance while preserving the fixed geometry.

Adopt only when the full stack improves; raw LPIPS alone is not enough.

### P5 — capacity only after the mechanisms pass

Try `base=64`, crop 384, or more pairs only after P0/P1. Pair count 90→180
already gained only +0.032 LB and did not improve LPIPS, so width/count without
new explanatory variables is low priority.

On 16 GB:

- use AMP;
- cache RGB/depth on CPU or memmap;
- compute source features per crop or at 1/4 scale;
- use gradient checkpointing for base 64/crop 384;
- avoid holding K full-resolution feature pyramids on GPU.

## 8. What is closed, and what is not

| Family | Correct conclusion |
|---|---|
| More ordinary refiner pairs | Count is nearly exhausted; new metadata/selection is not tested |
| K=5 inverse-distance DIBR | Hand-blending is harmful; learned K=5 selection is not tested |
| RIFE/FILM-style pure VFI | Closed by real RIFE pilot and simple baselines |
| Camera optimizer | Closed because fixed test world frame must be preserved |
| Current `3dgs-deblur` run | That pose-optimizing implementation is closed |
| Fixed-midpoint exposure model | Not tested; directly supported by chair diagnostics |
| Depth Anything loss | Monocular loss injection is negative |
| Multi-view dense learned initialization | Not tested in the proposed fixed-camera form |
| SSS globally | Closed; only bonsai benefits |
| Higher SH / reflective Gaussian model | Not tested on bonsai |
| Larger adversarial weight | Closed; 0.003 is best |
| Adversarial checkpoint iteration | Not validly selected; must be retested |
| Exact competition-shaped loss | Not tested |
| JPEG/encoding | Bounded to roughly 0.02 LB; freeze until final packaging |

## 9. Paper-search map

These are not generic reading recommendations; each maps to a measured failure.

### `bonsai`: source-view aggregation and sparse proxy geometry

Search terms:

```text
learned source view selection image based rendering
per-pixel multi-view attention novel view synthesis
sparse geometry depth completion image based rendering
visibility-aware source view transformer
pose-aware temporal source selection NVS
```

Starting points:

- [IBRNet: Learning Multi-View Image-Based Rendering](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_IBRNet_Learning_Multi-View_Image-Based_Rendering_CVPR_2021_paper.pdf):
  source-view feature aggregation and ray-level visibility reasoning.
- [SIBRNet: Learning Robust Image-Based Rendering on Sparse Scene Geometry via Depth Completion](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_Learning_Robust_Image-Based_Rendering_on_Sparse_Scene_Geometry_via_Depth_CVPR_2022_paper.html):
  sparse-depth completion, projection-bias correction, and learned light
  blending.

The implementation goal is not to port either paper wholesale. Borrow the
source-view attention/bias-correction mechanism and retain the current 3DGS
depth/DIBR/refiner pipeline.

### `chair`: blur-aware 3DGS without world-frame drift

Search terms:

```text
fixed pose motion blur gaussian splatting exposure integration
camera trajectory during exposure 3DGS
motion blurred handheld video novel view synthesis
rolling shutter gaussian splatting fixed midpoint pose
SE(3) exposure trajectory gaussian splatting
```

Starting points:

- [Gaussian Splatting on the Move](https://arxiv.org/abs/2403.13327):
  physical exposure/rolling-shutter rendering with camera velocities.
- [BAD-Gaussians](https://arxiv.org/abs/2403.11831):
  trajectory-based blurred image formation, but its bundle adjustment must be
  removed here.
- [CoMoGaussian](https://openaccess.thecvf.com/content/ICCV2025/html/Lee_CoMoGaussian_Continuous_Motion-Aware_Gaussian_Splatting_from_Motion-Blurred_Images_ICCV_2025_paper.html):
  continuous motion-aware reconstruction from blurred images.

When adapting these methods, freeze the official midpoint pose and world frame.
That constraint is more important than matching a paper's default config.

### Dense multi-view initialization

Search terms:

```text
multi-view transformer dense point map fixed camera gaussian initialization
VGGT depth alignment COLMAP
MASt3R dense point cloud gaussian splatting initialization
MVSFormer++ short baseline blurred images
confidence filtered depth fusion 3DGS
```

Starting points:

- [VGGT](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf):
  feed-forward cameras, depth, point maps, and tracks. Use its geometry, not its
  camera frame.
- [MASt3R official implementation](https://github.com/naver/mast3r).
- [MVSFormer](https://arxiv.org/abs/2208.02541).
- [DepthSplat official implementation](https://github.com/cvg/depthsplat):
  useful keywords and modules for combining multi-view depth and Gaussian
  prediction, though a full port is likely heavier than the proposed init-only
  test.

### Reflective `bonsai` table

Search terms:

```text
specular gaussian splatting view dependent appearance
reflective surface 3DGS shading normals
view adaptive gaussian splatting camera distance
directional encoding specular NeRF
```

Starting points:

- [GaussianShader](https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_GaussianShader_3D_Gaussian_Splatting_with_Shading_Functions_for_Reflective_Surfaces_CVPR_2024_paper.pdf):
  normals plus a compact shading function for reflective surfaces.
- [Scaffold-GS](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Scaffold-GS_Structured_3D_Gaussians_for_View-Adaptive_Rendering_CVPR_2024_paper.html):
  view-direction/distance-conditioned attributes.
- [SpecNeRF](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_SpecNeRF_Gaussian_Directional_Encoding_for_Specular_Reflections_CVPR_2024_paper.html):
  directional encoding for specular effects.

## 10. Anti-overfitting protocol

The competition has only two problematic scenes, so ordinary one-split tuning
can easily overfit.

Use:

1. **Development fold:** deterministic temporally distributed holdout.
2. **Confirmation fold:** disjoint names, selected before seeing experiment
   output.
3. **Slice metrics:** early/middle/late and pose-density bins defined from
   metadata, not result inspection.
4. **Two seeds** for any neural winner near the ±0.002 noise floor.
5. **Full-frame official metric** for decisions; crop loss is for training only.
6. **One winner per family** reaches the confirmation fold.
7. **No leaderboard tuning** of frame-specific parameters, temporal weights,
   shutter values, or checkpoint iterations.
8. Retrain the confirmed configuration on all supplied training frames exactly
   once, then package.

A useful ledger for each run:

```text
hypothesis
changed variables
fixed variables
development Score and components
confirmation Score and components
early/middle/late Score
runtime and peak VRAM
decision and reason
```

## 11. GPU recommendation

You do **not** need a better GPU for P0, P1, the RIFE conclusion, SH degree
4/5, or a careful fixed-midpoint blur pilot. The 4070 Ti SUPER 16 GB is enough.

A 24 GB 4090/3090 is helpful when:

- keeping K=5/6 full-resolution feature maps resident;
- training crop 384–512 with a base-64/96 refiner;
- running less aggressively chunked VGGT/MASt3R;
- retrying a high-capacity splatfacto model that previously OOMed at 16 GB.

A 48 GB A6000/L40S class GPU is useful, not mandatory, for:

- full-resolution long-sequence learned multi-view backbones;
- 4M Gaussian experiments with generous optimizer state;
- large source-view transformers without feature streaming.

More GPU will shorten iteration time and enable larger ablations, but it will
not resolve the current modeling omissions. Spend on a larger GPU only after a
low-resolution/crop pilot proves the mechanism.

## 12. Recommended order of work

| Order | Experiment | Stop/go criterion | Expected compute class |
|---:|---|---|---|
| 1 | Exact loss + full-frame checkpoint selection | ≥ +0.002 both-scene mean | hours |
| 2 | `bonsai` K=5 temporal/spatial attention, no GAN | ≥ +0.003 global, +0.008 early | hours |
| 3 | Re-enable adv 0.003 on P1 winner | preserves P1 gain | hours |
| 4 | `chair` fixed midpoint, M=5 | ≥ +0.005 global, +0.010 late | one training run |
| 5 | Confirm chair M=9 | stable mechanism | one training run |
| 6 | Fixed-camera learned dense init | ≥ +0.003 full-stack | preprocessing + run |
| 7 | `bonsai` SH4/SH5 | full-stack positive | two runs |
| 8 | GaussianShader/view-adaptive method | only if SH trend is positive | research port |
| 9 | Capacity scaling | only on a proven architecture | optional larger GPU |

If time is extremely short, do steps 1–3. If there is time for one serious
paper implementation, do fixed-midpoint exposure integration for `chair`.

## Final diagnosis

The repo's statement that “all lever families are closed” is too broad. The
completed experiments close several concrete implementations, but the data
audit exposes three untested variables:

1. **source capture time/temporal order** in the DIBR/refiner;
2. **camera motion during exposure** while preserving the fixed pose frame;
3. **the exact metric and valid checkpoint-selection signal** used to train the
   shipped adversarial refiner.

Those are the gaps to attack. The first is directly evidenced on `bonsai`; the
second is directly evidenced on `chair`; the third is a confirmed mismatch
between code, grader, and leaderboard behavior. They are more defensible and
less overfit-prone than another parameter sweep because each follows from a
measured failure mechanism rather than from hidden-test probing.
