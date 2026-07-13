# Finding F1: test GT images are RAW (radially distorted) — we render pinhole

*(2026-07-11, experiment X4, `03_x4_distortion_remap.py`, `X4_distortion_remap/`)*

## The defect

Every scene's COLMAP camera is `SIMPLE_RADIAL` (params `f, cx, cy, k`). The
provided `test_poses.csv` lists only `fx fy cx cy` — **`fx` equals the COLMAP
`f` to 3 decimals in all 13 scenes, i.e. the CSV was exported from the same
camera and simply omits `k`.** Our pipeline (and nerfstudio's convention)
undistorts train images for training and renders **pinhole** at test poses.
X4 proves the test GT is **not** undistorted: remapping our existing renders
into the distorted geometry massively improves every metric.

## Measured (hcm0034, existing locked-config renders, grader metric vgg+50)

| variant | PSNR | SSIM | LPIPS | Score | top-corner MSE |
|---|---|---|---|---|---|
| pinhole (all our submissions so far) | 21.428 | 0.7397 | 0.2393 | 0.6548 | 0.02314 |
| + distortion remap | **24.398** | **0.8433** | **0.2254** | **0.7092** | **0.00683** |
| delta | +2.97 dB | +0.104 | −0.014 | **+0.0544 (+5.4 LB pts)** | **−70%** |

This also solves the exp020 "frame-corner weakness" (2.5–2.9× corner error):
it was mostly unmodeled lens distortion (up to ~4.3 px displacement at corners
for k≈0.008–0.01, f≈926). exp021's conclusion that radial distortion "cannot
cause the top-bias" was correct about the *asymmetry* but missed that the bulk
of the corner error is radially symmetric misalignment. exp022's loss
reweighting failed because no loss can fix a geometric resampling mismatch.

## Per-scene k (from cameras.bin; CSV fx matches f in all 13)

| scene | k | max corner displacement (px, approx) |
|---|---|---|
| hcm0031/hcm0034/HCM0181/HCM0193/HCM0204 | +0.0078…+0.0100 | ~4–5 |
| HCM0249/HCM0254/HCM0276/HCM1439 | +0.0076…+0.0098 | ~4–5 |
| HNI0366/HNI0437 | +0.0120/+0.0138 | ~6–7 |
| **HNI0131 / HNI0265** | **−0.11479 / −0.11470** | **~35–40 (!)** |

The two HNI scenes with |k|≈0.115 have been submitted with renders that are
tens of pixels off across the whole frame periphery. This plausibly explains a
large share of the private-set score deficit (implied private mean 0.54 vs
public 0.63): 2 of 8 private scenes were being graded on massively misaligned
images. Expected gain there is far larger than the +0.054 measured on hcm0034.

## Fix (submission-ready, no retraining)

For each scene, remap every rendered test image from pinhole to SIMPLE_RADIAL
geometry using (f, cx, cy, k) from that scene's `cameras.bin` (fixed-point
inversion + bilinear sample — `distort_remap()` in `03_x4_distortion_remap.py`).
Optional refinement: render at 2× resolution and remap+downsample in one step
to avoid double-resampling blur (quantify before adopting).

Status: confirmed on ALL 5 public scenes (mean **+0.0596**, range +0.0505…+0.0743; see
`X4_distortion_remap/summary.json`). DIBR (X3) synthesizes directly in distorted geometry
(`out_k`), single resample.

**k=−0.115 scenes (HNI0131/HNI0265) — v1.1 fix DONE (2026-07-11, `08_hni_expanded_canvas_fix.py`):**
with negative k the distorted image sees rays *beyond* the pinhole render's FOV (measured up to
~87 px at corners — larger than the small-k estimate), so the v1.0 remap edge-replicated a border
band. Fixed by re-rendering all 112 test views on a 256 px-expanded canvas (same focal,
cx/cy+128) and remapping from that — border now has real content (visually verified; the fix even
recovers scene content the original pinhole framing cropped). The shipped
`exp030 submission_round1.zip` (467 MB, rebuilt 20:59) includes this fix. Note the Kaggle-trained
private checkpoints load locally via `eval_setup(update_config_callback=...)` path rewriting.
