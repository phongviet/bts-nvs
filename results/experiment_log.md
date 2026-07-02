# Experiment Log

Format per entry: hypothesis -> what ran -> result -> decision.

## exp001_baseline_splatfacto (Week 1)

- **Hypothesis:** vanilla `splatfacto` (default strategy, 30k iters) on the
  provided COLMAP sparse init gives a working, submittable baseline and lets
  us validate the full pipeline (pose convention, render, metric, package).
- **What ran:** Full 30k-iteration vanilla `splatfacto` on `hcm0034` (public
  scene), trained on the 240 filtered train-only images (see
  `src/data_prep/filter_colmap_train.py` — the raw sparse model contains all
  337 captured images including the 60 held-out test images, which the
  default Nerfstudio ColmapDataParser would otherwise leak into training).
  Rendered all 60 official test poses at their per-pose intrinsics/resolution
  and scored against the real public test GT.
- **Result:** PSNR=8.85, SSIM=0.167, LPIPS=0.933, **Score=0.143**. Visual
  inspection: pose convention confirmed correct (recognizable building
  content appears in the geometrically plausible position/orientation), but
  reconstruction quality is poor — one near-nadir test view rendered nearly
  blank. Ruled out near/far collider clipping as the cause (gsplat's
  `rasterization()` call hardcodes `near_plane=0.01, far_plane=1e10`,
  ignoring the model's configured collider). The low score is consistent
  with genuinely poor coverage from the untouched sparse COLMAP point cloud
  (no dense init, no sky masking, no appearance modeling) on a hard BTS
  aerial-drone scene with steep viewpoint changes.
- **Decision:** Infrastructure milestone achieved — pipeline is fully
  validated end-to-end (data loading with correct train/test isolation, pose
  conversion, training, per-pose-intrinsics rendering, exact-formula
  metrics). Quality is Week 2's job. This result matches
  `Documents/plan_overall.md`'s prediction that vanilla/sparse-init
  significantly underperforms dense init — **prioritize the Week 2 init
  ablation (dense COLMAP / depth-unprojected) first**, it's expected to be
  the single biggest lever. Also prioritize sky masking given the
  near-blank nadir view. `PSNR_MAX=40.0` in `metrics.py` is still an
  unconfirmed placeholder — must verify against official rules before
  trusting `PSNR_norm`/`Score` absolute values.
