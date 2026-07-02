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
  metrics). `PSNR_MAX=40.0` in `metrics.py` is still an unconfirmed
  placeholder — must verify against official rules before trusting
  `PSNR_norm`/`Score` absolute values.

- **⚠️ Critical bug found and fixed (same day):** the low initial score was
  NOT primarily reconstruction quality — `src/render.py`'s hand-rolled
  COLMAP→Nerfstudio pose conversion double-applied the world-coordinate axis
  swap. Nerfstudio 1.1.5's `ColmapDataParser` (with
  `assume_colmap_world_coordinate_convention=True`, our default) already
  bakes that swap into the saved `dataparser_transform`
  (`_get_all_images_and_cameras` returns `applied_transform`, composed into
  `transform_matrix` in `_generate_dataparser_outputs`). `render.py` was
  applying the swap by hand (with an additionally wrong permutation,
  `[1,0,2,3]` instead of the correct `[0,2,1,3]`) *and* via the saved
  transform, garbling every rendered pose. Caught by visually rendering a
  **seen/training view** and comparing to its GT — it should reconstruct
  near-perfectly and instead showed a small recognizable fragment crammed in
  one corner with the rest blank. Fixed by removing the manual axis-swap
  step; `colmap_pose_to_c2w` now applies only the OpenCV→OpenGL flip, and
  the (correct) axis swap comes from `dataparser_transform` alone.
  **Result after fix (same checkpoint, no retraining): Score 0.143 → 0.718**
  (PSNR 8.85→21.3, SSIM 0.167→0.734, LPIPS 0.933→0.156). This means the
  *actual* baseline reconstruction quality was always reasonable — Week 1's
  low number was a rendering-script bug, not a model problem.
- **Takeaway / process note:** always sanity-check any custom render/eval
  script against a **seen (training) view** before trusting scores on
  unseen views — a seen view should look close to perfect, and if it
  doesn't, the bug is in the eval path, not the model.
