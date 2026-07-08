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

## exp019_valsplit_validation (Week 2b, Jul 8)

- **Hypothesis:** a match-test val split (A.6b) ranks configs more consistently
  with real test GT than the every-Nth split, and can be trusted for
  private-scene model selection.
- **Ran:** render-only on existing checkpoints (no training): both pilots,
  11 variants (exp001/002/004), both split types at equal size (n=30),
  scored vs train photos, rankings compared to each variant's real-test-GT
  score. `scripts/run_exp019_valsplit_validation.py`.
- **Result:** match-test Spearman 1.0 + pairwise A/B agreement 20/20 on both
  pilots; every-Nth inverted one close pair on hcm0034 (scale_reg vs mcmc;
  Spearman 0.943, 13/14). Caveat: val views were seen in training, so this
  validates the pose-distribution component of the signal only.
- **Decision:** adopt match-test splits for all private-scene selection.
  Splits cut for all 8 private scenes same day (n=30; HCM1439 n=15 — it only
  ships 103 images). From now on private training must exclude
  `splits/val_ids.txt` or the val signal degrades to train-fit.

## exp011_postproc_encoding_sweep (Week 2b, Jul 8)

- **Hypothesis:** a global render post-process + encoding choice can buy
  +0.001–0.003 at zero training cost (A.7).
- **Ran:** 7 ops x 6 encoders on exp002 renders_test, all 5 public scenes,
  scored vs public GT (`src/postprocess/sweep_postprocess.py`,
  `results/week3_postproc_ablation.csv`). Round-trip check clean: identity
  jpeg95/98 >= 48 dB on GT, so IO/gamma path is trustworthy.
- **Result:** **unsharp_r1_p50 + jpeg98 wins all 5 scenes, mean +0.0039**
  (+0.0031..+0.0047) over identity+jpeg98. Sharp optimum: r1_p100 loses
  everywhere (~-0.006), r2 variants lose big (-0.038). Encoder axis nearly
  flat above jpeg90; png marginally best but banned in packaging (.JPG names).
- **Decision:** adopt `unsharp_r1_p50 + jpeg98` globally in the packaging
  path (`apply_postprocess.py`). Re-verify once on the next submission
  (delta-transfer check #2 rides on it).

## exp012_ensemble_pilot (Week 2b, Jul 8)

- **Hypothesis:** pixel-space render averaging trades LPIPS for PSNR/SSIM
  net-positive (A.8).
- **Ran:** cross-config variant (runs keep only final checkpoints): mean of
  {antialiased, dense} and {antialiased, dense, sky_mask} renders_test on
  both pilots, jpeg98, vs public GT. `runs/phase1/exp012_ensemble_pilot/`.
- **Result:** hcm0034 -0.0007, HCM0181 +0.0001 vs best single (antialiased).
  Averaging inflated LPIPS (~0.13 -> ~0.15-0.16); the 0.4-weighted LPIPS term
  eats the SSIM gain.
- **Decision:** **drop** (< +0.002 kill threshold on both pilots).
  Checkpoint-ensembling proper stays untested (single ckpt per run); only
  revisit if a future sweep saves multiple checkpoints for free.

## exp008 prep: transient masks built (Week 2b, Jul 8)

- Built SegFormer person+vehicle masks for both pilots (240/scene, 2x3
  tiling, dilate 4, QA overlays in `transient_masks_qa/`). Coverage sane:
  median ~0.25% pixels masked, max ~5%, no runaway false positives.
- Visual QA: catches motorbike clusters/pedestrians; occasional false
  positives on *static red objects* (mast beacon on hcm0034, parasols) —
  small and per-frame, other views still supervise those pixels, but read
  the exp008 A/B knowing masks are slightly over-eager on red.
- Staging symlink deliberately NOT created; exp008 links masks at training
  time.
