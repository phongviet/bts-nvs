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
- **REVERTED (Jul 9, LB datapoint #4):** the op scored −0.00502 on the
  leaderboard. Post-mortem: the local +0.0039 lived entirely in alex-LPIPS
  (−0.024); the LB uses VGG (confirmed), where the op is LPIPS-neutral
  (0.2393→0.2396) and its PSNR/SSIM costs dominate. Local vgg re-score
  delta = −0.0050 — matches the LB delta exactly. Lesson: score all
  decisions with vgg + psnr_max=50 (both now local defaults where it
  counts); a win concentrated in alex-LPIPS is a red flag, not a result.

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

## exp007_bilateral_grid + exp008_transient_mask (Week 2b, Jul 9, Kaggle)

- **Hypothesis:** (007) bilateral grid compensates per-image exposure variation;
  (008) masking people/vehicles out of the loss removes inconsistent content,
  win expected on HCM0181.
- **Ran:** both on Kaggle T4x2 (local 6 GB thrashed on the grid: 3.2 s/iter,
  14x control -- killed at 59%, whole experiment re-scoped). 5 cells: HCM0181
  control + bilateral x2 + masked x2, `kaggle-exp007-008-tierA.ipynb`, all
  scored vgg+50 in-notebook. Controls: hcm0034 0.6548, HCM0181 0.6344 (fresh,
  matches the antialiased proxy 0.6338).
- **Result:** bilateral LOSES both pilots (-0.0046 / -0.0081; PSNR -0.6..-0.8
  dB) -- same-flight DJI exposure is already consistent, the grid only eats
  capacity/optimization budget. masked is a wash on both (-0.0005 / -0.0003)
  -- even on the max-traffic scene; transients at this GSD are too small and
  too sparse to matter, and the masks' static-red false positives (mast
  beacon, parasols) cost a little valid supervision.
- **Decision:** **drop both.** Tier-A locked config stays dense + antialiased
  (+ splatfacto-big pending exp006 session 2). Week-3 headroom now rides on
  capacity/iters (exp006 s2), perceptual FT (exp009), and the enhancer chain.
  Notebook infra note: run cells must put the airace bin on subprocess PATH
  (fixed in run_sweep 4f804f5 + notebook).

## exp015_difix (Week 3, Jul 9, Kaggle)

- **Hypothesis (gate for the exp016 LoRA week):** off-the-shelf Difix3D+
  (nvidia/difix, single-step SD-Turbo img2img fixer) improves locked-config
  pilot renders. plan_overall_v3 expectation: wash or small win at best --
  its strength is extrapolation artifacts we mostly don't have.
- **Ran:** pure inference over exp004 antialiased renders_test on both
  pilots, `kaggle-exp015-difix.ipynb`, scored vgg+50 pre/post vs real test
  GT. Verification duties all passed: pipeline signature matches the model
  card (prompt="remove degradation", 1 step, timesteps=[199], cfg 0);
  1320x989 round-trip exact via reflect-pad+crop; fits T4 without tiling
  (~3.4 GiB at half res fp16, full res comfortably inside 16 GB).
- **Env archaeology (cost ~3 failed sessions; pins now in run_difix.py
  docstring):** nvidia/difix's model_index.json names `DifixPipeline`, which
  exists ONLY in github.com/nv-tlabs/Difix3D -- vendored to
  `src/enhancer/pipeline_difix.py`. trust_remote_code still required (the
  VAE is remote code). Difix3D's exact pins are load-bearing:
  diffusers==0.25.1 transformers==4.38.0 peft==0.9.0 huggingface-hub==0.25.1
  (diffusers>=0.31 removed FromOriginalVAEMixin the remote VAE needs;
  unpinned transformers>=4.56 removed FLAX_WEIGHTS_NAME diffusers<0.33
  imports). Validated locally end-to-end on CPU fp32 before re-renting; the
  local 1660 Ti outputs black frames in fp16 (GTX 16xx half-precision
  defect) -- difix never runs locally on GPU; NaN/black guards + --dtype
  fp32 added to run_difix.py.
- **Result: big LOSS on both pilots -- worst single-config result so far.**
  hcm0034 0.6548 -> 0.6209 (-0.0339); HCM0181 0.6338 -> 0.5964 (-0.0374).
  All three metrics worse: LPIPS +0.027/+0.031, SSIM -0.057/-0.065, PSNR
  -0.97/-0.92 dB. QA crops show the predicted failure mode exactly: output
  looks *cleaner* to the eye (speckle gone, edges crisp) but the texture is
  redrawn plausibly-wrong, so even vgg-LPIPS penalizes it. Interpolation-
  regime renders are already too artifact-light for a generic fixer to help.
- **Decision:** **drop exp015; SKIP exp016 (Difix LoRA week) entirely** --
  the <=0-on-both gate fired with margin ~10x the threshold. Freed P3 time
  reallocates to Tier-A (exp009/exp010). Revisit only if a coverage
  diagnostic ever flags an extrapolative private scene (none of the 13 is).
  P3 standing duty: confirm the nvidia/difix license checkbox in
  `docs/rules_and_constraints.md` at read-out.

## exp006 session-2 attempt #1: LOST AT THE WALL (Week 3, Jul 9-10, Kaggle)

- **Ran:** big x {60k, 100k} x both pilots as 4 cells in ONE T4x2 session.
  Measured train times: big_60k 4.0 h, big_100k 8.0 h -> the second scene's
  100k arm started at hour 8.3 and the 12 h wall killed the session before
  the packaging cell: **CSV, renders, and checkpoints all lost.**
- **Log-only salvage:** hcm0034 big_60k = big_100k = 0.7286. Two caveats:
  (1) alex+40 scale -- the session ran the dataset's stale pre-calibration
  code (vs big_30k alex 0.7274, so ~+0.0012); (2) an IDENTICAL 4-dp score
  from two different trainings is a saturation hint but also consistent
  with a scoring quirk -- treat as unverified.
- **Process lessons (cost: ~24 GPU-h):** (a) one scene per T4x2 session --
  60k+100k in parallel is ~8.5 h wall, both scenes serialize past the wall;
  (b) interrupt the sweep by hour 11 and run packaging on partials rather
  than trust the wall; (c) in-notebook scoring must be pinned to vgg+50 --
  self-heal the stale dataset code, don't re-score by memory.
- **Relaunch:** `kaggle/kaggle-exp006-session2-iters.ipynb` (per-scene,
  ckpt-keeping, metrics self-heal, partial-safe packaging), 2 accounts.

## exp014_camera_opt (Week 3, Jul 9, local)

- **Hypothesis (guarded):** SO3xR3 camera-optimizer refines residual pose
  error in the provided COLMAP, sharpening reconstruction -- guarded because
  refining TRAIN poses can drift the world frame away from the FIXED test
  poses, silently misaligning every graded render. Ships only if public-GT
  Score improves, else off permanently (plan_overall_v3 A.9).
- **Ran:** hcm0034 only (pilot), splatfacto + antialiased 30k +
  `--pipeline.model.camera-optimizer.mode SO3xR3`, scored vgg+50 vs real
  test GT.
- **Result: guard FIRED.** camopt_so3xr3 = 0.6349 vs control 0.6548
  (Delta -0.0199) -- PSNR -0.73 dB, SSIM -0.0417, LPIPS +0.0077, all three
  metrics worse, not just a PSNR/LPIPS tradeoff. Visual check (GT vs
  control vs camopt renders at the same pose) confirms the predicted
  failure mode exactly: the camopt render is visibly softer and desaturated
  relative to BOTH GT and the control -- classic symptom of the
  reconstruction's world frame drifting away from the fixed test-pose frame
  during pose optimization.
- **Decision:** **camera-optimizer OFF PERMANENTLY** per the pre-registered
  guard. DJI RTK poses + dense COLMAP init were already good; no further
  pose-refinement work planned this competition. exp014 closed.

## exp009 perceptual fine-tune: paused, suspected LR-schedule bug (Week 3, Jul 9, Kaggle)

- **Ran:** hcm0034 lpips_w005 + lpips_w010 (splatfacto-perceptual, +7k iters from the
  antialiased-30k base, LPIPS(vgg) loss weight 0.05/0.10 starting at step 30000).
  Each arm took ~5.9-6.0h (vs the notebook's ~30-45min estimate) -- LPIPS-VGG
  backward passes are genuinely expensive per step, this part is not itself
  suspicious. Session was interrupted before HCM0181 ran, for time-budget/
  session-wall reasons (see process note below); only the CSV was manually
  saved from the Kaggle output, not the checkpoints or renders.
- **Result: both arms scored 0.6548 -- byte-identical to 4 decimal places on
  ALL FOUR metric columns (psnr 21.4281, ssim 0.7397, lpips 0.2393, score
  0.6548) to each other AND to the untouched antialiased-30k control.** Two
  independently-trained arms with different loss weights (0.05 vs 0.10)
  producing an exact match this precise is not a plausible coincidence --
  it means the fine-tune had no measurable effect.
- **Leading hypothesis:** `run_sweep.py` passes `--max-num-iterations 37000`
  correctly on `--load-dir` resume, so training should run 7000 real steps
  past the loaded 29999-step checkpoint -- and the elapsed wall-time (~6h)
  confirms SOMETHING computationally expensive did happen. But nerfstudio's
  exponential LR schedule is recalculated against the NEW 37000-step horizon;
  by step 30000 (81% of that schedule) the learning rate may already be
  near its floor, making the extra 7k steps close to a no-op regardless of
  the added LPIPS term. Not yet confirmed (would need to inspect the actual
  LR value / loss curve from the training log, or diff fine-tuned vs base
  checkpoint weights -- neither is available since only the CSV survived).
- **Decision: PAUSED, not relaunched blind.** Excluded exp009 from tonight's
  (Jul 9 evening) exp006-s2 relaunch notebooks -- re-running the same recipe
  on HCM0181 would risk repeating ~6h of GPU-time for the same non-result.
  exp013 (independent recipe, no shared code path) still launches tonight
  on its own notebook. Before any exp009 relaunch: add instrumentation
  (log effective LR + LPIPS loss value every N steps) to confirm or rule
  out the schedule hypothesis, and/or use a fresh short LR schedule sized
  to the 7k fine-tune phase instead of a recalculated 37k-step one.

## exp006 session-2 relaunch + exp004_hcm0204_fill + exp013: launched (Week 3, Jul 9 evening, Kaggle)

- Two notebooks built to each fill a full ~12h T4x2 session across the
  team's two accounts, following the process lessons from session-2
  attempt #1's wall crash (one scene per session; package Stage 1 before
  attempting anything else; wall-time-gate any second stage).
- **Account A — `kaggle-exp006s2-hcm0034-plus-hcm0204.ipynb`:** Stage 1 =
  exp006-s2 hcm0034 (big x60k/100k, ~8.5h, keeps ckpts, packaged
  immediately). Stage 2 (gated: skips if <2.5h budget remains) =
  exp004_hcm0204_fill -- HCM0204 antialiased-30k, closing the one
  public-scene gap the 2026-07-09 weakness deep-dive had to exclude
  (`results/analysis_exp020_weakness/`). New dataset built:
  `kaggle-upload-hcm0204.zip` (~276 MB: fresh code copy + HCM0204 train
  images/dense-COLMAP-init/test GT; verified the dense init is already
  leak-filtered -- 240/240 registered images, matching train count exactly).
  New config: `configs/experiments/exp004_hcm0204_fill.yaml`.
- **Account B — `kaggle-exp006s2-HCM0181-plus-exp013.ipynb`:** Stage 1 =
  exp006-s2 HCM0181 (big x60k/100k, ~8.5h est. -- HCM0181's own timing
  wasn't confirmed by the crashed session, only hcm0034's was). Stage 2
  (same wall-time gate) = exp013 tpw_boost1 pilot (hcm0034, 30k from
  scratch, independent of the paused exp009 recipe). Reuses the existing
  `exp006-pilots` dataset, no new upload.
- Both notebooks reuse the proven env-provisioning cell verbatim, and
  package Stage 1's results in their own cell/zip BEFORE Stage 2 is even
  attempted, so a Stage-2 failure or a mistimed wall can never cost Stage
  1's output the way session-2 attempt #1 lost everything.

## exp006 session-2 relaunch + exp004_hcm0204_fill + exp013: read-out (Week 3, Jul 10)

- **Ran:** both Jul-9-evening notebooks completed successfully on their two
  accounts (see prior log entry for design). All 4 zips downloaded,
  cross-checked against the local master CSV (diffed, not blindly appended
  -- Account A's zip carried a superset of historical rows since its code
  copy started from a CSV snapshot; Account B's was already clean), merged
  (6 new rows), then extracted (checkpoints + 60 renders each, verified
  present for all 6 runs) and the 4 output zips deleted.
- **exp006-s2 iters, real read-out (vgg+50):** hcm0034 big_60k 0.6571
  (Delta +0.0002 vs big_30k 0.6569), big_100k 0.6578 (Delta +0.0009) --
  saturates by 30k. HCM0181 big_60k 0.6385 (Delta +0.0018 vs big_30k
  0.6367), big_100k 0.6399 (**Delta +0.0032** -- crosses the usual +0.003
  bar). This is a genuine cross-scene disagreement, not a repeat of the
  crashed session's identical-score artifact: all 4 values here are
  distinct (21.4514/21.4608/20.3239/20.3373 PSNR etc.), confirming that
  earlier "60k=100k=0.7286" was indeed a scoring/dataset-code fluke from
  the stale-metrics session, not a real saturation signal.
  **Decision: per-scene selection, not one global iters count** -- quiet/
  well-covered scenes (hcm0034-type) don't benefit past 30k-60k; busier/
  harder scenes (HCM0181-type) get a real, if modest, gain from 100k.
  This is consistent with (and gives a second data point for) the
  per-scene-config-selection lever in `plan_overall_v3.md` Tier A item 5.
- **exp013 tpw_boost1: dropped.** hcm0034 Score exactly matches control
  (0.6548 = 0.6548) but -- unlike exp009's suspicious identical-metrics
  result -- the underlying PSNR/SSIM/LPIPS genuinely differ and roughly
  offset (worse PSNR/SSIM, slightly better LPIPS), so this is a real wash,
  not a bug. Delta = 0.0000, below the +0.002 adoption gate. Test-pose-
  proximity sampling doesn't move this scene; no further exp013 work planned.
- **exp004_hcm0204_fill: done, no surprises.** Score 0.6339, PSNR 20.46,
  SSIM 0.70, LPIPS 0.246 -- squarely in range with the other 4 public
  scenes under the locked config. HCM0204 now has a checkpoint + renders
  like the rest; `results/analysis_exp020_weakness/analyze.py` can be
  re-run with all 5 public scenes for a 5th data point on the frame-corner
  and sharpness/PSNR findings.
- **Process note:** both notebooks' Stage-1-then-Stage-2 design with
  immediate Stage-1 packaging worked as intended -- no repeat of the
  session-2 attempt #1 wall crash, both stages completed inside the 12h
  budget on both accounts.
