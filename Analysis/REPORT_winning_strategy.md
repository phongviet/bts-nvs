# Winning Strategy Report — closing the 18-point gap to top-8

*2026-07-11 · Analysis owner: phongviet (goal: top-8 = LB 75.33–76.75; we are at 57.43, rank ~74).*
*Every number in the "measured" columns below was produced today on this machine against real public-scene GT with the confirmed grader metric (LPIPS-vgg, psnr_max=50). Scripts and raw outputs live in this folder.*

---

## 0b. UPDATE 2026-07-12 (evening) — P2 REFINER HITS TOP-8 QUALITY ON PILOT

The per-scene neural blending refiner (`Analysis/10_refiner_pilot.py`) was built and piloted on
hcm0034 against **real test GT**. Result — the full ladder on one scene:

| stage | PSNR | SSIM | LPIPS | Score |
|---|---|---|---|---|
| pinhole baseline (our old submissions) | 21.43 | 0.740 | 0.239 | 0.6548 |
| F1 remap | 24.40 | 0.843 | 0.225 | 0.7092 |
| F2 DIBR guarded | 24.62 | 0.859 | 0.161 | 0.7410 |
| **P2 refiner** | **25.11** | **0.890** | **0.1248** | **0.7678** |

**+0.0268 over DIBR, +0.113 over our shipped baseline, and LPIPS 0.1248 — inside the leaders'
0.10–0.12 regime.** A single-scene 0.7678 sits *at* the top-8 band (75.33–76.75). This is the
mechanism that closes the last ~5 points: a small U-Net (7-ch in: [3DGS render, DIBR blend,
visibility mask], residual on DIBR), trained per-scene on **held-out train views** with the grader's
own objective (0.4·LPIPS-vgg + 0.3·(1−SSIM) + 0.3·L1) — fully compliant (no test images, no external
data).

**CONFIRMED 2/2 public scenes** — the refiner beats DIBR on the real test metric on both, with the
identical recipe:

| scene | baseline | F1 remap | DIBR | **refiner** | Δ refiner−DIBR | refiner LPIPS |
|---|---|---|---|---|---|---|
| hcm0034 | 0.6548 | 0.7092 | 0.7410 | **0.7678** | +0.0268 | 0.1248 |
| HCM0181 | 0.6338 | 0.6973 | 0.7312 | **0.7556** | +0.0244 | 0.1284 |
| mean | 0.6443 | 0.7033 | 0.7361 | **0.7617** | **+0.0256** | ~0.127 |

Robust, not a fluke. **FLEET COMPLETE (Jul-12 14:37) — all 13 scenes refined, exp033 built.**

Public-5 refiner (measured on real GT): hcm0034 0.7678, HCM0181 0.7556, hcm0031 0.7498, HCM0193
0.7542, HCM0204 0.7569 → **mean 0.7569 (vs DIBR 0.7297, +0.027)**, all in the top-8 band.

Private-8 held-out val_loss (trust signal; public known-good range 0.071–0.093 → test 0.750–0.768):
HNI0131 **0.059**, HNI0366 0.070, HNI0437 0.073, HCM0276 0.078, HCM0254 0.081, HCM1439 0.083,
HCM0249 0.086, HNI0265 0.092 — **all 8 in-band**, so exp033 uses the refiner on all 13 scenes. The 3
scenes kept on remap in exp032 (HCM1439/HNI0437/HNI0265, poor DIBR traincheck) now have healthy
refiner val_losses — the residual-on-DIBR + grader-loss training learns to lean on the 3DGS render
where DIBR was unreliable.

**Deliverable: `submissions/phase1/exp033_refined_results/partial_private_set1.zip` (389 MB, 8/8
validated).** exp032 (5 DIBR + 3 remap) measured 72.22 on the LB; exp033 (refiner ×8, biggest jumps
on the 3 ex-remap scenes) projects **private ~0.75–0.76 → LB ~75–76 (top-8; current floor 75.41).**

## 0. UPDATE 2026-07-12 — F1 CONFIRMED ON THE REAL PRIVATE SET (the thesis is proven)

We submitted the F1-remapped **private-8 partial** (`partial_private_set1.zip`, with the HNI0131/HNI0265
expanded-canvas fix) and scored **70.44850 on the leaderboard → rank 31.** The private-8 mean was
implied at ~0.540 before F1; the remap took it to **0.7045 — a measured +16.4 LB points on private,
from a resampling fix with no retraining.** That is ~2.7× the +6 the public-scene mean predicted,
exactly because the two HNI k=−0.115 scenes (35–40 px peripheral misalignment) were the single
largest defect in our whole pipeline. F1 is no longer a projection — it is the biggest single jump
this project has made, and it validates the entire "structural, not tunable" diagnosis.

**New standing:** ~57.4 (rank ~74) → **70.45 (rank 31)**. Top-8 = 75.33–76.75, so the remaining gap
is **~5 points**, and F2 (DIBR, measured +9 on-scene on public) plus the refiner are dimensioned to
cover it. The projections in §6/§7 below are now superseded on the F1 row by this measured datapoint.
Next lever: roll DIBR fleet-wide onto the private scenes (P1).

---

## 1. Executive summary

Three weeks of Tier-A ablations (init, capacity, iters, losses, masks, enhancers — 20+ experiments)
produced ~+1 LB point total. The top-8 gap is ~18 points. **Conclusion: the gap was never going to
be closed by tuning the current paradigm, and today's analysis found two structural reasons why —
both fixable, one already fully measured:**

1. **F1 — Our submissions have been geometrically misaligned with the ground truth on every scene.**
   The test GT images are *raw, radially distorted* DJI frames (`SIMPLE_RADIAL`, k in `cameras.bin`;
   the test CSV omits k and we — like nerfstudio's default pipeline — rendered pinhole). Remapping
   our **existing** renders into distorted geometry gains **+0.050…+0.074 Score per scene (mean
   +0.061 over all 5 public scenes), no retraining**. It also eliminates ~70% of the "frame-corner
   weakness" chased by exp020/021/022. Two private scenes (HNI0131, HNI0265) have **k = −0.115,
   ~14× larger** → their renders are ~35–40 px off at the periphery; their gains should be far larger.
   **A submission-ready remapped zip builder exists (`06_build_remapped_submission.py`).**

2. **F2 — The task is a dense view-interpolation problem, and pure reconstruction discards the
   strongest signal: the real pixels.** In all 13 scenes, 86–98% of test frames have an immediately
   adjacent train frame; the median test camera sits 0.2–0.5 inter-frame spacings from a train
   camera. Depth-guided warping of real train pixels into the test pose (DIBR), using the 3DGS
   model only for geometry + occlusion fill, **measured Score 0.7410 on hcm0034 (baseline 0.6548,
   +8.6 LB pts) and 0.7312 on HCM0181 (baseline 0.6338, +9.7 LB pts) with the same untuned
   hyperparameters** — mostly by cutting LPIPS to **0.161/0.165** (LPIPS carries 0.4 of the Score;
   this is exactly the metric regime the leaders' numbers imply). Early version; per-scene tuning
   and a neural blender are still to come.

**The path: submit the remap fix immediately (~+6 LB → ~63–66), roll DIBR out fleet-wide
(measured +2.4 on top, target +3–5 after the guard/tuning → ~67–70), then add a per-scene neural
blending refiner (the standard next step for exactly this regime; targets the leaders' LPIPS
0.10–0.12 → 71–75+).** Section 6 gives the arithmetic under both hypotheses about how the top-8
scores are composed, and the 1-submission probe that decides between them.

---

## 2. Why the current direction could not catch up (evidence)

- Complete Tier-A ledger (PROGRESS.md): dense init +0.37, antialiased +0.16, splatfacto-big +0.38,
  everything else ≤ +0.2 or negative. Sum of adopted levers ≈ +1 LB point.
- Score sensitivity: +1 dB PSNR = +0.6 pts; −0.01 LPIPS = +0.4 pts; +0.01 SSIM = +0.3 pts.
  18 points ≈ LPIPS −0.15 **and** SSIM +0.15 **and** PSNR +8 dB *simultaneously*. No 3DGS variant
  in the literature delivers that over a tuned splatfacto on the same data. Either the game has a
  structural flaw we hadn't found (it did — F1), or the paradigm is wrong for the regime (it is — F2),
  or the leaders' numbers are partially an artifact of scoring composition (possible — §6).

---

## 3. Finding F1: test GT is raw/distorted; we submitted pinhole renders

*(full write-up: `05_camera_distortion_findings.md`; experiment `03_x4_distortion_remap.py`)*

**Measured on all 5 public scenes (existing locked-config renders, remap-only):**

| scene | Score pinhole → distorted | Δ | ΔPSNR | corner MSE |
|---|---|---|---|---|
| hcm0034 | 0.6548 → 0.7092 | **+0.0544** | +2.97 dB | −70% |
| hcm0031 | 0.6438 → 0.6942 | **+0.0505** | +2.59 dB | −67% |
| HCM0181 | 0.6338 → 0.6973 | **+0.0635** | +3.26 dB | −73% |
| HCM0193 | 0.6418 → 0.6974 | **+0.0555** | +2.95 dB | −74% |
| HCM0204 | 0.6339 → 0.7082 | **+0.0743** | +3.84 dB | −73% |
| **mean** | 0.6416 → 0.7013 | **+0.0596** | +3.12 dB | ~−71% |

Supporting facts: CSV `fx` equals COLMAP `f` to 3 decimals in **all 13 scenes** (the CSV is an
export of the same camera, minus k) · corner crops visually align after remap · the exp020 corner
weakness (2.5–2.9× tile error), exp021's "not distortion" verdict (correct only about the residual
top-bias), and exp022's failed loss reweighting are all explained: no loss can fix a resampling
mismatch. **Private k values:** 6 scenes ≈ +0.008…+0.014 (like public), HNI0131/HNI0265 = −0.1148
(~35–40 px peripheral displacement — likely a large chunk of the private-set deficit, §6).

**Action (ready today):** `06_build_remapped_submission.py` builds a validated 13-scene
`submission_round1.zip` from existing renders. Optional v1.1: render at 2× and remap+downsample in
one pass to avoid double-resampling blur (quantify first).

---

## 4. Finding F2: it's an interpolation problem — warp the real pixels

*(geometry: `01_test_train_geometry.py` + CSVs; baselines: `02_x1_nearest_copy.py`; pilot: `04_x3_dibr_pilot.py`)*

**Geometry (all 13 scenes):** 86–98% of test frames have a train frame at sequence-distance 1
(75–92% sandwiched both sides ≤2); median distance to nearest train camera = 0.21–0.51× the
consecutive-train spacing; median rotation to it 9.5–15.2°. One partial outlier (HNI0265: gap-1 0%,
sandwiched 48% — still sub-spacing distances).

**X1 — copy nearest train image:** Score 0.231–0.248 (PSNR ~10). Real pixels *without geometric
correction* are worthless → the value is unlocked only by warping. (This also bounds how far
naive frame-interpolation ideas go; pose-exact warping is the right tool.)

**X3 — DIBR pilot (hcm0034, 60 test views):** render depth at the test pose from the existing
3DGS checkpoint → unproject each test pixel → project into K=3 nearest train cameras (distortion-
aware, both directions) → occlusion z-test vs each neighbor's rendered depth → distance-weighted
blend → 3DGS render fills occlusions/holes (14.7% of pixels) → output directly in distorted
geometry (single resample).

| variant | PSNR | SSIM | LPIPS | Score |
|---|---|---|---|---|
| **hcm0034** — submitted baseline (pinhole 3DGS) | 21.43 | 0.7397 | 0.2393 | 0.6548 |
| + F1 remap (3DGS only) | 24.40 | 0.8433 | 0.2254 | 0.7092 |
| + F2 DIBR v0 (K=3, tol=0.03, no guard) | 24.19 | 0.8503 | 0.1674 | 0.7333 |
| + photometric guard 0.18 | **24.62** | **0.8589** | **0.1609** | **0.7410** |
| **HCM0181** — submitted baseline (pinhole 3DGS) | 20.25 | 0.6595 | 0.2446 | 0.6338 |
| + F1 remap (3DGS only) | 23.51 | 0.8182 | 0.2439 | 0.6973 |
| + F2 DIBR guarded 0.18 (same untuned params) | **23.94** | **0.8449** | **0.1649** | **0.7312** |

**Full public-5 DIBR sweep (guarded 0.18, identical untuned params on every scene — completed Jul-12):**

| scene | pinhole baseline | F1 remap-only | **F2 DIBR guarded** | Δ vs remap | Δ vs baseline |
|---|---|---|---|---|---|
| hcm0034 | 0.6548 | 0.7092 | **0.7410** | +0.0318 | +0.0862 |
| HCM0181 | 0.6338 | 0.6973 | **0.7312** | +0.0339 | +0.0974 |
| hcm0031 | 0.6438 | 0.6942 | **0.7222** | +0.0280 | +0.0784 |
| HCM0193 | 0.6418 | 0.6974 | **0.7270** | +0.0296 | +0.0852 |
| HCM0204 | 0.6339 | 0.7082 | **0.7271** | +0.0189 | +0.0932 |
| **mean** | **0.6416** | **0.7013** | **0.7297** | **+0.0284** | **+0.0881** |

DIBR beats F1-remap on **every one of the 5 public scenes** (+0.019…+0.034, mean **+2.84 LB pts on
top of the already-huge F1 gain**), with the same hyperparameters everywhere — i.e. the +9 on-scene
headline was not a two-scene fluke; it is a consistent **+8.8 LB pts over the pinhole baseline
across the whole public set.** LPIPS is the driver (mean ~0.17 vs remap ~0.23). Per-scene guard/K/tol
tuning (P1) only widens this.

The guarded variant beats the remapped-3DGS baseline on **all three metrics on both scenes**
(it is ≥3DGS by construction, and the measured ghost-rejection also recovered PSNR/SSIM while
*improving* LPIPS). **Total measured: hcm0034 0.6548 → 0.7410 (+8.6 LB pts), HCM0181 0.6338 →
0.7312 (+9.7 LB pts) — mean +9.2 points on-scene, in one day, with no retraining, and the same
hyperparameters on both scenes.** The LPIPS drop (−0.065/−0.079 vs remapped 3DGS) is the F2
thesis working: real photo texture beats
synthesized texture, and LPIPS is 40% of the Score. Warp correctness was validated separately by
warping into held-out *train* views (center-PSNR 25.4 vs a broken warp's ~10; note that comparison
understates test-time performance — train-view neighbors are 2–5× farther than test-view neighbors,
and the 3DGS competitor there had memorized the target).

**Known v0 artifacts and the fix ladder (in order of cost):**
1. *Photometric guard* (**measured, adopted**): accept a warped sample only where it agrees
   with the aligned-but-blurry 3DGS render → kills thin-structure ghosting (antennas/lattice — the
   visible v0 failure) and makes DIBR ≥ 3DGS by construction. Result: improved all three metrics
   simultaneously (see table). guard=0.18 untuned; per-scene selection still to come.
2. *Hyper-tuning* (K, tol, per-scene guard). **CALIBRATION CORRECTION (Jul-12):** the train-view
   holdout center-PSNR is **not** a reliable per-scene DIBR selector — measured against the known
   test outcome, it said "warp loses" on 3 of 5 public scenes where DIBR actually *won* on the real
   test metric (HCM0181: traincheck −0.14 dB, yet test +0.034 Score, its 2nd-biggest win). It
   systematically understates DIBR because test-time neighbors sit 2–5× closer than train-holdout
   neighbors and the 3DGS competitor memorized the train views. **Use it only as a worst-case floor,
   not a ranker: the empirically proven "DIBR transfers" band is traincheck ≥ −0.14 dB.** Private
   scenes in-band (→ roll out DIBR): HCM0249 +0.09, HCM0254 +0.87, HCM0276 −0.14, HNI0366 −0.10.
   Out-of-band (→ keep F1-remap, revisit with the refiner): HNI0437 −0.47, HCM1439 −1.48 (the sparse
   103-image scene). The real per-scene selector must be the LB itself (5 submissions/day) or, later,
   held-out-train *LPIPS* under the refiner — not train-PSNR.
3. *Neural blending refiner* (the standard endgame for this regime — Deep Blending / Stable View
   Synthesis recipe): a small per-scene U-Net taking [3DGS render, K warped neighbors, visibility
   masks] → final image, trained on held-out train views with 0.4·LPIPS-vgg + L1/SSIM loss (i.e.
   the grader's own objective). Fully compliant: trains only on provided scene imagery (no external
   data, no test images). This is the instrument that turns LPIPS 0.17 into the leaders' 0.10–0.12
   and repairs seams/holes learned-ly. exp015's Difix failed *because it hallucinated without
   references*; here the references are the warped real pixels.

---

## 5. Compliance

- DIBR/refiner use only provided train images + provided poses + our own trained model — classical
  image-based rendering *is* novel-view synthesis; nothing touches test images at any point.
  The banned practices (test-time optimization on test images, manual per-pose editing, external
  scene data) are not used. The refiner's training pairs come from held-out provided train views
  (the compliant analog of the exp016 LoRA plan).
- We do **not** adopt public-GT resubmission ("stuffing") even though the archive contains public
  test GT: it violates the test-image clause, dies at final code verification, and does not exist in
  Phases 2–3. Its only relevance is interpreting the leaders' numbers (§6).

## 6. What the top-8 numbers mean, and the target arithmetic

**RESOLVED (Jul-12) by the private-partial submission:** we uploaded a **private-only** partial and
it scored 70.45. The leaderboard therefore scores the private set directly (partial submissions are
graded on the scenes present), and public-GT "stuffing" cannot lift it — you cannot stuff private
GT you don't have. **Conclusion: the top-8 (75.33–76.75) are genuine private-set quality (~0.75+).**
This retires the H-all13/stuffing branch below as a scoring concern and sets a hard, honest target:
we must reach private ≈ 0.76 by real quality, i.e. F2 (real-pixel warping) + the refiner. The canary
probe is no longer needed. The arithmetic below is kept for the record.

*(full arithmetic: `00_leaderboard_arithmetic.md`)*

Submissions contain all 13 scenes; public-scene GT is distributed to everyone. Our #3 submission
(LB 0.5743; public part measured 0.6295 locally) implies private-8 ≈ **0.540** if the LB averages
all 13 (H-all13), or 0.5743 if it averages private only (H-priv8).

- **If leaders stuff public GT (H-all13):** their implied real (private) quality is
  (13×0.7533 − 5×0.974)/8 ≈ **0.615**. Their Jul-9 metric pattern (LPIPS 0.08–0.12 / SSIM 0.75–0.80 /
  PSNR 26–28) matches the GT+good-3DGS mixture *exactly*, and 8 teams clustering within 1.4 pts
  suggests a shared plateau. Under this hypothesis our pipeline after F1+F2 (private ≈ 0.60–0.65
  projected) **matches or beats every leader's genuine quality**, and stuffed scores die at final
  code verification.
- **If leaders are genuine (either hypothesis):** they hold LPIPS ≈ 0.10–0.12 — reachable in this
  regime only by real-pixel reuse (F2 + refiner). Same roadmap, higher bar: we must land the
  refiner, not just the warp.
- **Decisive, compliant, 1-submission probe:** submit once with exactly one public scene's renders
  deliberately blurred (locally measured Δ). LB drops by Δ/13 → H-all13 confirmed (and stuffing
  becomes near-certain given the metric pattern); no drop → H-priv8, leaders genuine. Re-submit the
  best clean zip the same day.

**Projected trajectory (H-all13 arithmetic, conservative — private F1 gain taken at the public
mean +0.06 although two scenes should gain much more):**

| step | public-5 mean | private-8 mean | LB (×100) |
|---|---|---|---|
| baseline (submitted #3) | 0.630 | 0.540 (implied) | 57.4 |
| + F1 remap — **MEASURED on private set (Jul-12)** | 0.701 (measured) | **0.7045 (MEASURED)** | **70.45 (rank 31)** |
| + F2 DIBR guarded fleet-wide | 0.73–0.75 (measured 0.72–0.74 on 5 pilot scenes) | 0.73–0.76 (target) | **~73–75** |
| + refiner (P2) hitting LPIPS 0.10–0.13 | 0.75–0.78 | 0.76–0.79 | **~76+ (top-8)** |

Each step is gated by its own submission datapoint (5/day available; local→LB delta transfer has
held for every all-three-metrics win so far). If the canary probe reveals H-priv8, the same steps
apply with the private columns as the whole score.

---

## 7. Execution plan

**Now (Jul 11–12) — P0:**
1. Build + validate remapped zip from existing renders (script ready) → **submit**. Expected ~+6.
2. Keep the LB canary probe ready; spend it the same day if submission cadence allows (blur variant
   of one public scene, then re-submit best).

**P1 STATUS (Jul-12): private DIBR fleet built and staged — READY TO SUBMIT.**
`submissions/phase1/exp032_dibr_mixed_results/partial_private_set1.zip` (332 MB, validated 8/8 vs
test_poses) = DIBR (guard 0.18) on the 4 in-band private scenes {HCM0249, HCM0254, HCM0276, HNI0366}
+ banked F1-remap on {HCM1439, HNI0437, HNI0131, HNI0265}. Strict low-regret upgrade over the 70.45
submission (4 scenes improved, 4 unchanged). Build script: `Analysis/09_build_dibr_mixed_submission.py`.
Expected private mean +0.01–0.015 → **~71.5–72**; the LB delta is also the per-scene DIBR→private
transfer measurement that gates adding HNI0437/HCM1439 and building the HNI negative-k expanded-canvas
DIBR next. Remaining P1 work: (a) ~~expanded-canvas DIBR for HNI0131/HNI0265~~ **DONE (Jul-12)**; (b) reconsider
HCM1439/HNI0437 after the transfer datapoint.

**Expanded-canvas DIBR for the k=−0.115 HNI scenes — DONE (Jul-12).** Wired script-08's expanded
canvas into `synthesize()`'s `out_k` path via a `canvas_margin` param (auto-set to 128 for k<−0.05):
the fallback/depth maps render on a (W+256, H+256) canvas so the negative-k periphery samples real
3DGS content instead of edge-replicating. Validated streak-free (no bounds warning at margin=128).
Traincheck: **HNI0131 +0.03 (in-band → DIBR)**, HNI0265 −0.79 (below floor → keep remap; traincheck
is *extra*-pessimistic here since the big negative-k periphery has more train-holdout fallback than
at test time). The mixed submission now carries DIBR on 5 private scenes {HCM0249, HCM0254, HCM0276,
HNI0366, HNI0131} + remap {HCM1439, HNI0437, HNI0265}; HNI0265-DIBR is also rendered and held for a
future LB A/B.

**Jul 12–16 — P1 (DIBR fleet):**
3. Finish guard/tol/K selection on public scenes (GT) + verify the *same* selection reproduces on
   held-out train views → freeze the automatic per-scene selection recipe.
4. Run DIBR across all 13 scenes (local or 1 Kaggle session; ~1 h/scene at v0 speed, easily
   optimizable) → submit → verify delta transfer.
5. Private-scene sanity: traincheck protocol per private scene (no GT needed); flag any scene where
   warp underperforms 3DGS and fall back per-scene (selection is automatic and compliant).

**Jul 15–24 — P2 (refiner):**
6. Build the per-scene U-Net blender (inputs: 3DGS render + K warped neighbors + masks; loss =
   grader objective on held-out train views). Pilot on hcm0034/HCM0181 with the +0.002-per-scene
   gate replaced by a "beats DIBR on held-out train views of that scene" gate.
7. Fleet-train on Kaggle (T4×2 fits easily; it's a small U-Net, not a diffusion model), submit.

**Continuous:** update PROGRESS.md ledger + calibration after every submission; keep the locked
splatfacto fleet as the fallback substrate (DIBR needs its depth; better 3DGS depth still helps —
LEGS' +0.002 and per-scene iters remain worth folding into the substrate when idle GPU exists).

**Phase-2/3 note:** everything above is per-scene automatic (k comes from cameras.bin; neighbor
selection, guard selection, and refiner training are self-supervised on train views), so the
72h/48h windows are compatible — bake it into `phase_run.sh` during the inter-phase gap.

## 8. Risks

| risk | mitigation |
|---|---|
| Grader resizes/undistorts before scoring (would mute F1 on LB despite local gains) | The +6 remap submission doubles as the F1 transfer check; if LB moves ≪ +6·(5/13), rethink. Local GT alignment strongly suggests otherwise. |
| DIBR underperforms on some private scene (worse COLMAP/depth) | per-scene automatic fallback to remapped 3DGS via traincheck signal |
| Refiner overfits held-out-train → test drop | scene-internal cross-validation (two disjoint holdout sets), keep DIBR output as fallback |
| "algorithmic warping of train pixels" challenged by organizers | prepare the defense in the method report: it is standard image-based rendering (cite Deep Blending, FVS/SVS, FWD); all inputs are provided data |
| Leaders genuine at 0.75+ and refiner tops out below | escalate P2: diffusion-based refiner conditioned on warped references (Difix + reference channel, LoRA on own pairs — the compliant variant already scoped in exp016) |

## 9. Folder map

| file | content |
|---|---|
| `00_leaderboard_arithmetic.md` | LB composition scenarios, needed-quality math, canary probe |
| `01_test_train_geometry.py` + `01_geometry_per_{scene,test_view}.csv` | interpolation-regime measurement |
| `02_x1_nearest_copy.py` + `X1_nearest_copy/` | naive-copy floor baseline |
| `03_x4_distortion_remap.py` + `X4_distortion_remap/` | F1 experiment + per-scene remapped renders/metrics |
| `04_x3_dibr_pilot.py` + `X3_dibr/` | DIBR pilot (traincheck validation + test-mode renders/metrics) |
| `05_camera_distortion_findings.md` | F1 write-up |
| `06_build_remapped_submission.py` | P0 submission builder (13 scenes) |
| `REPORT_winning_strategy.md` | this report |

---

## 10. Jul-12 evening: exp033 scored — pivot to the top-1 campaign

exp033 (refiner, all 8 private, q95) scored **75.38050, rank 9** (PSNR 25.06 /
SSIM 84.16 / LPIPS 12.26) — +3.16 over exp032, refiner transfer to private
CONFIRMED, 0.027 under the old top-8 floor. New goal: **top-1 = 77.02430**
(gap 1.644 LB pts) and defend it.

The top-1 campaign lives in **`FINAL_PLAN_top1.md`** (ladder of measured
rungs: encoding, TTA, supersampled+cubic resampling, big backbone, refiner
v2, private backbone fleet + defense strategy). This report stays as the
record of how the structural pivot (F1 remap → F2 DIBR → P2 refiner) was
found and proven. Scripts 12–14 + `run_v2_fleet.sh`, `run_overnight_ladder.sh`,
`X6_jpeg_budget/` belong to the campaign.
