# PROGRESS — single source of truth for experiment ↔ leaderboard sync

**Purpose:** one place where anyone (or any Claude session) can see what ran, what it scored locally, what it scored on the official leaderboard, and what's queued. Update rules:

1. **Every finished experiment** gets/updates a row in Table 1 the same day (owner = whoever ran it).
2. **Every submission** gets a row in Table 2 *the moment the leaderboard score appears* (owner: P2). Also backfill `submissions/phase1/SUBMISSION_LOG.md`.
3. After each submission, update the **calibration section** (offset + delta-transfer check).
4. Leaderboard scale is ×100 of the local Score formula (e.g. 57.19840 ≙ 0.57198). Store leaderboard values verbatim; derived /100 values in parentheses.
5. Statuses: `✅ done` · `🟢 adopted` · `🔴 dropped` · `🟡 running/partial` · `⬜ queued` · `⏸ deferred`.

---

## Table 1 — Experiments (local results)

Local public = mean Score over the 5 public scenes vs real test GT. Local val = mean over private-scene val splits (once cut).

| exp | config (delta vs previous) | local public | local val | leaderboard | status | decision / notes |
|---|---|---|---|---|---|---|
| exp001 | sparse init, vanilla splatfacto 30k | ~0.7036 (derived: dense mean − 0.0037; per-scene sparse only logged for hcm0034 = 0.7176) | — | **56.81480** (0.56815) | ✅ | Baseline control. Pose-bug fix included. |
| exp002 | + dense-COLMAP init | 0.70732 (per-scene: hcm0031 0.7090, hcm0034 0.7226, HCM0181 0.7004, HCM0193 0.7069, HCM0204 0.6977) | — | **57.19840** (0.57198) | 🟢 | Won all 5 public scenes. **Local Δ +0.0037 ↔ LB Δ +0.00384 — transfers.** |
| exp003 | VGGT pseudo-cloud init | lost on all 5 (−0.009 to −0.022) | — | — | 🔴 | Dropped. chunk_size=4 alignment instability noted. |
| exp004 | backend variants on dense init (4 public scenes) | antialiased +0.0016 mean (won 4/4); scale_reg negative 4/4; sky_mask ±0.0000; mcmc −0.003 (hcm0034, sparse-count regime only) | — | — | ✅ | Adopt antialiased; drop scale_reg, sky_mask. MCMC ⏸ deferred to exp006 (6 GB local card can't fit real cap_max). HCM0204 variant rows intentionally skipped. |
| exp005 | locked (dense + antialiased) → 8 private scenes | n/a (private) | — | **57.43380** (clean) / 56.93230 (pp) | 🟢 | **Submitted & scored Jul 9 — clean fleet is the new best (LB #3, +0.00235 over exp002 ✅ transfer check #2).** The pp variant (LB #4) scored WORSE (−0.00502) → exp011 refuted, see its row + calibration. **Re-submit the clean zip same day** (system keeps last upload). Fleet: 3 local + 5 rented scenes, verified 8/8. |
| exp006 | capacity/iters sweep: splatfacto-big; MCMC cap 2M/3M on dense; 30/60/100k iters | session 1: big_30k **0.7274 / 0.7050** (+0.0032/+0.0044); mcmc3M_30k 0.7207/0.6999 (−0.0035/−0.0007) | — | — | 🟡 rented | **Session-1 read-out (Jul 9): adopt splatfacto-big (won both pilots, mean +0.0038); MCMC dead on dense init too — drop.** Session 2 = big × 60k/100k on both pilots. **Jul 12 rent gate: condition met — lock in the paid 4090 workflow** (big needs >6 GB; fleet can't run on local). NOTE: output zip had renders/configs/metrics only, NO checkpoints — exp009's fine-tune load-dir must come from a session that keeps ckpts (bundle into session 2). |
| exp007 | bilateral grid A/B | — | — | — | ⬜ local | `--pipeline.model.use-bilateral-grid True` |
| exp008 | transient-mask A/B (person+vehicle out of loss) | — | — | — | ⬜ local | Expect win concentrated on HCM0181. **Masks built + QA'd for both pilots (Jul 8)**: coverage sane (median 0.25% px), catches bikes/pedestrians, slightly over-eager on static red objects (mast beacon, parasols). Staging symlink NOT made — link at training time. |
| exp009 | perceptual (LPIPS-loss) fine-tune, +5–10k iters, w ∈ {0.05, 0.1} | — | — | — | ⬜ rented | Kill if PSNR −0.3 dB. |
| exp010 | combined Tier-A config + fleet | — | — | — | ⬜ rented | Must beat each component alone. |
| exp011 | post-process + encoding sweep (JPEG quality / unsharp / denoise) | +0.0039 alex-local (all 5 public) | — | **−0.00502 — REFUTED** | 🔴 | **Adoption REVERTED (Jul 9, LB datapoint #4): the local win was an alex-LPIPS artifact** — sharpening games alex (−0.024) but not the LB's (VGG-consistent) LPIPS (−0.0017), while its PSNR/SSIM costs transfer. Lesson recorded in calibration. Encoder axis (jpeg≥90) genuinely neutral — keep jpeg98 encoding, drop the op. Optional revisit: rerun grid scored with vgg; adopt only a vgg-positive op. |
| exp012 | render ensembling (pilot: cross-config mean — runs keep only the final ckpt) | −0.0007 / +0.0001 vs best single | — | — | 🔴 | Dropped (< +0.002 kill threshold both pilots): averaging inflates LPIPS (~0.13→0.15) and the 0.4-weighted LPIPS term eats the SSIM gain. `runs/phase1/exp012_ensemble_pilot/`. Checkpoint-ensembling proper untested — revisit only if a sweep saves multiple ckpts for free. |
| exp013 | test-pose-weighted train sampling | — | — | — | ⬜ local | Stretch. |
| exp014 | camera-optimizer + drift guard | — | — | — | ⬜ local | Stretch. Public-GT Score must improve, else off permanently. |
| exp015 | Difix3D+ off-the-shelf | — | — | — | ⬜ rented | Week 3. |
| exp016 | Difix LoRA + per-scene gate | — | — | — | ⬜ rented | Week 3. Gate ≥0.003 (≥0.001 on under-covered scenes). |
| exp017 | final locked fleet (reproducibility run) | — | — | — | ⬜ rented | Week 4. |
| exp018 | SSS / UBS backend pilot | — | — | — | ⬜ rented | Inter-phase only. Promote on ≥0.005 on both pilots. |
| exp019 | val-split validation: does match-test split rank configs like real test GT? (vs every-Nth, equal n=30) | see `results/exp019_valsplit_summary.csv` | — | n/a (no submission) | 🟢 | **Gate PASSED (Jul 8) → adopt match-test splits for private-scene selection (A.6b unblocked).** Match-test ranked all 11 variants perfectly on both pilots (Spearman 1.0, A/B agreement 14/14 + 6/6); every-Nth mis-ranked one hcm0034 pair (scale_reg vs mcmc, Spearman 0.943, 13/14). Render-only on existing exp001/002/004 checkpoints. Caveat: val views were seen in training → validates the pose-distribution component of the signal only, not holdout generalization. `scripts/run_exp019_valsplit_validation.py` → `results/exp019_valsplit_{validation,summary}.csv`. |

## Table 2 — Submissions (leaderboard results)

| # | date | zip / config | exp | local public mean | leaderboard (÷100) | offset (LB − local public) | notes |
|---|---|---|---|---|---|---|---|
| 1 | 2026-07-07 | `exp001_baseline_splatfacto_results/submission_round1.zip` (sparse init, vanilla splatfacto) | exp001 | ~0.7036 | 56.81480 (0.56815) | −0.1355 | First datapoint. All uploads must be named `submission_round1.zip` per the rules — identify submissions by exp folder path, never by zip filename. |
| 2 | 2026-07-07 | `exp002_dense_colmap_init_results/submission_round1.zip` (dense-COLMAP init) | exp002 | 0.70732 | 57.19840 (0.57198) | −0.1353 | Δ transfer check #1: local +0.0037 vs LB +0.00384 ✅ |
| 3 | 2026-07-09 | `exp005_antialiased_dense_results/submission_round1.zip` (+ antialiased, full private fleet) | exp005 | 0.7089 (exp004 proxy, old psnr_max 40) | **57.43380** (0.57434) — PSNR 19.4269, SSIM 0.5514, LPIPS 0.2691 | ≈ −0.135 (proxy) | Δ transfer check #2: local +0.0016 vs LB Δ +0.00235 ✅ sign, magnitude ~1.5×. **First per-metric breakdown → psnr_max SOLVED = 50 (see calibration).** |
| 4 | 2026-07-09 | `exp005_antialiased_dense_pp_results/submission_round1.zip` (#3 + exp011 unsharp_r1_p50+jpeg98) | exp005+pp | predicted +0.0039..0.0046 | **56.93230** (0.56932) — PSNR 18.9973, SSIM 0.5410, LPIPS 0.2674 | — | **Δ transfer check #3: FAILED, sign flipped (LB Δ −0.00502).** Root cause: local gain was pure alex-LPIPS (−0.024); LB LPIPS moved only −0.0017 (VGG-consistent) while the op's PSNR/SSIM costs (−0.43 dB / −0.0105) transferred almost exactly. **exp011 adoption REVERTED.** ⚠️ System now holds this inferior zip — re-submit #3's zip same day (P2). |

## Calibration state (update after every submission)

- **`psnr_max` = 50.0 — SOLVED (2026-07-09).** The leaderboard now shows per-metric breakdowns; inverting Score = 0.4(1−LPIPS)+0.3·SSIM+0.3·psnr/psnr_max on submissions #3 and #4 gives psnr_max = 50.0001 and 50.0000. All local defaults updated 40→50 (`metrics.py`, `render_val.py`, `sweep_postprocess.py`). Per-dB weight is 0.006, not 0.0075 → PSNR matters ~20% less than assumed; LPIPS/SSIM gain relative weight.
- **LPIPS backbone: VGG — CONFIRMED (2026-07-09).** LB LPIPS ≈ 0.27 vs local alex ≈ 0.13; decisive probe: re-scoring the exp011 op on hcm0034 with vgg+psnr_max=50 gives delta **−0.0050 vs the LB's −0.00502 — an exact match** (under vgg the op's LPIPS "gain" vanishes: 0.2393→0.2396). Local vgg scoring predicts LB deltas almost perfectly. **All decision scoring now uses `--lpips-net vgg`.** Alex-scored decisions to sanity-check under vgg: exp011 (already refuted by LB), others were won on all-three-metrics so safe.
- **Delta transfer:** 2/3 checks passed. #1 init ✅ (+0.0037→+0.00384), #2 antialiased ✅ sign (+0.0016→+0.00235), #3 postproc ❌ **sign flip** (+0.004→−0.005) — root cause = alex/vgg backbone mismatch, not a general transfer failure. Rule going forward: distrust any local win that lives mostly in the alex-LPIPS column.
- **Offset local-public → LB-private:** −0.1355, −0.1353, ≈−0.135 (proxy) — stable. Will shift when local rescoring moves to psnr_max=50 + vgg; re-baseline then.
- **SSIM window:** still unconfirmed (local skimage win 11); LB SSIM 0.55 on private plausible for harder scenes, no contradiction signal available yet.

## Submission budget notes

- 5/day, 600 s cooldown, system keeps **last** pre-deadline upload → after any probe, re-submit the best-known config the same day.
- Used so far: 2 (Jul 7). Planned next: exp005 fleet (datapoint #3), exp010 Tier-A combo (#4), gated enhancer (#5), final (#6, Jul 28).
