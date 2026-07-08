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
| exp005 | locked (dense + antialiased) → 8 private scenes | n/a (private) | — | pending fleet finish | 🟡 | HCM0249 ✅, HCM0254 ✅, HCM0276 ✅ local (30k ckpt + 60 renders each). Remaining 5 (HCM1439, HNI0131, HNI0265, HNI0366, HNI0437) moved to rented GPU: `kaggle/kaggle-week2-dense-antialiased-train.ipynb` + `kaggle/kaggle-upload-week2-train.zip` (dataset). Re-integrate outputs into `runs/phase1/exp005_antialiased_dense/`, then `scripts/run_final_private_antialiased.sh` packages. Submit as LB datapoint #3 when done. |
| exp006 | capacity/iters sweep: splatfacto-big; MCMC cap 2M/3M on dense; 30/60/100k iters | — | — | — | ⬜ rented | Pilots hcm0034 + HCM0181. Launch via `kaggle/kaggle-exp006-capacity-sweep.ipynb` + `kaggle/kaggle-upload-exp006-pilots.zip` (runs `run_sweep.py --log-dir`, scores vs public GT in-notebook); session 1 = mcmc3M_30k + big_30k on both pilots, session 2 = winner ×60k/100k. Baselines to beat: hcm0034 0.7242, HCM0181 0.7006. |
| exp007 | bilateral grid A/B | — | — | — | ⬜ local | `--pipeline.model.use-bilateral-grid True` |
| exp008 | transient-mask A/B (person+vehicle out of loss) | — | — | — | ⬜ local | Expect win concentrated on HCM0181. |
| exp009 | perceptual (LPIPS-loss) fine-tune, +5–10k iters, w ∈ {0.05, 0.1} | — | — | — | ⬜ rented | Kill if PSNR −0.3 dB. |
| exp010 | combined Tier-A config + fleet | — | — | — | ⬜ rented | Must beat each component alone. |
| exp011 | post-process + encoding sweep (JPEG quality / unsharp / denoise) | — | — | — | ⬜ CPU | On existing renders, zero training cost. |
| exp012 | checkpoint/seed ensembling | — | — | — | ⬜ local | PSNR-favoring trade, gate per scene. |
| exp013 | test-pose-weighted train sampling | — | — | — | ⬜ local | Stretch. |
| exp014 | camera-optimizer + drift guard | — | — | — | ⬜ local | Stretch. Public-GT Score must improve, else off permanently. |
| exp015 | Difix3D+ off-the-shelf | — | — | — | ⬜ rented | Week 3. |
| exp016 | Difix LoRA + per-scene gate | — | — | — | ⬜ rented | Week 3. Gate ≥0.003 (≥0.001 on under-covered scenes). |
| exp017 | final locked fleet (reproducibility run) | — | — | — | ⬜ rented | Week 4. |
| exp018 | SSS / UBS backend pilot | — | — | — | ⬜ rented | Inter-phase only. Promote on ≥0.005 on both pilots. |
| exp019 | val-split validation: does match-test split rank configs like real test GT? (vs every-Nth, equal n=30) | — | — | n/a (no submission) | 🟡 local | Render-only on existing exp001/002/004 checkpoints, both pilots (11 variants). `scripts/run_exp019_valsplit_validation.py` → `results/exp019_valsplit_{validation,summary}.csv`. **Gates the A.6b private-scene split re-cut**: matched ranks ≥ nth (Spearman + A/B sign agreement) → adopt matched splits; worse → keep every-Nth. Caveat: val views seen in training → validates pose-distribution component only. |

## Table 2 — Submissions (leaderboard results)

| # | date | zip / config | exp | local public mean | leaderboard (÷100) | offset (LB − local public) | notes |
|---|---|---|---|---|---|---|---|
| 1 | 2026-07-07 | `exp001_baseline_splatfacto_results/submission_round1.zip` (sparse init, vanilla splatfacto) | exp001 | ~0.7036 | 56.81480 (0.56815) | −0.1355 | First datapoint. All uploads must be named `submission_round1.zip` per the rules — identify submissions by exp folder path, never by zip filename. |
| 2 | 2026-07-07 | `exp002_dense_colmap_init_results/submission_round1.zip` (dense-COLMAP init) | exp002 | 0.70732 | 57.19840 (0.57198) | −0.1353 | Δ transfer check #1: local +0.0037 vs LB +0.00384 ✅ |

## Calibration state (update after every submission)

- **Delta transfer:** 1/1 checks passed — sign ✅, magnitude ✅ (within 4%). Local public A/Bs are currently a reliable proxy for leaderboard deltas.
- **Offset local-public → LB-private:** −0.1355, −0.1353 → **stable at ≈ −0.135**. Bundles private-set difficulty + any metric-constant mismatch; stability matters more than its value. If a future submission moves the offset by >0.01 without a config change that explains it, investigate before trusting further local decisions.
- **`psnr_max` regression:** 2 datapoints, need ≥3 with *varied PSNR* to fit. Keep logging (local PSNR, LB score) pairs here:
  - exp001: PSNR ≈ 21.0 (public mean, derived) → 56.81480
  - exp002: PSNR = 21.16 (public mean) → 57.19840
- **LPIPS backbone / SSIM window:** unconfirmed; local uses alex + skimage win 11. No contradiction observed yet.

## Submission budget notes

- 5/day, 600 s cooldown, system keeps **last** pre-deadline upload → after any probe, re-submit the best-known config the same day.
- Used so far: 2 (Jul 7). Planned next: exp005 fleet (datapoint #3), exp010 Tier-A combo (#4), gated enhancer (#5), final (#6, Jul 28).
