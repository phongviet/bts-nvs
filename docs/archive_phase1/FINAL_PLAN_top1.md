# FINAL PLAN — Reach and defend LB #1 (77.02430)

**Status: DRAFT IN PROGRESS — measured numbers land as experiments finish (Jul-12).**

## 0. Where we stand (all numbers measured, LB scale = local Score × 100)

| date | submission | LB | rank |
|---|---|---|---|
| Jul 9 | exp005 baseline fleet (pinhole renders) | 57.43380 | ~74 |
| Jul 12 | exp030 F1 distortion remap | 70.44850 | 31 |
| Jul 12 | exp032 F2 DIBR (5 scenes) + remap | 72.22380 | 26 |
| Jul 12 | exp033 P2 refiner, all 8 private, q95 | **75.38050** | **9** |
| Jul 13 | exp034 full stack (big backbone + refiner v2 + TTA), all 8 private, q95 4:4:4 | **76.63890** | — |
| — | top-1 | 77.02430 | 1 |

exp034 measured LB: PSNR 25.3399 / SSIM 85.2489 / LPIPS 10.3493 (8/8 scenes).
Δ vs exp033 = +1.2584 LB (LPIPS −1.91 dominant). Landed ~0.007 under the 0.773
central projection → E3 resample survived the refiner below the assumed 60% and
fleet backbone gain slightly under pilot mean, but clearly positive. **Remaining
gap to top-1 = 0.3854 LB = 0.00385 Score.** Next: reserve levers R1–R6 (§4).

Gap = **1.644 LB pts = 0.01644 Score**. Equivalent single-metric moves:
−4.1 LPIPS pts, or +5.5 SSIM pts, or +2.74 dB PSNR. We attack all three with
a ladder of independent, individually-measured increments.

exp033 on the LB: PSNR 25.06 / SSIM 84.16 / LPIPS 12.26.
Public-5 local refiner mean (pre-JPEG): 0.7569 (PSNR 24.94 / SSIM 87.38 / LPIPS 13.73).
Private is SSIM-weak but LPIPS-strong relative to public — scene content, not a bug.

## 1. The ladder (exp034): each rung measured on public GT before scaling

| rung | what | status | measured Δ (public) |
|---|---|---|---|
| E1 encode | single-encode from float, q95 4:4:4 +optimize (was: q98→q95 4:2:0 double encode) | ✅ MEASURED | +0.0021 (hcm0034 0.76455→0.76704 incl. E2; size 307 MB < 350 cap) |
| E2 TTA | hflip self-ensemble at refiner apply | ✅ MEASURED | +0.0004 (hcm0034 0.7678→0.7682, float level) |
| E3 resample | ss=2 supersampled 3DGS canvas + cubic train-pixel gather (whole chain was bilinear @1×) | ✅ MEASURED ×2 | DIBR level: hcm0034 0.7410→**0.7553 (+0.0143)** LPIPS 0.161→0.138; HCM0181 0.7312→**0.7454 (+0.0142)** LPIPS 0.165→0.140 — replicates cross-scene |
| E4 backbone | DIBR on splatfacto-big (exp006, proven +0.0038 pre-DIBR; private fleet still on antialiased) | ✅ MEASURED ×2 | hcm0034 (quiet, big_60k): **+0.0032**; HCM0181 (busy, big_100k): **+0.0113** (0.7454→0.7567, fallback 19.2%→12.4%) — gain COMPOUNDS through depth/warp quality on busy scenes → E6 fleet strongly justified |
| E5 refiner v2 | base 32→48, 3k→6k iters, EMA 0.999, deterministic val, cosine LR | ✅ MEASURED | same inputs as v1 (hcm0034): v1+TTA 0.7682 → v2+TTA **0.7710 (+0.0028)**, LPIPS 0.125→0.1197; val still descending at 6k |
| E6 fleet | retrain 8 private backbones splatfacto-big (rented GPU), re-run DIBR+refiner v2 fleet | ⬜ pending E4 | — |

## 2. Projection to top-1 (updated as rungs land)

**Transfer calibration (why public-5 mean predicts the private LB):**

| submission | public-5 local mean (at shipped encode) | private LB /100 | offset |
|---|---|---|---|
| exp030 remap | 0.7013 | 0.70449 | +0.003 |
| exp033 refiner @q95 | 0.7539 | 0.75381 | **±0.000** |

Public-5 mean ≈ private LB within ±0.004 — so the ladder, measured on public
GT, projects the LB with ~±0.4 pt confidence. **Top-1 requires public-5 mean
≈ 0.7702; with margin, target ≥ 0.773.**

Running projection:
- exp033 public-5 @ shipped encode: 0.7539
- + E1 single-encode q95 4:4:4 + E2 TTA: **+0.0025 measured** → 0.7564
- + E3 resample: **+0.0143 measured at DIBR level** (hcm0034; refiner-level
  survival = E5b, overnight). If ≥60% survives the refiner → +0.009.
- + E4 big backbone: **+0.0032 measured post-DIBR** (hcm0034) → E6 applies it
  to the 8 LB scenes.
- + E5 refiner v2 trainer: **+0.0028 measured** (same-input A/B, hcm0034)
- + E4/E6 backbone: **+0.0032 (quiet) … +0.0113 (busy) measured** post-DIBR;
  conservative fleet mean +0.005.
- Central estimate: 0.7564 + 0.009 (E3 @60% survival) + 0.005 (E4/E6)
  + 0.0028 (E5) ≈ **0.773–0.776** before reserves → clear of the 0.7702
  top-1 line; reserves R1–R6 (§4) provide the defense margin. The E5b/E5c
  overnight stack measurements replace the survival assumption with data.
- DIBR-level cumulative (both pilots): hcm0034 0.7410→0.7585 (+0.0175);
  HCM0181 0.7312→0.7567 (**+0.0255**, LPIPS 0.165→0.127).
- DIBR-level full ladder (hcm0034): 0.7410 → cubic 0.7513 → ss2+cubic 0.7553
  → ss2+cubic+big 0.7585 (**+0.0175 total, LPIPS 0.161→0.133**).

## 3. Scale & runbook (how each rung reaches all 8 private scenes)

All artifacts exist and are validated (syntax + dry-run):

1. **exp034 v2 fleet (local, no new backbone)** — `Analysis/run_v2_fleet.sh`:
   per private scene, rebuild DIBR pairs with ss=2+cubic, train refiner v2
   (base 48, 6k iters, EMA, deterministic val), apply with TTA.
   ~2 h/scene on the 1660 Ti → private-8 ≈ 16 h (one night + day), or
   ~25 min/scene on a rented 4090. Public scenes optional (validation only).
2. **Backbone upgrade (rented)** — `configs/experiments/exp034_private_big_fleet.yaml`
   (dry-run validated, 8/8 commands): splatfacto-big 30k on dense init for the
   8 private scenes. ~5–6 h on one rented-4090 session. `run_v2_fleet.sh`
   auto-uses `runs/phase1/exp034_private_big_fleet/<scene>/big_30k` when
   present (per-scene, so a partial retrain still helps).
3. **Build** — `Analysis/14_build_v2_submission.py`: re-applies each scene's
   v2 net on cached test inputs (float), hflip TTA, **single-encode**
   q95 4:4:4 +optimize, auto-steps quality down if the private zip would
   exceed 340 MB, packages via the validated `src/package_submission.py`.
   Falls back per scene to v1 refiner → DIBR → remap, so partial fleets ship.
4. Trust signal for private scenes (no GT): refiner val_loss on held-out train
   views, proven band 0.059–0.092 across all 13 scenes; plus the public-5
   local scores measured at every rung.

New-scene cost (phase 2/3 future-proofing): COLMAP dense init (existing
pipeline) + big_30k train (~40 min 4090) + pairs+refiner v2 (~25 min 4090)
+ build. The entire method is per-scene and embarrassingly parallel.

### Concrete timeline — the full fleet runs on Kaggle (local is too slow)

The whole per-scene stack (big backbone + DIBR ss2·cubic + refiner v2) is
packaged for Kaggle T4×2 in two notebooks. See `kaggle/README_exp034_fleet.md`.

| when | action | where | outcome |
|---|---|---|---|
| once | build + upload dataset | `scripts/build_kaggle_exp034_upload.py` → Kaggle dataset (slug `exp034-fleet`) | `kaggle-upload-exp034.zip` (2.8 GB) |
| run A | HCM0249/0254/0276/1439 | `kaggle-exp034-fleet-A.ipynb` (T4×2, ~6–7 h) | `exp034_output.zip` (PNGs + ckpts) |
| run B | HNI0131/0265/0366/0437 | `kaggle-exp034-fleet-B.ipynb` (T4×2, ~6–7 h) | `exp034_output.zip` |
| after each | drop PNGs → build + **SUBMIT** | `Analysis/14_build_v2_submission.py --suffix _v2` | `exp034_v2_results/partial_private_set1.zip` |
| defend | hold reserves R1–R6, monitor LB daily | — | protect #1 |

The builder ships after **either** notebook (per-scene fallback v2 → v1 → DIBR
→ remap), so the LB moves incrementally as scenes return. Est. all-8 on the
full stack: public-5-calibrated projection **0.773–0.776 → LB ~77.3–77.6 (#1)**.

The Kaggle driver (`Analysis/kaggle_exp034_fleet.py`) is idempotent/resumable:
per scene it stages symlinks → `ns-train splatfacto-big` (skip if ckpt present)
→ `10_refiner_pilot.py --config <big> --ss 2 --sample cubic --base 48
--iters 6000 --ema 0.999 --tta --png` → collect. All commands tested locally
end-to-end (stage links, refiner train+PNG apply, builder PNG pickup).

## 4. Defense of #1 (future-proofing)

- **Margin, not parity**: ship only when projected ≥ 77.3 (top-1 77.02 + the
  0.3 cross-set noise band we've measured between public mean and private LB).
- **Reserve levers (built or specced, not yet spent)**:
  R1 per-scene iters (exp006: busy scenes gain +0.003 at 100k),
  R2 refiner seed-ensemble (2 seeds averaged — apply-time cost only),
  R3 K=5 neighbors + per-pixel z-margin source selection in DIBR,
  R4 extra refiner input channels (2nd-nearest blend, depth), R5 ss=3 canvas.
  Each is a measured-pilot-first increment on the same rails.
- **Cadence**: keep one validated zip better than the current LB position
  ready at all times; if overtaken, submit the next rung same-day (the
  private-partial zip builds in ~30 min from cached inputs).
- **Robustness**: per-scene fallback chain in the builder (v2 → v1 → DIBR →
  remap) means no single failed retrain can regress a scene below its banked
  quality; every submission is validated 8/8 scenes + ≤350 MB before upload.
- **Compliance defense** (audit-safe): pipeline uses only provided train
  images/poses + our own 3DGS + provided test poses. No test images anywhere
  in training or inference; LPIPS-vgg only inside the loss. Reproducible:
  configs + scripts + this ladder of measured public-GT deltas.

## 5. Compliance

DIBR + refiner use only: provided train images + train poses + our own 3DGS
model + test poses/intrinsics from the provided CSVs. No test images, no
external data, no pretrained enhancement nets (LPIPS-vgg is used only inside
our training loss, standard practice). Rasterizer = gsplat (allowed backend).
Submission ≤ 350 MB, JPEG, exact test filenames — enforced by the builder.
