# bonsai match-test hold-out validation (2026-07-18)

**First GT-based metric on the round-2 indoor regime.** These scenes ship no test
GT and have no same-regime public bench, so this is the only local score
available. Method + caveats: `scripts/val_holdout_run.sh`.

## Result (backbone only, no remap/DIBR/refiner)

```
VAL bonsai: PSNR=25.619  SSIM=0.8382  LPIPS=0.3232  Score=0.6759   (n=25)
```

Split: 223 train / 25 val, match-test (`n_val=25`, cost p50=0.074 max=0.172).
Backbone = same `configs/phase_locked.conf` as the submission run, retrained on
train-minus-holdout so val frames were never seen photometrically.

**The backbone is sound.** Compared with phase-1 drone scenes at the *same*
stage (backbone only, pinhole vs raw GT):

| scene | PSNR | SSIM | LPIPS | Score |
|---|---|---|---|---|
| **bonsai (val hold-out)** | **25.62** | **0.8382** | **0.3232** | **0.6759** |
| hcm0034 | 21.43 | 0.7397 | 0.2392 | 0.6548 |
| HCM0181 | 20.27 | 0.7017 | 0.2443 | 0.6344 |

bonsai is *ahead* on PSNR (+4.2 dB) and SSIM (+0.10) but clearly **behind on
LPIPS (0.323 vs ~0.24)**. Motion blur and the glossy table did **not** produce a
degenerate reconstruction — the fear going in. The signature (high PSNR/SSIM,
weak LPIPS) is "accurate but perceptually soft", which is precisely the axis the
F2 DIBR + P2 refiner stack attacks: on drone scenes it drove LPIPS 0.24 -> 0.12.
So headroom here is large, and it is concentrated in the metric carrying 0.4 of
the grader weight.

## The real finding: quality is strongly non-uniform along the capture

Per-val-frame score spread is wide (min 0.5427, med 0.6997, max 0.7569,
stdev 0.0632) and almost perfectly ordered by position in the capture:

| region | val score | val LPIPS |
|---|---|---|
| early (frames 310-730) | 0.54 - 0.62 | 0.36 - 0.44 |
| mid (860-1040) | 0.62 - 0.72 | 0.31 - 0.39 |
| late (1440-2650) | 0.67 - 0.76 | 0.24 - 0.32 |

Cause, from `results/round2_test_pose_coverage.csv` — the early capture is
**~10x more sparsely covered**:

| region | test poses | dist_frac | angle | nearby train views |
|---|---|---|---|---|
| early (<800) | 9 | 0.0320 | 4.44 deg | **2** |
| mid (800-1400) | 5 | 0.0099 | 5.65 deg | 9 |
| late (>=1400) | 14 | 0.0074 | 3.29 deg | **21** |

**9 of 28 graded test poses (32%) sit in the under-covered early region.**

### Why this matters for the pipeline

Our two big levers both consume *neighbouring real views*:
- **F2 DIBR** warps real train pixels from K neighbours. With only ~2 nearby
  views (vs 21 late), source material is scarce exactly where reconstruction is
  weakest -> expect high fallback-to-3DGS there, i.e. DIBR helps *least* where
  we need it *most*.
- **P2 refiner** builds leave-one-out training pairs from the same neighbours,
  so its supervision is also thin in that region.

This inverts the usual assumption: on the drone scenes DIBR gained most on the
hardest frames; on bonsai the hardest frames are hardest *because* neighbours
are missing, which is the one condition under which DIBR cannot help.

### Actionable next steps (not yet run)

1. Run DIBR on the bonsai val split and read **fallback % split by region** — if
   early-region fallback is near-total, the stack's bonsai gain will land almost
   entirely on the 19 late/mid test poses.
2. Consider a region-aware K for DIBR (larger K / looser tol in the early
   region) to scrape more source views, rather than one global setting.
3. The early region is where any generative/inpainting prior would pay off, and
   the only part of round-2 where extrapolation-like behaviour appears.

## Caveats (both conservative -- the real submission should be >= this)

- `points3D.bin` is the dense cloud fused by MVS that saw the val views, so the
  geometry *init* is mildly val-informed. Photometric loss strictly excludes val.
- Hold-outs are the *most test-adjacent* train frames, so this backbone trains
  without the 25 frames closest to the test poses; the submission backbone keeps
  them.
- `metrics_val_split.json`'s `per_image` list appends a `mean` row as its last
  element -- exclude `image == "mean"` when computing spreads (it silently
  skewed a first pass here).
