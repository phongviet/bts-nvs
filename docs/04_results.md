# 04 — Results, submissions, calibration

## Current best (2026-07-28)

| | |
|---|---|
| **Best submission** | `submissions/round2/round2_v7a_all_drones_adv/submission_round1.zip` — **LB 75.3793** (md5 `9375ced6`) |
| **Config** | 5 drones @ refiner `naf+evidence+adv 0.003`, indoor `bonsai` (SSS backbone) + `chair` (naf+evidence+adv 0.003), q98 4:4:4 pooled-knapsack encode, BUDGET_MIB 348 |
| **Standing decision** | **v7a stands / re-upload v7a.** All lever families are closed; no measured lever closes the top-8 gap before the deadline |
| **Deadline** | 2026-07-30 (quota refreshes 2026-08-01, after the deadline) |

## Submission history

| # | date | config | LB | notes |
|---|---|---|---|---|
| 1 | 07-07 | exp001 sparse init | 56.81480 | first datapoint (control) |
| 2 | 07-07 | exp002 dense-COLMAP init | 57.19840 | transfer ✅ local +0.0037 → LB +0.00384 |
| 3 | 07-09 | + antialiased, full private fleet | **57.43380** | per-metric breakdown → `psnr_max = 50` SOLVED |
| 4 | 07-09 | + exp011 unsharp+jpeg98 | 56.93230 | **transfer ❌ sign flip** → exp011 reverted (AlexNet-LPIPS artifact) |
| 5 | 07-12 | exp030 F1 distortion remap | **70.44850** | **+16.4, biggest lever ever.** Confirms LB grades private directly |
| 6 | 07-12 | exp032 DIBR hybrid | **72.22380** | +1.77 |
| 7 | 07-12 | exp033 neural refiner (all 8 private) | **75.38050** (rank 9) | +3.16 |
| R2-1 | 07-21 | round-2 all 7 scenes, full stack | **74.83130** | PSNR 25.2251 / SSIM 83.9707 / LPIPS 13.7374. Round 2 is a **harder set** (2 indoor), not a regression |
| R2-2 | 07-21 | indoor `--max-pairs 90→180` | **74.86320** | +0.032, SSIM+PSNR not LPIPS → pair count exhausted as an LPIPS lever |
| v4 | 07-23 | bonsai: SSS + naf + adv 0.003 | **75.1595** | +0.2963 (predicted +0.14) |
| v5 | 07-23 | + chair: naf + adv 0.003 | **75.2807** | +0.4175. PSNR 25.456 / SSIM 84.221 / LPIPS 13.148 |
| v6 | 07-23 | + chair: SSS backbone | **75.1478** 🔴 | −0.133 vs v5 (all 3 metrics worse). val_loss said better → the signal inverted |
| v7 | 07-23 | v5 + HCM0421 adv + q98 encode | **75.3200** | +0.039 (confounded: renders + encode changed together) |
| **v7a** | **07-23** | **+ the other 4 drones on adv** (q98) | **75.3793** 🏆 | +0.0593, **entirely LPIPS** (+0.0598), PSNR −0.0053. val_loss predicted −0.093 — sign wrong |

Round 1's 76.639 (PSNR 25.34 / SSIM 85.25 / LPIPS 10.35) was scored on the
retired private set; it is now a *method* validation, not a standing score.

## Calibration (the rules every A/B obeys)

- **`psnr_max = 50.0`** — SOLVED (per-dB weight 0.006).
- **LPIPS backbone = VGG** — CONFIRMED. All decision scoring `--lpips-net vgg`.
  Distrust any win concentrated in AlexNet-LPIPS.
- **Aggregation is per scene.** v4 changed only bonsai and gained 4.2× the
  frame-weighted prediction → indoor carries 28.6 % of the score. Solving
  `74.8632 = (5·drone + 2·indoor)/7` gives **drone ≈ 77.2–77.7, indoor ≈ 68–69.5**
  → **drones already exceed Round 1's winning 77.02; the entire gap is indoor.**
- **The 25-frame hold-out UNDER-predicts shipped gains ~2×** (v4 2.1×, v5 1.8×) —
  it trains on a weakened backbone and scores interpolated poses. **Treat
  hold-out ΔScore as a FLOOR.**
- **Refiner `val_loss` trust band 0.054–0.096 is DRONE-ONLY**; indoor sits at
  0.10–0.18 and still scores fine. Never alarm on an indoor val_loss.
- **val_loss ↓ does not imply test LPIPS ↓ indoors**; the metrics decouple on
  unseen poses. And val_loss is invalid for adversarial arms and backbone swaps
  (see [03_experiments](03_experiments.md)).
- v1→v5 decomposition (+0.4175): LPIPS +0.236 (56 %), PSNR +0.139, SSIM +0.075.
  Scene-points: bonsai +2.07, chair +0.85.

## Where the gap is, and why it is borderline

Top-8 needs ~76.1489 (gap ≈ +0.77 from v7a); Round-2 top-1 was ~77.25. The gap
is entirely in `bonsai` + `chair`, whose LPIPS/fidelity is dragged by motion
blur, a glossy reflective table, and short handheld baselines. The binding
constraint is that the **refiner already supplies the LPIPS gain**, so
backbone-side perceptual work is redundant with it, and every fidelity-shaped
backbone win inverts through the refiner (W11). No measured lever closes the gap
before the deadline; **v7a stands.**
