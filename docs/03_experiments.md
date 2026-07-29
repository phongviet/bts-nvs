# 03 — Experiment log & lever graveyard

Every experiment, its verdict, and the lesson kept. Status: 🟢 adopted ·
🔴 dropped/refuted · 🟡 partial · ⬜ queued/superseded. Decisions are scored
`vgg + psnr_max=50`; a win concentrated in AlexNet-LPIPS is a red flag, not a
result. The ±0.002 Score bar is the adoption gate.

## Phase-1 experiments (Round 1 — closed, kept for the lessons)

| exp | what | verdict | lesson |
|---|---|---|---|
| 001–002 | sparse → dense-COLMAP init | 🟢 | dense init +0.0037 local / +0.00384 LB. **The Week-1 "Score 0.143" was a render-script bug** (double-applied world-axis swap), not model quality — fixing it gave 0.718 same checkpoint. *Always sanity-check a render/eval path on a SEEN training view first.* |
| 003 | VGGT pseudo-cloud init | 🔴 | lost all 5 |
| 004 | backend variants | 🟢 | adopt **antialiased**; drop scale_reg, sky_mask |
| 006 | capacity/iters sweep | 🟢 | adopt **splatfacto-big** (+0.0038). MCMC dead on dense init. Iters: quiet scenes saturate at 30k; busy scenes (HCM0181) still gain +0.0032 at 100k → per-scene iters |
| 007 | bilateral grid | 🔴 | −0.005/−0.008; same-flight exposure already consistent |
| 008 | transient masks | 🔴 | wash on both pilots; transients too small/sparse at this GSD |
| 009 | LPIPS fine-tune | 🟡 harness no-op | two weights → byte-identical scores. Revived as **E1**, see below |
| 011 | postproc unsharp/denoise | 🔴 **refuted by LB** | local +0.0039 was pure AlexNet-LPIPS; VGG-neutral → LB −0.005. *This is how the VGG-vs-Alex lesson was learned.* Keep JPEG encoding, drop the op |
| 012 | render ensembling | 🔴 | pixel averaging inflates LPIPS; the 0.4 weight eats the SSIM gain |
| 013 | test-pose-weighted sampling | 🔴 | exact wash |
| 014 | camera-optimizer | 🔴 **permanent** | −0.0199. Optimizing train poses drifts the world frame off the FIXED test poses |
| 015/016 | Difix3D+ off-the-shelf | 🔴 | −0.034/−0.037 on drones; renders too artifact-light to benefit. LoRA arm (016) skipped. *A trained indoor LoRA was left unmeasured — revisited later* |
| 018 | SSS backend | 🟡 | wash with splatfacto-big at 2× cost; its one win is busy HCM0181. Later production-used on `bonsai`. See [06_sss_backbone](06_sss_backbone.md) |
| 019 | val-split validation | 🟢 | **match-test splits rank configs like real GT** (Spearman 1.0) — the basis of every hold-out A/B since |
| 021 | distortion-residual diagnostic | 🟢 | corner deficit is overlap/frustum, not lens distortion |
| 022 | periphery-weighted L1 | 🔴 | corners are data-under-constrained; spatial reweighting is dead |
| 023 | LEGS Laplacian loss | 🟡 | +0.0018/+0.0021 on sparse, at the bar; never retested on dense |
| 024 | FreGS frequency loss | 🔴 | OOM'd via densification blow-up; never re-run |
| 029 | MCMC on sparse init | 🔴 | MCMC is **not** init-robust (refutes the paper's headline premise on our data); dense MVS does real work |
| **030** | **F1 distortion remap** | 🟢 **+16.4 LB** | test GT is raw `SIMPLE_RADIAL`, CSV omits k. Biggest single lever found |
| **031/032** | **F2 DIBR hybrid** | 🟢 **+1.77 LB** | warp real train pixels via 3DGS depth + occlusion z-test; photometric guard fixes thin-structure ghosting |
| **033** | **neural refiner** | 🟢 **+3.16 LB** | U-Net on [render, DIBR, mask] with the grader objective. The top-8 mechanism |
| 034 | top-1 ladder | 🟢 | E1 single-encode q95 4:4:4 (+0.0025); E2 hflip TTA; **E3 ss2+cubic (+0.0103)**; E4 big backbone; E5 refiner v2 |
| 036 | DIBR reserves | 🟡 | +0.0009 < bar. Seed-ensemble refuted (averaging costs LPIPS) |
| 037 | JPEG knapsack | 🟢 built | cross-image allocation at equal bytes LOSES; what pays is flooring at shipped q and spending headroom upward |
| 039 | flow-residual pre-alignment | 🔴 | guard-reject is 0.1–0.4 % on drones — nothing for flow to rescue |
| 040 | refiner v3 naf+evidence | 🟡 | +0.0008 on drones (below bar), but naf+evidence+adv later **gained** indoors and on drones |
| 041/042 | RaDe-GS depth / 3DGUT | 🔴/⬜ | RaDe-GS depth inert on drones (far-field, tiny baseline/depth ratio → warp is near-homographic); 3DGUT superseded |

## Round-2 experiments

### GATE A — the DIBR+refiner stack transfers to indoor 🟢

Backbone-only → full stack: bonsai 0.6759 → 0.6913 (**+0.0154**), chair 0.6506 →
0.6650 (**+0.0144**). All three metrics improve on both.

### Refiner-side wins (the indoor levers)

- **W1 `--blocks naf --evidence`** 🟢 +0.0032–0.0035 on both indoor, LPIPS −0.005.
- **W3 `--adv 0.003`** 🟢 the biggest indoor lever: bonsai 0.6920 → 0.7016, chair
  0.6660 → 0.6724. PSNR, SSIM *and* LPIPS all improved (not the predicted trade).
- **W9 adv-weight ladder** (Kaggle, both indoor scenes, `ctrl/0.003/0.006/0.010`):
  **0.003 is optimal on both** Score and LPIPS; monotone-decreasing above. Audit
  CLOSED — the "0.003, no higher" rule is now genuinely measured.
- **Drone adv rollout** 🟢 **+0.0593 LB, 100 % LPIPS** (val_loss predicted −0.093;
  sign wrong — see the val_loss lessons below).

### Refuted on Round 2 🔴

- **Chair pair-selection** (sharp/stratified/count) — best +0.0003 vs a +0.002
  bar; LPIPS inert; sharp-only supervision *hurts*.
- **Indoor is NOT frame interpolation** — 3 zero-param 2D baselines score
  0.49/0.38 vs shipped 0.69/0.67. Rules out RIFE/FILM for ~$0; indoor is
  parallax-heavy.
- **SSS on drones / RaDe-GS depth on drones** — resampling blur / near-homographic
  warp; both inert-to-negative on far-field drones.
- **Scene-weighted knapsack** — −0.00039; bonsai's R-D curve is saturated, the
  unweighted greedy already allocates by marginal gain/byte.
- **Encode family bounded ~0.02 LB total** (grader-shaped knapsack study).
- **3dgs-deblur ("Gaussian Splatting on the Move") on bonsai** (Vast.ai A/B,
  2026-07-28) — **fidelity-NEGATIVE at raw poses**: baseline Score 0.6726 (≈ the
  0.6759 anchor, eval path validated) vs deblur 0.6395 (−1.17 dB, −0.065 SSIM,
  +0.017 LPIPS). Its camera/velocity optimizer refines TRAINING poses, drifting
  the gaussian frame off the raw COLMAP frame; its paper gains depend on
  `--optimize-eval-cameras` (test-time per-view pose opt) which this competition
  disallows. Box destroyed.

### Backbone-side family (the last unspent line) — CLOSED 🔴

Diagnosis: every closed lever acts *downstream of a frozen backbone*, and
splatfacto only *outputs* depth (no geometry supervision in training). Three arms
on chair (control raw Score 0.6506):

- **MCMC** — rejected without re-running: 0-for-3 on our own data (exp004/006/029).
- **E1 `splatfacto-perceptual`** (LPIPS-VGG 0.1, revives exp009) — raw win
  reproduced on two independent Kaggle sessions (+0.005…+0.006, chair 0.6558).
  **VOIDS exp009's null.**
- **E2 `splatfacto-depth`** — scale-and-shift-invariant Depth-Anything-V2
  supervision in disparity space (new code: `Analysis/26_export_mono_depth.py`,
  `src/models/splatfacto_depth.py`).

**W11 gate verdict (2026-07-27):** the raw backbone win **inverts through the
refiner** — PSNR deficit carries 94 %, SSIM 70 %, but the LPIPS advantage only
26 %, because the refiner already supplies ~−0.036 LPIPS. **The refiner, not the
backbone, is the binding LPIPS constraint.** RULE: only *fidelity-positive*
backbone levers are worth testing. Backbone-side family CLOSED.

### Difix indoor LoRA revival — CLOSED 🔴

Thesis: the phase-1 Difix refutation was on drones (LPIPS ~0.07, nothing to fix);
indoor is LPIPS ~0.27 = the regime a single-step SD-Turbo fixer targets.

- Root-caused a Kaggle crash: Difix's VAE decoder is skip-connected
  (`incoming_skip_acts`) and the trainer's generic img2img forward never wired
  them; now GPU-verified via `tests/test_difix_one_step.py` (mean |diff| 0.0000).
- The LoRA **did** train (repaired off-the-shelf Difix by −0.0641 LPIPS on center
  crops). But the full-frame 5-way gate showed the **refiner beats
  refiner+difix_lora on both indoor scenes** (loss is SSIM/PSNR, not LPIPS).
  Difix family CLOSED.

## The two `val_loss` lessons (do not relearn)

1. **`val_loss` is anti-correlated with LB for adversarial arms.** It is a
   regression loss; the critic's job is to move off the conditional mean. On 4
   drones it said all four got worse (projecting −0.093 LB); reality was +0.0593
   LB, 100 % LPIPS. Never use val_loss to judge an adversarial vs a regression arm.
2. **`val_loss` inverts on a backbone swap.** Refiner val pairs are leave-one-out
   TRAIN views; a higher-capacity backbone wins on val_loss by memorization that
   evaporates at novel poses. Confirmed twice (chair SSS val_loss −0.0187 → LB
   −0.133; H100→ read as Kaggle-equivalent SSS HCM0674 val_loss better → LB
   −0.266). val_loss is valid only for comparing REFINER variants on a FIXED
   backbone.

## SSS scoreboard (backbone-swap lever)

bonsai **+0.296** · chair **−0.133** · HCM0674 **−0.266**. The one win is the one
scene whose splatfacto baseline was capacity-starved (bonsai 0.25 gauss/px). ⇒
**SSS helps only where the splatfacto baseline is capacity-starved.** The chair
`splatfacto-big` vs `splatfacto` A/B was an exact tie (0.6513 = 0.6513) →
capacity is not chair's constraint, and every chair W1/W3 delta measured on plain
splatfacto transfers to the `-big` production backbone.

## Chair backbone diagnostic

Chair scores lowest of the 7 and the deficit is **in the backbone**: the refiner
adds the same +0.0144 to chair as to bonsai; the entire chair−bonsai gap is
present in the raw render. Oracle corrections (exposure, global shift, blur)
recover almost nothing (0 of 25 frames benefit from a global shift → registration
is perfect). The deficit is **distributed reconstruction error, not a correctable
global defect** — camera-optimizer and appearance modeling are both ruled out.
Chair is the only scene whose MVS stopped short of the point cap (short handheld
baseline). Ceiling if fully closed ≈ +0.45 LB, but **no measured lever exists**.
