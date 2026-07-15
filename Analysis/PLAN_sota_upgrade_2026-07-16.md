# PLAN — SOTA component upgrade campaign (drafted 2026-07-16)

**Inputs:** `RESEARCH_sota_components_2026-07-15.md` (our survey) + the Claude deep-research
report (`compass_artifact_wf-8c5e7258…_text_markdown.md`, same folder). This plan
supersedes neither `FINAL_PLAN_top1.md`'s rails (gates, calibration, fallback chain,
submission cadence) nor the reserves — it sequences the reserves *plus* the new
SOTA levers into waves sized for the Jul-30 Phase-1 deadline (14 days).

## 0. Situation and target

| | LB | Δ Score vs us |
|---|---|---|
| us (exp034, rank 12) | 76.63890 | — |
| top-8 line | 76.85270 | +0.00214 |
| top-1 | 78.57890 | +0.01940 |

- Public-5 mean predicts private LB within ±0.004 (exp030/exp033 calibration) —
  unchanged, still our decision metric (vgg + psnr_max 50).
- Promotion gate unchanged: **+0.002 public-5 pilot** (hcm0034 quiet + HCM0181 busy)
  before fleet scale-out; builder fallback chain keeps every submission monotone.
- Stack math to top-1: we need ~+0.019. Plausible sum if most waves land:
  reserves +0.002–0.005, flow-alignment +0.003–0.008, refiner v3 +0.003–0.008,
  depth A/B +0.002–0.006, encode knapsack +0.001–0.002, 3DGUT uncertain.
  **No single lever suffices — this is a parallel campaign, one wave per teammate.**

## 1. What the deep-research report changed vs our Jul-15 survey

1. **GADA and IBGS both release code** (github.com/siw00-lim/GADA,
   github.com/HoangChuongNguyen/ibgs) — but both are built on the **INRIA 3DGS +
   PGSR CUDA rasterizer** (`diff-plane-rasterization`), NOT gsplat → full port is
   HIGH effort. Report's own threshold: if port > ~1 week, use **SEA-RAFT
   pre-alignment** of warped views instead (captures much of the gain cheaply).
2. **GADA's headline mechanism stat:** deformable offsets (iterative K=5,
   σ_max = 7 px) + learned confidence raise usable warped-pixel density
   **33% → 79%**. Our analogue: photometric-guard rejections + 12–19% hole
   fallback are exactly the discarded-evidence pool it recovers.
3. **Depth ranking for warping revised:** PGSR (multi-view-consistent, most
   battle-tested, what IBGS/GADA consume) and **RaDe-GS** (DTU CD 0.68 mm;
   TSPE-GS finds it more stable than 2DGS/PGSR on **semi-transparent geometry —
   i.e. our see-through lattice masts**) beat 2DGS (CD ≈0.80, lower PSNR).
   → Depth A/B should test **RaDe-GS first**, not 2DGS. Both ship as standalone
   INRIA-fork repos; we only need their **depth maps**, so we run their repos
   as-is and export depth — no rasterizer port into our stack.
4. **3DGUT:** confirmed MCMC-only in gsplat; dense-init+MCMC combination is
   under-documented → empirical question. Alternative on-ramp:
   **spirulae-splat** (nerfstudio-compatible, 3DGUT default distortion handler,
   MCMC+IGS densifier, bilateral-grid/PPISP exposure) may cost less than wiring
   raw gsplat `simple_trainer`. Report gives revert thresholds: worse by
   >0.3 dB PSNR or +0.01 LPIPS → keep undistort+remap path.
5. **Refiner:** NAFNet confirmed still the ~1–17M-param efficiency SOTA in
   NTIRE 2025/2026 (frontier moved to big diffusion = banned for us);
   MambaIRv2 the one credible same-scale A/B. **Steal IBGS's exposure trick**
   (match nearest-neighbor view's exposure) — cheap, drone-relevant.
6. **JPEG:** mozjpeg trellis + **SSIM/MS-SSIM-tuned quant tables** (LPIPS-tuned
   tables measurably do NOT help — arXiv 2510.10970) + **per-image knapsack**
   rate allocation (greedy marginal Score/byte; low-SSIM images first). This
   upgrades reserve R6 from "per-scene quality" to per-image allocation.
7. **Licensing flag:** IBGS repo has no LICENSE (inherits INRIA non-commercial
   lineage). Mitigation: we lift *ideas* (offsets, confidence, τ-filter,
   exposure) into our own code — which is how our whole stack is built anyway —
   and only run their repos as external tools if needed for depth export.
   gsplat/3DGRT Apache-2.0, SEA-RAFT/NeuFlow/mozjpeg permissive. ✅

## 2. The waves (exp numbering continues from exp035)

### Wave 0 — re-enter top-8 with what's already specced (Jul 16–18)
Owner: P2 (infra) + one Kaggle account. **Goal: +0.003–0.005, SUBMIT by Jul 18.**

- **exp036 (R2+R3):** refiner seed-ensemble (2 seeds, apply-time average) +
  DIBR K=5 neighbors with per-pixel z-margin + **IBGS relative-τ depth-consistency
  filter** |z−z_w|/(z+z_w) ≤ τ (sweep τ ∈ {5e-4, 1e-3, 2e-3}; replaces absolute
  tol=0.03). Pilot both scenes → fleet on cached inputs (local/Kaggle, cheap).
- **exp037 (R6-upgraded encode):** mozjpeg (trellis on, MS-SSIM-tuned table) +
  per-image greedy knapsack across the ≤350 MB budget using measured per-image
  quality→(size, Score-proxy) curves. Public-GT validates the Score gain; the
  private allocation uses SSIM-proxy (no GT needed — allocate to low-SSIM-risk
  images = high-detail/busy frames). Zero model risk, strictly additive.
- **exp038 (R1):** 100k-iter big backbones for the 2 weakest busy scenes
  (HCM0249, HNI0265) on Kaggle (~6 h/scene T4×2); rerun DIBR+refiner on top.
- Ship as **exp036–038 combined submission** once each rung clears its pilot.

### Wave 1 — evidence recovery: flow alignment + refiner v3 (Jul 17–22)
Owner: P3 (perception) + local + one Kaggle account. **Goal: +0.005–0.012.**
The GADA-inspired wave: stop discarding misaligned real pixels; let the refiner
see per-neighbor evidence.

- **exp039 — SEA-RAFT pre-alignment (GADA approximation).** For each warped
  neighbor view: flow(warped_neighbor → 3DGS render), clamp |flow| ≤ 7 px
  (GADA's σ_max), apply residual warp, THEN blend/guard. Expectation: guard
  rejection rate and hole-fallback % drop; ghosting on antennas/lattice drops.
  - Instrumentation first: log current guard-rejection % per scene (the 33%→79%
    analogue) so the mechanism is measured, not just the Score.
  - SEA-RAFT: scene-agnostic pretrained, inference-only → **add provenance-table
    row before first use** (`docs/rules_and_constraints.md`). Runs fp32 on
    1660 Ti at our res or on Kaggle. NeuFlow v2 as fallback if VRAM/speed bites.
  - Guardrails: apply flow only where flow-confidence high & |flow| small;
    degenerate flow → keep unaligned pixel (never worse than today by
    construction, same philosophy as the photometric guard).
- **exp040 — refiner v3.** Input goes from 7ch [remap render | DIBR blend |
  vis-mask] to **evidence stack**: [render | per-neighbor aligned warps (K×3) |
  per-neighbor confidence/τ-consistency maps (K) | 3DGS depth (1)]; net learns
  the aggregation (IBGS's lesson). Architecture A/B at equal params:
  current conv U-Net base48 vs **NAFNet-block U-Net (~2–5M)**. Plus IBGS
  **exposure correction** (normalize target/neighbor exposure to nearest
  neighbor) as a toggle. Same recipe otherwise (grader loss, LOO pairs, 6k,
  EMA, cosine, TTA). MambaIRv2 only if NAFNet ties.
- exp039 feeds exp040 (aligned warps are the v3 inputs) but each is gated
  independently: exp039 measurable at DIBR level before any retraining.

### Wave 2 — depth-source A/B for the warp (Jul 19–24)
Owner: P1 (geometry) + one Kaggle account. **Goal: +0.002–0.006 (busy scenes).**

- **exp041 — RaDe-GS depth-only.** Train RaDe-GS (their repo, unmodified, Kaggle)
  on the 2 pilot scenes; export depth at test + neighbor poses; feed our DIBR
  as the depth/z-test source (RGB + hole fill stay splatfacto-big). Success
  metric: traincheck warp PSNR, guard-rejection %, thin-structure crops — then
  Score. PGSR as the alternate arm if RaDe-GS underwhelms (report: PGSR best
  multi-view-consistent depth; RaDe-GS best for see-through structures).
- Integration cost is LOW by design: our Warper already accepts a depth source
  (`depth_cache` is backbone-keyed since E4); a new exporter script maps their
  depth convention (scale/axes) onto ours — validate via traincheck before
  trusting (same validation that caught the double axis-swap in render.py).
- If depth wins at DIBR level, it compounds into exp039/exp040 (better depth →
  better alignment → more accepted evidence).

### Wave 3 — 3DGUT native-distortion pilot (Jul 20–26, gated/timeboxed)
Owner: P1 after Wave 2 pilot lands, one Kaggle account. **Goal: uncertain;
kills 2 resample losses; future-proofs Phase 2/3 camera models.**

- **exp042:** pilot on **HNI0131** (k=−0.115, biggest remap ceiling) +
  **HCM0181** (busy control). Two on-ramps, pick after a 1-day spike:
  (a) gsplat ≥ latest, `simple_trainer.py mcmc --with_ut --with_eval3d`,
  `radial_coeffs` from COLMAP SIMPLE_RADIAL, seed with dense cloud,
  `cap_max` ≈ our current Gaussian count; (b) spirulae-splat (nerfstudio-
  compatible, 3DGUT + MCMC/IGS + exposure handling built in).
- Train on RAW distorted images; render test views natively distorted → the
  whole F1 remap stage disappears for these arms. DIBR/refiner run unchanged
  on top (they already operate in distorted geometry).
- **Pre-registered revert rule (from the report):** worse by >0.3 dB PSNR or
  +0.01 LPIPS vs our undistort+remap pipeline on the pilot → drop the training
  arm; keep 3DGUT only as a candidate distorted-space *renderer*.
  Known risk: MCMC-only densifier (our dense-MCMC measured −0.0025) — this is
  precisely what the pilot measures.

### Wave 4 — full GADA/IBGS module port (decision Jul 23, likely Phase-2 asset)
Owner: whoever is free; default = **do not start before Jul 23**.

- If Waves 1–2 land ≥ +0.008 combined, a full learned deformable-offset +
  confidence module (our own implementation, GADA recipe: iterative K=5,
  σ_max=7, geometry-verified view aggregation) replaces flow-align + heuristic
  guard entirely. Est. 4–6 days — tight for Jul-30, valuable for Phase 2/3
  (72 h/48 h windows reward pre-built tooling; per-scene training fits windows).
- License-safe by construction: reimplemented in our codebase, ideas not code.

## 3. Compute & people map

| resource | Wave 0 | Wave 1 | Wave 2 | Wave 3 |
|---|---|---|---|---|
| local 1660 Ti | encode knapsack, τ/K sweeps on cached inputs | SEA-RAFT proto (fp32), pair rebuild | depth-convention validation | — |
| Kaggle acct A | exp038 100k backbones | — | — | exp042 3DGUT |
| Kaggle acct B | — | exp040 refiner v3 pilots+fleet | — | — |
| Kaggle acct C | — | — | exp041 RaDe-GS/PGSR | — |
| P1 geometry | — | — | exp041 | exp042 |
| P2 infra | exp036/037/038 + all submissions | builder updates (v3 in fallback chain) | — | — |
| P3 perception | — | exp039/exp040 | — | — |

Submission cadence: one validated zip strictly better than current LB position
at all times; submit after Wave 0, after Wave 1 fleet, and whenever overtaken.

## 4. Decision points & risks

- **Jul 18:** Wave-0 submitted? If reserves under-deliver (<+0.002), pull P2
  onto exp037 knapsack polish + help P3 (v3 has the largest remaining headroom).
- **Jul 20:** exp039 mechanism check — if guard-rejection % doesn't drop
  materially, flow-align is dead → all-in on refiner v3 evidence channels
  (the net can learn alignment implicitly via deformable-conv-like capacity;
  consider small learned offsets inside v3 = mini-GADA).
- **Jul 23:** Wave-4 go/no-go; also 3DGUT revert check.
- **Jul 27:** freeze — last 3 days are fleet re-runs + submission engineering
  only (the exp034 Kaggle fleet driver + builder already handle this shape).
- Risk: SEA-RAFT flow aligning real pixels toward 3DGS *blur* (target is the
  render). Mitigations: clamp, confidence gating, and the grader-loss refiner
  downstream can undo mild over-smoothing; mechanism metrics catch it early.
- Risk: RaDe-GS/PGSR depth convention mismatch — caught by traincheck (built).
- Risk: Kaggle quota (~30 h/wk/acct) — Waves are sized ≤ ~12 h/acct/wk; 100k
  backbones are the biggest ticket, capped at 2 scenes.
- Compliance: SEA-RAFT provenance row required before exp039 ships; all other
  new components trained from scratch on provided data; no diffusion priors.

## 5. Expected trajectory (honest ranges, public-5 mean)

| after | Δ range | LB projection |
|---|---|---|
| Wave 0 | +0.002–0.005 | 76.9–77.2 (top-8, maybe top-5) |
| + Wave 1 | +0.005–0.012 | 77.4–78.3 |
| + Wave 2 | +0.002–0.006 | 77.6–78.9 (top-1 contention) |
| + Wave 3 | 0…+? | option value + Phase-2/3 insurance |

Top-1 (78.579, and still moving) most likely requires Waves 0+1+2 all landing
in their mid ranges. Every rung stays individually gated and reversible.
