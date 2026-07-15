# SOTA component research — closing the new gap (2026-07-15)

**Trigger:** we were overtaken. LB standings moved under us in ~2 days:

| position | LB | Δ vs our 76.63890 |
|---|---|---|
| top-1 | 78.57890 | **+1.940 LB = 0.0194 Score** |
| top-8 line | 76.85270 | **+0.214 LB = 0.00214 Score** |
| us (exp034) | 76.63890 | rank 12 |

Gap arithmetic (Score = 0.4·(1−LPIPS) + 0.3·SSIM + 0.3·PSNR/50):

- **Top-8 re-entry** needs any of: −0.53 LPIPS pts, +0.71 SSIM pts, +0.36 dB PSNR.
  The already-specced reserves R1–R6 (`FINAL_PLAN_top1.md` §4) should cover this.
- **Top-1** needs any of: −4.85 LPIPS pts (10.35→5.5!), +6.5 SSIM pts, +3.2 dB PSNR.
  That is a *component-class* jump, not a tuning delta. Top-1 moved +1.55 in two
  days — they found a new mechanism (plausibly the same distortion/IBR family we
  found, executed harder, or a learned aggregation stack per §3 below).

**Conclusion driving this document:** spend the reserves to re-enter top-8 now,
but in parallel upgrade pipeline components to their 2025/2026 SOTA equivalents.
Each section: what we run today → SOTA candidate(s) → mechanism → expected gain →
port cost → compliance. Verified sources at the end.

---

## 1. Backbone (today: `splatfacto-big`, anti-aliased, dense-COLMAP init, 30k)

The backbone has **three distinct jobs** in our stack, and they pull toward
different SOTA branches:

| job | consumer | quality axis that matters |
|---|---|---|
| (a) depth for warp | DIBR unproject/z-test | **geometric depth accuracy** |
| (b) hole fill | DIBR fallback pixels (12–19% of busy scenes) | RGB fidelity |
| (c) input channel | refiner 7-ch input | RGB fidelity |

### 1a. Depth-accurate splatting variants — highest-leverage backbone axis
E4 measured that backbone gain **compounds through depth/warp quality**
(+0.0113 on busy HCM0181, fallback 19.2%→12.4%). Our depth is vanilla
alpha-blended expected depth — known to be biased at silhouettes/thin structures,
which is exactly where DIBR ghosts (antennas, lattice).

- **RaDe-GS** (arXiv 2406.01467, TOG): closed-form rasterized depth + normals on
  *unmodified* 3D Gaussians; DTU Chamfer ≈ NeuraLangelo at 3DGS speed; code
  `HKUST-SAIL/RaDe-GS` (INRIA-rasterizer-based → port cost, or depth-only reimpl).
- **2DGS** (SIGGRAPH 2024): surfel Gaussians, view-consistent depth; **already in
  gsplat** (`rasterization_2dgs`) → cheapest to wire into our stack; slight RGB
  quality loss vs 3DGS (would use it for depth only, keep splatfacto-big RGB).
- **GOF** (Gaussian Opacity Fields): best surfaces, but heavy — skip.

**Pilot (cheap, high info):** train one 2DGS model per pilot scene, use it *only*
as the DIBR depth/z-test source (RGB channels + hole fill stay splatfacto-big).
If traincheck/warp PSNR and thin-structure ghosting improve → fleet.
Expected: the residual 12–19% fallback + guard rejections are the LPIPS ceiling
of DIBR; this attacks both. **Priority: HIGH.**

### 1b. Training-recipe upgrades (RGB fidelity)
- **MVGS** (arXiv 2410.02103): multi-view-regulated per-step supervision,
  claims SOTA on MipNeRF360/T&T/DB, integrable into any 3DGS trainer. Cheap to
  try in splatfacto (multi-image batch per step). MEDIUM priority.
- **Densification line** (Pixel-GS exp025 still queued; AdpSplit arXiv
  2508.12313; Taming-3DGS): Pixel-GS's motivation is literally our frame-corner
  finding. MEDIUM — measurable but historically our loss/densifier levers
  returned ≤ +0.002.
- Already refuted here (do not revisit): MCMC-on-sparse, scale-reg, sky-mask,
  camera-opt (exp014 guard), LPIPS fine-tune at resume (exp009 bug),
  spatial loss reweighting (exp022), FreGS (pending re-run, expectations low).

## 2. Camera model / F1 distortion remap (today: pinhole render → SIMPLE_RADIAL remap, ss2 canvas + cubic)

F1 was our biggest single jump (+16.4 LB), but the *mechanism* still carries two
avoidable resampling losses: (i) train-time — supervision uses undistorted train
images (one resample of every GT pixel); (ii) render-time — pinhole→distorted
remap (mitigated but not eliminated by ss2+cubic).

- **3DGUT** (CVPR 2025, arXiv 2412.12507; **merged into gsplat**, `docs/3dgut.md`):
  replaces EWA projection with an Unscented Transform → rasterizes **directly
  under nonlinear camera models** (pinhole + `radial_coeffs` covers
  SIMPLE_RADIAL; also handles our HNI k=−0.115 fisheye-magnitude case natively).
  Train on RAW distorted images, render test views in distorted geometry — the
  remap stage disappears entirely.
  - Constraint: gsplat's 3DGUT currently supports **MCMC densification only**;
    our dense-init MCMC was −0.0025 vs antialiased default strategy — so the
    net is uncertain and must be measured, not assumed.
  - Port cost: gsplat ≥ 1.5 (we pin 1.4.0) + `with_ut=True, with_eval3d=True`
    plumbing into splatfacto or direct use of gsplat `simple_trainer.py`.
  - Also future-proofs Phase 2/3 (any camera model the organizers ship).
  **Priority: MEDIUM-HIGH pilot** (one busy + one HNI scene: HNI0131 is where
  remap error is largest, so the ceiling is highest there).

## 3. F2 DIBR (today: K=3 inverse-distance blend, z-test, photometric guard, 3DGS hole fill)

This is where the literature moved *toward us* in late 2025/2026 — two papers
formalize exactly our warp-real-pixels idea, and both beat our per-pixel
heuristics with learned components:

- **IBGS — Image-Based Gaussian Splatting** (NeurIPS 2025, arXiv 2511.14357,
  project `hoangchuongnguyen.github.io/ibgs`). Base 3DGS color + learned
  residual from neighboring *training images*. Deltas vs our v2 stack:
  1. **Warp points**: K=4 ray–Gaussian median-depth intersections (transmittance
     ≈ 0.5) instead of a single expected-depth point → robust at silhouettes.
  2. **Neighbor filter**: relative depth-consistency test
     |z−z_warp|/(z+z_warp) ≤ τ=0.001 → cleaner than our absolute-tol z-test.
  3. **Learned aggregation**: per-view PointNet-style features (2×32-dim linear)
     → max-pool across views → 9-layer conv decoder residual. Replaces our
     fixed 1/distance blend + binary photometric guard.
  - Reported: T&T PSNR 23.11→24.84 (+1.73 dB), LPIPS 0.184→0.148, with *fewer*
    Gaussians. Same order as our whole DIBR+refiner gain — on top of a stack
    that lacks our distortion/TTA/encode rungs.
- **GADA** (ICML 2026, arXiv 2607.00595): successor line; **deformable offsets**
  actively correct pixel-level warp misalignment + **implicit confidence
  weighting** instead of hard visibility thresholding ("discarding valid
  pixels" — literally our photometric-guard failure mode), 2.13× faster.

**Concrete upgrade ladder for our DIBR (ranked by port-cost/gain):**
1. **IBGS relative-τ neighbor filter + K=5 + per-pixel z-margin** — merges with
   reserve R3, hours of work. LOW risk.
2. **Flow-residual alignment**: run a generic pretrained optical-flow net
   (**SEA-RAFT**, ECCV 2024 — compliant: scene-agnostic pretrained, inference
   only, add to provenance table like SegFormer/VGGT) between each warped
   neighbor and the 3DGS render; apply the (small, ≤ a few px) residual flow to
   the warped pixels before blending. Attacks the exact ghosting the guard
   currently *discards* pixels over → guard rejections should drop, real
   texture coverage rise. This is GADA's "deformable offset" in a
   train-free form. MEDIUM cost, HIGH expected LPIPS effect.
3. **Median-depth multi-intersection warp points** (IBGS #1) — needs depth
   at multiple transmittance levels from the rasterizer; pairs naturally with
   the 2DGS/RaDe-GS depth pilot (§1a). MEDIUM cost.
4. **Learned aggregation** — subsumed by refiner v3 (§4): feed the refiner
   per-neighbor warped channels and let *it* learn the blend, rather than
   bolting a second net into DIBR.

## 4. P2 refiner (today: U-Net base48, 7ch [remap render | DIBR blend | vis-mask], residual, grader loss, LOO pairs, EMA, hflip TTA)

Our refiner is a 2018-era Deep-Blending recipe. SOTA-equivalents, in feasibility
order:

1. **Give it the evidence, not the verdict (refiner v3 = reserve R4 done right):**
   input per-neighbor warped images (K×3 ch) + per-neighbor confidence/depth-
   consistency maps + 3DGS depth, instead of the pre-blended 3-ch DIBR output.
   This *is* IBGS/GADA's learned aggregation, implemented inside the component
   we already trust. The net can then out-vote a wrong blend instead of merely
   polishing it. **Priority: HIGH** (pairs pipeline already caches neighbors).
2. **NAFNet blocks** (nonlinear activation-free; still the standard efficient
   restoration backbone through NTIRE 2025/2026 — used as the distill target in
   current challenge winners) replacing plain conv U-Net blocks at equal param
   budget. LOW cost, typically +0.2–0.5 dB on restoration tasks. Try with the
   same 6k/EMA/cosine recipe. **Priority: HIGH.**
3. **Seed/checkpoint ensemble** (reserve R2) + richer TTA (hflip is the only
   label-safe dihedral op given the measured top-corner asymmetry — do NOT add
   vflip blindly; test per-scene). LOW cost, ~+0.0005–0.001 each.
4. Transformer/SSM restorers (Restormer/X-Restormer, MambaIRv2 CVPR 2025):
   stronger on benchmarks but data-hungry; our per-scene LOO set is 100–240
   images. Only if v3 plateaus and we pool pairs across scenes
   (compliant — own provided imagery only, per rules v2 note). LOW priority.
5. Diffusion/reference-guided fixers (UniFixer 2026 etc.): the Difix evidence
   (exp015 −0.034) says our renders are too clean for generative fixers;
   grader-metric loss + PSNR term punishes hallucination. **Skip.**

## 5. Encode / packaging (today: single-encode q95 4:4:4 +optimize, 307 MB / 350 cap)

Engineering headroom, no research risk:
- **mozjpeg** (trellis quantization, tuned quant tables): ~5–10% better
  rate-distortion than libjpeg-turbo at equal quality setting → either smaller
  (headroom for q96/q97) or better SSIM/PSNR at the cap. LOW cost.
- **Per-scene/per-image quality allocation** (reserve R6): we have 43 MB of
  unused budget; allocate marginal MB by measured public-GT dScore/dMB curve
  (busy scenes gain more from +1q than quiet ones). LOW cost, strictly ≥ 0.
- JPEG XL / AVIF: check rules — submission format is JPEG; assume banned.

## 6. What probably happened above us (context for prioritization)

Top-1 +1.55 in two days after our own +1.26 public jump strongly suggests the
distortion-remap / real-pixel-reuse family is now discovered by multiple teams
(our exp030→exp034 LB trajectory was visible to everyone). If leaders are on an
IBGS-class learned-aggregation stack (published Nov 2025 — findable by any team
doing a literature pass), their remaining headroom is §3/§4's — same as ours.
Speed of iteration on the pilot ladder is now the differentiator.

## 7. Recommended execution order

| # | item | section | cost | expected Δ (public mean) |
|---|---|---|---|---|
| 0 | reserves R1/R2/R3/R6 (already specced) | FINAL_PLAN §4 | 1–2 days | +0.002–0.005 → **re-enter top-8** |
| 1 | refiner v3: per-neighbor channels + NAFNet blocks | §4.1–4.2 | 2–3 days (pilot 2 scenes) | +0.003–0.008 |
| 2 | DIBR: relative-τ filter, K=5, z-margin + SEA-RAFT flow-residual alignment | §3.1–3.2 | 2–3 days | +0.003–0.008 (LPIPS-heavy) |
| 3 | depth-source A/B: 2DGS (gsplat-native) depth for warp | §1a | 2 days (Kaggle) | +0.002–0.006 busy scenes |
| 4 | 3DGUT native distorted training pilot (HNI0131 + HCM0181) | §2 | 3–4 days (gsplat upgrade) | uncertain; removes 2 resample losses |
| 5 | MVGS / Pixel-GS backbone recipe | §1b | 2 days Kaggle | +0.001–0.003 |

Gates unchanged: every rung measured on public GT (grader metric, vgg+50)
before fleet scale-out; +0.002 promotion gate; per-scene fallback chain keeps
every submission monotone.

## 8. Compliance notes for new components

- SEA-RAFT / any optical-flow net: generic pretrained, inference-only → allowed
  under the same clause as SegFormer/VGGT; add provenance row before use.
- 2DGS/RaDe-GS/3DGUT/MVGS: architectures/rasterizers, trained from scratch on
  provided data only → no provenance issue; gsplat-family backends allowed.
- Refiner v3 / NAFNet: trained from scratch per scene on own (render, real)
  pairs from provided train images → compliant (same basis as refiner v1/v2).
- Cross-scene pair pooling (if ever needed for §4.4): allowed *only* with our
  own provided BTS scenes (rules v2 note bans external-scene pooling).

## Sources

- 3DGUT: https://arxiv.org/abs/2412.12507 · https://github.com/nerfstudio-project/gsplat/blob/main/docs/3dgut.md · https://research.nvidia.com/labs/toronto-ai/3DGUT/
- RaDe-GS: https://arxiv.org/abs/2406.01467 · https://github.com/HKUST-SAIL/RaDe-GS
- 2DGS: https://dl.acm.org/doi/10.1145/3641519.3657428
- IBGS: https://arxiv.org/abs/2511.14357 · https://hoangchuongnguyen.github.io/ibgs
- GADA: https://arxiv.org/abs/2607.00595 (ICML 2026)
- MVGS: https://arxiv.org/html/2410.02103
- AdpSplit densification: https://arxiv.org/pdf/2508.12313
- SEA-RAFT: https://link.springer.com/chapter/10.1007/978-3-031-72667-5_3
- NTIRE 2025/2026 restoration challenge reports (NAFNet/X-Restormer status): https://arxiv.org/html/2504.14600v1 · https://arxiv.org/pdf/2604.19445
- MambaIRv2 / UniRestore (CVPR 2025), UniFixer: https://arxiv.org/pdf/2605.12169
