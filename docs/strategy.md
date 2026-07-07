# Winning the Viettel AI Race 2026 "BTS Digital Twin" NVS Competition: A Complete Method & Experiment Strategy

> **Authority note (updated per `Documents/plan_overall_v2.md`).** This document preserves the original SOTA analysis (Sections 1–6), which is still valid. The **action plan, key findings, and benchmarks below have been updated to v2**. The decisive v2 change: reading both the 2nd-place Qualcomm *and* the 1st-place XiaomiEV RealADSim-NVS reports shows two independent top-2 teams converge on the same conclusion — **a generative post-processing/supervision stage on top of plain 3DGS is the dominant lever; 3DGS backend variants are not.** v2 adds (a) a vision-only pseudo-point-cloud init (VGGT-style) as a third init arm, and (b) an in-training generative pseudo-GT pattern as a second way to apply the generative lever. See Section 7 for the consolidated additions. Day-by-day execution lives in `Documents/plan_week1.md` and `Documents/plan_week2.md`.

## TL;DR
- **Your winning path is a carefully-initialized, MCMC-densified 3D Gaussian Splatting pipeline (gsplat/Nerfstudio `splatfacto` with the `mcmc` strategy) plus per-image appearance embeddings, monocular-depth regularization, and sky/transient masking — NOT an exotic new method.** The 2nd-place Qualcomm solution to the near-identical ICCV 2025 RealADSim-NVS challenge (scoring formula S = 0.4·(PSNR/100) + 0.3·SSIM + 0.3·(1−LPIPS), 28 teams) concluded verbatim: "we found that all these alternatives do not outperform the standard 3DGS implementation, provided that the model was initialized carefully" (alternatives tested: Gaussians vs. triangles, default vs. MCMC densification, monocular depth + surface-normal priors, and separate sky/ground treatment).
- **The single highest-leverage lever is point-cloud initialization**: Qualcomm reported "simply applying the basic COLMAP pipeline yields ≈6× more points, that boost the overall score from 0.319 to 0.343." A dense MVS/monocular-depth-fused point cloud attacks BTS-specific pain (thin lattice towers, textureless sky, weak-texture metal) far more than swapping the core representation.
- **Because your score weights LPIPS at 0.4 (the single largest term), a light perceptual/diffusion enhancement stage (à la Difix3D+/SD-Turbo, trained with an LPIPS+SSIM+MSE loss on your own renders) is the biggest differentiator** once your geometry is solid — but validate it against your local split, since Qualcomm found such iterative refinement improved visuals but not always metrics and excluded it from their final submission.

## Key Findings

1. **This is a *medium-view* problem, not extreme sparse-view.** With 100–300 posed training images per scene and provided COLMAP, vanilla 3DGS already works; the sparse-view specialist methods (DNGaussian, SparseGS, FSGS, CoR-GS) are tuned for 3–24 views and their aggressive regularization can *hurt* at your view counts. Borrow their *ideas* (depth priors, floater pruning) not their full pipelines.
2. **Scoring math dictates strategy.** Final Score = 0.4·(1−LPIPS) + 0.3·SSIM + 0.3·PSNR_norm. LPIPS is the dominant term. PSNR is normalized and clamped, so beyond a threshold extra PSNR gives diminishing returns, whereas LPIPS improvements pay full weight. **Optimize preferentially for perceptual quality: sharp edges, no floaters, correct high-frequency texture.**
3. **Appearance variation is likely the biggest silent score-killer** for outdoor BTS captures shot over time/with auto-exposure. Per-image appearance embeddings (Splatfacto-W / GLO / exposure affine) are essentially free wins on SSIM/LPIPS/PSNR.
4. **MCMC densification (Kheradmand et al., NeurIPS 2024) is the best default densifier**: on Mip-NeRF360 with SfM initialization it achieves 29.89 PSNR / 0.90 SSIM / 0.19 LPIPS, surpassing 3DGS (SfM) at 29.30 / 0.88 / 0.21 at equal Gaussian count, is robust to initialization, and gives you an explicit `cap_max` knob. It is built into gsplat and is now the Nerfstudio splatfacto default.
5. **Sky and thin metal are the two BTS-specific failure modes.** Sky generates floaters and wastes Gaussians; thin lattice towers are under-covered by SfM points and get blurred away. Both are addressed by sky-masking + depth/normal regularization + dense init.
6. **(v2) Generative post-processing/supervision, not backend exotica, is where two independent top-2 teams found their biggest wins.** XiaomiEV's own ablation: baseline 3DGS → +road prior → +pseudo-LiDAR init moved LPIPS 0.513→0.470 (geometry-only, modest); their generative pseudo-GT supervision + diffusion temporal-artifact remover then moved it 0.470→0.288 — **>75% of their total LPIPS gain came from generative stages.** This calibrates how to spend Weeks 2–4: geometry/backend gets a solid floor; the generative stage gets the majority of the remaining points.
7. **(v2) There are two distinct ways to inject a generative model, attacking different failure modes.** *Post-hoc enhancement (Qualcomm):* train once, apply to every render after 3DGS training — cheap, decoupled, easy to gate/fall back from; the lower-risk starting point. *In-training pseudo-GT (XiaomiEV):* periodically render at progressively-extrapolated poses during training, refine with a generative "fixer," feed back as extra supervision — this changes the learned geometry (can fix a *missing* Gaussian, not just a blurry one), but is more complex and needs L1 down-weighted vs LPIPS to avoid hallucinated detail hurting pixel metrics. Both use the same single-step diffusion "fixer" (Difix3D+/SD-Turbo family) in two different *places* in the pipeline. See Section 7.

## Details

### 1. State of the Art — What to use and what to skip

**Core representation — use vanilla 3DGS as your backbone (Kerbl et al., SIGGRAPH 2023).** The official INRIA repo now integrates Taming-3DGS speedups, fused-SSIM, depth regularization (from their Hierarchical 3DGS work), anti-aliasing, and per-image exposure compensation. This alone is a strong, compliant baseline.

**Densification: 3DGS-MCMC (arXiv 2404.09591, NeurIPS 2024 Spotlight).** Reframes Gaussians as MCMC samples with SGLD noise, replacing clone/split heuristics with relocation + opacity-based sampling. On Mip-NeRF360 it reached 29.89 PSNR / 0.90 SSIM / 0.19 LPIPS vs 3DGS's 29.30 / 0.88 / 0.21 at equal Gaussian count, and is robust to random init. **This is your recommended densifier.** Canonical hyperparameters from the official ubc-vision repo: `scale_reg=0.01`, `opacity_reg=0.01` (0.001 for Deep Blending-type scenes), `noise_lr=5e5`. gsplat's `MCMCStrategy` defaults: `cap_max=1_000_000`, `noise_lr=5e5`, `refine_start_iter=500`, `refine_stop_iter=25000`, `refine_every=100`, `min_opacity=0.005`. For large outdoor scenes push `cap_max` to ~3M (Mip-NeRF360 outdoor scenes need ~1.2M–6.1M Gaussians); avoid setting it so high that most Gaussians end at zero opacity (wasted capacity — one gsplat user reported ~75% zero-opacity Gaussians at cap_max=5M).

**Anti-aliasing: Mip-Splatting (Yu et al., CVPR 2024).** Because your test poses carry per-pose intrinsics (fx, fy) that may differ from training focal lengths/distances, the 3D smoothing filter + 2D Mip filter directly prevent zoom/sampling-rate artifacts that would hurt both PSNR and LPIPS. **Strongly recommended given the given per-pose intrinsics.**

**Appearance/exposure: Splatfacto-W (arXiv 2407.12306) or exposure-affine.** Per-Gaussian neural color features + per-image appearance embeddings + SH-based background model; the authors report their method "improves the Peak Signal-to-Noise Ratio (PSNR) by an average of 5.3 dB compared to 3DGS" on in-the-wild collections. For milder auto-exposure drift, the INRIA repo's per-image exposure affine transform is a cheaper option. WildGaussians and SWAG are alternatives. **Caution on evaluation:** these methods require a test-time appearance embedding; the standard NeRF-W protocol optimizes the embedding on the left half of the test image and scores the right half. You cannot do that here (no GT). Either (a) use a fixed/averaged embedding at test time, or (b) rely on exposure affine that defaults to identity. Test this carefully on your local split.

**2D Gaussian Splatting (Huang et al., SIGGRAPH 2024) — situational.** 2DGS gives view-consistent geometry and better thin-surface/surface reconstruction, valuable for the metal lattice, but typically *slightly lower* PSNR than 3DGS (e.g., one transparent-object study: 3DGS 36.06 dB vs 2DGS 34.91 dB). Consider it only if floaters/geometry on the tower dominate your errors, or ensemble it. AA-2DGS adds anti-aliasing to 2DGS.

**Depth-regularized 3DGS.** FSGS (Pearson-correlation monocular depth loss), DNGaussian (global-local depth normalization), and the official Hierarchical-3DGS depth regularization all use monocular depth priors from a pretrained model (Depth Anything V2 is the community standard). The INRIA maintainers note depth regularization is "particularly effective on untextured areas (e.g., roads) and helps eliminate floaters" — exactly the sky and textureless-metal problem. **Recommended, using Depth Anything V2 as prior.**

**NeRF alternatives — Zip-NeRF still leads on raw quality.** Barron et al. report that "on the mip-NeRF 360 benchmark, Zip-NeRF reduces error rates by as much as 19% and trains 24× faster than the previous state-of-the-art," and up to 77% on the multiscale variant — it remains state-of-the-art on image-quality benchmarks and NeRF generalizes better to views far from training cameras (NeRF-SH renders novel far views with fewer artifacts and better depth than Splatfacto). But it is far slower to train/render and harder to iterate on across 3 phases. **Use Nerfacto/Zip-NeRF only as a possible ensemble member or for scenes with extreme extrapolation; 3DGS is the practical workhorse.**

**2025/2026 competition-winning technique — the decisive lesson.** Qualcomm AI Research's "Hybrid Gaussian Splatting for Novel Urban View Synthesis" (Omran et al., arXiv 2510.12308) took **2nd of 28** in the ICCV 2025 RealADSim-NVS challenge with an almost-identical scoring formula ("our proposal reaches an aggregated score of 0.432, achieving the second place overall"). Their two-stage pipeline: (1) carefully-initialized standard 3DGS at full resolution, 30k iterations, L1+SSIM loss, dynamic objects masked with Grounded SAM 2; (2) a **single-step diffusion enhancer built from SD-Turbo via Difix3D+**, which they "finetune from the public checkpoint for 40k iterations with a learning rate of 2e-5 without any schedule," at resolution 720×1280, trained on 36,231 corrupted/clean frame pairs by "minimizing a weighted sum of MSE, SSIM and LPIPS objectives" between enhanced renders and clean GT. Crucially they found MCMC, triangle-splatting, monocular-depth/normal priors, and sky/ground separation *did not beat* well-initialized vanilla 3DGS on their data, and that init (dense COLMAP) drove score from 0.319→0.343. Their optional iterative 3DGS refinement improved visuals but not metrics, so they excluded it.

### 2. Practical pipeline engineering

**Use the provided COLMAP, but improve it.**
- Keep the provided poses/intrinsics (do not re-solve poses — test poses are defined in that coordinate frame). But **densify the point cloud**: run COLMAP dense (`image_undistorter` → `patch_match_stereo` → `stereo_fusion`) to go from sparse to millions of points, or unproject Depth Anything V2 depth (scale-aligned to SfM points) to add points on textureless/thin regions. This is the single highest-ROI step, per the Qualcomm 0.319→0.343 result.
- Optionally refine intrinsics/extrinsics with a few bundle-adjustment iterations, but **never move the test cameras**.
- Random init + short NeRF pretraining for depth supervision is a documented alternative that matches or beats COLMAP init, but with good SfM available, dense COLMAP init is simpler.

**BTS-specific challenges:**
- *Thin metal lattice towers:* dense init points on the structure; enable scale regularization (PhysGaussian-style, in splatfacto via `use_scale_regularization`) to kill long spikey Gaussians; consider 2DGS for the tower. Anisotropic Gaussians represent thin structures but need seed points there. Train at higher resolution to preserve thin geometry.
- *Sky / textureless background:* generate sky masks with a segmentation model (OneFormer/SegFormer/Mask2Former) and either exclude sky from the loss, penalize sky-region opacity toward 0, or initialize a large-radius "sky sphere" of Gaussians (R3GS uses 10× the foreground radius via Fibonacci sampling). This removes background floaters — a major LPIPS/SSIM win. Splatfacto-W's SH background model is an alternative.
- *Reflective antenna dishes:* view-dependent color needs SH degree 3; these regions benefit from more training iterations and appearance modeling. Reflections are inherently hard; do not over-regularize them away.
- *Repeated structures (lattice cross-members):* can confuse SfM matching → depth regularization and MVS priors help stabilize geometry.
- *Transient objects (people, vehicles, moving equipment):* mask with Grounded-SAM/SAM2 and exclude from loss.

**Floaters/geometry (SSIM/PSNR) vs perceptual artifacts (LPIPS) — make the two objectives explicit:**
- *To raise SSIM/PSNR (geometric accuracy):* depth/normal regularization, floater pruning (SparseGS-style), opacity reset, MCMC relocation, dense init, sky masking. These reduce structural error and large blobs.
- *To lower LPIPS (perceptual quality):* add an LPIPS/perceptual term to the loss (or a diffusion enhancement stage as Qualcomm did), train at full resolution, anti-alias (Mip-Splatting), and preserve high-frequency texture (avoid over-smoothing from too-strong depth regularization). Note the documented tension: overly strong depth regularization (DRGS/DNGS) produces "overly smooth renderings" that can help PSNR but hurt perceptual sharpness. **Tune the depth-loss weight on your local split against the *combined* score, not PSNR alone.**

### 3. Hyperparameter / training experiment design

Standard practice and defaults to anchor on: **30,000 iterations** (INRIA 3DGS, gsplat, splatfacto, and the Qualcomm 3DGS stage all use 30k); **SH degree 3** (the standard for most scenes, captures view-dependent reflections without excessive model size); densification start iter 500, stop 15,000 (heuristic) / 25,000 (MCMC), densify every 100; opacity reset every 3,000; `densify_grad_threshold≈0.0002`. 7,000 iters gives a usable-but-lower-quality checkpoint. Quality improves up to ~30k with diminishing returns after.

Prioritized experiment plan (each toggled on a fixed local val split, measured on the *combined* competition score):
1. **Init ablation (highest priority):** provided sparse vs COLMAP-dense vs depth-unprojected dense. Expect the biggest single gain here.
2. **Densifier:** heuristic ADC vs MCMC (`cap_max` ∈ {1M, 2M, 3M}). Pick by score/VRAM.
3. **Appearance modeling:** none vs exposure-affine vs full appearance embedding. Verify the test-time embedding strategy doesn't backfire.
4. **Depth regularization weight sweep:** {0, 0.05, 0.1, 0.2}; watch the PSNR↑/LPIPS tension.
5. **Sky handling:** none vs mask-loss vs sky-sphere init.
6. **Anti-aliasing (Mip filter):** on/off — likely on given varying test intrinsics.
7. **Perceptual/LPIPS loss term or diffusion enhancer:** off vs on (weight ~0.05 as in feed-forward GS works). Highest LPIPS leverage.
8. **Scale regularization:** on/off for spikey Gaussians on the tower.
9. Secondary: SH degree (2 vs 3), opacity reset interval, LR schedule (splatfacto: means lr 1.6e-4→1.6e-6 exp-decay over 30k, opacities 0.05, scales 0.005), resolution (train at full res if VRAM allows).

### 4. Evaluation / validation strategy

- **Build a held-out local split:** since test GT is hidden, hold out ~10–15% of training images (e.g., every 8th image, the Mip-NeRF360 convention) as a local test set. **Critically, pick held-out views whose poses resemble the *distribution* of `test_poses.csv`** (similar distances/angles) so your local metric predicts the leaderboard. Compute LPIPS (match the organizer's likely backbone — LPIPS-AlexNet or VGG; Zhang et al. CVPR 2018), SSIM, PSNR, and **the exact combined formula** locally.
- **Never fit anything to test poses.** Do not optimize per-test-pose parameters (that crosses into the banned "manual per-pose editing" territory if it becomes image-specific).
- **Match the metric implementation** (LPIPS net, SSIM window, `psnr_max` for normalization) as closely as you can infer from the rules; small implementation differences shift absolute numbers.
- **Ensembling:** average or select among (3DGS-MCMC, 2DGS, Nerfacto) per scene by local score; or feed all renders into the diffusion enhancer. Simplest effective ensemble: per-scene pick the model with best local combined score. Multi-view/appearance ensembles help most on hard scenes.

### 5. Compute and tooling

- **Primary stack: Nerfstudio + gsplat.** gsplat is CUDA-accelerated, ~4× less memory and ~10–15% faster than reference 3DGS, supports MCMC, absgrad, anti-aliasing, and depth rendering out of the box. `splatfacto` = default (~6GB VRAM); **`splatfacto-big` = more Gaussians / higher quality (~12GB VRAM, ~15 min to 30k on a 3090)** — use splatfacto-big for final runs. splatfacto now defaults to the MCMC strategy (`strategy="mcmc"`, `cull_alpha_thresh=0.005`, `stop_split_at=25000`).
- **Official INRIA gaussian-splatting** (the sanctioned baseline) — keep a run for reproducibility and because top teams may be asked to prove results from a published pipeline. It supports depth reg (via `-d <depth path>`), anti-aliasing, and per-image exposure affine compensation.
- **Splatfacto-W** for appearance-in-the-wild; **2DGS/SuGaR** for geometry; **Mip-Splatting** repo for anti-aliasing; **Depth Anything V2** for depth priors; **Grounded-SAM 2 / OneFormer** for transient & sky masks; **VGGT** (or similar generic feed-forward geometry model) for the v2 pseudo-point-cloud init arm; **Difix3D+ / SD-Turbo** for the generative enhancement/pseudo-GT stage. All of these are generically pretrained and scene-agnostic — record provenance in `docs/rules_and_constraints.md`.
- **GPU/VRAM:** a single 24GB GPU (3090/4090/A5000) handles 100–300 images at full res with splatfacto-big. Training ~15–40 min/scene at 30k iters. Budget for many ablation runs — the cross-phase bottleneck is *experiments*, not single-run time. Reduce resolution or `cap_max` if you hit OOM.

### 6. Risk areas vs anti-cheating rules

**Clearly allowed:**
- Training 3DGS/NeRF only on the provided images + provided COLMAP. This is the intended task.
- **Generically pretrained, scene-agnostic foundation models as priors** — monocular depth (Depth Anything V2), segmentation (SAM/OneFormer), even a generic diffusion enhancer (SD-Turbo/Difix3D+) — are standard practice and are *not* "external data of the same objects/scenes." They were pretrained on unrelated public data and never see the competition's ground-truth. **This is the same interpretation the 2nd-place Qualcomm ICCV'25 team relied on.** *However*, the rule text bans "external data" — so document clearly that these models are scene-agnostic and were not fine-tuned on any competition/BTS imagery, and be prepared to justify it. If you fine-tune the enhancer, fine-tune it ONLY on your own renders of the provided training images (as Qualcomm did on challenge-provided sequences), never on scraped BTS imagery.
- Self-supervised pretraining on the provided images only.

**Risky / avoid:**
- Scraping any BTS/telecom-tower imagery, or using other public 3D/telecom datasets of similar objects — explicitly banned.
- Any per-test-pose manual editing, compositing, or retouching — banned and must be provable-clean. Keep the pipeline fully automatic.
- Test-time optimization that uses test images (you have none) — not possible, but also don't try to infer/guess GT.
- A diffusion enhancer that "hallucinates" content not supported by the scene risks both metric penalties and reproducibility scrutiny; keep enhancement conservative and gated (a common technique applies the LPIPS loss only when render-vs-generated LPIPS is below a threshold to avoid instability from hallucinations).

**Reproducibility artifacts to keep from Day 1** (top teams must submit training/inference code, configs, dependency versions, checkpoints, logs):
- Pin exact library/commit versions (gsplat, nerfstudio, CUDA, torch) in a lockfile/Dockerfile.
- Save all config files, random seeds, and command lines per run.
- Archive checkpoints and training logs (loss curves, Gaussian counts, iteration counts) for every submitted result.
- Keep the preprocessing scripts (COLMAP commands, masking, depth generation) versioned in git with commit hashes matching each submission.
- Log which pretrained model weights (and their public sources/versions) were used, to prove they're scene-agnostic.

### 7. Updates from `plan_overall_v2.md` (XiaomiEV report)

*(Sections 1–6 preserve the original analysis; these are the v2 additions.)*

**7.1 Vision-only pseudo-point-cloud initialization (XiaomiEV) — a third init arm.** Instead of (or alongside) COLMAP dense, run a generic feed-forward geometry model (VGGT-class, generically pretrained — *not* trained on telecom/tower data) to predict per-image depth/pose, recover absolute scale by aligning its predicted camera poses to your provided COLMAP poses (Umeyama), then unproject + aggregate into a unified cloud. It attacks a *different* failure mode than dense COLMAP: not "too few points" (which dense fixes) but "SfM structurally can't triangulate this surface from your capture pattern" (thin lattice, textureless metal/sky). **Ablate it as a complement, not a replacement** — one is triangulation-based, the other a learned prior. Init is now a **three-way** comparison (sparse / dense COLMAP / VGGT-style) plus an optional (dense+VGGT) merge.

**7.2 Two flavors of generative integration.** Post-hoc enhancement (Qualcomm) is the simpler, lower-risk start; in-training pseudo-GT (XiaomiEV) is the higher-ceiling, more complex second step. Same tool (single-step diffusion fixer), two pipeline locations — see Key Findings 6–7.

**7.3 Road/temporal-specific machinery stays deprioritized.** XiaomiEV's 2D-SDF planar road prior and TIA-Net temporal-invariance network solve street-driving problems (flat roads, multi-traversal lighting drift) that don't map onto static BTS captures. **Do not implement these.** They are the genuinely non-transferable parts of XiaomiEV's pipeline, unlike the pseudo-cloud init and pseudo-GT ideas.

**7.4 Compliance — smaller compliant data pool (important).** You **cannot** pool external scenes the way Qualcomm did (25+79 external sequences) or use an external multi-traversal dataset the way XiaomiEV did (public Para-Lane for TIA-Net) — both are external data of similar scenes, barred by the rules. All generative-model fine-tuning pairs must come **strictly from your own provided BTS training images** (hold out a val subset, render 3DGS at those poses, pair with the real photos, pool across your Phase-1 scenes). This is a real data-scale disadvantage: expect to need **more conservative gating** (lower LPIPS-loss weight, or apply only where render-vs-real LPIPS exceeds a threshold) than either paper's numbers suggest. The generic feed-forward geometry model (VGGT) and the diffusion fixer sit in the same compliance bucket as Depth Anything V2 / SAM — document their provenance in `docs/rules_and_constraints.md`.

**7.5 The LPIPS/PSNR tension, restated with both reports agreeing.** Qualcomm's masked-loss ablation and XiaomiEV's explicit L1/LPIPS down-weighting are the same warning from two directions: generative supervision (either form) improves LPIPS but can hurt PSNR/SSIM via hallucinated detail. Both teams weight the loss toward LPIPS/perceptual terms and away from pixel-exact L1 wherever generative content enters.

## Recommendations — Phase 1 action plan (Jul 2 – Jul 30, 2026, ~4 weeks)

**Week 1 — Baseline + infrastructure.**
- Set up Nerfstudio + gsplat and the official INRIA repo in a pinned Docker image. Establish the git/log/checkpoint discipline now.
- Get vanilla `splatfacto`/3DGS running on provided data, produce a valid submission ZIP (correct folder/filename/dimension/count — a malformed or scene-count-mismatched ZIP voids scoring). **Submit early to confirm the pipeline end-to-end.**
- Build the local held-out validation split matching test-pose distribution; implement the exact combined-score metric locally.

**Week 2 — Initialization arms + backend floor.** *(v2 — detailed in `Documents/plan_week2.md`.)*
- Run the init ablation across **three arms**: (a) provided sparse COLMAP, (b) dense COLMAP (`image_undistorter → patch_match_stereo → stereo_fusion`), (c) VGGT-style pseudo-point-cloud scale-aligned to your COLMAP poses. Try a (b)+(c) merge if either single arm underperforms on BTS pain points (thin lattice, sky). Adopt the winning init (global or per-scene).
- Lock the backend on the winning init: `splatfacto-big` + MCMC (`cap_max` sweep), SH degree 3, 30k iters, Mip anti-aliasing, sky masking, tower scale regularization. A/B each feature.
- Build the pooled (3DGS render, real photo) pair set from the internal val split across all Phase-1 scenes, masked on both sides — this is the input Week 3 consumes.

**Week 3 — Generative stage: post-hoc enhancer first, in-training pseudo-GT as stretch.** *(v2)*
- Get an unmodified off-the-shelf Difix3D+/SD-Turbo fixer running on your renders (confirm plumbing), then finetune (LoRA on decoder, or full UNet if compute allows) on your pooled in-house pairs. Track LPIPS/SSIM/PSNR **pre- and post-enhancer, per scene** — first-class given the LPIPS/PSNR tension. Start conservative (favor LPIPS, down-weight L1) and **gate per-scene** if it regresses PSNR/SSIM.
- Stretch goal only if the post-hoc enhancer is stable: prototype XiaomiEV's in-training pseudo-GT on 2–3 scenes with a *progressive* pose-interpolation schedule (train poses → test poses, not a jump), down-weighted L1.

**Week 4 — Decide, scale, validate compliance, submit.** *(v2)*
- Decide post-hoc-only vs post-hoc + in-training pseudo-GT by which scored better *and more consistently* across scenes — don't carry an unstable component that only won on 2–3 scenes.
- Run the full locked pipeline across every scene; finalize `docs/rules_and_constraints.md` provenance for every generic pretrained model; verify reproducibility; package/submit; keep a per-scene "backend-only, no generative stage" fallback.

**Benchmarks / thresholds that change the plan (v2):**
- If the VGGT-style pseudo-cloud arm doesn't beat dense COLMAP by a meaningful margin, drop it — don't carry two init pipelines forward for a marginal gain.
- If appearance embedding *hurts* local score (bad test-time embedding), fall back to exposure-affine or drop it.
- If depth-reg weight >0.1 starts lowering LPIPS (over-smoothing), cap it low.
- If the post-hoc enhancer improves local combined score by <0.003 or introduces hallucinations, exclude it per-scene (as Qualcomm excluded their iterative refinement) — but if it improves LPIPS meaningfully it's your biggest edge given the 0.4 weight.
- If the in-training pseudo-GT stretch goal doesn't clearly beat post-hoc-only by Week 3, don't pursue it further in Phase 1 — revisit in Phase 2.
- If floaters/geometry dominate errors on the tower, promote 2DGS to primary for those scenes.

## Caveats
- **The exact scoring implementation details matter and are partly unknown**: the LPIPS backbone (AlexNet vs VGG), the SSIM window, and the `psnr_max` normalization constant all shift absolute numbers. Confirm from the official rules; my local-metric advice assumes you replicate their implementation.
- **The Qualcomm result is from an *urban driving* NVS challenge**, not telecom towers; its lessons (init dominates, enhancer helps LPIPS, exotic methods underperform) transfer well but are not guaranteed on BTS geometry. Their finding that MCMC/depth-priors didn't help was *on their data* — you must re-verify on yours, since other papers show clear depth-prior gains on textureless scenes.
- **Appearance-embedding evaluation is genuinely tricky without test GT**; the standard left-half-optimization protocol is unavailable, so there's real risk these methods behave unpredictably at test poses. Treat as an experiment, not a guaranteed win.
- **Some cited works are very recent (2025–2026 arXiv preprints)** and not all peer-reviewed; treat method claims (especially reported metric deltas) as indicative, not definitive.
- **Diminishing returns and metric trade-offs are real**: pushing PSNR past the normalization ceiling wastes effort; the 0.4 LPIPS weight means perceptual quality is where marginal effort pays most.
- **(v2) Your compliant pair/data pool for generative-model finetuning is structurally smaller than either paper's** (no external scene pooling permitted), so their reported finetuning score deltas are optimistic upper bounds — plan Week 3's gating and conservatism accordingly rather than assuming the same magnitude of gain.