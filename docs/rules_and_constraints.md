# Rules & Constraints Summary (fill in from official competition site)

Status: **DRAFT / INCOMPLETE** — fill in from `Documents/AI Race 2026 - ....html`
and any rules PDF/page before the Day-5 submission. Everything below marked
CONFIRM must be verified before trusting `package_submission.py` or `metrics.py`.

## ⚠️ Critical data gotcha (found Week 1, Day 2)
The provided `train/sparse/0/images.bin` contains poses for the scene's
**full capture**, not just the 240 train images: for `hcm0034` it has 337
posed entries = 240 train + 60 test + 37 extra registration-only frames.
Nerfstudio's default `ColmapDataParser` (`eval_mode="interval"`) splits
across ALL of images.bin, which **leaks the held-out test images straight
into training** if used unmodified. Always run
`src/data_prep/filter_colmap_train.py` first (writes a train-only filtered
sparse model to `data/processed/.../colmap_train_only/` + a staging dir with
symlinked `images/`+`sparse/0/`), and train with
`--pipeline.datamanager.dataparser.eval-mode=all` on that staging dir.
Verified zero leakage on `hcm0034` after filtering.

## Data (confirmed from provided archive)
- Phase 1: 13 scenes total — `public_set/` (5: hcm0031, hcm0034, HCM0181, HCM0193, HCM0204)
  and `private_set1/` (8: HCM0249, HCM0254, HCM0276, HCM1439, HNI0131, HNI0265, HNI0366, HNI0437).
- Per scene: 240 train images / 60 test images (80/20), images pre-scaled to 1/4 original size.
- `test_poses.csv`: `image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height` — per-pose intrinsics + desired render resolution.
- `public_set` test GT images are included in the archive (`test/images/`) — usable for local score reconciliation.

## Scoring (CONFIRM exact constants)
- `Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm` — confirmed formula shape from `plan_overall.md` research.
- CONFIRM: LPIPS backbone (AlexNet vs VGG).
- CONFIRM: PSNR normalization constant/clip range (`metrics.py` currently defaults `psnr_max=40.0` as a placeholder).
- CONFIRM: SSIM window size / implementation (skimage default `win_size=11` used currently).

## Submission format (confirmed from "AI Race 2026 - Submission" rules page, Round 1)
- ZIP named `submission_round1.zip`, one subfolder per scene, one image per required test pose:
  `submission_round1.zip / <scene>/ <image_name> ...`. The rules page shows illustrative
  `scene_001/0001.png` names, but the actual requirement is **filename = `image_name` column
  from that scene's `test_poses.csv`** (real files are e.g. `DJI_..._V.JPG`) and **image size
  = exact `width,height` from that same CSV row** — both verified locally before packaging
  (see `submissions/phase1/SUBMISSION_LOG.md`).
- All scenes' images must be present in one ZIP — missing any pose for any scene "sẽ ảnh hưởng
  đến kết quả" (affects scoring), i.e. all 13 scenes required per submission, not per-scene.
- Submission limits: 5 uploads/day, 600s cooldown between them, scored on GPU infra. System
  keeps the **last** submission before the round deadline (2026-07-30), so late re-submits are
  safe as long as they land before the cutoff.
- Round 1 data: 150-300 train images/scene, 40-70 test poses/scene, 200-300MB/scene.
- CONFIRM still open: exact LPIPS backbone, PSNR normalization constant, SSIM window (see
  Scoring section above) -- the submission-format doc above doesn't cover scoring internals.

## Anti-cheating rules (from plan_overall research — CONFIRM against official text)
- Allowed: training 3DGS/NeRF only on provided images + provided COLMAP; generically pretrained scene-agnostic foundation models (depth/segmentation/geometry/diffusion priors) as long as not fine-tuned on competition/BTS imagery.
- Banned: scraping BTS/telecom imagery or other public 3D/telecom datasets of similar objects; any per-test-pose manual editing/compositing/retouching; test-time optimization using test images.
- **(v2) No external-scene pooling for generative-model finetuning.** Qualcomm pooled 25+79 external sequences and XiaomiEV trained TIA-Net on the public Para-Lane multi-traversal dataset — both are external data of similar scenes and are **not compliant to reproduce**. Every (render, real) pair used to finetune any enhancer/pseudo-GT model must come **only from our own provided BTS training images** (hold out a val subset, render 3DGS at those poses, pair with the real photos, pool across Phase-1 scenes). This is why XiaomiEV's TIA-Net is deprioritized: irrelevant to BTS geometry *and* its training recipe isn't compliant.
- Top teams must be able to submit training/inference code, configs, dependency versions, checkpoints, logs — see `docs/reproducibility_checklist.md`.

## Pretrained-model provenance (generic / scene-agnostic — fill in as each is first used)
Every generically-pretrained model must be documented here with checkpoint source + training corpus + license, and an explicit statement it was **not** trained/finetuned on any competition/BTS imagery. Required before Week-4 compliance pass.

| Model | Role | Used from | Checkpoint / source | Training corpus (generic?) | License | CONFIRM |
|-------|------|-----------|---------------------|----------------------------|---------|---------|
| Depth Anything V2 | Monocular depth prior | Wk2/3 | — | generic (not BTS) | — | [ ] |
| SAM / Grounded-SAM 2 / OneFormer | Sky & transient masks | Wk2 | — | generic (not BTS) | — | [ ] |
| SegFormer-B0 (ADE20K) | Sky masks (Wk2 backend A/B) **and** transient masks for mask-out-of-loss training (Wk3, exp008) | Wk2 | `nvidia/segformer-b0-finetuned-ade-512-512` on HuggingFace Hub via `transformers.pipeline("image-segmentation")`, used strictly off-the-shelf for inference (no gradient updates) in `src/data_prep/build_sky_masks.py` and `src/data_prep/build_transient_masks.py` (tiled 2×3 inference) | ADE20K (generic scene-parsing dataset; not BTS) | Apache 2.0 (per HF model card, CONFIRM) | [ ] confirm license |
| VGGT-1B (Meta AI, Wang et al. CVPR 2025) | Pseudo-point-cloud init arm (v2, arm c) | Wk2 | `facebook/VGGT-1B` on HuggingFace Hub via `VGGT.from_pretrained`; repo `github.com/facebookresearch/vggt` vendored read-only under `third_party/vggt/`, unmodified | Generic multi-view/SfM-style training corpora per the VGGT paper; not trained/finetuned on BTS/telecom imagery. Used strictly off-the-shelf for inference (no gradient updates) in `src/data_prep/build_vggt_init.py` | CC BY-NC 4.0 (Meta, non-commercial research) | [ ] confirm license compatible with competition use |
| Difix3D+ / SD-Turbo | Diffusion "fixer" (post-hoc enhancer, exp015/016) | Wk3 | `nvidia/difix` on HuggingFace Hub via `diffusers.DiffusionPipeline.from_pretrained(..., trust_remote_code=True)` in `src/enhancer/run_difix.py`; single-step SD-Turbo-style img2img backbone. LoRA finetune (peft, UNet only) in `src/enhancer/train_difix_lora.py` | Generic image corpora per the Difix3D+ paper / SD-Turbo lineage; not BTS. **LoRA finetuned ONLY on our own (render, real) pairs** built from the organizers' provided train images by `src/enhancer/build_enhancer_pairs.py` (manifest.csv is the provenance record; no external-scene pooling per the v2 rule above) | NVIDIA license per HF model card (CONFIRM — check non-commercial terms) | [ ] confirm license |
| LPIPS backbones (AlexNet, VGG-16) | Perceptual metric (`alex`, local eval in `src/metrics.py`) and perceptual loss (`vgg`, exp009 `src/models/splatfacto_perceptual.py`) | Wk1/Wk3 | `lpips` pip package and `torchmetrics` LPIPS, both pulling torchvision ImageNet-pretrained backbones; frozen, no gradient updates to the backbones | ImageNet (generic; not BTS) | BSD (lpips repo) / torchvision BSD-3 | [ ] |

## Action items
- [ ] Read the official rules doc/site in full, replace every CONFIRM above with the actual value.
- [ ] Update `src/metrics.py --psnr-max` default once confirmed.
- [ ] Update `src/package_submission.py` layout once confirmed.
- [ ] Fill the pretrained-model provenance table as each model is first used (VGGT/Depth Anything V2/SAM in Week 2; Difix3D+/SD-Turbo in Week 3).
- [ ] Week-4 compliance pass: confirm all provenance rows complete + explicit statement that all finetuning data came only from our own provided BTS images (no Para-Lane-style external pooling).
