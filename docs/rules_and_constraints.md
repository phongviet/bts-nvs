# Rules & Constraints Summary (fill in from official competition site)

Status: **DRAFT / INCOMPLETE** — fill in from `Documents/AI Race 2026 - ....html`
and any rules PDF/page before the Day-5 submission. Everything below marked
CONFIRM must be verified before trusting `package_submission.py` or `metrics.py`.

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

## Submission format (CONFIRM before Day 5)
- CONFIRM: required ZIP folder structure (per-scene subfolder? flat?).
- CONFIRM: required filename convention (must match `test/images/*` names exactly, including case).
- CONFIRM: required image format/extension (JPG vs PNG) and whether dimensions must exactly match `width,height` from `test_poses.csv`.
- CONFIRM: whether all 13 scenes must be present in every submission, or per-scene submissions are allowed.

## Anti-cheating rules (from plan_overall.md research — CONFIRM against official text)
- Allowed: training 3DGS/NeRF only on provided images + provided COLMAP; generically pretrained scene-agnostic foundation models (depth/segmentation/diffusion priors) as long as not fine-tuned on competition/BTS imagery.
- Banned: scraping BTS/telecom imagery or other public 3D/telecom datasets of similar objects; any per-test-pose manual editing/compositing/retouching; test-time optimization using test images.
- Top teams must be able to submit training/inference code, configs, dependency versions, checkpoints, logs — see `docs/reproducibility_checklist.md`.

## Action items
- [ ] Read the official rules doc/site in full, replace every CONFIRM above with the actual value.
- [ ] Update `src/metrics.py --psnr-max` default once confirmed.
- [ ] Update `src/package_submission.py` layout once confirmed.
