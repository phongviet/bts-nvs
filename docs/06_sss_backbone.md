# 06 — SSS backbone ablation (`sss_experiment/`)

3D **S**tudent **S**platting and **S**cooping (SSS, CVPR'25, realcrane repo) vs
`splatfacto` as the backbone, scored with `src/metrics.py`
(LPIPS-vgg, `psnr_max=50`). The experiment code lives in `sss_experiment/`
(moved into this repo 2026-07-28); heavy artifacts (run tarballs, trained plys /
ckpts, COLMAP binaries, the vendored upstream repo) are gitignored.

## Verdict

**Do NOT adopt SSS as the global backbone.** At paper scale (cap_max 2 M, 40k
iters) it **ties** `splatfacto-big` for ~2× the training time — a wash, not a
loss. It ships in production for exactly **one** scene: `bonsai`.

## Paper-scale public-scene A/B (Kaggle T4×2, 2M / 40k)

| scene | SSS 2M/40k | splatfacto-antialiased | Δ | splatfacto-big 30k | Δ |
|---|---|---|---|---|---|
| hcm0031 | 0.6440 | 0.6438 | +0.0002 | — | — |
| hcm0034 | 0.6541 | 0.6548 | −0.0007 | 0.6569 | −0.0028 |
| HCM0181 | 0.6385 | 0.6338 | **+0.0047** | 0.6367 | +0.0018 |
| HCM0193 | 0.6435 | 0.6418 | +0.0017 | — | — |
| **mean** | | | **+0.0015** (4 sc) | | **−0.0005** (2 sc) |

- SSS's one clear win is the **busy/cluttered HCM0181** (Student-t heavy tails
  help complex geometry). Its loss is concentrated in **LPIPS**, its win in
  **SSIM** — a net loss under the grader's 0.4 LPIPS weight.
- The earlier "SSS loses by 0.016" was a **VRAM-starved artifact** (cap_max 600k
  on a 6 GB card); ~0.013 of that gap was capacity starvation. *Never issue a
  backend verdict from a VRAM-constrained proxy.*

## Why SSS ships on `bonsai` (and only `bonsai`)

SSS scoreboard across the round-2 scenes it was tried on: **bonsai +0.296 ·
chair −0.133 · HCM0674 −0.266**. The one win is the one scene whose splatfacto
baseline was **capacity-starved** (bonsai 0.25 gauss/px, least dense of all 7;
chair 0.89 and its `splatfacto-big` vs `splatfacto` A/B was an exact tie, proving
capacity is not chair's constraint). ⇒ **SSS helps only where the splatfacto
baseline is capacity-starved.** bonsai's production render channel is therefore an
SSS backbone (60k iters, cap 4 M = 4,000,001 gaussians, `-r 1`, nu 100,
C_burnin 5e5, C 120, burnin 7k); every other scene stays on `splatfacto-big`.

## Setup notes / gotchas

- Env: conda `sss` (torch 2.4.1+cu121). On Kaggle the base image ships an
  incompatible torch → provision a pinned env at `/opt/conda_sss`; build with
  `--no-build-isolation` and pre-patch `simple_knn.cu` with `#include <cfloat>`.
- Repo default `eval=True` silently holds out every 8th image → flip to False
  (full data bought +0.002).
- SSS needs **PINHOLE** input: `make_undistorted_scene.py` cv2-undistorts the
  `SIMPLE_RADIAL` train images (K unchanged) and filters `images.bin` to train
  frames only (the raw bin registers test frames → crash + leak risk).
- `render_test_csv.py` renders `test_poses.csv` (COLMAP w2c) from the SSS ply.
- **Stability fix (why runs complete at all):** post-burn-in SGHMC momentum
  diverges → NaN positions bypass the frustum cull → rasterizer tile-index
  corruption. The clamp/`nan_to_num` guard on `_xyz`, `_scaling`, and the
  momentum buffer must run **every iteration** (every-10 was not enough), plus
  rolling checkpoints + a 4-attempt auto-resume loop in the fleet driver.
- Timing: ~1.6 h / 30k at 600k components; ~4.3 h / 40k at 2 M (T4).

## Directory contents

- `kaggle_sss_fleet.py`, `build_kaggle_sss_upload.py` — the Kaggle fleet driver
  and upload builder.
- `make_undistorted_scene.py`, `render_test_csv.py`, `metrics.py` — pinhole prep,
  test-CSV rendering, scoring.
- `pod/` — provisioning + run scripts retained for reference (production SSS runs
  now go through the Kaggle fleet driver).
- `3D-student-splatting-and-scooping/` — vendored upstream repo (gitignored).
- `prod_out/`, `data/`, `*.tar.gz` — trained artifacts (gitignored, regenerable).
