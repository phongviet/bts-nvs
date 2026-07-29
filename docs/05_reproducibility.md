# 05 — Reproducibility & the Kaggle workflow

All compute runs on **Kaggle** (free T4×2 sessions, 12 h wall, ~30 h/week GPU
quota per account). Everything heavy (checkpoints, renders, dense point clouds,
submission zips, raw data) is gitignored and regenerable from the commands here.

## Environment

```bash
conda env create -f environment.yml      # creates the `airace` env
conda activate airace
```

From scratch:

```bash
conda create -n airace python=3.10 -y && conda activate airace
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nerfstudio gsplat
pip install lpips torchmetrics pillow numpy pandas tqdm pyyaml opencv-python plyfile
```

Pinned deps for the Difix env (only if reviving that arm — it is CLOSED):
`diffusers==0.25.1 transformers==4.38.0 peft==0.9.0 huggingface-hub==0.25.1
accelerate==0.27.2`, in a throwaway `--system-site-packages` venv off airace.

## Data layout

Extract the organizers' archives into `data/raw/` (read-only, never hand-edited):

```
data/raw/phase1/public_set/<scene>/{train,test}/...     # test GT present → local bench
data/raw/phase1/private_set1/<scene>/...                # test GT withheld (Round 1)
data/raw/VAI_NVS_DATA_ROUND2/<scene>/...                # Round 2 — the only target
```

**Always filter the COLMAP model first** to avoid the test-leak
([01_competition](01_competition.md)):

```bash
python src/data_prep/filter_colmap_train.py --scene <scene> ...
python src/data_prep/build_dense_colmap.py  --scene <scene> ...   # dense MVS init
```

## Reproduce the best result (v7a, LB 75.3793)

The whole 7-scene fleet is driven by one idempotent/resumable script. On Kaggle,
run **2 scenes per T4×2 session** (`PARALLEL=2`), ~3 sessions total.

```bash
# Per scene (drone example): backbone → F1 → DIBR → refiner, adv two-stage
python Analysis/kaggle_exp034_fleet.py \
    --phase round2 --scenes <scene> \
    --suffix _v2 --blocks naf --evidence --adv 0.003
```

Key per-scene settings baked into the driver: `splatfacto-big`, 30k,
`--rasterize-mode antialiased`, **`--downscale-factor 1`** (all 7); F1 per-scene
`k` for the 5 drones and **none** for `bonsai`/`chair`; DIBR ss=2 + cubic;
refiner `base=48`, 6k iters, EMA 0.999, hflip TTA, `--max-pairs 90`. `bonsai`
uses the SSS render-channel override (`--big-scenes`/SSS path — see
[06_sss_backbone](06_sss_backbone.md)). `--adv > 0` runs stage 1 for the warm
ckpt then stage 2 (both share the warp variant, so stage 2 reuses stage 1's pair
cache).

**Package the submission:**

```bash
python Analysis/24_build_round2_submission.py --suffix _v2
# floor q98 4:4:4, pooled knapsack, BUDGET_MIB 348, per-scene fallback chain
# → submissions/round2/<versioned>/submission_round1.zip  (≤ 350 MiB)
```

## The Kaggle fleet loop

1. Build the upload dataset: `scripts/build_kaggle_round2_upload.py`
   (`--scenes` filter; carries the fleet driver + each scene's staging + dense
   init, ~0.24–0.44 GB). Upload as a Kaggle dataset, note the slug.
2. Edit `SCENES` (and `--big-scenes`/`BIG_SCENES` for SSS) in the notebook
   `kaggle/kaggle-round2-fleet.ipynb`, set the `DATASET` slug, Run All.
3. Download the output zip, reintegrate each scene's refined renders + ckpt into
   `Analysis/X5_refiner/<scene>/`, verify counts / dims / stem-map.
4. When 7/7 are collected, run the packager above → submit.

**Quota discipline:** ~30 h/week/account, refreshing 2026-08-01 (after the
deadline). Run 2-scene sessions on T4×2 `PARALLEL=2` (~6 h, not ~11 h serial).

## Packaging rules (enforced by the packager)

- ONE zip named `submission_round1.zip`; always `--out` a **versioned** dir,
  never overwrite an old submission.
- Filename = the `image_name` column of `test_poses.csv` (never append a literal
  extension); size = exact `width,height` from that row. Extension case is MIXED
  (drone `.JPG`, indoor `.jpg`) — the builder restores names from the CSV.
- Budget in **MiB (2²⁰)**: build at a q98 floor; the cap is 350 MiB.

## Reproducibility checklist (before uploading any ZIP)

- [ ] `config_resolved.yaml`, `git_commit.txt`, `env_freeze.txt` in the run folder
- [ ] Checkpoint used for the submitted renders is saved and matches the render `--config`
- [ ] Metrics regenerated from the same checkpoint immediately before packaging
- [ ] Submission row added to [04_results](04_results.md) before uploading
- [ ] No manual edits under any `renders*/` dir
- [ ] Only `data/raw/<phase>/` used — no external / scene-specific fine-tuning
- [ ] Every generic pretrained model has a provenance row ([01_competition](01_competition.md))
- [ ] Scene-folder casing and every required test image (correct dims) present

## Gotchas that cost real time (do not relearn)

**Code / training**

- **nerfstudio auto-downscales any image with long side > 1600 px, then prompts
  interactively** → EOFError in a subprocess, dies in seconds, looks exactly like
  OOM. **Always pass `--downscale-factor 1`.** (bonsai 1920×1080 triggers it.)
- **`train_staging_holdout` must be uploaded** — `fix_paths` in `04` silently
  falls back to `train_staging_dense`, pairing a 223-image backbone with
  248-image staging → wrong dataparser transform → a control arm scored 0.6024
  instead of 0.6913. Provenance print + `.owner` assert added.
- **Refiner arms MUST run serially** — they share
  `X5_refiner/<scene>/depth_cache<tag>` keyed only by warp variant; parallel arms
  corrupt each other's `.npy`. (Different *scenes* parallelize fine.)
- Pair caches dominate disk: **5–13 GB per arm** (the 20-ch evidence stack is ~3×
  the 7-ch); Cell 2 disk guard sizes them and symlinks `X5_refiner` → `/tmp`.
- **Test-frame extension case is MIXED** (`.JPG` drone / `.jpg` indoor) — restore
  names from the pose CSV, never by appending.
- `--load` handles both refiner architectures (regression `d1.net.0.weight` and
  naf `d1.proj_in.weight`); broken symlinks are unlinked before repointing.

**Kaggle workflow**

- **Re-upload the notebook AND diff it against the local file before any
  multi-hour session.** A stale uploaded notebook is invisible until it fails —
  and it fails at whichever step you last edited (for a bug found by reading,
  always the end). A Difix session lost ~10 h this way.
- **No expensive artifact may live only in `/tmp`.** Write adapters/outputs to
  `/kaggle/working/...` and set session Persistence = "Files only", so a hard
  kill costs ≤ one eval interval, not the whole run.
- One T4×2 session ≈ 12 h wall. Package Stage 1 in its own cell **before**
  attempting any Stage 2; gate Stage 2 on remaining wall-time so a mistimed wall
  can never cost Stage 1's output.
