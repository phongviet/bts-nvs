# Round-2 test set analysis (extracted 2026-07-16)

Data: `data/raw/VAI_NVS_DATA_ROUND2/` — 7 scenes, **no test GT anywhere**
(all LB-graded blind). From now on submissions cover ONLY this set.

## Scene inventory

| scene | type | train | test | res (render) | camera | k (distortion) | points3D | registered |
|---|---|---|---|---|---|---|---|---|
| HCM0421 | drone/BTS | 240 | 60 | 1320×989 | SIMPLE_RADIAL f=926.4 | **+0.00894** | 171k | 350 (50 extra) |
| HCM0539 | drone/BTS | 240 | 60 | 1320×989 | SIMPLE_RADIAL f=925.4 | **+0.00810** | 219k | 398 (98 extra) |
| HCM0540 | drone/BTS | 240 | 60 | 1320×989 | SIMPLE_RADIAL f=926.7 | **+0.00887** | 203k | 358 (58 extra) |
| HCM0644 | drone/BTS | 240 | 60 | 1320×989 | SIMPLE_RADIAL f=925.5 | **+0.00903** | 210k | 381 (81 extra) |
| HCM0674 | drone/BTS | 240 | 60 | 1320×989 | SIMPLE_RADIAL f=925.3 | **+0.00881** | 154k | 311 (11 extra) |
| bonsai | indoor handheld video | 248 | 28 | 1920×1080 | **SIMPLE_PINHOLE** f=1650 | **0 (none)** | 54k | 276 (0 extra) |
| chair | indoor handheld video, portrait | 205 | 58 | **720×1280** | **SIMPLE_PINHOLE** f=1114 | **0 (none)** | 80k | 263 (0 extra) |

Total test frames to submit: 5×60 + 28 + 58 = **386** (phase-1 private-8 was 480).

## Key findings

1. **Drone scenes = phase-1 regime, same campaign** (DJI captures Dec 29–30
   2024, same 1320×989 quarter-scale, centered principal point, k in the
   phase-1 public band +0.0081…+0.0090 — no HNI-style k=−0.115 outlier in
   this drop). Test poses in CSV omit k as before → **F1 distortion remap
   applies verbatim** with per-scene k from `cameras.bin` (expect the same
   ~+0.05-class gain over pinhole).
2. **bonsai + chair are a NEW regime**: handheld indoor video (named like
   benchmark scenes but NOT the Mip-NeRF360/Blender data — real phone/cam
   footage, `frame_XXXXXX.jpg`). SIMPLE_PINHOLE, k=0 → **remap is a no-op;
   skip it** (running it would only resample/soften).
3. **Interpolation regime is even stronger than phase 1**, especially
   indoor: test frames are interleaved video frames, 100% inside the train
   frame range; median nearest-train angular gap **4.3° (bonsai) / 3.3°
   (chair)**, nearest-dist frac 0.012–0.020, 100% with a near neighbor.
   Drone scenes: median gap ~8–11°, 88–98% near-covered, 62–93%
   sequence-adjacent → **DIBR + per-scene refiner should transfer; indoor
   scenes are prime DIBR territory** (nearly view-duplicate).
4. **Image-quality watchouts**: bonsai has real motion blur (varLap min 35,
   median 161 vs ~3000 for drone frames) and a glossy black tabletop
   (strong view-dependent reflection) → expect this to be the hard scene;
   sharpness-aware sampling (exp013 TPW) may matter again. chair is
   portrait 720×1280 — check no landscape assumptions anywhere in the
   pipeline (render_utils, DIBR warp, refiner, packaging).
5. **Extra registered frames** (11–98 per drone scene) have no photo files
   — same pattern as phase 1 (registrations from a denser capture). Loaders
   that filter by filename are unaffected; anything iterating `images.bin`
   blindly must keep filtering.
6. Sparse clouds are thin for the indoor scenes (54k/80k) → the dense-MVS
   init lever (exp002, +0.0037) is likely worth MORE there; drone scenes
   ~150–220k as before.
7. CSV intrinsics exactly match `cameras.bin` for bonsai/chair (f, centered
   c) → rendering at CSV intrinsics is geometrically exact for them.

## Implications for the pipeline (phase_run / builder)

- Scene list + per-scene k table need updating; remap stage must be
  **conditional** (drone: remap with own k; indoor: bypass).
- Mixed resolutions in packaging/budget math: 60×1320×989 ×5, 28×1920×1080,
  58×720×1280. Budget-fit logic should re-knapsack per scene (X6 tooling).
- Refiner val-split protocol (match-test) carries over unchanged; indoor
  scenes give it an even denser pose match.
- Training cost: bonsai at 1920×1080 is ~1.6× phase-1 pixel count → Kaggle
  sessions/VRAM plan accordingly.

Repro: this file's numbers come from a one-shot analysis script (coverage
via `src/utils/pose_utils.py`); preview contact sheet was rendered during
the session. Analysis date: 2026-07-16 evening.
