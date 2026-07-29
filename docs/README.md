# bts-nvs documentation

Consolidated documentation for the **Viettel AI Race 2026 — BTS Digital Twin
(Novel View Synthesis)** entry. Everything the competition, the method, the
experiments, and the reproduction path needs lives in these files. All compute
in this project runs on **Kaggle** (free T4×2 GPU sessions) — no other
infrastructure is assumed.

| file | contents |
|---|---|
| [`01_competition.md`](01_competition.md) | Task, grader math, dataset (Round 1 & Round 2), submission format, compliance / anti-cheating rules, pretrained-model provenance |
| [`02_pipeline.md`](02_pipeline.md) | The shipped winning pipeline, stage by stage: backbone → F1 remap → DIBR → neural refiner → encode/package |
| [`03_experiments.md`](03_experiments.md) | Full experiment log and "lever graveyard" — everything tried, what won, what was refuted, and the lesson kept |
| [`04_results.md`](04_results.md) | Leaderboard history, submission log, per-metric calibration, and the current best result |
| [`05_reproducibility.md`](05_reproducibility.md) | Environment setup, exact commands to reproduce the best result, the Kaggle fleet workflow, packaging rules, and hard-won gotchas |
| [`06_sss_backbone.md`](06_sss_backbone.md) | The 3D Student-Splatting-and-Scooping (SSS) backbone ablation and its production use on `bonsai` |

## One-paragraph state (2026-07-28)

Round 2 is the only submission target (deadline **2026-07-30**). Best submission
= **v7a, LB 75.3793** (`submissions/round2/round2_v7a_all_drones_adv/`, md5
`9375ced6`). The competition is a view-**interpolation** problem: the biggest
levers were (1) correcting the camera distortion model (F1 remap, +16.4 LB),
(2) reprojecting real train pixels via 3DGS depth (DIBR, +1.8 LB), and (3) a
per-scene neural refiner trained against the grader objective (+3.2 LB). Every
downstream lever family (refiner adversarial weight, warp-side DIBR, encode
knapsack, SSAA, post-processing, and backbone-side perceptual/depth supervision)
has since been measured and **closed**. The remaining top-8 gap is entirely in
the two indoor scenes (`bonsai`, `chair`); no measured lever closes it before the
deadline. See [`04_results.md`](04_results.md) for the standing decision.
