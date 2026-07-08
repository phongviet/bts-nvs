# Phase-2/3 runbook — raw ZIP → submission, ≤2 h/scene

Everything a phase drop needs, in order. No hand-state: every stage reads
committed configs and is resumable (rerunning a scene skips completed
stages). Owner at phase start: P1 runs the fleet, P2 handles packaging and
upload, P3 spot-checks renders visually before packaging.

## 0. Before the phase drops (do in Phase-1 downtime)
- [ ] `configs/phase_locked.conf` frozen to the exp017 final config (method,
      iters, extra args, mask on/off). This file IS the strategy — freeze it.
- [ ] Rented-GPU images/env pre-pulled; `make env` passes there.
- [ ] Dry-run done: `make phase-run` on 2 Phase-1 scenes from the raw ZIP,
      timed. Target ≤2 h/scene wall-clock.

## 1. Data drop (hour 0)
```bash
unzip <drop>.zip -d data/raw/phase2/          # keep organizers' layout
ls data/raw/phase2/<split>/<scene>/train/images | wc -l   # sanity per scene
```
Expected per-scene layout: `train/images/`, `train/sparse/`, `test/test_poses.csv`.
If the layout differs, STOP and adapt `filter_colmap_train.py` first.

## 2. Per-scene run (parallel across scenes/GPUs)
```bash
make phase-run SCENE=<scene> SPLIT=<split> PHASE=phase2
```
Stages inside (each skipped if already done): COLMAP train-filter → coverage
tripwire → dense init → optional transient masks → train locked config →
render test poses. One scene per GPU; on a multi-GPU rented box launch one
shell per GPU with `CUDA_VISIBLE_DEVICES` set.

## 3. Tripwire — hour-0 regime check (runs automatically in stage 2)
The run prints a loud banner if a scene's verdict is **extrapolative**
(`results/phase2_test_pose_coverage_summary.csv` has the numbers). All 13
Phase-1 scenes were interpolative; the whole Tier-A strategy assumes it.
If ANY Phase-2/3 scene trips:
1. Post the banner + summary row to the team channel immediately.
2. That scene's renders get manual visual QA before packaging (P3).
3. Consider per-scene fallback in `configs/scene_overrides/<scene>.yaml`
   (e.g. camera-optimizer ON — exp014 — or more iters); decided at sync,
   not unilaterally.

## 4. Visual QA (P3, before packaging)
Open 3–5 renders per scene at 100% crop. Look for: floaters, sky bleed,
transient ghosts, color shift vs train images. Any FAIL → scene goes back
with an override, not into the zip.

## 5. Package + validate + upload (P2)
```bash
make phase-package SCENES="<scene1> <scene2> ..." PHASE=phase2 SPLIT=<split>
```
Validates pre- and post-zip (names/sizes/counts/decodability). Upload the
zip literally named `submission_round1.zip` (competition requirement);
record the exp folder path + LB score in `results/PROGRESS.md` and
`results/leaderboard_reconciliation.csv` immediately.

## 6. Chaos checklist (test these in the dry-run, know the recovery)
| Failure | Recovery |
|---|---|
| Training OOM on rented GPU | Lower `DENSE_MAX_POINTS` / cap-max in `phase_locked.conf`; rerun scene (stage skip resumes data prep) |
| COLMAP dense init crashes | Scene falls back to sparse staging: point `phase_run.sh` at `train_staging` for that scene via an override; flag at sync |
| Run killed mid-train | Just rerun `make phase-run` — incomplete runs are detected and restarted |
| Renders missing images | `package_submission.py` validation fails loudly with the exact missing names; re-render that scene |
| Rented box dies | All state is under `runs/` + `data/processed/` — rsync them, resume on another box |
