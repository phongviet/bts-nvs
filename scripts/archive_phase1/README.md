# Archived phase-1 scripts

One-off drivers from the phase-1 campaign (week-2 sweeps, the backend ablation, the
final private fleets). Kept for reproducibility of the results recorded in
`results/PROGRESS.md` and `results/experiment_log.md` — **older log entries still cite
these by their original `scripts/<name>` path; they now live here.**

Nothing in the round-2 pipeline calls them. The live drivers are:

| script | role |
|---|---|
| `scripts/phase_run.sh` | per-scene raw → test renders (the locked config). `phase_run.sh <scene> all round2` |
| `scripts/run_local_wave_2026-07-17.sh` | local experiment queue (wave 1) |
| `scripts/run_local_wave2_2026-07-17.sh` | local experiment queue (wave 2, guarded compositions) |
| `scripts/build_kaggle_exp034_upload.py` | Kaggle fleet upload builder |
| `scripts/run_sweep.py` | generic config sweep |
