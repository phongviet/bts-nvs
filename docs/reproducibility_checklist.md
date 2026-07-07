# Reproducibility Checklist

Run through this before uploading any submission ZIP.

- [ ] `config_resolved.yaml`, `git_commit.txt`, `env_freeze.txt` present in the run folder
- [ ] Checkpoint used for the submitted renders is saved and matches `render.py`'s `--config` input path
- [ ] `metrics_summary.csv` (or `metrics_val.json`) regenerated from the same checkpoint immediately before packaging
- [ ] `SUBMISSION_LOG.md` row added before uploading the ZIP
- [ ] No manual edits to any file under `renders_test/` (spot-check file mtimes if unsure)
- [ ] Only data under `data/raw/<phase>/` was used — no external images/models fine-tuned on scene-specific data
- [ ] Every generic pretrained model used (depth/segmentation/geometry/diffusion) has a provenance row in `docs/rules_and_constraints.md`, and any finetuned model was trained only on our own renders of the provided images — no external-scene pooling
- [ ] Scene folder name casing in the ZIP exactly matches `data/raw/phase1/<split>/<scene>` (e.g. `hcm0034` vs `HCM0193`)
- [ ] Every required test image for every required scene is present, correct dimensions
