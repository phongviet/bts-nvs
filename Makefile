SHELL := /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate airace

.PHONY: env train-check render-check freeze phase-run phase-package

# Sanity check: env + gsplat CUDA extension
env:
	$(CONDA_ACTIVATE) && python -c "import torch; print('cuda:', torch.cuda.is_available())" && python -c "import gsplat; print('gsplat ok')"

# Day 2 pose-convention check: usage `make train-check SCENE=hcm0034 CONFIG=runs/.../config.yml`
train-check:
	$(CONDA_ACTIVATE) && python src/render.py --config $(CONFIG) --mode train_check \
		--sparse-dir data/raw/phase1/public_set/$(SCENE)/train/sparse/0 \
		--out runs/_pose_check/$(SCENE)

# usage: make render SCENE=hcm0034 SPLIT=public_set CONFIG=runs/.../config.yml OUT=runs/.../renders_test
render:
	$(CONDA_ACTIVATE) && python src/render.py --config $(CONFIG) --mode test \
		--poses-csv data/raw/phase1/$(SPLIT)/$(SCENE)/test/test_poses.csv --out $(OUT)

freeze:
	$(CONDA_ACTIVATE) && conda env export > environment.yml && pip freeze > docs/pip_freeze_week1.txt

# Phase-2/3 per-scene automation (raw scene -> test renders, resumable).
# usage: make phase-run SCENE=HCM0249 SPLIT=private_set1 [PHASE=phase2]
# See docs/phase_runbook.md. Locked config: configs/phase_locked.conf.
PHASE ?= phase1
phase-run:
	scripts/phase_run.sh $(SCENE) $(SPLIT) $(PHASE)

# Package + validate the submission zip across scenes.
# usage: make phase-package SCENES="HCM0249 HCM0254 ..." [PHASE=phase2] [SPLIT=private_set1]
SPLIT ?= private_set1
phase-package:
	$(CONDA_ACTIVATE) && python src/package_submission.py \
		--runs-dir runs/$(PHASE)/phase_locked \
		--scenes $(SCENES) \
		--poses-root data/raw/$(PHASE)/$(SPLIT) \
		--out submissions/$(PHASE)/phase_locked/submission_round1.zip
