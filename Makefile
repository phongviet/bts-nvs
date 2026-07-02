SHELL := /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate airace

.PHONY: env train-check render-check freeze

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
