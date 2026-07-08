import csv
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_sweep import (append_row, build_train_cmd, existing_cells,  # noqa: E402
                       load_sweep, training_complete)

SWEEP = {
    "experiment": "exp999_test",
    "results_csv": "results/test.csv",
    "gpu": "local",
    "split": "public_set",
    "scenes": ["hcm0034"],
    "data_template": "data/processed/phase1/{scene}/train_staging_dense",
    "eval": "public_gt",
    "base_args": ["--pipeline.model.rasterize-mode", "antialiased"],
    "variants": {
        "mcmc2M": {"method": "splatfacto-mcmc", "iters": 30000,
                   "args": ["--pipeline.model.cap-max", "2000000"]},
        "ft": {"method": "splatfacto-perceptual", "iters": 37000,
               "load_dir_template": "runs/x/{scene}/models"},
    },
}


def test_build_train_cmd_basic():
    cmd = build_train_cmd(SWEEP, "mcmc2M", "hcm0034")
    assert cmd[0:2] == ["ns-train", "splatfacto-mcmc"]
    assert "data/processed/phase1/hcm0034/train_staging_dense" in cmd
    assert "--pipeline.model.cap-max" in cmd and "2000000" in cmd
    assert "antialiased" in cmd
    assert cmd[-5:] == ["colmap", "--eval-mode", "all", "--colmap-path", "sparse/0"]


def test_build_train_cmd_load_dir_templated():
    cmd = build_train_cmd(SWEEP, "ft", "hcm0034")
    i = cmd.index("--load-dir")
    assert cmd[i + 1] == "runs/x/hcm0034/models"


def test_load_sweep_rejects_missing_key(tmp_path):
    bad = {k: v for k, v in SWEEP.items() if k != "eval"}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit, match="eval"):
        load_sweep(p)


def test_load_sweep_rejects_bad_eval(tmp_path):
    bad = dict(SWEEP, eval="leaderboard")
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit, match="public_gt or val_split"):
        load_sweep(p)


def test_csv_roundtrip_and_skip(tmp_path):
    p = tmp_path / "res.csv"
    assert existing_cells(p) == set()
    append_row(p, {"scene": "hcm0034", "exp": "exp999_test", "variant": "mcmc2M",
                   "psnr": "21.0", "ssim": "0.71", "lpips": "0.16", "score": "0.70",
                   "train_hours": "1.5", "gpu": "local", "timestamp": "t"})
    assert ("hcm0034", "exp999_test", "mcmc2M") in existing_cells(p)
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["score"] == "0.70"


def test_training_complete(tmp_path):
    run = tmp_path / "run" / "splatfacto" / "ts" / "nerfstudio_models"
    run.mkdir(parents=True)
    assert not training_complete(tmp_path / "run", 30000)
    (run / "step-000029999.ckpt").touch()
    assert training_complete(tmp_path / "run", 30000)
    assert not training_complete(tmp_path / "run", 60000)
