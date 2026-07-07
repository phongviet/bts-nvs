import json
from pathlib import Path


import numpy as np


def test_method_specs_importable():
    from src.register_custom_methods import (
        splatfacto_mcmc_method, splatfacto_mcmc_perceptual_method,
        splatfacto_perceptual_method, splatfacto_tpw_method,
    )
    for spec, name in [
        (splatfacto_mcmc_method, "splatfacto-mcmc"),
        (splatfacto_perceptual_method, "splatfacto-perceptual"),
        (splatfacto_mcmc_perceptual_method, "splatfacto-mcmc-perceptual"),
        (splatfacto_tpw_method, "splatfacto-tpw"),
    ]:
        assert spec.config.method_name == name
        assert spec.config.max_num_iterations == 30000


def test_perceptual_config_defaults():
    from src.models.splatfacto_perceptual import (
        SplatfactoMCMCPerceptualModelConfig, SplatfactoPerceptualModelConfig,
    )
    cfg = SplatfactoPerceptualModelConfig()
    assert cfg.lpips_loss_weight == 0.1 and cfg.lpips_patch_size == 512
    mcfg = SplatfactoMCMCPerceptualModelConfig()
    assert mcfg.cap_max == 1_000_000  # inherits the MCMC knobs too


def test_weighted_sampling_prefers_boosted_images(tmp_path):
    from src.data.weighted_datamanager import (
        WeightedFullImageDatamanager, WeightedFullImageDatamanagerConfig,
    )

    class FakeDataset:
        def __init__(self, n):
            self.image_filenames = [Path(f"img{i}.JPG") for i in range(n)]

        def __len__(self):
            return len(self.image_filenames)

    n = 10
    weights = {f"img{i}.JPG": (2.0 if i < 2 else 1.0) for i in range(n)}
    wpath = tmp_path / "train_weights.json"
    wpath.write_text(json.dumps(weights))

    dm = object.__new__(WeightedFullImageDatamanager)
    dm.config = WeightedFullImageDatamanagerConfig(
        data=tmp_path, weights_path=wpath, train_cameras_sampling_seed=42)
    dm.train_dataset = FakeDataset(n)

    draws = []
    for _ in range(200):  # 200 epochs of 10
        epoch = dm.sample_train_cameras()
        assert len(epoch) == n
        draws.extend(epoch)
    counts = np.bincount(draws, minlength=n)
    # boosted images (w=2) should be drawn ~2x the unboosted ones
    boosted = counts[:2].mean()
    normal = counts[2:].mean()
    assert 1.6 < boosted / normal < 2.5
    # every image still gets sampled (weight floor)
    assert counts.min() > 0


def test_weight_floor_rescues_zero_weights(tmp_path):
    from src.data.weighted_datamanager import (
        WeightedFullImageDatamanager, WeightedFullImageDatamanagerConfig,
    )

    class FakeDataset:
        def __init__(self, n):
            self.image_filenames = [Path(f"img{i}.JPG") for i in range(n)]

        def __len__(self):
            return len(self.image_filenames)

    wpath = tmp_path / "train_weights.json"
    wpath.write_text(json.dumps({"img0.JPG": 0.0, "img1.JPG": 4.0}))
    dm = object.__new__(WeightedFullImageDatamanager)
    dm.config = WeightedFullImageDatamanagerConfig(
        data=tmp_path, weights_path=wpath, weight_floor=0.25, train_cameras_sampling_seed=0)
    dm.train_dataset = FakeDataset(2)
    probs = dm._load_weights()
    assert probs[0] > 0  # floored to 0.25 * max
    assert abs(probs.sum() - 1.0) < 1e-9


def test_compute_train_weights():
    from src.data_prep.compute_train_weights import compute_weights
    from src.utils.pose_utils import PoseSet

    train = PoseSet(
        names=[f"t{i}.JPG" for i in range(10)],
        centers=np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], axis=1),
        view_dirs=np.tile([1.0, 0, 0], (10, 1)),
    )
    test = PoseSet(
        names=["q0"], centers=np.array([[5.2, 0, 0]]), view_dirs=np.array([[1.0, 0, 0]]),
    )
    w = compute_weights(train, test, boost=1.0, dist_frac_thresh=0.1, angle_thresh_deg=20)
    assert w["t5.JPG"] == 2.0        # covers the test pose -> max boost
    assert w["t0.JPG"] == 1.0        # far away -> neutral
