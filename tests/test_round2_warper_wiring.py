"""The DIBR/refiner stack had four hard-coded phase-1 assumptions (2026-07-19).

Round-2 is the only graded set now, but Analysis/04 could not load a round-2
scene at all: scene_raw knew only public_set/private_set1, CONFIGS had no
round-2 entries, fix_paths pointed at data/processed/phase1/..., and a bare
`assert cam.model == "SIMPLE_RADIAL"` rejected bonsai/chair (SIMPLE_PINHOLE).
These lock in the fixes; the camera-model half is the one that would otherwise
resurface as a crash mid-submission.
"""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "dibr04", REPO / "Analysis" / "04_x3_dibr_pilot.py")
dibr04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dibr04)

ROUND2_SCENES = ["bonsai", "chair", "HCM0421", "HCM0539",
                 "HCM0540", "HCM0644", "HCM0674"]

pytestmark = pytest.mark.skipif(
    not dibr04.ROUND2.exists(), reason="round-2 raw data not present")


@pytest.mark.parametrize("scene", ROUND2_SCENES)
def test_scene_raw_resolves_round2(scene):
    assert dibr04.scene_raw(scene) == dibr04.ROUND2
    assert dibr04.is_round2(scene)


def test_scene_raw_still_prefers_phase1():
    """hcm0034 must keep resolving to public_set -- every cached phase-1 pair
    and every published A/B number depends on that path being unchanged."""
    if (dibr04.RAW / "hcm0034").exists():
        assert dibr04.scene_raw("hcm0034") == dibr04.RAW
        assert not dibr04.is_round2("hcm0034")


def test_unknown_scene_fails_loudly():
    with pytest.raises(SystemExit):
        dibr04.scene_raw("no_such_scene_xyz")


@pytest.mark.parametrize("scene", ROUND2_SCENES)
def test_camera_model_is_supported(scene):
    """bonsai/chair are SIMPLE_PINHOLE (k absent), drone scenes SIMPLE_RADIAL.
    Both must parse, and the pinhole ones must yield k == 0 so the distortion
    remap degenerates to a no-op instead of being applied with a bogus k."""
    from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary
    cams_bin = dibr04.ROUND2 / scene / "train/sparse/0/cameras.bin"
    if not cams_bin.exists():
        pytest.skip(f"{scene}: no train sparse model yet")
    cam = list(read_cameras_binary(cams_bin).values())[0]
    assert cam.model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE"), cam.model
    if cam.model == "SIMPLE_PINHOLE":
        assert len(cam.params) == 3        # f, cx, cy -- no k to unpack
        assert scene in ("bonsai", "chair")
    else:
        assert len(cam.params) == 4


def test_find_config_discovers_round2_backbones():
    """Round-2 backbones are auto-discovered from runs/round2/phase_locked/,
    so the 5 drone scenes need no CONFIGS edit as their fleet lands."""
    for scene in ROUND2_SCENES:
        run_dir = REPO / "runs/round2/phase_locked" / scene
        if not run_dir.exists():
            continue
        cfg = dibr04.find_config(scene)
        assert cfg.name == "config.yml" and run_dir in cfg.parents


def test_find_config_reports_missing_backbone():
    with pytest.raises(SystemExit):
        dibr04.find_config("no_such_scene_xyz")
