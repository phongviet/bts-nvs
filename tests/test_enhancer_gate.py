from src.enhancer.gate_enhancer import decide_scene


def _m(score, psnr=21.0, ssim=0.71, lpips=0.16):
    return {"score": score, "psnr": psnr, "ssim": ssim, "lpips": lpips}


def test_gate_applies_on_clear_win():
    d = decide_scene({"render_only": _m(0.700), "offtheshelf": _m(0.705)}, under_covered=False)
    assert d["enhancer"] == "offtheshelf"


def test_gate_rejects_below_threshold():
    d = decide_scene({"render_only": _m(0.700), "offtheshelf": _m(0.702)}, under_covered=False)
    assert d["enhancer"] == "off"


def test_loose_gate_for_under_covered():
    stages = {"render_only": _m(0.700), "finetuned": _m(0.7015)}
    assert decide_scene(stages, under_covered=False)["enhancer"] == "off"
    assert decide_scene(stages, under_covered=True)["enhancer"] == "finetuned"


def test_severe_psnr_regression_blocks():
    d = decide_scene({"render_only": _m(0.700, psnr=21.0),
                      "finetuned": _m(0.706, psnr=20.5)}, under_covered=False)
    assert d["enhancer"] == "off"
    assert "psnr" in d["reason"]


def test_picks_best_stage():
    d = decide_scene({"render_only": _m(0.700), "offtheshelf": _m(0.704),
                      "finetuned": _m(0.708)}, under_covered=False)
    assert d["enhancer"] == "finetuned"


def test_no_baseline_row():
    d = decide_scene({"offtheshelf": _m(0.7)}, under_covered=False)
    assert d["enhancer"] == "off"
