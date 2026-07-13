"""Local unit tests for exp023 (LEGS) and exp024 (FreGS) loss mixins + their
registration. Run: python tests/test_loss_experiments.py  (no pytest needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.models.splatfacto_legs import LaplacianWeightedLossMixin, SplatfactoLEGSModelConfig
from src.models.splatfacto_fregs import FrequencyRegLossMixin, SplatfactoFreGSModelConfig


# ---- a minimal fake SplatfactoModel base providing exactly what the mixins call ----
class FakeSSIM:
    def __call__(self, a, b):  # returns a scalar "ssim" in [0,1]
        return torch.tensor(0.8)


class FakeBase:
    """Stands in for SplatfactoModel: parent get_loss_dict + gt/pred prep helpers."""
    def __init__(self, config, step=0, training=True):
        self.config = config
        self.step = step
        self.training = training
        self.device = "cpu"
        self.ssim = FakeSSIM()

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        # mimic splatfacto's plain main_loss so we can compare control vs treatment
        gt = self.get_gt_img(batch["image"])
        pred = outputs["rgb"]
        Ll1 = torch.abs(gt - pred).mean()
        simloss = 1 - self.ssim(None, None)
        return {"main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss}

    def composite_with_background(self, img, background):
        return img

    def get_gt_img(self, image):
        return image

    def _downscale_if_required(self, x):
        return x


class LEGSUnderTest(LaplacianWeightedLossMixin, FakeBase):
    pass


class FreGSUnderTest(FrequencyRegLossMixin, FakeBase):
    pass


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def _img(seed, H=48, W=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(H, W, 3, generator=g)


def test_legs_weight_map():
    cfg = SplatfactoLEGSModelConfig(legs_beta=10.0, legs_gamma=1.0)
    m = LEGSUnderTest(cfg)
    # flat image -> zero Laplacian everywhere -> normalization is degenerate but finite,
    # weight must stay >= 1 and finite (no NaN from the eps guard)
    flat = torch.full((32, 32, 3), 0.5)
    Wf = m._legs_weight(flat)
    assert torch.isfinite(Wf).all(), "LEGS weight NaN/inf on flat image"
    assert float(Wf.min()) >= 1.0 - 1e-6, "LEGS weight must be >= 1 (Eq. 7)"
    # a vertical edge -> high Laplacian at the edge columns, weight there ~ 1+beta
    edge = torch.zeros(32, 32, 3); edge[:, 16:, :] = 1.0
    We = m._legs_weight(edge)
    assert float(We.max()) > 5.0, f"edge weight should be strongly boosted, got {float(We.max())}"
    assert float(We.min()) >= 1.0 - 1e-6
    # gamma<1 (C3-style concave) must lift mid-responses vs linear C1
    cfg_g = SplatfactoLEGSModelConfig(legs_beta=10.0, legs_gamma=0.5)
    Wg = LEGSUnderTest(cfg_g)._legs_weight(edge)
    assert float(Wg.mean()) >= float(We.mean()) - 1e-4, "concave map should not lower mean weight"
    print("  [ok] LEGS weight map: finite, >=1, edge-boosted, gamma monotone")


def test_legs_control_equivalence():
    """beta=0 must reproduce the baseline main_loss byte-identically."""
    gt, pred = _img(1), _img(2)
    batch, outputs = {"image": gt}, {"rgb": pred, "background": torch.zeros(3)}
    base_loss = FakeBase(SplatfactoLEGSModelConfig(legs_beta=0.0)).get_loss_dict(outputs, batch)["main_loss"]
    ctrl = LEGSUnderTest(SplatfactoLEGSModelConfig(legs_beta=0.0)).get_loss_dict(outputs, batch)["main_loss"]
    assert approx(ctrl, base_loss, 0.0), "LEGS beta=0 must equal baseline exactly"
    treat = LEGSUnderTest(SplatfactoLEGSModelConfig(legs_beta=10.0)).get_loss_dict(outputs, batch)["main_loss"]
    assert float(treat) > float(base_loss), "LEGS beta=10 should raise the weighted L1 term"
    assert torch.isfinite(torch.tensor(float(treat)))
    print(f"  [ok] LEGS control==baseline ({float(ctrl):.6f}); beta=10 raises loss ({float(treat):.6f})")


def test_fregs_zero_when_identical():
    """pred == gt -> amplitude & phase discrepancy zero -> no freq loss contribution."""
    gt = _img(3)
    outputs = {"rgb": gt.clone(), "background": torch.zeros(3)}
    m = FreGSUnderTest(SplatfactoFreGSModelConfig(freq_weight=0.05), step=20000)
    ld = m.get_loss_dict(outputs, {"image": gt})
    assert "freq_loss" in ld, "freq_loss missing when freq_weight>0"
    assert float(ld["freq_loss"]) < 1e-5, f"freq_loss should ~0 for identical images, got {float(ld['freq_loss'])}"
    print(f"  [ok] FreGS freq_loss ~0 when pred==gt ({float(ld['freq_loss']):.2e})")


def test_fregs_gated_off_during_densification():
    """The OOM fix: freq loss must be INACTIVE before freq_start_step (== stop_split_at=15000),
    so it can't inflate DefaultStrategy's split decisions during densification."""
    gt, pred = _img(4), _img(5)
    outputs = {"rgb": pred, "background": torch.zeros(3)}
    cfg = SplatfactoFreGSModelConfig(freq_weight=0.05)  # freq_start_step defaults to 15000
    for step in (0, 500, 7000, 14999):
        ld = FreGSUnderTest(cfg, step=step).get_loss_dict(outputs, {"image": gt})
        assert "freq_loss" not in ld, f"freq loss must be OFF at step {step} (< freq_start_step)"
    on = FreGSUnderTest(cfg, step=15000).get_loss_dict(outputs, {"image": gt})
    assert "freq_loss" in on and float(on["freq_loss"]) > 0, "freq loss must activate at freq_start_step"
    print("  [ok] FreGS gated OFF <15000 (no densification interference), ON at 15000")


def test_fregs_positive_finite_and_progressive():
    gt, pred = _img(4), _img(5)
    outputs = {"rgb": pred, "background": torch.zeros(3)}
    # control
    ctrl = FreGSUnderTest(SplatfactoFreGSModelConfig(freq_weight=0.0), step=20000).get_loss_dict(outputs, {"image": gt})
    assert "freq_loss" not in ctrl, "freq_weight=0 must add nothing (control)"
    # start of the active window (== freq_T0): low band only
    low = FreGSUnderTest(SplatfactoFreGSModelConfig(freq_weight=0.05, freq_start_step=15000, freq_T0=15000, freq_T=27000), step=15000)
    le = low.get_loss_dict(outputs, {"image": gt})["freq_loss"]
    # later in the window (> T0): high band added -> strictly larger
    late = FreGSUnderTest(SplatfactoFreGSModelConfig(freq_weight=0.05, freq_start_step=15000, freq_T0=15000, freq_T=27000), step=26000)
    ll = late.get_loss_dict(outputs, {"image": gt})["freq_loss"]
    for v in (le, ll):
        assert torch.isfinite(v).all() and float(v) > 0, f"freq_loss must be positive finite, got {float(v)}"
    assert float(ll) > float(le), "later step should include the high-frequency band term"
    print(f"  [ok] FreGS positive/finite; low-only={float(le):.4e} < low+high={float(ll):.4e}")


def test_registration():
    from src.register_custom_methods import splatfacto_legs_method, splatfacto_fregs_method
    for meth, name, mtype in [
        (splatfacto_legs_method, "splatfacto-legs", "SplatfactoLEGSModelConfig"),
        (splatfacto_fregs_method, "splatfacto-fregs", "SplatfactoFreGSModelConfig"),
    ]:
        assert meth.config.method_name == name, meth.config.method_name
        assert type(meth.config.pipeline.model).__name__ == mtype
    print("  [ok] splatfacto-legs & splatfacto-fregs register with correct model configs")


if __name__ == "__main__":
    tests = [test_legs_weight_map, test_legs_control_equivalence, test_fregs_zero_when_identical,
             test_fregs_gated_off_during_densification,
             test_fregs_positive_finite_and_progressive, test_registration]
    print(f"Running {len(tests)} loss-experiment tests...")
    for t in tests:
        t()
    print("ALL PASSED")
