"""Exact-grader surrogate regression tests (also runnable without pytest)."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "refiner10_metric", REPO / "Analysis/10_refiner_pilot.py")
refiner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refiner)


class MetricAlignmentTest(unittest.TestCase):
    def test_metric_ssim_matches_skimage(self):
        rng = np.random.default_rng(7)
        a = rng.random((2, 31, 37, 3), dtype=np.float32)
        b = np.clip(a + rng.normal(0, 0.05, a.shape), 0, 1).astype(np.float32)
        got = refiner.metric_ssim(
            torch.from_numpy(a).permute(0, 3, 1, 2),
            torch.from_numpy(b).permute(0, 3, 1, 2)).numpy()
        expected = np.asarray([
            structural_similarity(a[i], b[i], data_range=1.0,
                                  channel_axis=2, win_size=11)
            for i in range(len(a))])
        np.testing.assert_allclose(got, expected, rtol=0, atol=2e-5)

    def test_psnr_loss_is_negative_clamped_normalized_psnr_per_image(self):
        pred = torch.zeros(3, 3, 16, 16)
        target = torch.stack([
            torch.full((3, 16, 16), 0.1),
            torch.full((3, 16, 16), 0.01),
            torch.full((3, 16, 16), 1e-4),
        ])
        mse = (pred - target).square().flatten(1).mean(1)
        expected = -(-10 * torch.log10(mse)).clamp(max=50) / 50
        torch.testing.assert_close(
            refiner.normalized_psnr_loss(pred, target), expected)

    def test_legacy_default_matches_original_formula_and_gradient(self):
        class DummyLPIPS(torch.nn.Module):
            def forward(self, a, b):
                return (a - b).square().mean((1, 2, 3), keepdim=True)

        torch.manual_seed(4)
        target = torch.rand(2, 3, 20, 24)
        window = refiner._gauss_window(3)
        pred_new = torch.rand(2, 3, 20, 24, requires_grad=True)
        pred_old = pred_new.detach().clone().requires_grad_(True)
        lpips_fn = DummyLPIPS()

        got, parts = refiner.refiner_objective(
            pred_new, target, lpips_fn, window)
        expected = (
            0.4 * lpips_fn(pred_old * 2 - 1, target * 2 - 1).mean()
            + 0.3 * (1 - refiner.ssim(pred_old, target, window))
            + 0.3 * torch.nn.functional.l1_loss(pred_old, target)
        )
        got.backward()
        expected.backward()
        torch.testing.assert_close(got.detach(), expected.detach(), rtol=0, atol=0)
        # CPU grouped-convolution backward reduction can differ below 1e-9
        # across otherwise identical graphs.
        torch.testing.assert_close(pred_new.grad, pred_old.grad, rtol=1e-6, atol=5e-10)
        self.assertTrue(np.isnan(parts["psnr_norm"]))


if __name__ == "__main__":
    unittest.main()
