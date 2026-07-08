import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_exp019_valsplit_validation import (  # noqa: E402
    aggregate_split, pairwise_agreement, spearman,
)


def _row(score):
    return {"psnr": 21.0, "ssim": 0.7, "lpips": 0.13, "score": score}


def test_aggregate_split_means_only_available_ids():
    per_image = {"a.JPG": _row(0.70), "b.JPG": _row(0.72)}
    # c.JPG requested but never rendered (unregistered in COLMAP) -> skipped
    agg = aggregate_split(per_image, ["a.JPG", "b.JPG", "c.JPG"])
    assert agg["n"] == 2
    assert agg["score"] == pytest.approx(0.71)


def test_aggregate_split_empty_raises():
    with pytest.raises(RuntimeError):
        aggregate_split({"a.JPG": _row(0.7)}, ["z.JPG"])


def test_pairwise_agreement_grades_only_non_ties():
    test = [0.720, 0.710, 0.7101]  # pair (1,2) delta 0.0001 -> tie, ungraded
    val_good = [0.68, 0.66, 0.67]  # agrees on both graded pairs
    val_bad = [0.66, 0.68, 0.67]   # inverts both graded pairs
    assert pairwise_agreement(val_good, test, min_delta=0.001) == (2, 2)
    assert pairwise_agreement(val_bad, test, min_delta=0.001) == (0, 2)


def test_spearman_perfect_and_inverted():
    assert spearman([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert spearman([4, 3, 2, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(-1.0)
