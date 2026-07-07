import numpy as np

from src.data_prep.make_val_split import match_test_split
from src.utils.pose_utils import PoseSet


def _line_poses(prefix, xs, dirs=None):
    n = len(xs)
    return PoseSet(
        names=[f"{prefix}{i}.JPG" for i in range(n)],
        centers=np.stack([np.asarray(xs, dtype=float), np.zeros(n), np.zeros(n)], axis=1),
        view_dirs=np.tile([1.0, 0.0, 0.0], (n, 1)) if dirs is None else np.asarray(dirs, dtype=float),
    )


def test_match_test_picks_nearest_without_duplicates():
    train = _line_poses("t", np.arange(20))
    # test poses cluster near x=5 and x=15
    test = _line_poses("q", [4.9, 5.1, 15.0, 15.2])
    val_idx, rows = match_test_split(train, test, n_val=4)
    assert len(val_idx) == len(set(val_idx)) == 4
    picked_x = sorted(train.centers[i, 0] for i in val_idx)
    # two picks near 5, two near 15
    assert picked_x[0] in (4, 5) and picked_x[1] in (4, 5, 6)
    assert picked_x[2] in (14, 15) and picked_x[3] in (14, 15, 16)
    assert all(r["cost"] < 0.2 for r in rows)


def test_match_test_subsamples_when_nval_lt_ntest():
    train = _line_poses("t", np.arange(50))
    test = _line_poses("q", np.linspace(0, 49, 10) + 0.3)
    val_idx, rows = match_test_split(train, test, n_val=5)
    assert len(val_idx) == 5
    # evenly spread across the test range, not clustered at one end
    xs = sorted(train.centers[i, 0] for i in val_idx)
    assert xs[0] < 10 and xs[-1] > 40


def test_match_test_nval_capped():
    train = _line_poses("t", np.arange(3))
    test = _line_poses("q", [0.1, 1.1, 2.1, 2.2, 2.3])
    val_idx, _ = match_test_split(train, test, n_val=10)
    assert len(val_idx) == 3  # capped at len(train)
