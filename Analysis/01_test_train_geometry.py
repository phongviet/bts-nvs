"""Analysis 01: How do test poses relate to train poses, per scene (all 13)?

Motivation: the top-8 leaderboard gap (~18 pts vs our 57.4) is far beyond any
tuning lever we've measured (all Tier-A levers sum to ~+1 pt). If test views are
temporally/spatially sandwiched between train views (a video-interpolation
regime), then image-based rendering (warping real train pixels) or video frame
interpolation can beat pure 3DGS reconstruction — especially on LPIPS (0.4 of
the Score). This script measures that regime precisely.

Outputs: Analysis/01_geometry_per_scene.csv (one row per scene) and
Analysis/01_geometry_per_test_view.csv (one row per test view), plus a printed
summary table.

Per test view:
  - seq gap: distance in DJI sequence numbers to nearest train frame
    (1 = the immediately adjacent captured frame is a train frame).
  - sandwiched: has a train frame within <=2 seq steps on BOTH sides.
  - d_nn / spacing: Euclidean distance from test camera center to nearest train
    camera center, normalized by the scene's median consecutive-train-frame
    spacing (<1 = closer to a train view than train views are to each other).
  - rot_nn_deg: rotation angle to that nearest train pose.

Run: conda run -n airace python Analysis/01_test_train_geometry.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "phase1"
OUT_DIR = Path(__file__).resolve().parent

from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary  # noqa: E402


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def cam_center(qvec, tvec):
    R = qvec2rotmat(qvec)
    return -R.T @ np.asarray(tvec)


def rot_angle_deg(q1, q2):
    R = qvec2rotmat(q1) @ qvec2rotmat(q2).T
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


SEQ_RE = re.compile(r"_(\d{4})_V\.JPG$", re.IGNORECASE)


def seq_num(name: str):
    m = SEQ_RE.search(name)
    return int(m.group(1)) if m else None


def load_scene(scene_dir: Path):
    ims = read_images_binary(scene_dir / "train" / "sparse" / "0" / "images.bin")
    train_names = {p.name for p in (scene_dir / "train" / "images").iterdir()}
    train = []
    for im in ims.values():
        if im.name in train_names:  # images.bin may contain extra/test entries
            train.append((im.name, list(im.qvec), list(im.tvec)))
    test = []
    with open(scene_dir / "test" / "test_poses.csv") as f:
        for r in csv.DictReader(f):
            test.append((
                r["image_name"],
                [float(r[k]) for k in ("qw", "qx", "qy", "qz")],
                [float(r[k]) for k in ("tx", "ty", "tz")],
            ))
    return train, test


def main():
    scene_dirs = sorted(RAW.glob("*/*"))
    per_scene_rows, per_view_rows = [], []
    for sd in scene_dirs:
        if not (sd / "test" / "test_poses.csv").exists():
            continue
        scene, split = sd.name, sd.parent.name
        train, test = load_scene(sd)
        tr_seq = {seq_num(n): i for n, _, _ in train for i in [0]} if False else {}
        tr_seqs = sorted(s for s in (seq_num(n) for n, _, _ in train) if s is not None)
        tr_seq_set = set(tr_seqs)
        te_seqs = sorted(s for s in (seq_num(n) for n, _, _ in test) if s is not None)

        C_tr = np.stack([cam_center(q, t) for _, q, t in train])
        # median spacing between consecutive train frames (by seq order)
        order = np.argsort([seq_num(n) if seq_num(n) is not None else 10**9 for n, _, _ in train])
        C_sorted = C_tr[order]
        spacing = np.median(np.linalg.norm(np.diff(C_sorted, axis=0), axis=1))

        gaps, sandw, dnn_ratio, rots, d2nn_ratio = [], [], [], [], []
        for name, q, t in test:
            s = seq_num(name)
            if s is not None and tr_seq_set:
                below = max((x for x in tr_seq_set if x < s), default=None)
                above = min((x for x in tr_seq_set if x > s), default=None)
                gap = min([abs(s - x) for x in (below, above) if x is not None] or [999])
                gaps.append(gap)
                sandw.append(below is not None and above is not None
                             and (s - below) <= 2 and (above - s) <= 2)
            C = cam_center(q, t)
            d = np.linalg.norm(C_tr - C, axis=1)
            i0 = int(np.argmin(d))
            d_sorted = np.sort(d)
            dnn_ratio.append(d_sorted[0] / spacing)
            d2nn_ratio.append(d_sorted[1] / spacing)
            rots.append(rot_angle_deg(q, train[i0][1]))
            per_view_rows.append(dict(
                scene=scene, split=split, image=name, seq=s,
                seq_gap=gaps[-1] if s is not None else None,
                sandwiched=sandw[-1] if s is not None else None,
                d_nn_over_spacing=round(float(d_sorted[0] / spacing), 3),
                d_2nn_over_spacing=round(float(d_sorted[1] / spacing), 3),
                rot_nn_deg=round(rots[-1], 2),
            ))
        row = dict(
            scene=scene, split=split, n_train=len(train), n_test=len(test),
            median_seq_gap=float(np.median(gaps)) if gaps else None,
            max_seq_gap=int(np.max(gaps)) if gaps else None,
            pct_gap1=round(100 * np.mean([g == 1 for g in gaps]), 1) if gaps else None,
            pct_sandwiched=round(100 * np.mean(sandw), 1) if sandw else None,
            median_dnn_over_spacing=round(float(np.median(dnn_ratio)), 3),
            p90_dnn_over_spacing=round(float(np.percentile(dnn_ratio, 90)), 3),
            median_d2nn_over_spacing=round(float(np.median(d2nn_ratio)), 3),
            median_rot_nn_deg=round(float(np.median(rots)), 2),
            p90_rot_nn_deg=round(float(np.percentile(rots, 90)), 2),
        )
        per_scene_rows.append(row)
        print(f"{scene:9s} ({split:12s}) gap1={row['pct_gap1']}% sandwiched={row['pct_sandwiched']}% "
              f"d_nn/spacing med={row['median_dnn_over_spacing']} p90={row['p90_dnn_over_spacing']} "
              f"rot med={row['median_rot_nn_deg']}deg")

    for fname, rows in [("01_geometry_per_scene.csv", per_scene_rows),
                        ("01_geometry_per_test_view.csv", per_view_rows)]:
        with open(OUT_DIR / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {OUT_DIR/'01_geometry_per_scene.csv'} and per-view CSV.")


if __name__ == "__main__":
    main()
