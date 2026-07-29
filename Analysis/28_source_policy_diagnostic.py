"""Diagnose source-view ranking on sequential Round-2 indoor scenes.

This is a zero-training, no-test-GT proxy.  It leaves each supplied train frame
out, selects sources using only pose/name metadata, and measures how well source
brightness/sharpness match the omitted real frame.  It also reports source time
gaps on the hidden-GT test poses.  The test images are never accessed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dibr04", REPO / "Analysis/04_x3_dibr_pilot.py")
dibr04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dibr04)


def pose(qvec, tvec):
    r = dibr04.qvec2rotmat(np.asarray(qvec, dtype=np.float64))
    return -r.T @ np.asarray(tvec, dtype=np.float64), r.T


def appearance(path: Path, max_side=512):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"cannot read {path}")
    scale = min(1.0, max_side / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def summarize_scene(scene, policies, k):
    scene_dir = dibr04.scene_raw(scene) / scene
    train_dir = scene_dir / "train"
    on_disk = {p.name for p in (train_dir / "images").iterdir() if p.is_file()}
    records = dibr04.read_images_binary(train_dir / "sparse/0/images.bin")
    records = sorted((r for r in records.values() if r.name in on_disk), key=lambda r: r.name)
    names = [r.name for r in records]
    poses = [pose(r.qvec, r.tvec) for r in records]
    centers = np.stack([x[0] for x in poses])
    rotations = np.stack([x[1] for x in poses])
    extent = max(float(np.linalg.norm(np.ptp(centers, axis=0))), 1e-9)
    feats = np.asarray([appearance(train_dir / "images" / name) for name in names])
    brightness = feats[:, 0]
    log_sharpness = np.log1p(feats[:, 1])
    frames = [dibr04.frame_index(name) for name in names]

    result = {"n_train": len(names), "policies": {}}
    for policy in policies:
        rows = []
        for i, name in enumerate(names):
            idx = dibr04.select_source_indices(
                names, centers, rotations, centers[i], rotations[i], name, k,
                policy=policy, exclude_names={name})
            dist = np.linalg.norm(centers[idx] - centers[i], axis=1)
            weights = 1.0 / np.maximum(dist, 1e-9)
            weights /= weights.sum()
            gaps = [abs(frames[j] - frames[i]) for j in idx
                    if frames[j] is not None and frames[i] is not None]
            angles = dibr04.rotation_gaps_deg(rotations[idx], rotations[i])
            rows.append({
                "frame": frames[i],
                "brightness_error": abs(float(weights @ brightness[idx]) - brightness[i]),
                "log_sharpness_error": abs(float(weights @ log_sharpness[idx]) - log_sharpness[i]),
                "brightness_pred": float(weights @ brightness[idx]),
                "sharpness_pred": float(weights @ log_sharpness[idx]),
                "time_gaps": gaps,
                "distances": (dist / extent).tolist(),
                "angles": angles.tolist(),
            })
        all_gaps = np.asarray([v for r in rows for v in r["time_gaps"]], dtype=float)
        all_dist = np.asarray([v for r in rows for v in r["distances"]], dtype=float)
        all_angle = np.asarray([v for r in rows for v in r["angles"]], dtype=float)
        order = np.argsort([r["frame"] if r["frame"] is not None else n for n, r in enumerate(rows)])
        thirds = np.array_split(order, 3)
        result["policies"][policy] = {
            "brightness_mae": float(np.mean([r["brightness_error"] for r in rows])),
            "log_sharpness_mae": float(np.mean([r["log_sharpness_error"] for r in rows])),
            "brightness_spearman": float(spearmanr(
                [r["brightness_pred"] for r in rows], brightness).statistic),
            "sharpness_spearman": float(spearmanr(
                [r["sharpness_pred"] for r in rows], log_sharpness).statistic),
            "time_gap_p50": percentile(all_gaps, 50),
            "time_gap_p90": percentile(all_gaps, 90),
            "time_gap_max": float(all_gaps.max()) if len(all_gaps) else float("nan"),
            "distance_frac_p50": percentile(all_dist, 50),
            "distance_frac_p90": percentile(all_dist, 90),
            "angle_p50": percentile(all_angle, 50),
            "angle_p90": percentile(all_angle, 90),
            "brightness_mae_thirds": [
                float(np.mean([rows[i]["brightness_error"] for i in ids])) for ids in thirds],
            "log_sharpness_mae_thirds": [
                float(np.mean([rows[i]["log_sharpness_error"] for i in ids])) for ids in thirds],
        }

    test_rows = dibr04.load_test_poses(scene_dir / "test/test_poses.csv")
    test_stats = {}
    for policy in policies:
        gaps, distances, angles, brackets = [], [], [], 0
        for row in test_rows:
            center, rotation = pose(row["qvec"], row["tvec"])
            idx = dibr04.select_source_indices(
                names, centers, rotations, center, rotation, row["image_name"], k,
                policy=policy)
            tf = dibr04.frame_index(row["image_name"])
            sf = [frames[j] for j in idx]
            if tf is not None:
                gaps.extend(abs(x - tf) for x in sf if x is not None)
                brackets += int(any(x < tf for x in sf if x is not None)
                                and any(x > tf for x in sf if x is not None))
            distances.extend((np.linalg.norm(centers[idx] - center, axis=1) / extent).tolist())
            angles.extend(dibr04.rotation_gaps_deg(rotations[idx], rotation).tolist())
        test_stats[policy] = {
            "n_test": len(test_rows),
            "bracketed": brackets,
            "time_gap_p50": percentile(gaps, 50),
            "time_gap_p90": percentile(gaps, 90),
            "time_gap_max": float(max(gaps)) if gaps else float("nan"),
            "distance_frac_p50": percentile(distances, 50),
            "distance_frac_p90": percentile(distances, 90),
            "angle_p50": percentile(angles, 50),
            "angle_p90": percentile(angles, 90),
        }
    result["test_poses"] = test_stats
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["bonsai", "chair"])
    ap.add_argument("--policies", nargs="+", default=["spatial", "pose", "temporal"])
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=REPO / "results/source_policy_diagnostic.json")
    args = ap.parse_args()
    out = {scene: summarize_scene(scene, args.policies, args.K) for scene in args.scenes}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    for scene, scene_result in out.items():
        print(f"\n{scene} (K={args.K})")
        print("policy     brightMAE sharpMAE sharp-rho  train-gap-p90  test-gap-p90  test-bracket")
        for policy in args.policies:
            tr = scene_result["policies"][policy]
            te = scene_result["test_poses"][policy]
            print(f"{policy:10s} {tr['brightness_mae']:9.3f} {tr['log_sharpness_mae']:8.3f} "
                  f"{tr['sharpness_spearman']:9.3f} {tr['time_gap_p90']:14.1f} "
                  f"{te['time_gap_p90']:13.1f} {te['bracketed']:5d}/{te['n_test']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
