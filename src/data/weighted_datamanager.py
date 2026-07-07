"""exp013 (A.6c stretch): test-pose-weighted train-image sampling.

FullImageDatamanager samples one full train image per step from an epoch
permutation (sample_train_cameras). This subclass replaces the uniform epoch
with weighted draws-with-replacement, so train views that cover test poses
get proportionally more optimization steps.

Weights come from a JSON file {image_filename: weight} written by
src/data_prep/compute_train_weights.py (weight 1.0 = neutral). Images absent
from the file get weight 1.0; a floor keeps every image sampled occasionally
so no region of the scene collapses. Epoch length stays num_train_cameras,
so iteration counts stay comparable with the uniform baseline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

import numpy as np
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager, FullImageDatamanagerConfig,
)


@dataclass
class WeightedFullImageDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: WeightedFullImageDatamanager)
    weights_path: Optional[Path] = None
    """JSON {image_filename: weight}. Default: <data>/train_weights.json."""
    weight_floor: float = 0.25
    """Minimum relative sampling weight so every train image keeps coverage."""


class WeightedFullImageDatamanager(FullImageDatamanager):
    config: WeightedFullImageDatamanagerConfig

    def _load_weights(self) -> np.ndarray:
        path = self.config.weights_path or (Path(self.config.data) / "train_weights.json")
        names = [Path(p).name for p in self.train_dataset.image_filenames]
        weights = np.ones(len(names), dtype=np.float64)
        if Path(path).exists():
            table = json.loads(Path(path).read_text())
            for i, n in enumerate(names):
                weights[i] = float(table.get(n, 1.0))
            n_boosted = int((weights > 1.0).sum())
            print(f"[weighted-dm] {path}: {n_boosted}/{len(names)} images boosted, "
                  f"max weight {weights.max():.2f}")
        else:
            print(f"[weighted-dm] WARNING: {path} not found -- sampling uniformly. "
                  f"Run src/data_prep/compute_train_weights.py first.")
        weights = np.maximum(weights, self.config.weight_floor * weights.max())
        return weights / weights.sum()

    def sample_train_cameras(self):
        if not hasattr(self, "_sampling_probs"):
            self._sampling_probs = self._load_weights()
            self._weighted_rng = np.random.default_rng(self.config.train_cameras_sampling_seed)
        n = len(self.train_dataset)
        return self._weighted_rng.choice(n, size=n, replace=True, p=self._sampling_probs).tolist()
