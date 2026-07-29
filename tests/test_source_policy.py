"""Pose/temporal source-policy tests (also runnable without pytest)."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "dibr04_source_policy", REPO / "Analysis/04_x3_dibr_pilot.py")
dibr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dibr)


class SourcePolicyTest(unittest.TestCase):
    def setUp(self):
        self.names = ["frame_000.jpg", "frame_010.jpg", "frame_020.jpg",
                      "frame_030.jpg", "frame_100.jpg"]
        self.centers = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                                   [3, 0, 0], [2.01, 0, 0]], dtype=float)
        self.rotations = np.repeat(np.eye(3)[None], len(self.names), axis=0)

    def test_temporal_policy_reserves_previous_and_next(self):
        got = dibr.select_source_indices(
            self.names, self.centers, self.rotations,
            np.asarray([2.0, 0, 0]), np.eye(3), "frame_025.jpg", 3,
            policy="temporal")
        self.assertEqual([self.names[i] for i in got[:2]],
                         ["frame_020.jpg", "frame_030.jpg"])
        # The almost-coincident repeated-orbit frame remains available as the
        # pose-ranked third source, but cannot displace temporal bracketing.
        self.assertEqual(self.names[got[2]], "frame_100.jpg")

    def test_spatial_default_is_unchanged(self):
        got = dibr.select_source_indices(
            self.names, self.centers, self.rotations,
            np.asarray([2.0, 0, 0]), np.eye(3), "frame_025.jpg", 2,
            policy="spatial")
        self.assertEqual([self.names[i] for i in got],
                         ["frame_020.jpg", "frame_100.jpg"])

    def test_temporal_falls_back_when_name_has_no_index(self):
        temporal = dibr.select_source_indices(
            self.names, self.centers, self.rotations,
            np.asarray([2.0, 0, 0]), np.eye(3), "target", 3,
            policy="temporal")
        pose = dibr.select_source_indices(
            self.names, self.centers, self.rotations,
            np.asarray([2.0, 0, 0]), np.eye(3), "target", 3,
            policy="pose")
        self.assertEqual(temporal, pose)


if __name__ == "__main__":
    unittest.main()
