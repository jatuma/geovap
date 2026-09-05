"""Which frames see a point / a tile; nearest frame in time."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .cloud_store import TileInfo
from .config import R_MAX
from .geometry import frame_rotations
from .poses import Poses
from .rig import IDENTITY, RigModel


class FrameIndex:
    def __init__(self, poses: Poses, rig: RigModel = IDENTITY):
        self.poses = poses
        self.rig = rig
        self.R, self.C = frame_rotations(poses, rig)  # [M,3,3], [M,3]
        self.t = poses.t + rig.dt_s
        self._tree = cKDTree(self.C[:, :2])

    def __len__(self) -> int:
        return len(self.poses)

    def frames_for_points(self, P: np.ndarray, r_max: float = R_MAX) -> list[np.ndarray]:
        """For each point, indices of frames whose camera centre is within r_max (2D)."""
        res = self._tree.query_ball_point(np.asarray(P)[:, :2], r_max)
        return [np.asarray(r, dtype=np.int64) for r in res]

    def frames_in_bbox(self, bbox: tuple[float, float, float, float], margin: float = R_MAX) -> np.ndarray:
        minE, minN, maxE, maxN = bbox
        c = self.C
        m = (c[:, 0] >= minE - margin) & (c[:, 0] <= maxE + margin) & (c[:, 1] >= minN - margin) & (c[:, 1] <= maxN + margin)
        return np.flatnonzero(m)

    def frames_for_tile(self, tile: TileInfo, r_max: float = R_MAX) -> np.ndarray:
        return self.frames_in_bbox(tile.bbox, r_max)

    def nearest_in_time(self, gps_time: np.ndarray) -> np.ndarray:
        """Index of the frame whose (offset-corrected) timestamp is closest to each gps_time."""
        t = self.t  # sorted
        j = np.searchsorted(t, gps_time)
        j1 = np.clip(j, 0, len(t) - 1)
        j0 = np.clip(j - 1, 0, len(t) - 1)
        pick0 = np.abs(gps_time - t[j0]) <= np.abs(gps_time - t[j1])
        return np.where(pick0, j0, j1)
