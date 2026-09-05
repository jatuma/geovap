"""Camera poses from export.csv: sorted by time, segmented into passes, interpolable in time.

Wraps `experiments/common/io_data.load_frames` (pure csv, no pandas) instead of re-parsing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import compat

# Frames are distance-triggered (~5 m). Gaps > 5 s (29 of them) split the trajectory into the
# 30 driving passes the docs refer to; the spatial jump catches teleports at the same time.
PASS_GAP_S = 5.0
PASS_JUMP_M = 15.0


@dataclass
class Poses:
    filename: np.ndarray  # [M] object (str)
    t: np.ndarray  # [M] float64, seconds of GPS week (same base as LAZ gps_time)
    origin: np.ndarray  # [M,3] float64 (E, N, H) S-JTSK
    roll: np.ndarray  # [M] deg
    pitch: np.ndarray  # [M] deg
    yaw: np.ndarray  # [M] deg, mathematical azimuth CCW from +E
    pass_id: np.ndarray  # [M] int32
    speed: np.ndarray  # [M] m/s, from neighbours within the pass

    def __len__(self) -> int:
        return len(self.t)

    def path(self, i: int) -> str:
        from .config import PANO_DIR

        return str(PANO_DIR / str(self.filename[i]))

    # ------------------------------------------------------------------ interpolation
    def pose_at(self, idx: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pose of frames `idx` at time t_idx + dt, interpolated linearly within the same pass.

        dt == 0 returns the stored values exactly (regression anchor). Returns
        (origin[K,3], roll[K], pitch[K], yaw[K]) with yaw wrapped to (-180, 180].
        """
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        if dt == 0.0:
            return self.origin[idx].copy(), self.roll[idx].copy(), self.pitch[idx].copy(), self.yaw[idx].copy()
        return self.interp(self.t[idx] + dt, self.pass_id[idx])

    def interp(self, t_query: np.ndarray, pass_hint: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trajectory pose at arbitrary times: piecewise-linear along the frame sequence of the pass
        (yaw unwrapped within the pass). Outside a pass the end pose is extrapolated linearly from
        its last segment. pass_hint selects the pass for each query (default: the pass of the nearest frame)."""
        t_query = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
        if pass_hint is None:
            j = np.clip(np.searchsorted(self.t, t_query), 0, len(self) - 1)
            j0 = np.clip(j - 1, 0, len(self) - 1)
            j = np.where(np.abs(t_query - self.t[j0]) < np.abs(t_query - self.t[j]), j0, j)
            pass_hint = self.pass_id[j]
        origin = np.empty((len(t_query), 3))
        roll = np.empty(len(t_query))
        pitch = np.empty(len(t_query))
        yaw = np.empty(len(t_query))
        for p in np.unique(pass_hint):
            q = pass_hint == p
            sel = np.flatnonzero(self.pass_id == p)
            tt = self.t[sel]
            tq = t_query[q]
            if len(sel) == 1:
                origin[q] = self.origin[sel[0]]
                roll[q], pitch[q], yaw[q] = self.roll[sel[0]], self.pitch[sel[0]], self.yaw[sel[0]]
                continue
            yaw_unw = np.degrees(np.unwrap(np.radians(self.yaw[sel])))

            def lin(vals):
                # np.interp clamps; extrapolate linearly beyond the ends from the end segments
                out = np.interp(tq, tt, vals)
                lo = tq < tt[0]
                hi = tq > tt[-1]
                if lo.any():
                    out[lo] = vals[0] + (vals[1] - vals[0]) / (tt[1] - tt[0]) * (tq[lo] - tt[0])
                if hi.any():
                    out[hi] = vals[-1] + (vals[-1] - vals[-2]) / (tt[-1] - tt[-2]) * (tq[hi] - tt[-1])
                return out

            origin[q] = np.stack([lin(self.origin[sel, i]) for i in range(3)], 1)
            roll[q] = lin(self.roll[sel])
            pitch[q] = lin(self.pitch[sel])
            yaw[q] = (lin(yaw_unw) + 180.0) % 360.0 - 180.0
        return origin, roll, pitch, yaw


def _segment_passes(t: np.ndarray, origin: np.ndarray) -> np.ndarray:
    dt = np.diff(t)
    jump = np.linalg.norm(np.diff(origin[:, :2], axis=0), axis=1)
    brk = (dt > PASS_GAP_S) | (jump > PASS_JUMP_M)
    pass_id = np.zeros(len(t), dtype=np.int32)
    pass_id[1:] = np.cumsum(brk)
    return pass_id


def _speeds(t: np.ndarray, origin: np.ndarray, pass_id: np.ndarray) -> np.ndarray:
    speed = np.zeros(len(t), dtype=np.float64)
    for p in np.unique(pass_id):
        sel = np.flatnonzero(pass_id == p)
        if len(sel) < 2:
            continue
        tt = t[sel]
        xy = origin[sel, :2]
        v = np.gradient(xy, tt, axis=0)
        speed[sel] = np.linalg.norm(v, axis=1)
    return speed


def load_poses() -> Poses:
    compat.ensure_experiments_on_path()
    from common import io_data

    f = io_data.load_frames()  # already sorted by timestamp
    pass_id = _segment_passes(f.timestamp, f.origin_enh)
    speed = _speeds(f.timestamp, f.origin_enh, pass_id)
    return Poses(
        filename=f.filename,
        t=f.timestamp,
        origin=f.origin_enh,
        roll=f.roll_deg,
        pitch=f.pitch_deg,
        yaw=f.yaw_deg,
        pass_id=pass_id,
        speed=speed,
    )
