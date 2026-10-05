"""Camera poses: the pose table as a value object, with pass segmentation and interpolation.

Pure data and maths. *Reading* a vendor export lives in `geovap.io.adapters.poses`; reading and
writing our own pose-table CSV lives in `geovap.runtime.pose_table`; choosing which table to load
lives in `geovap.runtime.settings`. Nothing here touches the filesystem.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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
    source: str = "export"  # "export" or the stem of the pose-table CSV this was loaded from
    # Dense trajectory backing this pose table, or None (linear interpolation only). Typed loosely
    # (object, not `Trajectory`) so this module has no hard dependency on `mapping.trajectory`
    # (written later, S3). Expected interface: `covers(pass_id, t_query) -> bool[K]` and
    # `camera_pose(t_query, pass_id) -> (origin[K,3], roll[K], pitch[K], yaw[K])`.
    traj: object | None = field(default=None, compare=False)
    # Path to the pass_transforms.json this table's points are registered against (S5b), or None
    # ("export", and any table without a "registration" entry in its sidecar). Set by
    # `read_pose_table`; the corrected poses are only geometrically consistent with a `CloudStore`
    # opened with `registration=` this path (see `cloud_store.open_store`).
    registration: Path | None = field(default=None, compare=False)

    def __len__(self) -> int:
        return len(self.t)

    def hash(self) -> str:
        """Stable identity of the pose table: sha1 of t (exact) and origin/roll/pitch/yaw rounded
        to 1e-3 m / 1e-4 deg, so re-derivations that round-trip through CSV text hash identically."""
        payload = json.dumps(
            {
                "t": [repr(float(x)) for x in self.t],
                "origin": [[round(float(v), 3) for v in row] for row in self.origin],
                "roll": [round(float(x), 4) for x in self.roll],
                "pitch": [round(float(x), 4) for x in self.pitch],
                "yaw": [round(float(x), 4) for x in self.yaw],
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    def pass_of_time(self, t: np.ndarray) -> np.ndarray:
        """Pass id of arbitrary query times: boundaries at the midpoint of each inter-pass gap;
        queries before the first / after the last boundary clamp to the first / last pass."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        passes = np.unique(self.pass_id)
        brk = np.flatnonzero(np.diff(self.pass_id) != 0)
        bounds = (self.t[brk] + self.t[brk + 1]) / 2.0
        idx = np.clip(np.searchsorted(bounds, t), 0, len(passes) - 1)
        return passes[idx]

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
        its last segment. pass_hint selects the pass for each query (default: the pass of the nearest frame).

        If `self.traj` is set and covers a query (pass, t), its dense pose is used instead of the
        linear fallback; uncovered queries fall through to the existing behaviour unchanged."""
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
            tq = t_query[q]
            traj_mask = None
            if self.traj is not None:
                traj_mask = np.asarray(self.traj.covers(p, tq), dtype=bool)
                if traj_mask.any():
                    o_t, r_t, pt_t, y_t = self.traj.camera_pose(tq[traj_mask], p)
                    idx_q = np.flatnonzero(q)[traj_mask]
                    origin[idx_q] = o_t
                    roll[idx_q] = r_t
                    pitch[idx_q] = pt_t
                    yaw[idx_q] = y_t
                if traj_mask.all():
                    continue
            sel = np.flatnonzero(self.pass_id == p)
            # queries handled by the trajectory above are skipped here
            q_lin = ~traj_mask if traj_mask is not None else np.ones(len(tq), dtype=bool)
            tq_lin = tq[q_lin]
            idx_lin = np.flatnonzero(q)[q_lin]
            if len(sel) == 1:
                origin[idx_lin] = self.origin[sel[0]]
                roll[idx_lin], pitch[idx_lin], yaw[idx_lin] = self.roll[sel[0]], self.pitch[sel[0]], self.yaw[sel[0]]
                continue
            tt = self.t[sel]
            yaw_unw = np.degrees(np.unwrap(np.radians(self.yaw[sel])))

            def lin(vals):
                # np.interp clamps; extrapolate linearly beyond the ends from the end segments
                out = np.interp(tq_lin, tt, vals)
                lo = tq_lin < tt[0]
                hi = tq_lin > tt[-1]
                if lo.any():
                    out[lo] = vals[0] + (vals[1] - vals[0]) / (tt[1] - tt[0]) * (tq_lin[lo] - tt[0])
                if hi.any():
                    out[hi] = vals[-1] + (vals[-1] - vals[-2]) / (tt[-1] - tt[-2]) * (tq_lin[hi] - tt[-1])
                return out

            origin[idx_lin] = np.stack([lin(self.origin[sel, i]) for i in range(3)], 1)
            roll[idx_lin] = lin(self.roll[sel])
            pitch[idx_lin] = lin(self.pitch[sel])
            yaw[idx_lin] = (lin(yaw_unw) + 180.0) % 360.0 - 180.0
        return origin, roll, pitch, yaw


def segment_passes(t: np.ndarray, origin: np.ndarray) -> np.ndarray:
    dt = np.diff(t)
    jump = np.linalg.norm(np.diff(origin[:, :2], axis=0), axis=1)
    brk = (dt > PASS_GAP_S) | (jump > PASS_JUMP_M)
    pass_id = np.zeros(len(t), dtype=np.int32)
    pass_id[1:] = np.cumsum(brk)
    return pass_id


def speeds(t: np.ndarray, origin: np.ndarray, pass_id: np.ndarray) -> np.ndarray:
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


def yaw_rates(poses: Poses) -> np.ndarray:
    """|dyaw/dt| in deg/s, central difference inside a pass, one-sided at pass ends.

    Moved here from `mapping/quality.py:78`: it is a pure function of a pose table, and keeping it
    in the frame-screening stage forced pose refinement to import that stage.
    """
    dy = np.full(len(poses), np.nan)
    n = len(poses)
    for k in range(n):
        a = k - 1 if k - 1 >= 0 and poses.pass_id[k - 1] == poses.pass_id[k] else k
        b = k + 1 if k + 1 < n and poses.pass_id[k + 1] == poses.pass_id[k] else k
        if a == b:
            continue
        dy[k] = abs(((poses.yaw[b] - poses.yaw[a] + 180) % 360 - 180) / (poses.t[b] - poses.t[a]))
    return dy
