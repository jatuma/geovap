"""Camera poses from export.csv: sorted by time, segmented into passes, interpolable in time.

Wraps `experiments/common/io_data.load_frames` (pure csv, no pandas) instead of re-parsing.

`load_poses(source=None)` selects among pose tables: "export" (default, the regression anchor,
unchanged behaviour) or a corrected table written by `write_pose_table` (name "corrected" or any
CSV path). Downstream code opts into a corrected table only explicitly.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

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

    def path(self, i: int) -> str:
        from .config import PANO_DIR

        return str(PANO_DIR / str(self.filename[i]))

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


def _load_export() -> Poses:
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
        source="export",
    )


def read_pose_table(path: str | Path) -> Poses:
    """Read a pose table written by `write_pose_table`. `pass_id` is taken from the table as-is
    (never re-segmented); `filename`/`t` order is whatever was written, expected to match export
    ordering so frame indices stay stable. Speed is recomputed. Source is the file stem. If a
    sidecar JSON of the same stem names a "trajectory" and the npz exists, it is attached as `traj`
    (best-effort: mapping.trajectory may not exist yet)."""
    path = Path(path)
    filenames, ts, E, N, H, roll, pitch, yaw, pass_id = [], [], [], [], [], [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            filenames.append(row["filename"])
            ts.append(float(row["t"]))
            E.append(float(row["E"]))
            N.append(float(row["N"]))
            H.append(float(row["H"]))
            roll.append(float(row["roll"]))
            pitch.append(float(row["pitch"]))
            yaw.append(float(row["yaw"]))
            pass_id.append(int(row["pass_id"]))
    t = np.asarray(ts, dtype=np.float64)
    origin = np.stack([np.asarray(E), np.asarray(N), np.asarray(H)], axis=1).astype(np.float64)
    pass_id = np.asarray(pass_id, dtype=np.int32)
    speed = _speeds(t, origin, pass_id)
    poses = Poses(
        filename=np.asarray(filenames, dtype=object),
        t=t,
        origin=origin,
        roll=np.asarray(roll, dtype=np.float64),
        pitch=np.asarray(pitch, dtype=np.float64),
        yaw=np.asarray(yaw, dtype=np.float64),
        pass_id=pass_id,
        speed=speed,
        source=path.stem,
    )
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            prov = json.loads(sidecar.read_text())
            traj_name = prov.get("trajectory")
            if traj_name:
                traj_path = path.parent / traj_name
                if traj_path.exists():
                    from .trajectory import Trajectory

                    poses.traj = Trajectory.load(traj_path)
        except Exception:
            pass  # traj stays None; mapping.trajectory may not exist yet, or npz missing/corrupt
        try:
            reg = json.loads(sidecar.read_text()).get("registration")
            if reg:
                reg_path = reg.get("path") if isinstance(reg, dict) else reg
                if reg_path:
                    poses.registration = Path(reg_path)
        except Exception:
            pass  # registration stays None
    return poses


def write_pose_table(poses: Poses, per_frame_meta: dict[str, np.ndarray], provenance: dict, path: str | Path) -> Path:
    """Write a pose table CSV (frame, filename, t, E, N, H, roll, pitch, yaw, pass_id, <per_frame_meta
    columns>) plus a sidecar `<path>.json` with `provenance` extended by `poses_hash`, git rev and the
    sha1 of export.csv."""
    from .config import EXPORT_CSV

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_cols = list(per_frame_meta.keys())
    fieldnames = ["frame", "filename", "t", "E", "N", "H", "roll", "pitch", "yaw", "pass_id", *meta_cols]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fieldnames)
        for i in range(len(poses)):
            row = [
                i,
                poses.filename[i],
                repr(float(poses.t[i])),
                repr(float(poses.origin[i, 0])),
                repr(float(poses.origin[i, 1])),
                repr(float(poses.origin[i, 2])),
                repr(float(poses.roll[i])),
                repr(float(poses.pitch[i])),
                repr(float(poses.yaw[i])),
                int(poses.pass_id[i]),
                *[per_frame_meta[c][i] for c in meta_cols],
            ]
            w.writerow(row)

    def _sha1_file(p: Path) -> str:
        h = hashlib.sha1()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        git_rev = "unknown"

    prov = {
        **provenance,
        "poses_hash": poses.hash(),
        "git": git_rev,
        "export_csv_sha1": _sha1_file(EXPORT_CSV) if EXPORT_CSV.exists() else None,
    }
    path.with_suffix(".json").write_text(json.dumps(prov, indent=2, default=str))
    return path


def load_poses(source: str | Path | None = None) -> Poses:
    """Load a pose table. `source`: None -> `config.POSES_SOURCE` (env `GEOVAP_POSES`, default
    "export"); "export" -> current behaviour (regression anchor, unchanged); "corrected" ->
    `POSES_DIR/poses_corrected.csv`; any other str/Path -> that CSV path."""
    from .config import POSES_DIR, POSES_SOURCE

    if source is None:
        source = POSES_SOURCE
    source = str(source)
    if source == "export":
        return _load_export()
    if source == "corrected":
        return read_pose_table(POSES_DIR / "poses_corrected.csv")
    return read_pose_table(source)
