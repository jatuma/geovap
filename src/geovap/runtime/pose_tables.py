"""Our own pose-table artifacts: reading and writing the CSVs the registration stage produces.

`geovap.domain.model.poses.Poses` is pure data and maths; the vendor export is read by
`geovap.io.adapters.poses` (a `PoseSource` adapter chosen by the descriptor); this module is the
third leg -- our own pose-table CSV, written by the registration stage and re-read by anything that
opts into a corrected trajectory.

`load(source)` selects among pose tables: "export" (the regression anchor, unchanged behaviour,
read through the descriptor's configured `PoseSource` adapter) or a corrected table written by
`write` (name "corrected" or any CSV path). Downstream code opts into a corrected table only
explicitly.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from geovap.domain.model.poses import Poses, speeds

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings

#: How a pose table's dense-trajectory sidecar is loaded back into `Poses.traj`.
#:
#: `Poses.traj` is typed `object | None` precisely so `domain` need not know what a trajectory is,
#: and the class that implements one (~1000 lines of scipy plane fitting) belongs to the
#: registration stage -- which sits ABOVE `runtime` in the layering and so cannot be imported from
#: here. `geovap.stages.register` closes the loop by calling `set_trajectory_loader` on import.
#: When it is not installed, or the dataset never had a trajectory fitted, `traj` simply stays
#: `None` -- exactly the best-effort behaviour the legacy `try/except` import gave.
_trajectory_loader: "Callable[[Path], object] | None" = None


def set_trajectory_loader(fn: "Callable[[Path], object] | None") -> None:
    """Register the reader for the `.trajectory.npz` sidecar. Called by `geovap.stages.register`."""
    global _trajectory_loader
    _trajectory_loader = fn


def read(path: str | Path, *, traj_loader: "Callable[[Path], object] | None" = None) -> Poses:
    """Read a pose table written by `write`. `pass_id` is taken from the table as-is (never
    re-segmented): a corrected table may have been hand-edited or produced by a registration pass
    that moved frames between passes on purpose, and re-deriving `pass_id` from the (now corrected)
    origins could silently disagree with whatever the writer intended. `filename`/`t` order is
    whatever was written, expected to match export ordering so frame indices stay stable. Speed is
    recomputed. Source is the file stem. If a sidecar JSON of the same stem names a "trajectory" and
    the npz exists, it is attached as `traj` (best-effort: `geovap.domain.model.trajectory` may not
    exist yet). If the sidecar names a "registration", it is attached as `registration`: the corrected
    poses are only geometrically consistent with a `CloudStore` opened against that path."""
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
    speed = speeds(t, origin, pass_id)
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
            loader = traj_loader if traj_loader is not None else _trajectory_loader
            if traj_name and loader is not None:
                traj_path = path.parent / traj_name
                if traj_path.exists():
                    poses.traj = loader(traj_path)
        except Exception:
            pass  # traj stays None: no loader registered, or the npz is missing/corrupt
        try:
            reg = json.loads(sidecar.read_text()).get("registration")
            if reg:
                reg_path = reg.get("path") if isinstance(reg, dict) else reg
                if reg_path:
                    poses.registration = Path(reg_path)
        except Exception:
            pass  # registration stays None
    return poses


def _sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(poses: Poses, per_frame_meta: dict[str, np.ndarray], provenance: dict, path: str | Path) -> Path:
    """Write a pose table CSV (frame, filename, t, E, N, H, roll, pitch, yaw, pass_id, <per_frame_meta
    columns>) plus a sidecar `<path>.json` with `provenance` extended by `poses_hash` (see
    `Poses.hash` for why this is stable across a CSV round-trip), the current git rev, and the sha1
    of the export.csv this correction was derived from."""
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

    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        git_rev = "unknown"

    from geovap.runtime import settings as _settings

    s = _settings.get()
    # Provenance of the vendor table this correction descends from. `config.py` hashed a hardcoded
    # `EXPORT_CSV`; the adapter now declares its own source file, so the key is named for what it is
    # rather than for Dražkov's filename.
    src = s.poses.source_file()
    prov = {
        **provenance,
        "poses_hash": poses.hash(),
        "git": git_rev,
        "pose_source_file": str(src) if src else None,
        "pose_source_sha1": _sha1_file(src) if src and src.exists() else None,
    }
    path.with_suffix(".json").write_text(json.dumps(prov, indent=2, default=str))
    return path


def path_for(name: str, *, s: "Settings | None" = None) -> Path:
    """Resolve a bare table name (e.g. "corrected") to a file under the workspace, mirroring the
    non-"export" branches of `load` exactly: "corrected" -> `<workspace>/out/poses/poses_corrected.csv`;
    anything else is treated as a path already and returned as-is."""
    if s is None:
        from geovap.runtime import settings as _settings

        s = _settings.get()
    if name == "corrected":
        return s.workspace.poses / "poses_corrected.csv"
    return Path(name)


def load(source: str | Path | None = None, *, s: "Settings | None" = None) -> Poses:
    """Load a pose table. `source`: None -> `s.pose_table` (was `config.POSES_SOURCE`, env
    `GEOVAP_POSES`, default "export"); "export" -> the descriptor's configured `PoseSource` adapter
    (current behaviour, regression anchor, unchanged); "corrected" ->
    `<workspace>/out/poses/poses_corrected.csv`; any other str/Path -> that CSV path.

    `s` defaults to `geovap.runtime.settings.get()`, imported here (not at module scope) so this
    module stays importable standalone."""
    from geovap.runtime import settings as _settings

    if s is None:
        s = _settings.get()
    if source is None:
        source = s.pose_table
    source = str(source)
    if source == "export":
        return s.poses.load()
    if source == "corrected":
        return read(path_for("corrected", s=s))
    return read(source)
