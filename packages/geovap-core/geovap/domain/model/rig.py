"""Rig model: constant camera-to-body boresight, lever arm and time offset.

All three are zero for the identity rig, which reproduces the verified pilot geometry exactly.
Products derived under a rig (depth panoramas, colorized clouds) carry `RigModel.hash()`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RigModel:
    # boresight (deg): rotation body -> camera, applied after the per-frame vehicle rotation.
    # omega about x (forward), phi about y (left), kappa about z (up)
    boresight_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # lever arm of the camera centre in body axes (x forward, y left, z up), metres
    lever_arm_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # camera time offset: the image was exposed at t_csv + dt_s
    dt_s: float = 0.0
    note: str = field(default="", compare=False)

    @property
    def is_identity(self) -> bool:
        return not any(self.boresight_deg) and not any(self.lever_arm_m) and self.dt_s == 0.0

    def as_vector(self, with_lever_arm: bool = False) -> np.ndarray:
        v = [*self.boresight_deg, self.dt_s]
        if with_lever_arm:
            v += list(self.lever_arm_m)
        return np.asarray(v, dtype=np.float64)

    @classmethod
    def from_vector(cls, v, with_lever_arm: bool = False, note: str = "") -> "RigModel":
        v = np.asarray(v, dtype=np.float64)
        la = (float(v[4]), float(v[5]), float(v[6])) if with_lever_arm else (0.0, 0.0, 0.0)
        return cls(boresight_deg=(float(v[0]), float(v[1]), float(v[2])), lever_arm_m=la, dt_s=float(v[3]), note=note)

    def hash(self) -> str:
        payload = json.dumps(
            {"b": [round(x, 6) for x in self.boresight_deg], "l": [round(x, 5) for x in self.lever_arm_m], "dt": round(self.dt_s, 6)},
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({**asdict(self), "hash": self.hash()}, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "RigModel":
        d = json.loads(Path(path).read_text())
        return cls(
            boresight_deg=tuple(d["boresight_deg"]),
            lever_arm_m=tuple(d["lever_arm_m"]),
            dt_s=float(d["dt_s"]),
            note=d.get("note", ""),
        )


IDENTITY = RigModel()
