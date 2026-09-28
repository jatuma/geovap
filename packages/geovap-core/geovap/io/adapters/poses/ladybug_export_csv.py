"""Pose source for a Ladybug-style `export.csv`: one row per frame, columns given by position.

Ported from `experiments/common/io_data.load_frames()`. There the column indices (0, 1, 2, 3, 4,
11, 12, 13) and the "exactly 17 columns" assertion were hardcoded for Dražkov's export; here both
come from the descriptor's `[poses]` table, so a differently-shaped export from another project
only needs a different `columns` mapping, not a new adapter.
"""
from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np

from geovap.domain.model.poses import Poses, segment_passes, speeds
from geovap.io.registry import register

if TYPE_CHECKING:
    from geovap.io.descriptor import Descriptor

_REQUIRED_COLUMNS = ("t", "file", "e", "n", "h", "roll", "pitch", "yaw")


@register("poses", "ladybug_export_csv")
class LadybugExportCsvPoseSource:
    def __init__(self, descriptor: "Descriptor", table: dict):
        self._descriptor = descriptor
        self._table = table
        try:
            self._columns: dict[str, int] = table["columns"]
        except KeyError as exc:
            raise ValueError("[poses] table is missing 'columns'") from exc
        missing = [c for c in _REQUIRED_COLUMNS if c not in self._columns]
        if missing:
            raise ValueError(f"[poses].columns is missing: {', '.join(missing)}")
        try:
            self._path = descriptor.data_root / table["file"]
        except KeyError as exc:
            raise ValueError("[poses] table is missing 'file'") from exc
        self._expect_columns = table.get("expect_columns")

    def load(self) -> Poses:
        col = self._columns
        filenames, timestamps, origins, rolls, pitches, yaws = [], [], [], [], [], []
        with open(self._path, encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            if self._expect_columns is not None and len(header) != self._expect_columns:
                raise ValueError(
                    f"{self._path}: expected {self._expect_columns} columns in the export.csv header, "
                    f"got {len(header)}"
                )
            for row in reader:
                if not row or not row[0]:
                    continue
                timestamps.append(float(row[col["t"]]))
                filenames.append(row[col["file"]])
                origins.append((float(row[col["e"]]), float(row[col["n"]]), float(row[col["h"]])))
                rolls.append(float(row[col["roll"]]))
                pitches.append(float(row[col["pitch"]]))
                yaws.append(float(row[col["yaw"]]))

        t = np.array(timestamps, dtype=np.float64)
        origin = np.array(origins, dtype=np.float64)
        order = np.argsort(t)
        t = t[order]
        origin = origin[order]
        filename = np.array(filenames, dtype=object)[order]
        roll = np.array(rolls, dtype=np.float64)[order]
        pitch = np.array(pitches, dtype=np.float64)[order]
        yaw = np.array(yaws, dtype=np.float64)[order]

        pass_id = segment_passes(t, origin)
        return Poses(
            filename=filename,
            t=t,
            origin=origin,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            pass_id=pass_id,
            speed=speeds(t, origin, pass_id),
            source="export",
        )

    def describe(self) -> dict:
        if not self._path.exists():
            return {"file": str(self._path), "exists": False}
        poses = self.load()
        return {
            "file": str(self._path),
            "exists": True,
            "count": len(poses),
            "t_min": float(poses.t.min()) if len(poses) else None,
            "t_max": float(poses.t.max()) if len(poses) else None,
            "passes": int(poses.pass_id.max()) + 1 if len(poses) else 0,
            "extent_e": (float(poses.origin[:, 0].min()), float(poses.origin[:, 0].max())) if len(poses) else None,
            "extent_n": (float(poses.origin[:, 1].min()), float(poses.origin[:, 1].max())) if len(poses) else None,
        }
