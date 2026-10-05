"""Reference-vectors adapter for a JVF/ZPS GeoJSON export (Czech cadastral/technical-map taxonomy).

Ported from `experiments/common/io_data.load_jvf_objects` (geometry parsing) and
`experiments/common/class_map.py` (the code -> class-info table). The table used to be a Python
dict frozen at import time; it is now data (`jvf_zps_cz.toml`, shipped as package data next to this
module) so a different JVF export, or a fork of the taxonomy, can point `[reference].codes` at its
own file instead of editing code.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from geovap.io.protocols import RefObject
from geovap.io.registry import register

if TYPE_CHECKING:
    from geovap.io.descriptor import Descriptor

_DEFAULT_CODES = Path(__file__).resolve().parent / "jvf_zps_cz.toml"


@dataclass(frozen=True)
class ClassInfo:
    name_cz: str
    public: str | None = None  # None = no good public-taxonomy equivalent (own class)
    note: str = ""


@register("reference", "jvf_zps")
class JvfZpsReferenceVectors:
    def __init__(self, descriptor: "Descriptor", table: dict):
        try:
            self._path = descriptor.data_root / table["file"]
        except KeyError as exc:
            raise ValueError("[reference] table is missing 'file'") from exc
        codes = table.get("codes")
        # A bare filename (no path separator) is package data shipped next to this adapter; a path
        # is resolved relative to the dataset's own data_root, for a project-specific override.
        if not codes:
            self._codes_path = _DEFAULT_CODES
        elif "/" in codes or "\\" in codes:
            self._codes_path = descriptor.data_root / codes
        else:
            self._codes_path = _DEFAULT_CODES.with_name(codes)

    def objects(self) -> list[RefObject]:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        objects = []
        for feat in data["features"]:
            props = feat.get("properties", {}) or {}
            code = props.get("jvfcode")
            if not code:
                continue
            geom = feat.get("geometry")
            if geom is None:
                continue
            gtype = geom["type"]
            raw = geom["coordinates"]
            if gtype == "Point":
                coords = np.array([raw], dtype=np.float64)
            elif gtype == "LineString":
                coords = np.array(raw, dtype=np.float64)
            elif gtype == "Polygon":
                coords = np.array(raw[0], dtype=np.float64)
            else:
                continue
            if coords.shape[1] == 2:
                # missing Z: keep the row (still usable for planar counts) with a NaN height
                coords = np.concatenate([coords, np.full((coords.shape[0], 1), np.nan)], axis=1)
            base = code.split("-")[0]
            objects.append(
                RefObject(
                    code=base,
                    code_full=code,
                    geom_type=gtype,
                    coords=coords,
                    name=props.get("rc") or props.get("name") or "",
                    uid=props.get("uniqueID", ""),
                )
            )
        return objects

    def classes(self) -> dict[str, ClassInfo]:
        with open(self._codes_path, "rb") as f:
            data = tomllib.load(f)
        return {code: ClassInfo(**info) for code, info in data.get("codes", {}).items()}

    def describe(self) -> dict:
        if not self._path.is_file():
            return {"file": str(self._path), "exists": False}
        objects = self.objects()
        by_geom: dict[str, int] = {}
        for obj in objects:
            by_geom[obj.geom_type] = by_geom.get(obj.geom_type, 0) + 1
        return {
            "file": str(self._path),
            "exists": True,
            "count": len(objects),
            "by_geom_type": by_geom,
            "distinct_codes": len({o.code for o in objects}),
            "codes_file": str(self._codes_path),
        }
