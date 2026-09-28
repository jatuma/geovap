"""Tile source for a flat directory of LAZ files, one per tile.

Ported from the tile-name parsing in `mapping/cloud_store.py:_tile_name_from_laz` (`stem.split("_")
[1][-3:]`, Dražkov's `ID3432_000037_JTSK.laz` -> `"037"`) and the tile-grid GeoJSON reader in the
same module's `_load_layout`. The slice is replaced by a named-group regex from the descriptor, and
output names come from `TileNaming` (`domain.model.tiles`) instead of the literal `"ID3432_000"`
prefix that used to be duplicated across modules.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from typing import TYPE_CHECKING

from geovap.domain.model.tiles import TileId, TileNaming, TileRef
from geovap.io.registry import register

if TYPE_CHECKING:
    from geovap.io.descriptor import Descriptor


@register("tiles", "laz_dir")
class LazDirTileSource:
    def __init__(self, descriptor: "Descriptor", table: dict):
        try:
            self._dir = descriptor.data_root / table["dir"]
            pattern = table["id_regex"]
            template = table["out_name"]
        except KeyError as exc:
            raise ValueError(f"[tiles] table is missing {exc}") from exc
        self._glob = table.get("glob", "*.laz")
        self._id_regex = re.compile(pattern)
        if "id" not in self._id_regex.groupindex:
            raise ValueError(f"[tiles].id_regex {pattern!r} has no named group 'id'")
        self._naming = TileNaming(template)
        grid = table.get("grid")
        self._grid_path = descriptor.data_root / grid if grid else None

    def tiles(self) -> list[TileRef]:
        rings = self._load_grid() if self._grid_path is not None else {}
        refs = []
        for path in sorted(self._dir.glob(self._glob)):
            match = self._id_regex.search(path.name)
            if match is None:
                raise ValueError(f"{path}: filename does not match [tiles].id_regex {self._id_regex.pattern!r}")
            tile_id = TileId(match.group("id"))
            # `_load_grid` keys by normalised id, so normalise this side too -- looking up the raw
            # id only happens to work when `id_regex` already strips padding, and fails silently
            # (every bbox None) as soon as it does not.
            refs.append(TileRef(id=tile_id, path=path, ring=rings.get(_normalize_id(tile_id.value))))
        return refs

    def out_name(self, tile: TileId, kind: str = "") -> str:
        return self._naming.out_name(tile, kind)

    def describe(self) -> dict:
        if not self._dir.is_dir():
            return {"dir": str(self._dir), "exists": False}
        refs = self.tiles()
        return {
            "dir": str(self._dir),
            "exists": True,
            "count": len(refs),
            "ids": [t.id.value for t in refs],
            "grid": str(self._grid_path) if self._grid_path else None,
        }

    def _load_grid(self) -> dict[str, "np.ndarray"]:
        """[id -> ring[K,2]] from a tile-grid GeoJSON: tile outlines as LineString rings, each tile's id
        as a Point label inside its ring -- mirrors `mapping/cloud_store.py:_load_layout`.

        The grid's label text (e.g. "001") and the `id_regex` capture from a LAZ filename (e.g.
        "1", once `id_regex`'s own leading-zero stripping has run) need not use the same width, so
        both are normalised to a bare integer string before matching rather than compared as text."""
        from matplotlib.path import Path as MplPath

        if not self._grid_path.is_file():
            return {}
        data = json.loads(self._grid_path.read_text(encoding="utf-8"))
        rings = [
            f["geometry"]["coordinates"]
            for f in data["features"]
            if f["geometry"]["type"] == "LineString"
        ]
        labels = [
            (str(f["properties"].get("name", "")), f["geometry"]["coordinates"])
            for f in data["features"]
            if f["geometry"]["type"] == "Point"
        ]
        out: dict[str, np.ndarray] = {}
        for ring in rings:
            arr = np.asarray(ring, dtype=np.float64)[:, :2]
            poly = MplPath(arr)
            inside = [n for n, (px, py, *_r) in labels if poly.contains_point((px, py))]
            if len(inside) != 1:
                raise ValueError(f"{self._grid_path}: ring has {len(inside)} labels: {inside}")
            out[_normalize_id(inside[0])] = arr
        return out


def _normalize_id(text: str) -> str:
    """A label like "001" and a filename capture like "1" identify the same tile; compare them as
    integers (falling back to the raw text for a non-numeric id) rather than as zero-padded strings."""
    digits = re.search(r"\d+", text)
    return str(int(digits.group())) if digits else text
