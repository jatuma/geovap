"""Tile identity as a value object.

Previously a tile's name was recovered by slicing its filename -- `stem.split("_")[1][-3:]` in
`mapping/cloud_store.py:355`, `stem[:3]` in `mapping/report.py:22` -- and its *output* name was
rebuilt from the literal prefix `ID3432_000` in ten different modules. Both assumptions are
Dražkov's, and a dataset named differently would have been written out under the wrong filenames
without any error. Parsing now happens once, in the tile adapter; everything downstream carries a
`TileId` and asks `TileNaming` for filenames.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, order=True)
class TileId:
    """A dataset's own identifier for one tile, e.g. "037". Opaque: never sliced or parsed."""

    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TileRef:
    """A tile as the dataset presents it: its id, its source file, and its extent if known."""

    id: TileId
    path: Path
    bbox: tuple[float, float, float, float] | None = None  # (min_e, min_n, max_e, max_n)


@dataclass(frozen=True)
class TileNaming:
    """Builds output filenames from a template, e.g. "ID3432_000{id}{kind}.laz".

    `{id}` is the TileId; `{kind}` is an optional product suffix ("_colored", "_labels", ...) and is
    empty for the consolidated product. A template without `{kind}` is accepted and the suffix is
    appended before the extension.
    """

    template: str

    def out_name(self, tile: TileId, kind: str = "") -> str:
        if "{kind}" in self.template:
            return self.template.format(id=tile.value, kind=kind)
        name = self.template.format(id=tile.value)
        if not kind:
            return name
        stem, dot, ext = name.rpartition(".")
        return f"{stem}{kind}{dot}{ext}" if dot else f"{name}{kind}"
