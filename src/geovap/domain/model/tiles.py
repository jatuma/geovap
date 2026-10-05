"""Tile identity as a value object.

Previously a tile's name was recovered by slicing its filename -- `stem.split("_")[1][-3:]` in
`mapping/cloud_store.py:355`, `stem[:3]` in `mapping/report.py:22` -- and its *output* name was
rebuilt from the literal prefix `ID3432_000` in ten different modules. Both assumptions are
Dražkov's, and a dataset named differently would have been written out under the wrong filenames
without any error. Parsing now happens once, in the tile adapter; everything downstream carries a
`TileId` and asks `TileNaming` for filenames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True, order=True)
class TileId:
    """A dataset's own identifier for one tile, e.g. "037". Opaque: never sliced or parsed."""

    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TileRef:
    """A tile as the dataset presents it: its id, its source file, and its extent if known.

    `ring` is the tile's outline as the dataset's grid defines it, NOT a bounding box: Drazkov's
    tiles are skewed parallelograms, and the store build tests each tile's centre for containment in
    its own ring and persists the polygon alongside the tile. Reducing that to a bbox would
    over-cover neighbouring tiles. `bbox` is derived from the ring when not supplied, for the cheap
    rejection tests that only need an extent.
    """

    id: TileId
    path: Path
    ring: np.ndarray | None = None  # [K,2] float64, world coordinates, first point repeated last
    bbox: tuple[float, float, float, float] | None = None  # (min_e, min_n, max_e, max_n)

    def __post_init__(self) -> None:
        if self.bbox is None and self.ring is not None:
            r = np.asarray(self.ring, dtype=np.float64)
            object.__setattr__(
                self, "bbox", (float(r[:, 0].min()), float(r[:, 1].min()),
                               float(r[:, 0].max()), float(r[:, 1].max())),
            )


@dataclass(frozen=True)
class TileNaming:
    """Builds output filenames from templates, e.g. "ID3432_000{id}{kind}.laz".

    `{id}` is the TileId; `{kind}` is an optional product suffix ("_colored", "_seg", ...) and is
    empty for the consolidated product. A template without `{kind}` is accepted and the suffix is
    appended before the extension.

    `variants` holds the products whose filename is not the main template with a suffix -- the
    clustering stage writes `objects_t000037.laz`, a different prefix entirely. They are named in
    the descriptor's `[tiles.names]` table rather than spelled out in the stage, so a second dataset
    renames them in one place.
    """

    template: str
    variants: Mapping[str, str] = field(default_factory=dict)

    def out_name(self, tile: TileId, kind: str = "", *, variant: str | None = None) -> str:
        template = self.template if variant is None else self._variant(variant)
        if "{kind}" in template:
            return template.format(id=tile.value, kind=kind)
        name = template.format(id=tile.value)
        if not kind:
            return name
        stem, dot, ext = name.rpartition(".")
        return f"{stem}{kind}{dot}{ext}" if dot else f"{name}{kind}"

    def _variant(self, name: str) -> str:
        try:
            return self.variants[name]
        except KeyError:
            known = ", ".join(sorted(self.variants)) or "(none)"
            raise KeyError(
                f"[tiles.names] has no template {name!r}; defined variants: {known}"
            ) from None


def tile_of(name: str, tiles: Iterable[TileId], naming: TileNaming, kind: str = "",
            *, variant: str | None = None) -> TileId | None:
    """Which tile a product filename belongs to, by matching against the names the tiles WOULD be
    written under.

    Templates are not parsed back. `"ID3432_000{id}{kind}.laz"` cannot be inverted unambiguously --
    `ID3432_000037_colored.laz` could be tile "037" of kind "_colored" or tile "037_colored" of no
    kind -- and the old code papered over this by slicing three characters off the name, which is
    the assumption this module exists to remove. The tile set is always known, so generating and
    comparing is both exact and dataset-independent.
    """
    wanted = {naming.out_name(t, kind, variant=variant): t for t in tiles}
    return wanted.get(name)


def id_from_sidecar(path: Path | str, suffix: str = "_meta") -> str:
    """Tile id from one of our own per-tile sidecars, `<id>_meta.json`.

    Replaces `p.stem[:3]` / `p.name[:3]`, which silently assumed a three-character tile id. This is
    OUR filename, not the dataset's, so it needs no template -- only the suffix stripped.
    """
    return Path(path).stem.removesuffix(suffix)
