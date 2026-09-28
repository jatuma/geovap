"""Interfaces a dataset adapter must satisfy.

Before this module, every stage imported `mapping.io_data` (or `experiments/common/io_data.py`)
directly and got Dražkov's paths, csv layout and geojson shape baked in. `runtime`/`stages` code
now depends only on these Protocols; `io.registry` resolves the concrete class named in a
descriptor. Runtime-checkable so `isinstance(x, PoseSource)` works for adapters that don't
explicitly subclass anything (structural typing, matching how `domain` already works).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from geovap.domain.model.poses import Poses
from geovap.domain.model.tiles import TileId, TileRef


@runtime_checkable
class PoseSource(Protocol):
    """Reads one dataset's camera trajectory."""

    def load(self) -> Poses: ...

    def source_file(self) -> Path | None:
        """The file this trajectory was read from, for provenance. `None` for an adapter that has no
        single file (a database, a directory of per-pass tables). A pose table written downstream
        records its hash, which is how a run can prove which vendor export it descends from."""
        ...

    def describe(self) -> dict:
        """Summary for `geovap doctor`: frame count, time span, spatial extent, source file."""
        ...


@runtime_checkable
class TileSource(Protocol):
    """Enumerates a dataset's point-cloud tiles and names their output products."""

    def tiles(self) -> list[TileRef]: ...

    def out_name(self, tile: TileId, kind: str = "") -> str: ...

    def describe(self) -> dict: ...


@runtime_checkable
class PanoSource(Protocol):
    """Resolves panorama filenames (as they appear in a pose table) to files on disk."""

    def path(self, filename: str) -> Path: ...

    def describe(self) -> dict: ...


@dataclass(frozen=True)
class RefObject:
    """One reference-vector feature, mirroring `experiments/common/io_data.JvfObject` but named
    generically: a `ReferenceVectors` adapter for a non-JVF dataset produces the same shape."""

    code: str  # base code, e.g. "0100000162" (adapter-specific taxonomy, before class_map lookup)
    code_full: str  # e.g. "0100000162-01"
    geom_type: str  # "LineString" | "Point" | "Polygon"
    coords: np.ndarray  # [K, 3] float64 (E, N, H); K=1 for Point
    name: str  # human-readable hint (a code, label or property value)
    uid: str


@runtime_checkable
class ReferenceVectors(Protocol):
    """Optional: a taxonomy of reference vector objects for evaluation. Not every dataset has one,
    so `io.registry.build_reference` returns `None` rather than a stub implementation."""

    def objects(self) -> list[RefObject]: ...

    def classes(self): ...

    def describe(self) -> dict: ...
