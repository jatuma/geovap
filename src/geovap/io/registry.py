"""Adapter registry: maps the adapter name a descriptor names (e.g. `"laz_dir"`) to the class that
implements it, so `Descriptor` stays a plain data object with no knowledge of the adapter modules.

Adapters register themselves with `@register(...)` on import; this module only imports the
built-in adapter modules once, at the bottom, so registration happens as a side effect of
importing `geovap.io.registry` itself.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from geovap.io.descriptor import Descriptor
    from geovap.io.protocols import PanoSource, PoseSource, ReferenceVectors, TileSource

_POSES: dict[str, type] = {}
_TILES: dict[str, type] = {}
_PANOS: dict[str, type] = {}
_REFERENCE: dict[str, type] = {}

_KINDS = {"poses": _POSES, "tiles": _TILES, "panos": _PANOS, "reference": _REFERENCE}

T = TypeVar("T")


class UnknownAdapterError(ValueError):
    def __init__(self, kind: str, name: str, registry: dict[str, type]):
        known = ", ".join(sorted(registry)) or "(none registered)"
        super().__init__(f"unknown {kind} adapter {name!r}; registered adapters: {known}")


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: `@register("tiles", "laz_dir")` on `LazDirTileSource`."""
    if kind not in _KINDS:
        raise ValueError(f"unknown adapter kind {kind!r}; expected one of {sorted(_KINDS)}")

    def decorator(cls: type[T]) -> type[T]:
        _KINDS[kind][name] = cls
        return cls

    return decorator


def _build(kind: str, registry: dict[str, type], descriptor: "Descriptor", table: dict):
    try:
        name = table["adapter"]
    except KeyError as exc:
        raise ValueError(f"[{kind}] table is missing 'adapter'") from exc
    try:
        cls = registry[name]
    except KeyError as exc:
        raise UnknownAdapterError(kind, name, registry) from exc
    return cls(descriptor, table)


def build_pose_source(descriptor: "Descriptor") -> "PoseSource":
    return _build("poses", _POSES, descriptor, descriptor.poses)


def build_tile_source(descriptor: "Descriptor") -> "TileSource":
    return _build("tiles", _TILES, descriptor, descriptor.tiles)


def build_pano_source(descriptor: "Descriptor") -> "PanoSource":
    return _build("panos", _PANOS, descriptor, descriptor.panos)


def build_reference(descriptor: "Descriptor") -> "ReferenceVectors | None":
    """`None` when the descriptor has no `[reference]` table: reference vectors are optional."""
    if not descriptor.reference:
        return None
    return _build("reference", _REFERENCE, descriptor, descriptor.reference)


# Import for registration side effects only (each module decorates its class with `@register`).
from geovap.io.adapters.panos import equirect_dir  # noqa: E402, F401
from geovap.io.adapters.poses import ladybug_export_csv  # noqa: E402, F401
from geovap.io.adapters.reference import jvf_zps  # noqa: E402, F401
from geovap.io.adapters.tiles import laz_dir  # noqa: E402, F401
