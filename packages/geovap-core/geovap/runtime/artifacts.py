"""The interfaces between team streams: every artifact one stage writes and another reads.

These are the only things the streams share. Stream B (semantics) never imports stream A's Python;
it reads A's pose table. Stream C (clustering) imports no project code at all; it reads LAZ tiles
and writes a `cluster_id` dimension. So these paths and column names -- not any function signature
-- are what must not drift, and they are declared once here rather than being spelled out as string
literals at each end.

Each entry gives the producing stage, the consuming streams, where the artifact lives for a given
`Settings`, and the schema a `verify` test asserts. `docs/artifacts.md` is generated from this
module, so the documentation cannot go stale independently of the code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings


@dataclass(frozen=True)
class Artifact:
    """One inter-stream contract."""

    name: str
    producer: str  # stage name
    consumers: tuple[str, ...]  # stream letters, as in the restructuring plan
    summary: str
    #: Where it lives, given the resolved settings. A directory or a file.
    locate: Callable[["Settings"], Path]
    #: For a CSV: its required columns. For an .npy/.npz set: its required member names.
    columns: tuple[str, ...] = ()
    #: Sidecar JSON keys that must be present (provenance a consumer relies on).
    meta_keys: tuple[str, ...] = ()
    optional: bool = False
    per_tile: bool = False
    extra: dict = field(default_factory=dict)

    def path(self, s: "Settings") -> Path:
        return self.locate(s)

    def exists(self, s: "Settings") -> bool:
        return self.path(s).exists()


#: Columns of a pose table, in order, as `runtime.pose_tables.write` emits them. Stages may append
#: their own per-frame diagnostic columns after `pass_id`; a consumer must tolerate extras and must
#: not depend on their order.
POSE_TABLE_COLUMNS = ("frame", "filename", "t", "E", "N", "H", "roll", "pitch", "yaw", "pass_id")

#: Provenance a pose table's sidecar always carries. `poses_hash` is the identity every downstream
#: artifact records so a mismatch is detectable rather than silent.
POSE_TABLE_META = ("poses_hash", "git")

#: Per-tile columns of the point store. Frozen: an existing store must keep opening.
STORE_COLUMNS = (
    "xyz", "intensity", "classification", "rgb", "gps_time", "psid", "orig_index", "cell_starts",
)
#: Added later; a store built before them opens with these set to None.
STORE_OPTIONAL_COLUMNS = (
    "user_data", "scan_angle_rank", "return_number", "time_order", "time_bucket_starts",
)

#: Quality classes of the frame screening. Every one is a list of frame indices.
CLEAN_FRAME_CLASSES = ("clean", "unverified", "usable", "reject")


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        name="pose_table",
        producer="assemble",
        consumers=("B", "D", "E", "F"),
        summary="Corrected camera trajectory: one row per panorama, plus a provenance sidecar.",
        locate=lambda s: s.workspace.poses,
        columns=POSE_TABLE_COLUMNS,
        meta_keys=POSE_TABLE_META,
    ),
    Artifact(
        name="point_store",
        producer="store",
        consumers=("A", "B", "D", "E"),
        summary=(
            "Columnar, cell-indexed memmap of the whole cloud, one directory per tile, keyed by "
            "TileId. Global point_id = tile.row_offset + local row."
        ),
        locate=lambda s: s.workspace.store,
        columns=STORE_COLUMNS,
        meta_keys=("total", "cell_size", "tiles"),
        per_tile=True,
    ),
    Artifact(
        name="frame_products",
        producer="products",
        consumers=("A", "B", "D"),
        summary=(
            "Per-frame depth and point-id panoramas at z-buffer resolution -- the inverse mapping "
            "every projection stage reads. One .npz per frame; a corrected pose table writes into "
            "a hash-suffixed subdirectory so export products are never overwritten."
        ),
        locate=lambda s: s.workspace.frames,
        columns=("depth_mm", "point_id", "meta"),
    ),
    Artifact(
        name="clean_frames",
        producer="screen",
        consumers=("B", "D", "F"),
        summary="Frame screening verdict: which panoramas are usable for colour and for training.",
        locate=lambda s: s.workspace.clean_frames_json,
        columns=CLEAN_FRAME_CLASSES,
    ),
    Artifact(
        name="coloured_tiles",
        producer="colorize",
        consumers=("E",),
        summary="Per-tile fused colour with its QA dimensions, plus per-tile stats.",
        locate=lambda s: s.workspace.out / "colour",
        meta_keys=("n", "seconds", "n_frames", "coverage"),
        per_tile=True,
    ),
    Artifact(
        name="point_labels",
        producer="label",
        consumers=("E",),
        summary=(
            "One uint8 semantic class per store point, in store row order, per tile. The sidecar "
            "carries the poses_hash it was projected under; merging labels from a different pose "
            "table is a detectable error, not a silent one."
        ),
        locate=lambda s: s.workspace.segds / "point_labels",
        meta_keys=("rules_hash", "total", "tiles"),
        optional=True,
        per_tile=True,
    ),
    Artifact(
        name="object_clusters",
        producer="cluster",
        consumers=("E",),
        summary=(
            "Per-tile LAZ carrying a `cluster_id` extra dimension. Produced by code that imports "
            "no project Python at all (numpy, laspy, scipy only), which is what makes clustering a "
            "detachable branch."
        ),
        locate=lambda s: s.workspace.out / "clusters",
        optional=True,
        per_tile=True,
    ),
    Artifact(
        name="consolidated_tiles",
        producer="merge",
        consumers=("F",),
        summary=(
            "THE PRODUCT: one LAZ per tile carrying every dimension -- colour, semantics, objects, "
            "reference RGB and colour provenance. Replaces the three parallel LAZ sets (tiles/, "
            "objects/, vendor/) that duplicated the same 585 M XYZ triples three times over."
        ),
        locate=lambda s: s.workspace.consolidated_tiles,
        per_tile=True,
    ),
)

BY_NAME: dict[str, Artifact] = {a.name: a for a in ARTIFACTS}


def get(name: str) -> Artifact:
    try:
        return BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown artifact {name!r}; known: {', '.join(sorted(BY_NAME))}") from None


def as_markdown() -> str:
    """The table in `docs/artifacts.md`, generated so it cannot drift from this module."""
    rows = [
        "| artifact | producer | consumers | optional | summary |",
        "|---|---|---|---|---|",
    ]
    for a in ARTIFACTS:
        rows.append(
            f"| `{a.name}` | `{a.producer}` | {', '.join(a.consumers)} | "
            f"{'yes' if a.optional else 'no'} | {a.summary} |"
        )
    return "\n".join(rows)


#: The generated table in `docs/artifacts.md` sits between these markers; the prose around them is
#: written by hand and is not touched by regeneration.
DOC_BEGIN = "<!-- BEGIN GENERATED: geovap.runtime.artifacts -->"
DOC_END = "<!-- END GENERATED -->"


def render_doc(existing: str) -> str:
    """`existing` with the region between the markers replaced by the current table."""
    try:
        head, rest = existing.split(DOC_BEGIN, 1)
        _, tail = rest.split(DOC_END, 1)
    except ValueError:
        raise ValueError(f"docs/artifacts.md is missing the {DOC_BEGIN} / {DOC_END} markers") from None
    return f"{head}{DOC_BEGIN}\n\n{as_markdown()}\n\n{DOC_END}{tail}"


if __name__ == "__main__":  # `uv run python -m geovap.runtime.artifacts` regenerates the doc
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/artifacts.md")
    target.write_text(render_doc(target.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"regenerated {target}")
