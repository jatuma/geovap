# Artifact contracts

These are the interfaces between the team streams. A stream owns whole directories and may change
anything inside them; what it may not change alone is the shape of an artifact another stream reads.

Streams never import each other's Python. Stream B (semantics) does not call stream A's registration
code — it reads A's pose table off disk. Stream C (clustering) imports no project code at all: it
reads LAZ tiles and writes a `cluster_id` dimension, which is what makes it droppable. So the paths
and column names below, not any function signature, are the coupling that matters.

The table is generated from `geovap/runtime/artifacts.py`, where each entry also carries the code
that locates it for a given dataset. Regenerate with:

    uv run python -m geovap.runtime.artifacts

<!-- BEGIN GENERATED: geovap.runtime.artifacts -->

| artifact | producer | consumers | optional | summary |
|---|---|---|---|---|
| `pose_table` | `assemble` | B, D, E, F | no | Corrected camera trajectory: one row per panorama, plus a provenance sidecar. |
| `point_store` | `store` | A, B, D, E | no | Columnar, cell-indexed memmap of the whole cloud, one directory per tile, keyed by TileId. Global point_id = tile.row_offset + local row. |
| `frame_products` | `products` | A, B, D | no | Per-frame depth and point-id panoramas at z-buffer resolution -- the inverse mapping every projection stage reads. One .npz per frame; a corrected pose table writes into a hash-suffixed subdirectory so export products are never overwritten. |
| `clean_frames` | `screen` | B, D, F | no | Frame screening verdict: which panoramas are usable for colour and for training. |
| `coloured_tiles` | `colorize` | E | no | Per-tile fused colour with its QA dimensions, plus per-tile stats. |
| `point_labels` | `label` | E | yes | One uint8 semantic class per store point, in store row order, per tile. The sidecar carries the poses_hash it was projected under; merging labels from a different pose table is a detectable error, not a silent one. |
| `object_clusters` | `cluster` | E | yes | Per-tile LAZ carrying a `cluster_id` extra dimension. Produced by code that imports no project Python at all (numpy, laspy, scipy only), which is what makes clustering a detachable branch. |
| `consolidated_tiles` | `merge` | F | no | THE PRODUCT: one LAZ per tile carrying every dimension -- colour, semantics, objects, reference RGB and colour provenance. Replaces the three parallel LAZ sets (tiles/, objects/, vendor/) that duplicated the same 585 M XYZ triples three times over. |

<!-- END GENERATED -->

## Rules

**A producer may add, never rename or remove.** Consumers must tolerate columns and dimensions they
do not know about. A pose table carrying extra per-frame diagnostics is still a valid pose table.

**Every derived artifact records the `poses_hash` it was produced under.** Two artifacts built from
different pose tables must not be merged; because the hash travels with each, that is a detectable
error rather than a silently wrong product. This is why `Poses.hash()` is defined to survive a
round-trip through CSV text.

**Optional artifacts are absent, not empty.** A dataset with no reference vectors produces no point
labels, and the stages that would consume them report `available() is False`. A run completes
without them; it does not fail and does not write placeholder files.

**Paths come from `Settings`, filenames from `TileSource.out_name`.** No stage builds an output path
from a literal. See `geovap/domain/model/tiles.py` for why the name cannot simply be parsed back out
of a filename.

## Testing against them

Every stage is runnable against the generated fixture dataset, which needs no real data:

    uv run python -m geovap.io.datasets.synthetic /tmp/fixture
    GEOVAP_SYNTHETIC_ROOT=/tmp/fixture uv run geovap doctor --dataset synthetic

A schema test for each artifact is run by both its producer and its consumers, so one stream cannot
change a shared format without failing another stream's suite.
