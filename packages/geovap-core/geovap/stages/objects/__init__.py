"""Stream C: segmenting the point cloud into discrete objects.

The most independent part of the pipeline. `cluster_laz` and `merge_tiles` import numpy, laspy,
scipy and skimage and nothing else -- no project code at all -- so this stream can be worked on by
someone with no Geovap context, can run in parallel with everything else from day one, or can be
dropped entirely: the stage is `optional`, and `merge` already tolerates missing cluster input.

Its only contract with the rest of the pipeline is the per-tile LAZ carrying a `cluster_id`
dimension (see `docs/artifacts.md`); the tile names come from the descriptor.

The stage modules are deliberately NOT imported here -- see `geovap.stages.base.discovery`.
"""
