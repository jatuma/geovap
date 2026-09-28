"""Stage group: semantics -- pseudo-ground-truth, zero-shot segmentation, and label projection.

The only distribution allowed to import torch/transformers/shapely (see the `core-without-torch`
import-linter contract). Split into three sub-groups by what they do:

  pseudogt   JVF pseudo-ground-truth dataset (areas -> rasters -> point/ERP labels -> views)
  segment    zero-shot panorama segmentation + its benchmark against the pseudo-GT
  label      project the 2D segmentation masks into the 3D point cloud

Unlike the single-level groups (`prepare`, `colour`, ...), `geovap.stages.base.discovery` cannot
just walk this package's own top-level modules: there are none, only the three sub-packages above.
`discovery.GROUPS` therefore lists `semantics.pseudogt`, `semantics.segment` and `semantics.label`
directly rather than `semantics` -- each is walked for its own stage modules, exactly like a
single-level group. `segment` builds on `pseudogt` and `label` builds on `segment`, which is the one
import chain the `stage-independence` contract allows within one stage group.
"""
