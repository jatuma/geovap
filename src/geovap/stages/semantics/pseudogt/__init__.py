"""Stage group: semantics.pseudogt -- the JVF pseudo-ground-truth dataset.

  segds (`dataset.py`)   areas -> rasters -> point labels -> ERP labels -> gnomonic views ->
                         dataset metadata (+ the nearfield alignment check)

`areas`, `rasters`, `points`, `erp`, `views` and `nearfield` are libraries this stage's `run()`
calls in sequence, not stages of their own -- see `dataset.py`'s module docstring for why they
were not split into separate resumable steps.

Not imported here: see `geovap.stages.base.discovery`, and this package's own entry in
`discovery.GROUPS` (`semantics.pseudogt`, alongside `semantics.segment` and `semantics.label`) for
why a `semantics` sub-group needs telling apart from the single-level groups.
"""
