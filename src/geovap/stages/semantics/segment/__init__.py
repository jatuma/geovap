"""Stage group: semantics.segment -- zero-shot panorama segmentation and its benchmark.

  seg-eval (`evaluate.py`)   run the registered HF checkpoints (`models.py`) over the levelled
                             gnomonic views, fuse back to ERP (`fusion.py`), evaluate against the
                             JVF pseudo-GT (`stages.semantics.pseudogt`) and report.

`models.py`, `fusion.py` and `bench.py` are libraries this stage's `run()` uses, not stages of
their own. Not imported here -- see `geovap.stages.base.discovery`.
"""
