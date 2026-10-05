"""Rig-calibration library for the `register` stage group: not stages themselves (no `StageSpec`
here), just the edge-ICP / chamfer / objective machinery `align.py`, `refine.py` and `passes.py`
build their stages on top of. Mirrors `stages.prepare`'s note about `render`/`masks`: these modules
are libraries in this group, not stages, so `stages.register.__init__` does not import them either."""
