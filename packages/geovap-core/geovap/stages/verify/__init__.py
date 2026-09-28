"""Stage group: verify — cross-checks on the delivered product and the camera model.

Ported from `pointcloud-tools/validate/*.py` and `mapping/pose_report.py` (+
`mapping/cli/pose_report.py`), which lived outside every package and reached into `mapping/` with a
`sys.path.insert` hack. Inside `geovap.stages.verify` they simply import.

This group deliberately does not import `geovap.stages.deliver`: an import-linter contract forbids
stage groups importing each other, so `checks.py` reads the delivered files directly rather than
calling into the `merge`/`octree` stages that produced them.
"""
from __future__ import annotations
