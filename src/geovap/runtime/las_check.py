"""Verifying a written tile against the store it came from.

Moved verbatim from `mapping/las_out.py:verify` (the rewrite of that module,
`geovap.io.las_writer`, only WRITES tiles). Verification needs both a `TileData` (the store) and
the writer's provenance VLR (the LAZ on disk), so it cannot live in `io` -- `io` sits below
`runtime` and knows nothing about a point store -- and it cannot live in `stages.deliver` either,
since `stages.deliver` and `stages.verify` are forbidden from importing each other. `runtime` is
the only layer both can reach, so this is where the check lives.
"""
from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from geovap.io.las_writer import PROVENANCE_USER_ID
from geovap.runtime.store import SCALE, TileData


def verify_tile(out_path: Path, td: TileData, expect_class_exact: bool = True, xyz_mode: str = "exact", tol_mm: float = 0.0) -> dict:
    """`xyz_mode="exact"`: written XYZ must be bit-identical to the source LAZ (default, pre-S5 behaviour).
    `xyz_mode="registered"`: written XYZ is compared to `td.xyz_m()` (honours `td.registration`) rounded to
    LAS integers; reports `shift_mm` (max/p50/p99 3D distance in mm) and `n_moved` (points beyond `tol_mm`),
    `xyz_exact` = all points within `tol_mm`."""
    las = laspy.read(str(out_path))
    src = laspy.read(td.info.laz)
    inv = np.asarray(td.orig_index)  # store row -> source position
    out = {"n": len(las.points), "n_src": len(src.points)}
    if xyz_mode == "exact":
        ok_xyz = np.array_equal(las.X, src.X) and np.array_equal(las.Y, src.Y) and np.array_equal(las.Z, src.Z)
        out["xyz_exact"] = bool(ok_xyz)
    elif xyz_mode == "registered":
        xyz_reg_store = np.round(td.xyz_m() / SCALE).astype(np.int64)  # store order
        xyz_reg_src = np.empty_like(xyz_reg_store)
        xyz_reg_src[inv] = xyz_reg_store
        actual = np.stack([np.asarray(las.X), np.asarray(las.Y), np.asarray(las.Z)], axis=1).astype(np.int64)
        diff = (actual - xyz_reg_src).astype(np.float64) * (SCALE * 1000.0)  # mm
        shift_mm = np.sqrt((diff**2).sum(axis=1))
        out["shift_mm"] = {"max": float(shift_mm.max()), "p50": float(np.percentile(shift_mm, 50)), "p99": float(np.percentile(shift_mm, 99))}
        out["n_moved"] = int((shift_mm > tol_mm).sum())
        out["xyz_exact"] = bool(shift_mm.max() <= tol_mm)
        # how far registration moved the points away from the vendor (source) coordinates -- the number a
        # reader actually wants to see; 0 here would mean the registration was NOT applied
        src_xyz = np.stack([np.asarray(src.X), np.asarray(src.Y), np.asarray(src.Z)], axis=1).astype(np.int64)
        dsrc = np.sqrt((((actual - src_xyz).astype(np.float64) * (SCALE * 1000.0)) ** 2).sum(axis=1))
        out["shift_vs_source_mm"] = {"max": float(dsrc.max()), "p50": float(np.percentile(dsrc, 50)), "p99": float(np.percentile(dsrc, 99)),
                                     "frac_moved": float((dsrc > 1.0).mean())}
    else:
        raise ValueError(f"unknown xyz_mode: {xyz_mode!r}")
    ok_cls = np.array_equal(np.asarray(las.classification), np.asarray(src.classification)) if expect_class_exact else True
    out["class_exact"] = bool(ok_cls)
    prov = None
    for v in las.header.vlrs:
        if v.user_id == PROVENANCE_USER_ID:
            import json

            prov = json.loads(bytes(v.record_data).decode())
    out["provenance"] = prov is not None
    return out
