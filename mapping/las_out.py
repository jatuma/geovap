"""LAS 1.4 PF7 writer: source points bit-exact + our colours + diagnostic extra dims + provenance VLR."""
from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np

from geovap.runtime.store import SCALE, TileData

EXTRA_DIMS = [
    ("ref_r", np.uint8, "TerraScan RGB (reference), 8 bit"),
    ("ref_g", np.uint8, ""),
    ("ref_b", np.uint8, ""),
    ("nt_r", np.uint8, "nearest-in-time single frame, with occlusion"),
    ("nt_g", np.uint8, ""),
    ("nt_b", np.uint8, ""),
    ("dE00_med", np.float32, "CIEDE2000 median-top-k product vs reference"),
    ("dE00_nt", np.float32, "CIEDE2000 nearest-in-time (occlusion) vs reference"),
    ("dE00_nt_noocc", np.float32, "CIEDE2000 nearest-in-time (no occlusion) vs reference"),
    ("src_image", np.uint16, "frame index of best-scoring sample (product)"),
    ("n_views", np.uint8, "number of visible frames fused (0 = not coloured)"),
    ("col_conf", np.uint8, "fusion confidence 0-255"),
    ("cam_dist", np.float32, "camera distance, nearest-in-time frame [m]"),
    ("inc_angle", np.uint8, "incidence angle [deg], nearest-in-time frame (255 = unknown)"),
    ("img_grad", np.float32, "image gradient at the sampled pixel, nearest-in-time frame"),
]

# semantic-segmentation product (mapping.seg.project): `classification` carries the common15 label
SEG_EXTRA_DIMS = [
    ("seg_conf", np.uint8, "label vote share 0-255 (winning class)"),
    ("seg_n_views", np.uint8, "number of frames that voted (0 = unlabelled)"),
    ("seg_src_frame", np.uint16, "frame index of the strongest vote"),
    ("src_class", np.uint8, "original LAS classification"),
]

# consolidated product (mapping.merge): classification carries the common15 label (255 = unlabelled), rgb the
# tw45 colorization product (or reference RGB where n_views == 0)
CONS_EXTRA_DIMS = [
    ("src_class", np.uint8, "original LAS classification"),
    ("seg_conf", np.uint8, "label vote share 0-255 (winning class)"),
    ("seg_n_views", np.uint8, "number of frames that voted (0 = unlabelled)"),
    ("cluster_id", np.int32, "object cluster id (-1 = none)"),
    ("obj_class", np.uint8, "object class"),
    ("hag", np.float32, "height above ground [m]"),
    ("ref_r", np.uint8, "TerraScan RGB (reference), 8 bit"),
    ("ref_g", np.uint8, ""),
    ("ref_b", np.uint8, ""),
    ("dE00_med", np.float32, "CIEDE2000 median-top-k product vs reference"),
    ("n_views", np.uint8, "number of visible frames fused (0 = not coloured)"),
    ("col_conf", np.uint8, "fusion confidence 0-255"),
]

# object-cluster product (mapping.merge): rgb carries the cluster palette
OBJ_EXTRA_DIMS = [
    ("cluster_id", np.int32, "object cluster id (-1 = none)"),
    ("obj_class", np.uint8, "object class"),
]

PROVENANCE_USER_ID = "geovap_map"
PROVENANCE_RECORD_ID = 1


def write_tile(td: TileData, out_path: Path, product_rgb: np.ndarray, extras: dict[str, np.ndarray], provenance: dict,
               extra_dims: list[tuple] = EXTRA_DIMS, classification: np.ndarray | None = None, description: str = "colorization provenance",
               xyz: np.ndarray | None = None) -> Path:
    """`product_rgb`, `extras`, `classification` and `xyz` arrays are in STORE (cell-sorted) order; written in source order.

    `classification` replaces the source classification when given (e.g. semantic labels); otherwise it is copied bit-exact.
    `product_rgb` is u8 [n,3] (scaled to 16 bit, `*256`) unless it is already `uint16` (e.g. a cluster palette), in which
    case it passes through unchanged.
    `xyz` is int32 [n,3] LAS integers (same scale/offset as the source, see `geovap.runtime.store.SCALE`); when given it replaces
    the bit-exact source copy (e.g. the S5 per-pass registered coordinates); otherwise `src.X/Y/Z` is copied bit-exact.
    """
    src = laspy.read(td.info.laz)
    n = len(src.points)
    assert n == len(td)
    inv = np.asarray(td.orig_index)  # store row -> source position

    header = laspy.LasHeader(version="1.4", point_format=7)
    header.scales = src.header.scales
    header.offsets = src.header.offsets
    for name, dt, desc in extra_dims:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dt, description=desc[:31]))
    header.vlrs.append(laspy.vlrs.VLR(user_id=PROVENANCE_USER_ID, record_id=PROVENANCE_RECORD_ID, description=description[:31], record_data=json.dumps(provenance).encode()))

    las = laspy.LasData(header)
    if xyz is not None:
        xyz_src = np.empty((n, 3), dtype=np.int32)
        xyz_src[inv] = np.asarray(xyz, dtype=np.int32)
        las.X, las.Y, las.Z = xyz_src[:, 0], xyz_src[:, 1], xyz_src[:, 2]
    else:
        las.X, las.Y, las.Z = src.X, src.Y, src.Z
    for dim in ("intensity", "return_number", "number_of_returns", "scan_direction_flag", "edge_of_flight_line", "classification", "synthetic", "key_point", "withheld", "scan_angle", "user_data", "point_source_id", "gps_time"):
        try:
            if dim == "scan_angle":
                las.scan_angle = np.asarray(src.scan_angle_rank, dtype=np.int16) * 166  # 0.006 deg units in PF6+
            else:
                setattr(las, dim, getattr(src, dim))
        except Exception:
            pass

    def to_src(a):
        out = np.empty((n, *a.shape[1:]), a.dtype)
        out[inv] = a
        return out

    product_rgb = np.asarray(product_rgb)
    rgb = to_src(product_rgb).astype(np.uint16)
    if product_rgb.dtype != np.uint16:
        rgb = rgb * 256
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if classification is not None:
        las.classification = to_src(np.asarray(classification, np.uint8))
    for name, dt, _ in extra_dims:
        setattr(las, name, to_src(extras[name]).astype(dt))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_path))
    return out_path


def verify(out_path: Path, td: TileData, expect_class_exact: bool = True, xyz_mode: str = "exact", tol_mm: float = 0.0) -> dict:
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
            prov = json.loads(bytes(v.record_data).decode())
    out["provenance"] = prov is not None
    return out
