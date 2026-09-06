"""LAS 1.4 PF7 writer: source points bit-exact + our colours + diagnostic extra dims + provenance VLR."""
from __future__ import annotations

import json
from pathlib import Path

import laspy
import numpy as np

from .cloud_store import TileData

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

PROVENANCE_USER_ID = "geovap_map"
PROVENANCE_RECORD_ID = 1


def write_tile(td: TileData, out_path: Path, product_rgb: np.ndarray, extras: dict[str, np.ndarray], provenance: dict,
               extra_dims: list[tuple] = EXTRA_DIMS, classification: np.ndarray | None = None, description: str = "colorization provenance") -> Path:
    """`product_rgb` u8 [n,3], `extras` and `classification` arrays are in STORE (cell-sorted) order; written in source order.

    `classification` replaces the source classification when given (e.g. semantic labels); otherwise it is copied bit-exact.
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

    rgb = to_src(product_rgb).astype(np.uint16) * 256
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if classification is not None:
        las.classification = to_src(np.asarray(classification, np.uint8))
    for name, dt, _ in extra_dims:
        setattr(las, name, to_src(extras[name]).astype(dt))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_path))
    return out_path


def verify(out_path: Path, td: TileData, expect_class_exact: bool = True) -> dict:
    las = laspy.read(str(out_path))
    src = laspy.read(td.info.laz)
    ok_xyz = np.array_equal(las.X, src.X) and np.array_equal(las.Y, src.Y) and np.array_equal(las.Z, src.Z)
    ok_cls = np.array_equal(np.asarray(las.classification), np.asarray(src.classification)) if expect_class_exact else True
    prov = None
    for v in las.header.vlrs:
        if v.user_id == PROVENANCE_USER_ID:
            prov = json.loads(bytes(v.record_data).decode())
    return {"n": len(las.points), "n_src": len(src.points), "xyz_exact": bool(ok_xyz), "class_exact": bool(ok_cls), "provenance": prov is not None}
