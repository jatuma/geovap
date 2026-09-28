"""Writing a tile out: LAS 1.4 point format 7, source points bit-exact, our dimensions alongside.

The dimension lists are here, and there is now ONE of them where there used to be four.

`mapping/las_out.py` carried `EXTRA_DIMS`, `SEG_EXTRA_DIMS`, `CONS_EXTRA_DIMS` and `OBJ_EXTRA_DIMS`,
overlapping heavily, because the delivered product was three parallel LAZ sets over the same 585 M
points: `tiles/` with the fused colour, `objects/` with a cluster palette in RGB, and `vendor/` with
the reference RGB. 75 GB, of which two thirds was the same XYZ written three times so that a viewer
could be handed a different colour. `objects/`'s `cluster_id`/`obj_class` were already in `tiles/`,
and `vendor/`'s RGB was already `ref_r/g/b`.

`CONSOLIDATED_DIMS` is their union, with the duplicated dimensions stored once and the full
colourisation provenance added. One LAZ per tile carries everything; the viewer shades by attribute
instead of loading a second cloud. Extra bytes per point go from 21 to 45, so the tile set grows to
roughly 40-45 GB while `objects/` and `vendor/` (43 GB) disappear entirely -- and one octree is built
instead of three. The gain is the elimination of duplicated XYZ, not smaller tiles.

This module takes a source LAZ path and a store-order permutation rather than a `TileData`, so that
`io` stays below `runtime` in the layering: writing a LAS file is a format concern, and it should not
need to know what a point store is.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import laspy
import numpy as np

#: Colourisation diagnostics, as written by the colour stage's per-tile product.
COLOUR_DIMS: list[tuple] = [
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

#: Semantic-segmentation product: `classification` carries the common15 label.
SEG_DIMS: list[tuple] = [
    ("seg_conf", np.uint8, "label vote share 0-255 (winning class)"),
    ("seg_n_views", np.uint8, "number of frames that voted (0 = unlabelled)"),
    ("seg_src_frame", np.uint16, "frame index of the strongest vote"),
    ("src_class", np.uint8, "original LAS classification"),
]

#: Object clustering product: `cluster_id` is what the clustering stream contributes.
OBJECT_DIMS: list[tuple] = [
    ("cluster_id", np.int32, "object cluster id (-1 = none)"),
    ("obj_class", np.uint8, "object class"),
    ("hag", np.float32, "height above ground [m]"),
]

#: THE product. One LAZ per tile, every dimension, nothing duplicated.
#:
#: standard LAS fields carry: XYZ, intensity, GPS time, `classification` = the common15 semantic
#: label (255 = unlabelled), RGB = the fused colour (reference RGB where n_views == 0).
CONSOLIDATED_DIMS: list[tuple] = [
    *SEG_DIMS,
    *OBJECT_DIMS,
    *COLOUR_DIMS,
]

#: Bytes of extra data per point, for sizing a run before it is started.
EXTRA_BYTES_PER_POINT = sum(np.dtype(dt).itemsize for _n, dt, _d in CONSOLIDATED_DIMS)

PROVENANCE_USER_ID = "geovap_map"
PROVENANCE_RECORD_ID = 1

#: LAS integer scale the store and every product are written at: 1 mm, offset 0.
SCALE = 0.001


def _check_no_duplicates() -> None:
    names = [n for n, _dt, _d in CONSOLIDATED_DIMS]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"CONSOLIDATED_DIMS declares {dupes} more than once")


_check_no_duplicates()


def write_tile(
    source_laz: Path | str,
    store_order: np.ndarray,
    out_path: Path | str,
    product_rgb: np.ndarray,
    extras: dict[str, np.ndarray],
    provenance: dict,
    *,
    extra_dims: Sequence[tuple] = CONSOLIDATED_DIMS,
    classification: np.ndarray | None = None,
    xyz: np.ndarray | None = None,
    description: str = "geovap consolidated tile",
) -> Path:
    """Write one tile, keeping the source points bit-exact unless explicitly replaced.

    `store_order` is the store's `orig_index`: store row -> position in the source LAZ. Every array
    passed in (`product_rgb`, `extras`, `classification`, `xyz`) is in STORE order and is permuted
    back to source order here, so the written file's point order matches the vendor's file exactly.
    That is what makes a delivered tile diffable against its input.

    `product_rgb` is uint8 [n,3] and is scaled to 16 bit; a uint16 array passes through unchanged.
    `xyz` is int32 [n,3] LAS integers at the same scale/offset as the source; when given it replaces
    the bit-exact copy (the per-pass registered coordinates), otherwise `src.X/Y/Z` is copied
    verbatim.
    """
    src = laspy.read(str(source_laz))
    n = len(src.points)
    inv = np.asarray(store_order)
    if len(inv) != n:
        raise ValueError(f"{source_laz}: store order has {len(inv)} rows, source LAZ has {n} points")

    header = laspy.LasHeader(version="1.4", point_format=7)
    header.scales = src.header.scales
    header.offsets = src.header.offsets
    for name, dt, desc in extra_dims:
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dt, description=desc[:31]))
    header.vlrs.append(laspy.vlrs.VLR(
        user_id=PROVENANCE_USER_ID, record_id=PROVENANCE_RECORD_ID,
        description=description[:31], record_data=json.dumps(provenance).encode(),
    ))

    las = laspy.LasData(header)
    if xyz is not None:
        xyz_src = np.empty((n, 3), dtype=np.int32)
        xyz_src[inv] = np.asarray(xyz, dtype=np.int32)
        las.X, las.Y, las.Z = xyz_src[:, 0], xyz_src[:, 1], xyz_src[:, 2]
    else:
        las.X, las.Y, las.Z = src.X, src.Y, src.Z

    for dim in ("intensity", "return_number", "number_of_returns", "scan_direction_flag",
                "edge_of_flight_line", "classification", "synthetic", "key_point", "withheld",
                "scan_angle", "user_data", "point_source_id", "gps_time"):
        try:
            if dim == "scan_angle":
                # PF6+ stores scan angle in 0.006 deg units; the source is a PF<6 rank in degrees.
                las.scan_angle = np.asarray(src.scan_angle_rank, dtype=np.int16) * 166
            else:
                setattr(las, dim, getattr(src, dim))
        except Exception:  # noqa: BLE001 - a source without this dimension simply does not carry it
            pass

    def to_src(a: np.ndarray) -> np.ndarray:
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

    missing = [name for name, _dt, _d in extra_dims if name not in extras]
    if missing:
        raise KeyError(f"{out_path}: missing extra dimension(s) {missing}")
    for name, dt, _desc in extra_dims:
        setattr(las, name, to_src(np.asarray(extras[name])).astype(dt))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_path))
    return out_path


def read_provenance(path: Path | str) -> dict | None:
    """The provenance VLR a tile was written with, or None. Every consumer checks the `poses_hash`
    in here before merging a product: two artifacts built from different pose tables must not be
    combined, and because the hash travels with each file that is detectable rather than silent."""
    las = laspy.read(str(path))
    for vlr in las.header.vlrs:
        if vlr.user_id == PROVENANCE_USER_ID:
            return json.loads(bytes(vlr.record_data).decode())
    return None
