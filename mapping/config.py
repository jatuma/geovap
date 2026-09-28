"""Paths and numeric constants shared by the whole package."""
from __future__ import annotations

import os
from pathlib import Path

from geovap.domain.model.sensor import NO_POINT, Sensor, Tuning  # noqa: F401  (NO_POINT re-exported)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

DATA_ROOT = Path(os.environ.get("GEOVAP_DATA", "/home/jatuma/repos/Geovap/Geovap_data/DTM_Dražkov"))
PANO_DIR = DATA_ROOT / "LB5, Camera Ladybug"
EXPORT_CSV = PANO_DIR / "export.csv"
LAZ_DIR = DATA_ROOT / "LAZ_Dražkov_ground"
TILE_LAYOUT_GEOJSON = DATA_ROOT / "Klad_LAZ_Dražkov.geojson"
JVF_GEOJSON = DATA_ROOT / "1_ZPS_GAD.geojson"


def _default_cache_root() -> Path:
    for candidate in (REPO_ROOT.parent / "Geovap_cache", Path("/mnt/Geovap_cache")):
        if candidate.exists():
            return candidate
    return Path("/home/jatuma/repos/Geovap/Geovap_cache")


CACHE_ROOT = Path(os.environ["GEOVAP_CACHE"]) if os.environ.get("GEOVAP_CACHE") else _default_cache_root()
STORE_DIR = CACHE_ROOT / "store"
FRAMES_DIR = CACHE_ROOT / "frames"
GRAY_DIR = CACHE_ROOT / "gray"
RENDERS_DIR = CACHE_ROOT / "renders"
OUT_DIR = CACHE_ROOT / "out"
POSES_DIR = OUT_DIR / "poses"
PIPELINE_DIR = OUT_DIR / "pipeline"
CONSOLIDATED_DIR = OUT_DIR / "consolidated"

CLEAN_FRAMES_JSON = Path(os.environ.get("GEOVAP_CLEAN_FRAMES", REPO_ROOT / "dataset" / "clean_frames.json"))
QUALITY_CSV = REPO_ROOT / "dataset" / "frame_quality.csv"
POTREE_OUTPUT_DIR = Path(os.environ.get("POTREE_OUTPUT", "/home/jatuma/repos/Geovap/potree_output"))  # fast local drive since 2026-09-17; /mnt (slow) holds the cache

# pose source: "export" (default, the regression anchor) or a corrected pose-table name/path.
# override with env GEOVAP_POSES; see mapping/poses.py:load_poses.
POSES_SOURCE = os.environ.get("GEOVAP_POSES", "export")

# panorama
PANO_W = 8000
PANO_H = 4000
DEG_PER_PX = 360.0 / PANO_W  # 0.045 deg

# z-buffer (depth panorama)
ZB_W = 2000
ZB_H = 1000

# geometry limits
R_MIN = 1.0  # m, closer points are camera/scanner parallax garbage
R_MAX = 40.0  # m

# visibility tolerance: r <= depth + max(TOL_ABS, TOL_REL * r)
TOL_ABS = 0.15  # m
TOL_REL = 0.03

# splat: radius in radians = SPLAT_K * POINT_SPACING / r, clamped to [SPLAT_MIN_PX, SPLAT_MAX_PX] in z-buffer px
POINT_SPACING = 0.051  # m (382 pts/m^2)
SPLAT_K = 1.2
SPLAT_MIN_PX = 1
SPLAT_MAX_PX = 8

# scoring
SCORE_R0 = 8.0  # m
INCIDENCE_MAX_DEG = 80.0

# fusion
TOP_K = 5
MAD_CUTOFF = 2.5

# cloud store
CELL_SIZE = 4.0  # m
EXPECTED_TOTAL_POINTS = 584_809_840

# NO_POINT now lives in geovap.domain.model.sensor and is re-exported above.

# Transitional: the same numbers as the constants above, as the value object `geovap.domain` takes.
# Disappears with this module once every caller reads them from `geovap.runtime.settings`.
SENSOR = Sensor(
    pano_w=PANO_W, pano_h=PANO_H, zb_w=ZB_W, zb_h=ZB_H,
    r_min=R_MIN, r_max=R_MAX, point_spacing=POINT_SPACING, cell_size=CELL_SIZE,
    tol_abs=TOL_ABS, tol_rel=TOL_REL,
    splat_k=SPLAT_K, splat_min_px=SPLAT_MIN_PX, splat_max_px=SPLAT_MAX_PX,
)
TUNING = Tuning(score_r0=SCORE_R0, incidence_max_deg=INCIDENCE_MAX_DEG, top_k=TOP_K, mad_cutoff=MAD_CUTOFF)


def source_dir(base: Path, poses) -> Path:
    """Per-pose-source output root: `base` itself for the default "export" pose table (byte-identical
    paths, the regression anchor), a hash-suffixed sibling directory for any corrected one, so a
    corrected run's outputs never overwrite (or mix with) the export ones. `poses` is any object with
    `.source` and `.hash()` (a `mapping.poses.Poses`); mirrors `mapping.products.frames_dir`."""
    base = Path(base)
    if poses.source == "export":
        return base
    return base.with_name(base.name + "_" + poses.hash()[:6])
