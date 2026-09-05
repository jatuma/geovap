"""Paths and numeric constants shared by the whole package."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

DATA_ROOT = Path(os.environ.get("GEOVAP_DATA", "/home/jatuma/repos/Geovap/Geovap_data/DTM_Dražkov"))
PANO_DIR = DATA_ROOT / "LB5, Camera Ladybug"
EXPORT_CSV = PANO_DIR / "export.csv"
LAZ_DIR = DATA_ROOT / "LAZ_Dražkov_ground"
TILE_LAYOUT_GEOJSON = DATA_ROOT / "Klad_LAZ_Dražkov.geojson"
JVF_GEOJSON = DATA_ROOT / "1_ZPS_GAD.geojson"

CACHE_ROOT = Path(os.environ.get("GEOVAP_CACHE", "/home/jatuma/repos/Geovap/Geovap_cache"))
STORE_DIR = CACHE_ROOT / "store"
FRAMES_DIR = CACHE_ROOT / "frames"
GRAY_DIR = CACHE_ROOT / "gray"
RENDERS_DIR = CACHE_ROOT / "renders"
OUT_DIR = CACHE_ROOT / "out"

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

NO_POINT = 0xFFFFFFFF  # uint32 sentinel in point_id panoramas
