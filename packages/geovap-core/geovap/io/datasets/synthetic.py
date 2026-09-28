"""Generates a tiny, geometrically coherent stand-in for the Dražkov dataset shape: posed
equirectangular panoramas + tiled MLS LAZ ground points + JVF/ZPS reference vectors.

Every team stream needs to run its stage without the real ~600 GB dataset mounted. This module
builds a scene small enough to commit (a few MB) but real enough that a stage which registers
panoramas against the point cloud, colours points from photos, or splits queries across tiles gets
a meaningful, checkable answer rather than noise:

  - a ~60 m straight street: a ground plane, two building facades, and a few vertical poles, all
    sampled as point-cloud surfaces in a projected CRS (not a random point soup);
  - ~12 camera poses driven along the street centreline, 5 m apart;
  - one panorama PER POSE, rendered by projecting the scene's own points through
    `geovap.domain.model.geometry`'s camera model and splatting them -- so a stage that projects
    the point cloud back into a panorama lands on genuinely matching pixels, not luck;
  - >=3 LAZ tiles on a skewed (non-axis-aligned) grid, so cross-tile queries are exercised and nobody
    can get away with treating a tile as an axis-aligned bbox;
  - a handful of JVF/ZPS reference-vector features using real codes from `jvf_zps_cz.toml`.

`synthetic.toml` (next to this file) is the descriptor that reads what `build()` writes; `[paths]`
there uses `${GEOVAP_SYNTHETIC_ROOT}` etc., so this fixture works from any directory it happens to
be generated into. `geovap doctor --dataset synthetic` is the intended smoke test once a fixture
exists on disk.

Everything here is deterministic given `seed`: no wall-clock time, no filesystem iteration order
sensitivity (tiles are written by explicit id, not glob order), no non-deterministic dict ordering
in written JSON (`sort_keys=True` throughout).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import laspy
import numpy as np

from geovap.domain.model.geometry import vehicle_rotation, world_to_pano

# --------------------------------------------------------------------------------- scene layout
# World origin of the scene, in an arbitrary projected CRS (metres). Not (0, 0, 0): a fixture whose
# coordinates are all small/round numbers would silently forgive an adapter or stage that mishandles
# a real dataset's large offsets.
E0, N0, H0 = 1000.0, 500.0, 300.0

STREET_LEN = 60.0          # m, along +E
POSE_SPACING = 5.0         # m
CAM_HEIGHT = 1.6           # m above ground
SPEED_MPS = 2.0            # m/s, for a plausible gps_time
T0 = 100_000.0             # s, arbitrary GPS-week-seconds base

GROUND_HALF_W = 2.0        # m, ground plane extends N in [-2, 2] (street + verge)
FACADE_N = 5.0             # m, facades set back at N = +-FACADE_N
FACADE_H = 4.0             # m, facade height
POLE_N = 4.5               # m, poles between street edge and facades
POLE_H = 3.0               # m

DE = 0.05                  # m, along-street spacing -- the dataset's `point_spacing`
DCROSS = 0.25              # m, cross-track spacing (ground width / facade height): MLS clouds are
                            # dense along-track, coarser across, same as the real export

N_TILES = 4
TILE_LEN = STREET_LEN / N_TILES   # 15 m
SHEAR = 0.15                # tile grid skew (m of E per m of N); real tile grids are not axis-aligned
RING_N_HALF = 6.5           # m, half-width used for the tile ring polygons (> any point's |N|)

PANO_W, PANO_H = 512, 256
R_MAX = 40.0                # m, matches Dražkov's sensor.r_max order of magnitude at this scale
SPLAT_ITER = 1              # dilation iterations -> ~1 px splat radius

GROUND_RGB = (120, 118, 112)
FACADE_RGB = (150, 92, 70)
POLE_RGB = (70, 70, 74)
GROUND_CLS, FACADE_CLS, POLE_CLS = 2, 6, 15  # ASPRS: ground, building, transmission tower (poles)

# Reference-vector codes, taken from geovap.io.adapters.reference.jvf_zps_cz.toml
CODE_ROAD = "0100000304"       # hranice dopravní stavby nebo plochy
CODE_BUILDING = "0100000001"   # budova
CODE_FENCE = "0100000162"      # plot
CODE_VEGETATION = "0100000215"  # udržovaná plocha zeleně


def build(root: Path, *, seed: int = 0) -> Path:
    """Generate the complete fixture under `root` (created if needed) and return `root`.

    Deterministic: the same `seed` regenerates byte-identical files.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)

    points = _build_points(rng)
    poses = _build_poses(rng)
    tiles = _assign_tiles(points)

    (root / "panos").mkdir(parents=True, exist_ok=True)
    (root / "tiles").mkdir(parents=True, exist_ok=True)
    (root / "reference").mkdir(parents=True, exist_ok=True)

    _write_poses_csv(root / "panos" / "export.csv", poses)
    _render_panos(root / "panos", poses, points)
    _write_tiles(root / "tiles", points, tiles)
    _write_grid(root / "tiles_grid.geojson")
    _write_reference(root / "reference" / "vectors.geojson")

    return root


# ------------------------------------------------------------------------------- point cloud
def _grid(e_vals: np.ndarray, cross_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ee, cc = np.meshgrid(e_vals, cross_vals, indexing="xy")
    return ee.ravel(), cc.ravel()


def _build_points(rng: np.random.Generator) -> dict:
    """All scene points, relative to (E0, N0, H0): dict of arrays (e, n, h, rgb[N,3], classification,
    intensity, gps_time)."""
    e_along = np.arange(0.0, STREET_LEN + 1e-9, DE)

    parts_e, parts_n, parts_h, parts_rgb, parts_cls = [], [], [], [], []

    # ground plane: flat, N in [-GROUND_HALF_W, GROUND_HALF_W]
    n_cross = np.arange(-GROUND_HALF_W, GROUND_HALF_W + 1e-9, DCROSS)
    e, n = _grid(e_along, n_cross)
    parts_e.append(e)
    parts_n.append(n)
    parts_h.append(np.zeros_like(e))
    parts_cls.append(np.full(e.shape, GROUND_CLS, dtype=np.uint8))
    parts_rgb.append(np.tile(np.array(GROUND_RGB, dtype=np.int64), (len(e), 1)))

    # two facades, vertical planes at fixed N
    z_cross = np.arange(0.0, FACADE_H + 1e-9, DCROSS)
    for side in (+1.0, -1.0):
        e, z = _grid(e_along, z_cross)
        n = np.full_like(e, side * FACADE_N)
        parts_e.append(e)
        parts_n.append(n)
        parts_h.append(z)
        parts_cls.append(np.full(e.shape, FACADE_CLS, dtype=np.uint8))
        # faint per-column banding so the facade is not a flat colour field
        band = (np.sin(e / 3.0) * 10.0).astype(np.int64)
        rgb = np.array(FACADE_RGB, dtype=np.int64)[None, :] + band[:, None]
        parts_rgb.append(rgb)

    # poles: a handful of vertical lines, alternating sides, at tile centres
    z_pole = np.arange(0.0, POLE_H + 1e-9, DE)
    pole_e_right = np.array([TILE_LEN * (k + 0.5) for k in range(N_TILES)])
    pole_e_left = np.array([TILE_LEN * k + TILE_LEN * 0.9 for k in range(N_TILES)])
    for pole_es, side in ((pole_e_right, +1.0), (pole_e_left, -1.0)):
        for pe in pole_es:
            z = z_pole
            e = np.full_like(z, pe)
            n = np.full_like(z, side * POLE_N)
            parts_e.append(e)
            parts_n.append(n)
            parts_h.append(z)
            parts_cls.append(np.full(e.shape, POLE_CLS, dtype=np.uint8))
            parts_rgb.append(np.tile(np.array(POLE_RGB, dtype=np.int64), (len(e), 1)))

    e = np.concatenate(parts_e)
    n = np.concatenate(parts_n)
    h = np.concatenate(parts_h)
    cls = np.concatenate(parts_cls)
    rgb = np.concatenate(parts_rgb, axis=0)

    # deterministic colour jitter and intensity
    jitter = rng.integers(-8, 9, size=rgb.shape)
    rgb = np.clip(rgb + jitter, 0, 255).astype(np.uint8)
    intensity = rng.integers(50, 300, size=e.shape[0]).astype(np.uint16)
    gps_time = T0 + e / SPEED_MPS

    return {
        "e": e, "n": n, "h": h, "rgb": rgb, "classification": cls,
        "intensity": intensity, "gps_time": gps_time,
    }


def _assign_tiles(points: dict) -> np.ndarray:
    """Tile index [0, N_TILES) per point, by the same shear used for the tile-grid ring polygons."""
    e_shift = points["e"] - SHEAR * points["n"]
    idx = np.clip((e_shift // TILE_LEN).astype(np.int64), 0, N_TILES - 1)
    return idx


def _tile_ring(k: int) -> np.ndarray:
    """World-coordinate ring [K,2] (closed) for tile `k`, a skewed parallelogram sheared by SHEAR."""
    e0, e1 = k * TILE_LEN, (k + 1) * TILE_LEN
    nh = RING_N_HALF
    verts_local = [
        (e0 + SHEAR * (-nh), -nh),
        (e0 + SHEAR * nh, nh),
        (e1 + SHEAR * nh, nh),
        (e1 + SHEAR * (-nh), -nh),
        (e0 + SHEAR * (-nh), -nh),
    ]
    return np.array([[E0 + e, N0 + n] for e, n in verts_local], dtype=np.float64)


def _tile_id(k: int) -> str:
    return f"{k + 1:04d}"


# ------------------------------------------------------------------------------------- poses
def _build_poses(rng: np.random.Generator) -> dict:
    e_vals = np.arange(POSE_SPACING, STREET_LEN + 1e-9, POSE_SPACING)
    n = len(e_vals)
    t = T0 + e_vals / SPEED_MPS
    # small deterministic attitude jitter so the fixture is not degenerately perfect
    roll = rng.uniform(-0.5, 0.5, size=n)
    pitch = rng.uniform(-0.5, 0.5, size=n)
    yaw = rng.uniform(-1.0, 1.0, size=n)
    origin = np.stack([E0 + e_vals, np.full(n, N0), np.full(n, H0 + CAM_HEIGHT)], axis=1)
    filenames = [f"pano_{i:04d}.jpg" for i in range(n)]
    return {"filename": filenames, "t": t, "origin": origin, "roll": roll, "pitch": pitch, "yaw": yaw}


# `[poses].columns` in synthetic.toml: deliberately NOT Dražkov's indices/order/width.
_CSV_COLUMNS = ("gpstime", "filename", "easting", "northing", "height", "yaw_deg", "pitch_deg", "roll_deg", "quality", "notes")


def _write_poses_csv(path: Path, poses: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CSV_COLUMNS)
        n = len(poses["t"])
        for i in range(n):
            e, nn, h = poses["origin"][i]
            w.writerow([
                f"{poses['t'][i]:.3f}",
                poses["filename"][i],
                f"{e:.4f}",
                f"{nn:.4f}",
                f"{h:.4f}",
                f"{poses['yaw'][i]:.4f}",
                f"{poses['pitch'][i]:.4f}",
                f"{poses['roll'][i]:.4f}",
                "1.0",
                "synthetic",
            ])


# ------------------------------------------------------------------------------------ panoramas
def render_pano(P: np.ndarray, rgb: np.ndarray, R: np.ndarray, C: np.ndarray, w: int, h: int) -> np.ndarray:
    """Render one panorama by projecting world points `P[N,3]` (with colours `rgb[N,3]` uint8)
    through the SAME camera model a consumer would invert with (`geometry.world_to_pano`), keeping
    only the nearest point per pixel, then splatting by a small dilation.
    """
    u, v, r, _el = world_to_pano(P, R, C, w, h, dtype=np.float64)
    keep = r <= R_MAX
    u, v, r, rgb = u[keep], v[keep], r[keep], rgb[keep]
    uu = np.clip(np.floor(u).astype(np.int64), 0, w - 1)
    vv = np.clip(np.floor(v).astype(np.int64), 0, h - 1)
    flat = vv * w + uu

    # nearest point wins per pixel: sort by (pixel, range) and take the first of each group
    order = np.lexsort((r, flat))
    flat_sorted = flat[order]
    first = np.concatenate(([True], flat_sorted[1:] != flat_sorted[:-1]))
    winners = order[first]

    img = np.zeros((h * w, 3), dtype=np.uint8)
    img[flat[winners]] = rgb[winners]
    img = img.reshape(h, w, 3)

    if SPLAT_ITER:
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.dilate(img, kernel, iterations=SPLAT_ITER)
    return img


def _render_panos(pano_dir: Path, poses: dict, points: dict) -> None:
    P = np.stack([E0 + points["e"], N0 + points["n"], H0 + points["h"]], axis=1)
    rgb = points["rgb"]
    n = len(poses["t"])
    for i in range(n):
        R = vehicle_rotation(poses["yaw"][i], poses["roll"][i], poses["pitch"][i])
        C = poses["origin"][i]
        img_rgb = render_pano(P, rgb, R, C, PANO_W, PANO_H)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        out = pano_dir / poses["filename"][i]
        ok = cv2.imwrite(str(out), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError(f"failed to write {out}")


# ---------------------------------------------------------------------------------------- tiles
def _write_tiles(tiles_dir: Path, points: dict, tile_idx: np.ndarray) -> None:
    scale = 0.001
    for k in range(N_TILES):
        m = tile_idx == k
        n_pts = int(m.sum())
        if n_pts == 0:
            raise RuntimeError(f"tile {k}: no points assigned -- scene/grid mismatch")

        e = E0 + points["e"][m]
        n = N0 + points["n"][m]
        h = H0 + points["h"][m]
        rgb = points["rgb"][m]
        cls = points["classification"][m]
        intensity = points["intensity"][m]
        gps_time = points["gps_time"][m]

        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = [scale, scale, scale]
        header.offsets = [0.0, 0.0, 0.0]
        las = laspy.LasData(header)
        las.X = np.round(e / scale).astype(np.int32)
        las.Y = np.round(n / scale).astype(np.int32)
        las.Z = np.round(h / scale).astype(np.int32)
        las.intensity = intensity
        las.classification = cls
        las.gps_time = gps_time
        las.point_source_id = np.ones(n_pts, dtype=np.uint16)
        las.red = rgb[:, 0].astype(np.uint16) * 256
        las.green = rgb[:, 1].astype(np.uint16) * 256
        las.blue = rgb[:, 2].astype(np.uint16) * 256

        name = f"SYN_TILE_{_tile_id(k)}_JTSK.laz"
        las.write(str(tiles_dir / name))


def _write_grid(path: Path) -> None:
    features = []
    for k in range(N_TILES):
        ring = _tile_ring(k)
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": ring.tolist()},
            "properties": {"name": _tile_id(k)},
        })
        centroid = ring[:-1].mean(axis=0)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": centroid.tolist()},
            "properties": {"name": _tile_id(k)},
        })
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, sort_keys=True), encoding="utf-8")


# ------------------------------------------------------------------------------------ reference
def _write_reference(path: Path) -> None:
    def pt(e, n, h):
        return [E0 + e, N0 + n, H0 + h]

    road_line = [pt(e, 0.0, 0.0) for e in (0.0, 20.0, 40.0, STREET_LEN)]
    building_poly = [
        pt(10.0, FACADE_N - 0.2, 0.0),
        pt(10.0, FACADE_N + 0.2, 0.0),
        pt(20.0, FACADE_N + 0.2, 0.0),
        pt(20.0, FACADE_N - 0.2, 0.0),
        pt(10.0, FACADE_N - 0.2, 0.0),
    ]
    fence_line = [pt(e, -POLE_N, 0.0) for e in (0.0, 30.0, STREET_LEN)]
    tree_point = pt(30.0, GROUND_HALF_W - 0.5, 0.0)

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": road_line},
            "properties": {"jvfcode": f"{CODE_ROAD}-01", "rc": "street centreline"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [building_poly]},
            "properties": {"jvfcode": f"{CODE_BUILDING}-01", "rc": "facade footprint"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": fence_line},
            "properties": {"jvfcode": f"{CODE_FENCE}-01", "rc": "left-side fence"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": tree_point},
            "properties": {"jvfcode": f"{CODE_VEGETATION}-01", "rc": "street tree"},
        },
    ]
    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, sort_keys=True), encoding="utf-8")


# ------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir", type=Path, help="output directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    root = build(args.dir, seed=args.seed)

    n_tiles = N_TILES
    n_poses = len(_build_poses(np.random.default_rng(args.seed))["t"])
    laz_bytes = sum(p.stat().st_size for p in (root / "tiles").glob("*.laz"))
    jpg_bytes = sum(p.stat().st_size for p in (root / "panos").glob("*.jpg"))
    total_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"synthetic dataset built at {root}")
    print(f"  poses:      {n_poses}")
    print(f"  tiles:      {n_tiles} ({laz_bytes / 1e6:.2f} MB)")
    print(f"  panoramas:  {n_poses} ({jpg_bytes / 1e6:.2f} MB)")
    print(f"  total size: {total_bytes / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
