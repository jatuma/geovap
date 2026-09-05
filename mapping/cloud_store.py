"""Columnar, cell-indexed memmap store of the whole point cloud.

Why: the z-buffer of a frame needs every point within R_MAX of the camera regardless of tile
borders, workers must fork without copying 585 M points, and outputs must be written back in
the original per-tile order. A one-off conversion (`build_store`) gives all three.

Layout (`STORE_DIR`):
    tiles.json                         list of tiles: name, laz path, n, row_offset, bbox, polygon, grid
    tiles/NN/xyz.npy      int32 [n,3]  raw LAS integers (offset 0, scale 1 mm) -> bit-exact
    tiles/NN/intensity.npy   u16
    tiles/NN/classification.npy u8
    tiles/NN/rgb.npy       u8 [n,3]    TerraScan RGB >> 8
    tiles/NN/gps_time.npy  f64
    tiles/NN/psid.npy      u16
    tiles/NN/orig_index.npy u32        position in the source LAZ
    tiles/NN/cell_starts.npy u32 [ny*nx+1]  CSR over the tile's dense CELL_SIZE grid (rows sorted by cell)

Global point_id = tile.row_offset + local row (fits uint32).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CELL_SIZE, LAZ_DIR, STORE_DIR, TILE_LAYOUT_GEOJSON

SCALE = 0.001  # m per LAS integer unit (all tiles: scale 0.001, offset 0 - asserted at build time)
COLUMNS = ("xyz", "intensity", "classification", "rgb", "gps_time", "psid", "orig_index")


@dataclass
class TileInfo:
    name: str  # "037"
    laz: str
    n: int
    row_offset: int
    bbox: tuple[float, float, float, float]  # minE, minN, maxE, maxN (m)
    polygon: np.ndarray  # [K,2] closed ring (m)
    grid_origin: tuple[float, float]  # (E0, N0) of cell (0,0), m
    grid_shape: tuple[int, int]  # (ny, nx)

    @property
    def dir(self) -> Path:
        return STORE_DIR / "tiles" / self.name


class TileData:
    """Memmapped columns of one tile (read-only)."""

    def __init__(self, info: TileInfo):
        self.info = info
        d = info.dir
        self.xyz = np.load(d / "xyz.npy", mmap_mode="r")
        self.intensity = np.load(d / "intensity.npy", mmap_mode="r")
        self.classification = np.load(d / "classification.npy", mmap_mode="r")
        self.rgb = np.load(d / "rgb.npy", mmap_mode="r")
        self.gps_time = np.load(d / "gps_time.npy", mmap_mode="r")
        self.psid = np.load(d / "psid.npy", mmap_mode="r")
        self.orig_index = np.load(d / "orig_index.npy", mmap_mode="r")
        self.cell_starts = np.load(d / "cell_starts.npy")

    def __len__(self) -> int:
        return self.info.n

    def release(self) -> None:
        """Drop the pages this process has touched in the tile's memmaps (they stay valid and are re-read
        on demand). Keeps per-process RSS small: a process-tree memory watchdog counts memmapped file
        pages in every worker's RSS, so 4 workers x an 18 GB store look like 72 GB."""
        import mmap

        for a in (self.xyz, self.intensity, self.classification, self.rgb, self.gps_time, self.psid, self.orig_index):
            mm = getattr(a, "_mmap", None)
            if mm is not None:
                try:
                    mm.madvise(mmap.MADV_DONTNEED)
                except (AttributeError, OSError, ValueError):
                    pass

    def xyz_m(self, rows=slice(None)) -> np.ndarray:
        return np.asarray(self.xyz[rows], dtype=np.float64) * SCALE

    def rows_in_disc(self, cx: float, cy: float, radius: float) -> np.ndarray:
        """Local row indices of points whose cell may intersect the disc (superset; refine by distance)."""
        e0, n0 = self.info.grid_origin
        ny, nx = self.info.grid_shape
        half_diag = CELL_SIZE * np.sqrt(0.5)
        ix = np.arange(nx)
        iy = np.arange(ny)
        cxs = e0 + (ix + 0.5) * CELL_SIZE
        cys = n0 + (iy + 0.5) * CELL_SIZE
        d2 = (cxs[None, :] - cx) ** 2 + (cys[:, None] - cy) ** 2
        cells = np.flatnonzero(d2 <= (radius + half_diag) ** 2)
        if len(cells) == 0:
            return np.empty(0, dtype=np.int64)
        starts = self.cell_starts[cells]
        ends = self.cell_starts[cells + 1]
        lengths = ends - starts
        keep = lengths > 0
        starts, lengths = starts[keep], lengths[keep]
        if len(starts) == 0:
            return np.empty(0, dtype=np.int64)
        # expand ranges
        total = int(lengths.sum())
        out = np.empty(total, dtype=np.int64)
        pos = 0
        for s, l in zip(starts.tolist(), lengths.tolist()):
            out[pos : pos + l] = np.arange(s, s + l)
            pos += l
        return out


class CloudStore:
    def __init__(self, root: Path = STORE_DIR):
        self.root = Path(root)
        meta = json.loads((self.root / "tiles.json").read_text())
        self.tiles: list[TileInfo] = [
            TileInfo(
                name=t["name"],
                laz=t["laz"],
                n=t["n"],
                row_offset=t["row_offset"],
                bbox=tuple(t["bbox"]),
                polygon=np.asarray(t["polygon"], dtype=np.float64),
                grid_origin=tuple(t["grid_origin"]),
                grid_shape=tuple(t["grid_shape"]),
            )
            for t in meta["tiles"]
        ]
        self.by_name = {t.name: t for t in self.tiles}
        self.total = sum(t.n for t in self.tiles)
        self._data: dict[str, TileData] = {}

    def tile(self, name: str) -> TileData:
        if name not in self._data:
            self._data[name] = TileData(self.by_name[name])
        return self._data[name]

    def tiles_near(self, cx: float, cy: float, radius: float) -> list[TileInfo]:
        out = []
        for t in self.tiles:
            minE, minN, maxE, maxN = t.bbox
            dx = max(minE - cx, 0.0, cx - maxE)
            dy = max(minN - cy, 0.0, cy - maxN)
            if dx * dx + dy * dy <= radius * radius:
                out.append(t)
        return out

    def query_disc(self, cx: float, cy: float, radius: float, exact: bool = True) -> list[tuple[TileInfo, np.ndarray]]:
        """[(tile, local_rows)] of points within `radius` (2D) of (cx, cy), across tile borders."""
        res = []
        for t in self.tiles_near(cx, cy, radius):
            td = self.tile(t.name)
            rows = td.rows_in_disc(cx, cy, radius)
            if exact and len(rows):
                xy = np.asarray(td.xyz[rows, :2], dtype=np.float64) * SCALE
                m = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2 <= radius * radius
                rows = rows[m]
            if len(rows):
                res.append((t, rows))
        return res

    def release(self) -> None:
        """Release touched memmap pages of all opened tiles (see TileData.release)."""
        for td in self._data.values():
            td.release()

    def drop_cache(self, extra_dirs: tuple[Path, ...] = ()) -> None:
        """Ask the kernel to drop cached pages of the store (and optional other dirs, e.g. FRAMES_DIR).

        The memmapped store is 18 GB; a long run fills the page cache with it, and a memory watchdog that
        looks at *free* rather than *available* memory then kills the job. Cached pages are reclaimable,
        so dropping them costs only re-reads."""
        import os

        dirs = [self.root / "tiles" / t.name for t in self.tiles] + list(extra_dirs)
        for d in dirs:
            for p in Path(d).glob("*.np*"):
                try:
                    fd = os.open(p, os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    os.close(fd)
                except OSError:
                    pass

    def global_ids(self, tile: TileInfo, rows: np.ndarray) -> np.ndarray:
        return (tile.row_offset + rows).astype(np.uint32)

    def locate(self, point_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """global point_id -> (tile index, local row)"""
        offs = np.array([t.row_offset for t in self.tiles] + [self.total], dtype=np.int64)
        ti = np.searchsorted(offs, point_id.astype(np.int64), side="right") - 1
        return ti, point_id.astype(np.int64) - offs[ti]


# ------------------------------------------------------------------------------------- build
def _load_layout() -> list[tuple[str, np.ndarray]]:
    """[(name, ring[K,2])] from Klad geojson: rings are LineStrings, names are label Points inside them."""
    from matplotlib.path import Path as MplPath

    d = json.loads(Path(TILE_LAYOUT_GEOJSON).read_text())
    rings = [np.asarray(f["geometry"]["coordinates"], dtype=np.float64)[:, :2] for f in d["features"] if f["geometry"]["type"] == "LineString"]
    labels = [(f["properties"]["name"], np.asarray(f["geometry"]["coordinates"][:2], dtype=np.float64)) for f in d["features"] if f["geometry"]["type"] == "Point"]
    out = []
    for ring in rings:
        path = MplPath(ring)
        inside = [name for name, p in labels if path.contains_point(p)]
        assert len(inside) == 1, f"ring has {len(inside)} labels: {inside}"
        out.append((inside[0], ring))
    return out


def _tile_name_from_laz(path: Path) -> str:
    # ID3432_000037_JTSK.laz -> "037"
    return path.stem.split("_")[1][-3:]


def build_tile(laz_file: str, out_dir: str) -> dict:
    """Convert one LAZ tile into the columnar store. Returns tile metadata (without row_offset)."""
    import laspy

    laz_path = Path(laz_file)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with laspy.open(laz_path) as reader:
        h = reader.header
        assert np.allclose(h.scales, SCALE) and np.allclose(h.offsets, 0.0), (h.scales, h.offsets)
        n = h.point_count
        las = reader.read()
    xyz = np.stack([las.X, las.Y, las.Z], axis=1).astype(np.int32)
    assert np.array_equal(xyz[:, 0], las.X) and np.array_equal(xyz[:, 1], las.Y) and np.array_equal(xyz[:, 2], las.Z), "int32 overflow"
    rgb16 = np.stack([las.red, las.green, las.blue], axis=1)
    assert np.all(rgb16 % 256 == 0), "RGB not 8bit*256"
    rgb = (rgb16 >> 8).astype(np.uint8)

    # dense cell grid over the tile bbox
    e_m = xyz[:, 0].astype(np.float64) * SCALE
    n_m = xyz[:, 1].astype(np.float64) * SCALE
    e0 = np.floor(e_m.min() / CELL_SIZE) * CELL_SIZE
    n0 = np.floor(n_m.min() / CELL_SIZE) * CELL_SIZE
    ix = ((e_m - e0) // CELL_SIZE).astype(np.int64)
    iy = ((n_m - n0) // CELL_SIZE).astype(np.int64)
    nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
    key = iy * nx + ix
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    cell_starts = np.searchsorted(key_sorted, np.arange(nx * ny + 1)).astype(np.uint32)

    np.save(out / "xyz.npy", xyz[order])
    np.save(out / "intensity.npy", np.asarray(las.intensity, dtype=np.uint16)[order])
    np.save(out / "classification.npy", np.asarray(las.classification, dtype=np.uint8)[order])
    np.save(out / "rgb.npy", rgb[order])
    np.save(out / "gps_time.npy", np.asarray(las.gps_time, dtype=np.float64)[order])
    np.save(out / "psid.npy", np.asarray(las.point_source_id, dtype=np.uint16)[order])
    np.save(out / "orig_index.npy", order.astype(np.uint32))
    np.save(out / "cell_starts.npy", cell_starts)
    return {
        "name": _tile_name_from_laz(laz_path),
        "laz": str(laz_path),
        "n": int(n),
        "bbox": [float(e_m.min()), float(n_m.min()), float(e_m.max()), float(n_m.max())],
        "grid_origin": [float(e0), float(n0)],
        "grid_shape": [ny, nx],
        "gps_time_range": [float(las.gps_time.min()), float(las.gps_time.max())],
        "classes": {int(c): int(k) for c, k in zip(*np.unique(las.classification, return_counts=True))},
    }


def build_store(workers: int = 8, root: Path = STORE_DIR, laz_dir: Path = LAZ_DIR) -> None:
    from multiprocessing import Pool

    root = Path(root)
    (root / "tiles").mkdir(parents=True, exist_ok=True)
    laz_files = sorted(Path(laz_dir).glob("*.laz"))
    layout = dict(_load_layout())
    jobs = [(str(p), str(root / "tiles" / _tile_name_from_laz(p))) for p in laz_files]
    # largest first for better packing
    jobs.sort(key=lambda j: -Path(j[0]).stat().st_size)
    with Pool(workers) as pool:
        metas = pool.starmap(build_tile, jobs)
    metas.sort(key=lambda m: m["name"])
    offset = 0
    for m in metas:
        m["row_offset"] = offset
        offset += m["n"]
        ring = layout[m["name"]]
        # sanity: tile bbox centre inside its layout polygon
        from matplotlib.path import Path as MplPath

        c = ((m["bbox"][0] + m["bbox"][2]) / 2, (m["bbox"][1] + m["bbox"][3]) / 2)
        assert MplPath(ring).contains_point(c), f"tile {m['name']} centre not inside layout ring"
        m["polygon"] = ring.tolist()
    (root / "tiles.json").write_text(json.dumps({"total": offset, "cell_size": CELL_SIZE, "tiles": metas}))
    print(f"store built: {len(metas)} tiles, {offset} points -> {root}")
