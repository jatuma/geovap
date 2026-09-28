"""Columnar, cell-indexed memmap store of the whole point cloud.

Why: the z-buffer of a frame needs every point within R_MAX of the camera regardless of tile
borders, workers must fork without copying 585 M points, and outputs must be written back in
the original per-tile order. A one-off conversion (`build_store`) gives all three.

Layout (the store root):
    tiles.json                         list of tiles: name, laz path, n, row_offset, bbox, polygon, grid
    tiles/NN/xyz.npy      int32 [n,3]  raw LAS integers (offset 0, scale 1 mm) -> bit-exact
    tiles/NN/intensity.npy   u16
    tiles/NN/classification.npy u8
    tiles/NN/rgb.npy       u8 [n,3]    TerraScan RGB >> 8
    tiles/NN/gps_time.npy  f64
    tiles/NN/psid.npy      u16
    tiles/NN/orig_index.npy u32        position in the source LAZ
    tiles/NN/cell_starts.npy u32 [ny*nx+1]  CSR over the tile's dense cell_size grid (rows sorted by cell)
    tiles/NN/user_data.npy       u8    (S1, optional - old stores open without it)
    tiles/NN/scan_angle_rank.npy i8    (S1, optional)
    tiles/NN/return_number.npy   u8    return_number | number_of_returns<<4 (S1, optional)
    tiles/NN/time_order.npy      u32   argsort of gps_time (S1, optional)
    tiles/NN/time_bucket_starts.npy u32 [n_buckets+1]  CSR over 10 ms gps_time buckets, in time_order (S1, optional)
    tiles/NN/time_meta.json      {"t_min": float, "bucket_s": float, "n_buckets": int} (S1, optional)

Global point_id = tile.row_offset + local row (fits uint32).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geovap.domain.model.tiles import TileId

SCALE = 0.001  # m per LAS integer unit (all tiles: scale 0.001, offset 0 - asserted at build time)
COLUMNS = ("xyz", "intensity", "classification", "rgb", "gps_time", "psid", "orig_index")
# S1 extra columns: lazily loaded, None if the .npy is missing (old stores keep opening fine)
EXTRA_COLUMNS = ("user_data", "scan_angle_rank", "return_number")
COLUMNS = COLUMNS + EXTRA_COLUMNS
TIME_BUCKET_S = 0.01  # 10 ms buckets for the per-tile time index (S1)


@dataclass
class TileInfo:
    name: str  # "037"
    root: Path  # store root; `dir` is derived from this instead of a global
    laz: str
    n: int
    row_offset: int
    bbox: tuple[float, float, float, float]  # minE, minN, maxE, maxN (m)
    polygon: np.ndarray  # [K,2] closed ring (m), or shape (0, 2) when the tile has no ring
    grid_origin: tuple[float, float]  # (E0, N0) of cell (0,0), m
    grid_shape: tuple[int, int]  # (ny, nx)
    cell_size: float  # m, the store's build-time cell edge (from tiles.json, not the live descriptor)
    gps_time_range: tuple[float, float] | None = None  # (min, max) gps_time in the tile, from tiles.json

    @property
    def id(self) -> TileId:
        return TileId(self.name)

    @property
    def dir(self) -> Path:
        return self.root / "tiles" / self.name


class PassRegistration:
    """S5 pass-to-pass registration applied to point coordinates: per photo-pass 4-DoF rigid
    transform (dE, dN, dH, dyaw about a per-pass centre), selected by `poses.pass_of_time(gps_time)`.
    Built from a `pass_transforms.json`-shaped dict (see `geovap.stages.register.passes`); `poses`
    defaults to the pose table named in the transforms file (`poses_source`), else the default
    `load()`."""

    def __init__(self, transforms: dict, poses=None, poses_source: str | None = None):
        self.transforms = {int(k): v for k, v in transforms.get("passes", transforms).items() if str(k).lstrip("-").isdigit()}
        if poses is None:
            from geovap.runtime.pose_tables import load

            poses = load(poses_source or transforms.get("poses_source"))
        self.poses = poses
        self.hash = hashlib.sha1(json.dumps(transforms, sort_keys=True, default=str).encode()).hexdigest()[:10]

    def apply(self, xyz: np.ndarray, gps_time: np.ndarray) -> np.ndarray:
        if len(xyz) == 0:
            return xyz
        out = xyz.copy()
        passes = self.poses.pass_of_time(np.asarray(gps_time))
        for p in np.unique(passes):
            tr = self.transforms.get(int(p))
            if tr is None:
                continue
            m = passes == p
            cx, cy = tr["centre"]
            dE, dN, dH = tr["t"]
            a = np.radians(tr["yaw_deg"])
            c, s = np.cos(a), np.sin(a)
            Rz = np.array([[c, -s], [s, c]])
            xy = out[m, :2] - np.array([cx, cy])
            out[m, :2] = xy @ Rz.T + np.array([cx, cy]) + np.array([dE, dN])
            out[m, 2] += dH
        return out

    @staticmethod
    def load(source) -> "PassRegistration":
        """`source`: dict (already-loaded transforms), or a Path/str to a pass_transforms.json."""
        if isinstance(source, PassRegistration):
            return source
        if isinstance(source, dict):
            return PassRegistration(source)
        data = json.loads(Path(source).read_text())
        return PassRegistration(data)


class TileData:
    """Memmapped columns of one tile (read-only)."""

    def __init__(self, info: TileInfo, registration: "PassRegistration | None" = None):
        self.info = info
        self.registration = registration
        d = info.dir
        self.xyz = np.load(d / "xyz.npy", mmap_mode="r")
        self.intensity = np.load(d / "intensity.npy", mmap_mode="r")
        self.classification = np.load(d / "classification.npy", mmap_mode="r")
        self.rgb = np.load(d / "rgb.npy", mmap_mode="r")
        self.gps_time = np.load(d / "gps_time.npy", mmap_mode="r")
        self.psid = np.load(d / "psid.npy", mmap_mode="r")
        self.orig_index = np.load(d / "orig_index.npy", mmap_mode="r")
        self.cell_starts = np.load(d / "cell_starts.npy")
        # S1 extra columns: None if the tile predates them (old stores must still open)
        for name in EXTRA_COLUMNS:
            p = d / f"{name}.npy"
            setattr(self, name, np.load(p, mmap_mode="r") if p.exists() else None)
        # S1 time index: None if not built yet
        self.time_order = np.load(d / "time_order.npy", mmap_mode="r") if (d / "time_order.npy").exists() else None
        self.time_bucket_starts = np.load(d / "time_bucket_starts.npy") if (d / "time_bucket_starts.npy").exists() else None
        self._time_meta = json.loads((d / "time_meta.json").read_text()) if (d / "time_meta.json").exists() else None

    def __len__(self) -> int:
        return self.info.n

    def release(self) -> None:
        """Drop the pages this process has touched in the tile's memmaps (they stay valid and are re-read
        on demand). Keeps per-process RSS small: a process-tree memory watchdog counts memmapped file
        pages in every worker's RSS, so 4 workers x an 18 GB store look like 72 GB."""
        import mmap

        arrays = [self.xyz, self.intensity, self.classification, self.rgb, self.gps_time, self.psid, self.orig_index]
        arrays += [getattr(self, name) for name in EXTRA_COLUMNS]
        arrays += [self.time_order]
        for a in arrays:
            if a is None:
                continue
            mm = getattr(a, "_mmap", None)
            if mm is not None:
                try:
                    mm.madvise(mmap.MADV_DONTNEED)
                except (AttributeError, OSError, ValueError):
                    pass

    def xyz_m(self, rows=slice(None)) -> np.ndarray:
        """Point coordinates (m). With `registration` set (S5, off by default), applies the per-pass
        rigid transform selected by each row's gps_time; default behaviour (registration=None) is
        byte-identical to before S5."""
        xyz = np.asarray(self.xyz[rows], dtype=np.float64) * SCALE
        if self.registration is None:
            return xyz
        return self.registration.apply(xyz, np.asarray(self.gps_time[rows]))

    def rows_in_disc(self, cx: float, cy: float, radius: float) -> np.ndarray:
        """Local row indices of points whose cell may intersect the disc (superset; refine by distance)."""
        e0, n0 = self.info.grid_origin
        ny, nx = self.info.grid_shape
        cell_size = self.info.cell_size
        half_diag = cell_size * np.sqrt(0.5)
        ix = np.arange(nx)
        iy = np.arange(ny)
        cxs = e0 + (ix + 0.5) * cell_size
        cys = n0 + (iy + 0.5) * cell_size
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

    def rows_in_time(self, t0: float, t1: float) -> np.ndarray:
        """Local row indices (sorted ascending) with gps_time in [t0, t1]. Requires the time index
        (time_order.npy / time_bucket_starts.npy / time_meta.json); raises if missing."""
        if self.time_order is None or self.time_bucket_starts is None or self._time_meta is None:
            raise RuntimeError(f"tile {self.info.name}: time index not built (run store_add_columns)")
        t_min = self._time_meta["t_min"]
        bucket_s = self._time_meta["bucket_s"]
        n_buckets = self._time_meta["n_buckets"]
        b0 = max(0, int(np.floor((t0 - t_min) / bucket_s)))
        b1 = min(n_buckets, int(np.floor((t1 - t_min) / bucket_s)) + 1)  # exclusive upper bucket
        if b0 >= b1:
            return np.empty(0, dtype=np.int64)
        s = int(self.time_bucket_starts[b0])
        e = int(self.time_bucket_starts[b1])
        cand = np.asarray(self.time_order[s:e], dtype=np.int64)
        if len(cand) == 0:
            return cand
        g = np.asarray(self.gps_time[cand])
        m = (g >= t0) & (g <= t1)
        return np.sort(cand[m])


class CloudStore:
    def __init__(self, root: Path, *, registration: "Path | str | dict | PassRegistration | None" = None):
        self.root = Path(root)
        self._registration = PassRegistration.load(registration) if registration is not None else None
        meta = json.loads((self.root / "tiles.json").read_text())
        cell_size = meta["cell_size"]
        self.tiles: list[TileInfo] = [
            TileInfo(
                name=t["name"],
                root=self.root,
                laz=t["laz"],
                n=t["n"],
                row_offset=t["row_offset"],
                bbox=tuple(t["bbox"]),
                polygon=np.asarray(t["polygon"], dtype=np.float64).reshape(-1, 2),
                grid_origin=tuple(t["grid_origin"]),
                grid_shape=tuple(t["grid_shape"]),
                cell_size=cell_size,
                gps_time_range=tuple(t["gps_time_range"]) if "gps_time_range" in t else None,
            )
            for t in meta["tiles"]
        ]
        self.by_name = {t.name: t for t in self.tiles}
        self.total = sum(t.n for t in self.tiles)
        self._data: dict[str, TileData] = {}

    def tile(self, name: str) -> TileData:
        if name not in self._data:
            self._data[name] = TileData(self.by_name[name], registration=self._registration)
        return self._data[name]

    @property
    def registration_hash(self) -> str | None:
        return self._registration.hash if self._registration is not None else None

    def xyz_registered(self, tile: TileInfo, rows) -> np.ndarray:
        """Convenience: `self.tile(tile.name).xyz_m(rows)`, honouring `registration`."""
        return self.tile(tile.name).xyz_m(rows)

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

    def query_time(self, t0: float, t1: float, bbox: tuple[float, float, float, float] | None = None) -> list[tuple[TileInfo, np.ndarray]]:
        """[(tile, local_rows)] of points with gps_time in [t0, t1], restricted to tiles whose known
        gps_time range overlaps [t0, t1] (tiles without a cached range are always checked); rows sorted.
        `bbox` (minE, minN, maxE, maxN) additionally restricts to tiles whose bbox overlaps it."""
        res = []
        for t in self.tiles:
            if t.gps_time_range is not None and (t.gps_time_range[1] < t0 or t.gps_time_range[0] > t1):
                continue
            if bbox is not None:
                minE, minN, maxE, maxN = t.bbox
                if maxE < bbox[0] or minE > bbox[2] or maxN < bbox[1] or minN > bbox[3]:
                    continue
            td = self.tile(t.name)
            rows = td.rows_in_time(t0, t1)
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


def open_store(s=None, poses=None) -> CloudStore:
    """`CloudStore` honouring `poses.registration` (the S5b `pass_transforms.json` a corrected pose
    table's poses are only geometrically consistent with -- see `geovap.runtime.pose_tables` and
    `PassRegistration` above): `poses=None` or an export-sourced table (`.registration` is None) gives
    a plain, unregistered store, byte-identical to `CloudStore(root)`; any corrected table gives a
    store whose `xyz_m()` applies that table's per-pass rigid transform. Use this instead of a bare
    `CloudStore(root)` wherever a pose table is also in play, so the two never drift apart.

    `s`: `Settings` naming the dataset whose store to open; defaults to the process-wide one
    (`geovap.runtime.settings.get()`). The root is `s.workspace.store`."""
    from geovap.runtime import settings

    if s is None:
        s = settings.get()
    registration = poses.registration if poses is not None else None
    return CloudStore(s.workspace.store, registration=registration)


# ------------------------------------------------------------------------------------- build
def build_tile(laz_file: str, out_dir: str, cell_size: float, name: str) -> dict:
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
    e0 = np.floor(e_m.min() / cell_size) * cell_size
    n0 = np.floor(n_m.min() / cell_size) * cell_size
    ix = ((e_m - e0) // cell_size).astype(np.int64)
    iy = ((n_m - n0) // cell_size).astype(np.int64)
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
        "name": name,
        "laz": str(laz_path),
        "n": int(n),
        "bbox": [float(e_m.min()), float(n_m.min()), float(e_m.max()), float(n_m.max())],
        "grid_origin": [float(e0), float(n0)],
        "grid_shape": [ny, nx],
        "gps_time_range": [float(las.gps_time.min()), float(las.gps_time.max())],
        "classes": {int(c): int(k) for c, k in zip(*np.unique(las.classification, return_counts=True))},
    }


def build_store(s, workers: int = 8) -> None:
    """Build the columnar store for `s`'s tiles (`s.tiles.tiles()`, a `TileSource`) under
    `s.workspace.store`, at `s.sensor.cell_size`."""
    from multiprocessing import Pool

    root = Path(s.workspace.store)
    cell_size = s.sensor.cell_size
    (root / "tiles").mkdir(parents=True, exist_ok=True)
    refs = s.tiles.tiles()
    jobs = [(str(ref.path), str(root / "tiles" / ref.id.value), cell_size, ref.id.value) for ref in refs]
    rings = {ref.id.value: ref.ring for ref in refs}
    # largest first for better packing
    jobs.sort(key=lambda j: -Path(j[0]).stat().st_size)
    with Pool(workers) as pool:
        metas = pool.starmap(build_tile, jobs)
    metas.sort(key=lambda m: m["name"])
    offset = 0
    for m in metas:
        m["row_offset"] = offset
        offset += m["n"]
        ring = rings[m["name"]]
        if ring is None:
            m["polygon"] = []
        else:
            # sanity: tile bbox centre inside its layout polygon
            from matplotlib.path import Path as MplPath

            c = ((m["bbox"][0] + m["bbox"][2]) / 2, (m["bbox"][1] + m["bbox"][3]) / 2)
            assert MplPath(ring).contains_point(c), f"tile {m['name']} centre not inside layout ring"
            m["polygon"] = np.asarray(ring, dtype=np.float64).tolist()
    (root / "tiles.json").write_text(json.dumps({"total": offset, "cell_size": cell_size, "tiles": metas}))
    print(f"store built: {len(metas)} tiles, {offset} points -> {root}")
