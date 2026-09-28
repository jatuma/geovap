"""`products`: per-frame depth panorama + point-id panorama at z-buffer resolution.

These are the inverse-mapping enablers: `pano_to_world` and `pano_to_point` (client code) read them,
and the visibility test in colorization uses the closed depth. Products are tied to a rig model
(hash in meta); a refit invalidates them.

Ported from `mapping/products.py` + `mapping/cli/build_frames.py` and turned into the `products`
stage of the `prepare` group -- the first per-frame artifact every other stream reads (`frame_products`
in `geovap.runtime.artifacts`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from geovap.domain.math import depth as zbuffer
from geovap.domain.model import geometry
from geovap.domain.model.frames import FrameIndex
from geovap.domain.model.poses import Poses
from geovap.domain.model.rig import IDENTITY, RigModel
from geovap.runtime import pose_tables
from geovap.runtime.store import CloudStore, open_store
from geovap.stages.base.cli import stage_main
from geovap.stages.base.spec import StageSpec, registry

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings


def product_path(frame: int, root: Path) -> Path:
    return Path(root) / f"f{frame:04d}.npz"


@dataclass
class FrameProducts:
    frame: int
    depth_mm: np.ndarray  # uint16 [ZB_H, ZB_W], 0 = empty
    point_id: np.ndarray  # uint32 [ZB_H, ZB_W], NO_POINT = empty
    meta: dict
    _closed: np.ndarray | None = None
    _spread: np.ndarray | None = None

    @property
    def depth_m(self) -> np.ndarray:
        return zbuffer.depth_from_mm(self.depth_mm)

    @property
    def depth_closed(self) -> np.ndarray:
        if self._closed is None:
            self._closed = zbuffer.close_depth(self.depth_m)
        return self._closed

    @property
    def spread(self) -> np.ndarray:
        if self._spread is None:
            self._spread = zbuffer.depth_spread(self.depth_closed)
        return self._spread

    @property
    def scale(self) -> float:
        """full-res px -> z-buffer px.

        Read from `meta["pano_w"]` -- the panorama width the product was actually BUILT at -- rather
        than from the currently configured `Settings.sensor.pano_w`: a product built at a different
        panorama size (an older run, a different dataset revision) must still be readable, and
        re-deriving this from live settings would silently misinterpret it instead of reading it
        correctly (or raising, via the rig/poses hash checks in `load` below).
        """
        return self.depth_mm.shape[1] / self.meta["pano_w"]

    def save(self, root: Path) -> Path:
        p = product_path(self.frame, root)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, depth_mm=self.depth_mm, point_id=self.point_id, meta=json.dumps(self.meta))
        return p

    @classmethod
    def load(
        cls,
        frame: int,
        rig: RigModel | None = None,
        *,
        root: Path,
        allow_stale: bool = False,
        poses: Poses | None = None,
    ) -> "FrameProducts":
        with np.load(product_path(frame, root)) as z:
            meta = json.loads(str(z["meta"]))
            fp = cls(frame=frame, depth_mm=z["depth_mm"], point_id=z["point_id"], meta=meta)
        if rig is not None and meta["rig_hash"] != rig.hash() and not allow_stale:
            raise RuntimeError(f"frame {frame}: products built for rig {meta['rig_hash']}, requested {rig.hash()}")
        if poses is not None and meta.get("poses_hash") != poses.hash() and not allow_stale:
            raise RuntimeError(f"frame {frame}: products built for poses {meta.get('poses_hash')}, requested {poses.hash()}")
        return fp

    # ----------------------------------------------------------------- inverse mapping
    def cell(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.scale
        h, w = self.depth_mm.shape
        cv = np.clip((np.asarray(v) * s).astype(np.int64), 0, h - 1)
        cu = np.mod((np.asarray(u) * s).astype(np.int64), w)
        return cv, cu

    def range_at(self, u, v, closed: bool = False) -> np.ndarray:
        cv, cu = self.cell(u, v)
        d = self.depth_closed if closed else self.depth_m
        return d[cv, cu]

    def point_at(self, u, v) -> np.ndarray:
        cv, cu = self.cell(u, v)
        return self.point_id[cv, cu]

    def visible(self, r: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Visibility of points at full-res (u, v) with range r.

        Tolerances and range limits are read from `meta` (recorded at build time), for the same
        reason `scale` reads `meta["pano_w"]`: a product is self-describing, not dependent on
        whatever `Settings` happens to be active when it is later read.
        """
        s = self.scale
        return zbuffer.visible(
            r, np.asarray(u) * s, np.asarray(v) * s, self.depth_closed,
            self.meta["tol_abs"], self.meta["tol_rel"], spread_m=self.spread,
        ) & zbuffer.range_filter(r, self.meta["r_min"], self.meta["r_max"])


def load_products(frame: int, poses: Poses, rig: RigModel | None = IDENTITY, *, s: "Settings | None" = None, **kw) -> FrameProducts:
    """`FrameProducts.load` scoped to a pose table: root=`s.workspace.frames_dir(poses)`, `poses=poses`
    (so products built for the wrong pose table raise loudly instead of silently mixing sources). Use
    this instead of a bare `FrameProducts.load(k, ...)` wherever a pose table is also in play.

    `s` defaults to `geovap.runtime.settings.get()`, imported here (not at module scope) so this
    module stays importable standalone."""
    from geovap.runtime import settings

    if s is None:
        s = settings.get()
    return FrameProducts.load(frame, rig, root=s.workspace.frames_dir(poses), poses=poses, **kw)


# ------------------------------------------------------------------------------------- building
TIME_WINDOW_S = 45.0  # occluders come only from points scanned within this window of the frame (same pass):
# the cloud merges 29 passes, a photo is one instant - gates, parked cars, people from other passes must not occlude.


def gather_candidates(store: CloudStore, C: np.ndarray, r_max: float, t_frame: float | None = None, time_window_s: float | None = None):
    """Points within r_max of camera centre C (2D), from all tiles; optionally only those scanned within
    time_window_s of t_frame. Returns (xyz_m f64 [N,3], point_id u32 [N])."""
    parts = store.query_disc(float(C[0]), float(C[1]), r_max)
    if not parts:
        return np.empty((0, 3)), np.empty(0, np.uint32)
    if t_frame is not None and time_window_s is not None:
        filt = []
        for t, rows in parts:
            g = np.asarray(store.tile(t.name).gps_time[rows])
            rows = rows[np.abs(g - t_frame) <= time_window_s]
            if len(rows):
                filt.append((t, rows))
        parts = filt
        if not parts:
            return np.empty((0, 3)), np.empty(0, np.uint32)
    xyz = np.concatenate([store.tile(t.name).xyz_m(rows) for t, rows in parts])
    pid = np.concatenate([store.global_ids(t, rows) for t, rows in parts])
    return xyz, pid


def build_frame(s: "Settings", frame: int, store: CloudStore, fi: FrameIndex, rig: RigModel = IDENTITY, r_max: float | None = None, time_window_s: float | None = TIME_WINDOW_S) -> FrameProducts:
    sensor = s.sensor
    r_max = sensor.r_max if r_max is None else r_max
    R, C = fi.R[frame], fi.C[frame]
    xyz, pid = gather_candidates(store, C, r_max, fi.t[frame], time_window_s)
    u, v, r, el = geometry.world_to_pano(xyz, R, C, sensor.pano_w, sensor.pano_h)
    keep = zbuffer.range_filter(r, sensor.r_min, r_max)
    zb_s = sensor.zb_w / sensor.pano_w
    depth, ids = zbuffer.splat(u[keep] * zb_s, v[keep] * zb_s, r[keep], pid[keep], sensor.zb_w, sensor.zb_h, sensor)
    meta = {
        "frame": frame,
        "filename": str(fi.poses.filename[frame]),
        "rig_hash": rig.hash(),
        "rig": {"boresight_deg": list(rig.boresight_deg), "lever_arm_m": list(rig.lever_arm_m), "dt_s": rig.dt_s},
        "poses_hash": fi.poses.hash(),
        "poses_source": fi.poses.source,
        "pano_w": sensor.pano_w,  # the panorama width this product was splatted at; see `FrameProducts.scale`
        "pano_h": sensor.pano_h,
        "zb": [sensor.zb_h, sensor.zb_w],
        "r_min": sensor.r_min,
        "r_max": r_max,
        "tol_abs": sensor.tol_abs,
        "tol_rel": sensor.tol_rel,
        "time_window_s": time_window_s,
        "n_candidates": int(len(pid)),
        "n_splatted": int(keep.sum()),
        "git": _git_rev(),
    }
    return FrameProducts(frame=frame, depth_mm=zbuffer.depth_to_mm(depth), point_id=ids, meta=meta)


def _git_rev() -> str:
    from geovap.runtime import manifest

    return manifest.git_rev() or "unknown"


_G: dict = {}


def _init_worker(dataset_env: dict[str, str], rig_vec, with_la: bool, poses_source: str | None) -> None:
    """Re-resolve `Settings` in the child: the pool may not use `fork`, so the parent's in-memory
    `Settings` (built from CLI flags, not necessarily mirrored into `os.environ`) would otherwise be
    invisible here. `dataset_env` is `Settings.env()` from the parent -- the same mechanism
    `geovap.runtime.procs` uses to hand a dataset to a stage subprocess.

    `poses_source` is ALSO passed explicitly (not just implied by `dataset_env`'s `GEOVAP_POSES`) so a
    `poses_source` given at call time is honoured even if the pool start method is not "fork" --
    ported unchanged from `mapping/products.py:_init_worker`'s reasoning.
    """
    import os

    os.environ.update(dataset_env)
    from geovap.runtime import settings

    settings.reset()
    s = settings.get()
    _G["s"] = s
    _G["poses"] = pose_tables.load(poses_source, s=s)
    _G["store"] = open_store(s, poses=_G["poses"])  # registered cloud when poses carries a "registration"
    _G["rig"] = RigModel.from_vector(rig_vec, with_lever_arm=with_la)
    _G["fi"] = FrameIndex(_G["poses"], _G["rig"])


def _build_and_save(frame: int) -> int:
    s = _G["s"]
    fp = build_frame(s, frame, _G["store"], _G["fi"], _G["rig"])
    fp.save(s.workspace.frames_dir(_G["poses"]))
    return fp.meta["n_splatted"]


def build_all_frames(s: "Settings", frames=None, rig: RigModel = IDENTITY, workers: int = 16, poses_source: str | None = None, skip_existing: bool = True) -> None:
    from geovap.runtime.procs import pool_context

    from tqdm import tqdm

    poses = pose_tables.load(poses_source, s=s)
    frames = list(range(len(poses))) if frames is None else list(frames)
    root = s.workspace.frames_dir(poses)
    root.mkdir(parents=True, exist_ok=True)
    rig.to_json(root / "rig.json")
    n_requested = len(frames)
    if skip_existing:
        frames = [f for f in frames if not product_path(f, root).exists()]
    n_skipped = n_requested - len(frames)
    n_built = len(frames)
    print(f"build_all_frames: {n_requested} requested, {n_skipped} skipped (already exist), {n_built} to build")
    if not frames:
        return
    with pool_context().Pool(workers, initializer=_init_worker, initargs=(s.env(), rig.as_vector(True), True, poses_source)) as pool:
        for _ in tqdm(pool.imap_unordered(_build_and_save, frames, chunksize=4), total=len(frames), desc="frames"):
            pass


# ================================================================================================ stage
class Products:
    spec = StageSpec(
        name="products", after=("store",), est_min=90,
        summary="per-frame depth + point-id panoramas",
    )

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        if s.pose_table == "export":
            poses_path = s.poses.source_file()
        else:
            poses_path = pose_tables.path_for(s.pose_table, s=s)
        return {"poses": poses_path, "tiles": s.workspace.store / "tiles.json"}

    def outputs(self, s: "Settings") -> list[Path]:
        poses = pose_tables.load(s=s)
        return [s.workspace.frames_dir(poses) / "rig.json"]

    def metrics(self, s: "Settings") -> dict[str, Any]:
        try:
            poses = pose_tables.load(s=s)
            root = s.workspace.frames_dir(poses)
            n_expected = len(poses)
            n_built = sum(1 for f in range(n_expected) if product_path(f, root).exists())
            return {"root": str(root), "n_expected": n_expected, "n_built": n_built}
        except Exception as e:  # noqa: BLE001 - metrics must always be reportable
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", *, workers: int = 16, frames: list[int] | None = None, rig: RigModel = IDENTITY, force: bool = False) -> None:
        build_all_frames(s, frames=frames, rig=rig, workers=workers, poses_source=s.pose_table, skip_existing=not force)


STAGE = registry.add(Products())


def _add_options(p) -> None:
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--force", action="store_true", help="rebuild even if the product file already exists")
    p.add_argument("--frames", default=None, help="comma-separated frame indices (default: all frames in the pose table)")


def _to_opts(args) -> dict:
    return {
        "workers": args.workers,
        "force": args.force,
        "frames": [int(f) for f in args.frames.split(",")] if args.frames else None,
    }


def main(argv=None) -> int:
    return stage_main(STAGE, argv, add_options=_add_options, to_opts=_to_opts)


if __name__ == "__main__":
    raise SystemExit(main())
