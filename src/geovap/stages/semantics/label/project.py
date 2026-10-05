"""`seg-project`: push 2D segmentation masks (bench/<tag>/f%04d_common.png) into the 3D cloud, tile by tile.

Per tile: for each clean frame with a prediction, project the tile's points within r_max, test visibility
against the frame's depth panorama, drop vehicle-body samples, read the predicted common class and its
confidence at the point's z-buffer cell and add a weighted vote:

    w = conf/255 * 1/(1+(r/score_r0)^2) * min(1, d_edge/EDGE_PX)

`d_edge` is the pixel distance to the nearest class boundary of the mask (suppresses 1-px label bleed onto far
points at depth discontinuities); `sky` votes get weight 0 (a 3D point is never sky) but are counted.
Votes are accumulated as a float32 [n, C] weighted histogram; label = argmax, conf = winning share.

Outputs (store row order, like segds/point_labels): out_dir/labels/NN.npy (uint8, 255 = unlabelled),
NN_conf.npy, NN_nviews.npy, NN_meta.json (written last = resume marker), and a LAS 1.4 PF7 tile with
`classification` = common id, extra dims seg_conf / seg_n_views / seg_src_frame / src_class.

Folds `mapping/cli/seg_project.py`'s `run` subcommand in as this module's `main()`, and declares the
`seg-project` `StageSpec` (named for the marker file the old `mapping.cli.pipeline` driver already
wrote, `seg-project.json`). `eval`/`render`/`potree-classes` live in `report.py`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from geovap.domain.model import geometry
from geovap.domain.math import depth as zbuffer
from geovap.io import las_writer
from geovap.runtime import manifest, settings
from geovap.stages.prepare import products
from geovap.runtime.store import CloudStore, TileInfo, open_store
from geovap.domain.model.frames import FrameIndex
from geovap.runtime.pose_tables import load as load_poses
from geovap.domain.model.rig import IDENTITY
from geovap.domain.scheme import taxonomy as T
from geovap.stages.base.spec import StageSpec, registry
from geovap.stages.semantics.segment.bench import bench_dir, dataset_seg_dir
from geovap.domain.model.tiles import id_from_sidecar

N_CLASSES = len(T.COMMON)
SKY_ID = T.COMMON_ID["sky"]
EDGE_PX = 3.0
TW45_TAG = "tw45"


def _out_name(name: str, kind: str = "", *, variant: str | None = None) -> str:
    """Product filename for a tile, from the dataset descriptor. Replaces the literal `ID3432_000`
    prefix, which was Dražkov's and was duplicated across ten modules -- on another dataset it wrote
    every product under the wrong name, with no error anywhere."""
    from geovap.domain.model.tiles import TileId

    return settings.get().tiles.out_name(TileId(name), kind, variant=variant)


# Pose-source aware, like `pseudogt.areas.segds_dir()`: the plain path for the default "export" pose
# table, a hash-suffixed sibling for a corrected one, so the two runs' outputs never mix. Functions,
# not constants: resolving them at import time is what made the old code unable to choose a dataset
# after argparse had run, and made merely importing this module fail on a machine with none
# configured.
def seg_out_dir(s=None) -> Path:
    s = s or settings.get()
    return s.workspace.source_dir(s.workspace.out / "seg_eomt", load_poses(s=s))


def las_dir(s=None) -> Path:
    s = s or settings.get()
    return s.workspace.source_dir(s.paths.publish / "seg_eomt", load_poses(s=s)) / "tiles"


@dataclass
class Options:
    bench_tag: str = "eomt_city"
    frames: list[int] | None = None  # default: clean frames that have a prediction
    r_max: float | None = None  # None -> s.sensor.r_max
    occlusion: bool = True
    vehicle_mask: bool = True
    min_conf: int = 0
    edge_px: float = EDGE_PX
    rgb: str = "ref"  # ref = TerraScan RGB from the store | tw45 = colorization product
    out_dir: Path = field(default_factory=seg_out_dir)
    las_dir: Path = field(default_factory=las_dir)
    write_las: bool = True
    subsample: int | None = None
    frame_limit: int | None = None
    poses_source: str | None = None  # None -> env GEOVAP_POSES / "export"; explicit initarg for worker pools
    tw45_tag: str = TW45_TAG  # colorization run tag under workspace out/<tag>/tiles, used when rgb="tw45"


@dataclass
class TileResult:
    name: str
    n: int
    seconds: float
    n_frames: int
    coverage: float
    counts: list[int] = field(default_factory=list)
    sky_votes: int = 0
    vehicle_samples: int = 0


def clean_frames() -> list[int]:
    return settings.get().workspace.clean_frames("clean")


def mask_paths(tag: str, k: int) -> tuple[Path, Path]:
    d = bench_dir() / tag
    return d / f"f{k:04d}_common.png", d / f"f{k:04d}_conf.png"


def frames_with_predictions(opt: Options) -> list[int]:
    ks = clean_frames() if opt.frames is None else list(opt.frames)
    return [k for k in ks if mask_paths(opt.bench_tag, k)[0].exists()]


def edge_factor(common: np.ndarray, edge_px: float) -> np.ndarray:
    """[H,W] float32 in [0,1]: distance to the nearest class boundary / edge_px, clipped. 255 counts as a class."""
    if edge_px <= 0:
        return np.ones(common.shape, np.float32)
    k = np.ones((3, 3), np.uint8)
    edge = (cv2.erode(common, k) != common) | (cv2.dilate(common, k) != common)
    dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    return np.minimum(dist / edge_px, 1.0).astype(np.float32)


def load_mask(tag: str, k: int, edge_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pc, pf = mask_paths(tag, k)
    common = cv2.imread(str(pc), cv2.IMREAD_GRAYSCALE)
    conf = cv2.imread(str(pf), cv2.IMREAD_GRAYSCALE)
    if common is None or conf is None:
        raise FileNotFoundError(pc)
    return common, conf, edge_factor(common, edge_px)


def _score(r: np.ndarray, score_r0: float) -> np.ndarray:
    return (1.0 / (1.0 + (r / score_r0) ** 2)).astype(np.float32)


class VoteHist:
    """float32 weighted class histogram per point + strongest single vote (provenance)."""

    def __init__(self, n: int, n_classes: int = N_CLASSES):
        self.n = n
        self.wsum = np.zeros((n, n_classes), np.float32)
        self.n_views = np.zeros(n, np.uint8)
        self.best_w = np.zeros(n, np.float32)
        self.best_frame = np.zeros(n, np.uint16)

    def update(self, rows: np.ndarray, label: np.ndarray, w: np.ndarray, frame: int) -> None:
        if len(rows) == 0:
            return
        np.add.at(self.wsum, (rows, label.astype(np.int64)), w)
        self.n_views[rows] = np.minimum(self.n_views[rows] + 1, 255)
        m = w > self.best_w[rows]
        self.best_w[rows[m]] = w[m]
        self.best_frame[rows[m]] = frame

    def finalize(self) -> dict:
        tot = self.wsum.sum(1)
        has = tot > 0
        label = self.wsum.argmax(1).astype(np.uint8)
        label[~has] = T.IGNORE
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(has, self.wsum.max(1) / np.maximum(tot, 1e-9), 0.0)
        conf = np.clip(np.round(255.0 * share), 0, 255).astype(np.uint8)
        return {"label": label, "conf": conf, "n_views": self.n_views, "src_frame": self.best_frame}


def _tw45_rgb(td, opt: Options, s) -> np.ndarray | None:
    p = s.workspace.out / opt.tw45_tag / "tiles" / _out_name(td.info.name, "_colored")
    if not p.exists():
        return None
    import laspy

    las = laspy.read(str(p))
    rgb_src = np.stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)], 1) >> 8
    return rgb_src[np.asarray(td.orig_index)].astype(np.uint8)  # source order -> store order


def project_tile(tile: TileInfo, store: CloudStore, fi: FrameIndex, opt: Options, log=print) -> TileResult:
    s = settings.get()
    r_max = s.sensor.r_max if opt.r_max is None else opt.r_max
    pano_w, pano_h = s.sensor.pano
    r_min = s.sensor.r_min
    t0 = time.time()
    td = store.tile(tile.name)
    rows_all = np.arange(len(td)) if opt.subsample is None else np.arange(0, len(td), opt.subsample)
    n = len(rows_all)
    xyz = td.xyz_m(rows_all)
    src_cls = np.asarray(td.classification[rows_all]).copy()
    rgb = None
    if opt.write_las and opt.subsample is None:
        rgb = _tw45_rgb(td, opt, s) if opt.rgb == "tw45" else None
        if rgb is None:
            if opt.rgb == "tw45":
                log(f"tile {tile.name}: tw45 product missing, falling back to reference RGB")
            rgb = np.asarray(td.rgb).copy()
    store.release()

    cand = fi.frames_for_tile(tile, r_max)
    have = set(frames_with_predictions(opt))
    frames = [int(k) for k in cand if int(k) in have]
    if opt.frame_limit:
        frames = frames[: opt.frame_limit]
    acc = VoteHist(n)
    n_sky = 0
    n_veh = 0
    vmask = None
    if opt.vehicle_mask:
        from geovap.stages.prepare.masks import VehicleMask

        _mask_path = s.workspace.vehicle_mask
        if _mask_path.exists():
            vmask = VehicleMask(_mask_path, *s.sensor.pano)
        else:
            log("WARNING: vehicle mask not built, continuing without it")

    for k in frames:
        R, C = fi.R[k], fi.C[k]
        d2 = (xyz[:, 0] - C[0]) ** 2 + (xyz[:, 1] - C[1]) ** 2
        sel = np.flatnonzero(d2 <= r_max**2)
        if len(sel) == 0:
            continue
        u, v, r, _el = geometry.world_to_pano(xyz[sel], R, C, pano_w, pano_h)
        inr = zbuffer.range_filter(r, r_min, r_max)
        sel, u, v, r = sel[inr], u[inr], v[inr], r[inr]
        if len(sel) == 0:
            continue
        fp = products.load_products(k, fi.poses, IDENTITY)
        if opt.occlusion:
            vis = fp.visible(r, u, v)
            sel, u, v, r = sel[vis], u[vis], v[vis], r[vis]
            if len(sel) == 0:
                continue
        if vmask is not None:
            on_vehicle = vmask(u, v)
            n_veh += int(on_vehicle.sum())
            keep = ~on_vehicle
            sel, u, v, r = sel[keep], u[keep], v[keep], r[keep]
            if len(sel) == 0:
                continue
        common, conf, ef = load_mask(opt.bench_tag, k, opt.edge_px)
        cv, cu = fp.cell(u, v)
        lab = common[cv, cu]
        cf = conf[cv, cu]
        ok = (lab != T.IGNORE) & (cf >= opt.min_conf)
        is_sky = lab == SKY_ID
        n_sky += int((ok & is_sky).sum())
        ok &= ~is_sky
        if not ok.any():
            continue
        w = (cf[ok].astype(np.float32) / 255.0) * _score(r[ok], s.tuning.score_r0) * ef[cv[ok], cu[ok]]
        acc.update(sel[ok], lab[ok], w, k)

    store.drop_cache((s.workspace.frames_dir(fi.poses),))
    out = acc.finalize()
    label = out["label"]
    counts = np.bincount(label, minlength=256)
    coverage = float((label != T.IGNORE).mean()) if n else 0.0

    if opt.subsample is None:
        lab_dir = Path(opt.out_dir) / "labels"
        lab_dir.mkdir(parents=True, exist_ok=True)
        np.save(lab_dir / f"{tile.name}.npy", label)
        np.save(lab_dir / f"{tile.name}_conf.npy", out["conf"])
        np.save(lab_dir / f"{tile.name}_nviews.npy", out["n_views"])
        if opt.write_las:
            extras = {"seg_conf": out["conf"], "seg_n_views": out["n_views"], "seg_src_frame": out["src_frame"], "src_class": src_cls}
            prov = {
                "product": "semantic_labels", "classification_scheme": "common15", "classes": T.COMMON, "unlabelled": T.IGNORE,
                "bench_tag": opt.bench_tag, "checkpoint": _checkpoint(opt.bench_tag),
                "weights": f"conf/255 * 1/(1+(r/{s.tuning.score_r0})^2) * min(1, d_edge/{opt.edge_px}px); sky votes dropped",
                "r_max": r_max, "occlusion": opt.occlusion, "vehicle_mask": vmask is not None, "rgb": opt.rgb,
                "git": manifest.git_rev(), "frames": frames, "poses_source": fi.poses.source, "poses_hash": fi.poses.hash(),
            }
            las_writer.write_tile(
                td.info.laz, np.asarray(td.orig_index),
                Path(opt.las_dir) / _out_name(tile.name, "_seg"), rgb, extras, prov,
                extra_dims=las_writer.SEG_DIMS, classification=label,
                description="semantic labels provenance")
        meta = {"n": n, "seconds": time.time() - t0, "n_frames": len(frames), "coverage": coverage,
                "counts": counts[:N_CLASSES].tolist(), "unlabelled": int(counts[T.IGNORE]), "sky_votes": n_sky, "vehicle_samples": n_veh,
                "nviews_hist": np.bincount(out["n_views"], minlength=32)[:32].tolist(), "frames": frames,
                "poses_source": fi.poses.source, "poses_hash": fi.poses.hash()}
        (lab_dir / f"{tile.name}_meta.json").write_text(json.dumps(meta))  # last: resume marker

    res = TileResult(name=tile.name, n=n, seconds=time.time() - t0, n_frames=len(frames), coverage=coverage,
                     counts=counts[:N_CLASSES].tolist(), sky_votes=n_sky, vehicle_samples=n_veh)
    log(f"tile {tile.name}: {n} pts, {len(frames)} frames, {res.seconds:.0f} s, coverage {coverage:.3f}, "
        f"top classes {_top(counts)}")
    return res


def _top(counts: np.ndarray, k: int = 4) -> str:
    c = counts[:N_CLASSES]
    idx = np.argsort(-c)[:k]
    tot = max(int(c.sum()), 1)
    return ", ".join(f"{T.COMMON[i]} {100 * c[i] / tot:.0f}%" for i in idx if c[i] > 0)


def _checkpoint(tag: str) -> str | None:
    p = dataset_seg_dir() / f"id2label_{tag}.json"
    return json.loads(p.read_text()).get("checkpoint") if p.exists() else None


# ----------------------------------------------------------------------------- multiprocessing driver
_G: dict = {}


def _init(opt: Options):
    # explicit poses_source on opt (not just env inheritance via fork) so it is honoured even if
    # the pool start method is not "fork".
    poses = load_poses(opt.poses_source)
    _G["store"] = open_store(poses=poses)
    _G["fi"] = FrameIndex(poses, IDENTITY)
    _G["opt"] = opt


def _run_tile(name: str) -> TileResult:
    store = _G["store"]
    return project_tile(store.by_name[name], store, _G["fi"], _G["opt"])


def run(tiles: list[str] | None, opt: Options, workers: int = 5) -> list[TileResult]:
    from multiprocessing import Pool

    store = open_store(poses=load_poses(opt.poses_source))
    names = [t.name for t in store.tiles] if tiles is None else list(tiles)
    names = sorted(names, key=lambda nm: -store.by_name[nm].n)  # largest first
    (Path(opt.out_dir) / "labels").mkdir(parents=True, exist_ok=True)
    if opt.subsample is None:
        done = {id_from_sidecar(p) for p in (Path(opt.out_dir) / "labels").glob("*_meta.json")}
        if done:
            print(f"resuming: {len(done)} tiles already done")
            names = [nm for nm in names if nm not in done]
    if workers <= 1:
        _init(opt)
        return [_run_tile(nm) for nm in names]
    with Pool(workers, initializer=_init, initargs=(opt,)) as pool:
        return list(pool.imap_unordered(_run_tile, names))


# ================================================================================================ stage
class SegProject:
    spec = StageSpec(
        name="seg-project", after=("seg-eval",), optional=True, est_min=15,
        summary="project the primary model's 2D masks into the point cloud, tile by tile",
    )
    cli_args: tuple[str, ...] = ()

    def available(self, s) -> bool:
        """Same precondition as `segds`/`seg-eval`: there is nothing to project onto without a
        `[reference]`-derived pseudo-GT to validate against."""
        return s.reference is not None

    def inputs(self, s) -> dict[str, Path]:
        return {"clean_frames": s.workspace.clean_frames_json}

    def outputs(self, s) -> list[Path]:
        return [seg_out_dir(s) / "eval.json"]

    def metrics(self, s) -> dict:
        try:
            return json.loads((seg_out_dir(s) / "eval.json").read_text())
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s, *, tag: str = "eomt_city", workers: int = 8) -> None:
        from geovap.stages.semantics.label import report as report_mod

        opt = Options(bench_tag=tag, poses_source=s.pose_table)
        run(None, opt, workers=workers)
        report_mod.evaluate()
        report_mod.render_all(tag=tag)


STAGE = registry.add(SegProject())


# ================================================================================================ cli
def main(argv=None) -> int:
    """Folds `mapping/cli/seg_project.py` in.

    uv run python -m geovap.stages.semantics.label.project run [--tiles 037 ...] [--workers 5] [--rgb ref|tw45] [--subsample N] [--frame-limit N]
    uv run python -m geovap.stages.semantics.label.project eval
    uv run python -m geovap.stages.semantics.label.project render
    uv run python -m geovap.stages.semantics.label.project potree-classes

    `eval`/`render`/`potree-classes` delegate to `report.py`.
    """
    import argparse

    from geovap.stages.base.cli import add_dataset_flags, configure_from, describe

    ap = argparse.ArgumentParser(description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=False)
    r = sub.add_parser("run")
    r.add_argument("--tiles", nargs="*", default=None)
    r.add_argument("--workers", type=int, default=5)
    r.add_argument("--tag", default="eomt_city", help="bench tag under segds/bench")
    r.add_argument("--rgb", default="ref", choices=["ref", "tw45"])
    r.add_argument("--rmax", type=float, default=None)
    r.add_argument("--min-conf", type=int, default=0)
    r.add_argument("--edge-px", type=float, default=EDGE_PX)
    r.add_argument("--no-occlusion", action="store_true")
    r.add_argument("--no-las", action="store_true")
    r.add_argument("--subsample", type=int, default=None)
    r.add_argument("--frame-limit", type=int, default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--las-dir", default=None)
    sub.add_parser("eval")
    rn = sub.add_parser("render")
    rn.add_argument("--frames", default=None, help="comma list of frames for the ERP round-trip panels")
    rn.add_argument("--tiles", nargs="*", default=None, help="tiles for BEV / oblique renders")
    pc = sub.add_parser("potree-classes")
    pc.add_argument("--out", default=None, help="directory of the Potree octree (default s.paths.publish / eomt_city_seg)")
    add_dataset_flags(ap)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    s = configure_from(a)

    if a.cmd is None:
        if not STAGE.available(s):
            if STAGE.spec.optional:
                print(f"{STAGE.spec.name}: unavailable for dataset {s.name!r}; skipping (stage is optional)")
                return 0
            ap.error(f"{STAGE.spec.name}: unavailable for dataset {s.name!r}")
        if a.status:
            print(describe(STAGE, s))
        return 0

    if a.cmd == "run":
        opt = Options(bench_tag=a.tag, r_max=a.rmax, occlusion=not a.no_occlusion, min_conf=a.min_conf, edge_px=a.edge_px, rgb=a.rgb,
                      write_las=not a.no_las, subsample=a.subsample, frame_limit=a.frame_limit, poses_source=a.poses)
        if a.out:
            opt.out_dir = Path(a.out)
        if a.las_dir:
            opt.las_dir = Path(a.las_dir)
        run(a.tiles, opt, workers=a.workers)
    elif a.cmd == "eval":
        from geovap.stages.semantics.label import report as report_mod

        report_mod.evaluate(poses_source=a.poses)
    elif a.cmd == "render":
        from geovap.stages.semantics.label import report as report_mod

        frames = [int(x) for x in a.frames.split(",")] if a.frames else None
        report_mod.render_all(frames=frames, tiles=a.tiles, poses_source=a.poses)
    elif a.cmd == "potree-classes":
        from geovap.stages.semantics.label import report as report_mod

        report_mod.write_potree_classes(Path(a.out) if a.out else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
