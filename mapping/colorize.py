"""Client B: colour the point cloud from panoramas (pano -> cloud), tile by tile, frame by frame.

Per tile: for each candidate frame decode the image once, project the tile's points within R_MAX,
test visibility against the frame's depth panorama, sample, and feed three accumulators:
  - median top-K (product)
  - nearest-in-time, with occlusion      (validation vs TerraScan)
  - nearest-in-time, without occlusion   (the pilot's baseline, A1 ablation)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geometry, las_out, metrics, products, zbuffer
from .accumulate import ColourTopK, NearestInTime
from .cloud_store import CloudStore, TileInfo
from .config import INCIDENCE_MAX_DEG, OUT_DIR, R_MAX, R_MIN, SCORE_R0, TOP_K
from .frame_select import FrameIndex
from .poses import load_poses
from .rig import IDENTITY, RigModel
from .sample import PanoSampler, load_pano_rgb, srgb_to_linear


@dataclass
class Options:
    rig: RigModel = IDENTITY
    r_max: float = R_MAX
    top_k: int = TOP_K
    occlusion: bool = True  # A1: visibility test for the product / nt variant
    sampling: str = "footprint"  # A4: nearest | bilinear | footprint  (product); nt variants use nearest (pilot)
    nt_sampling: str = "nearest"
    write_las: bool = True
    subsample: int | None = None  # every n-th point (fast experiments)
    frame_limit: int | None = None
    out_dir: Path = OUT_DIR
    tag: str = "run"
    mirror: bool = False  # regression check: mirrored azimuth must give median CIE76 ~ 17
    vehicle_mask: bool = True  # drop samples that fall on the vehicle body


@dataclass
class TileResult:
    name: str
    n: int
    seconds: float
    n_frames: int
    coverage: dict
    hist: dict = field(default_factory=dict)  # variant -> StrataHist
    sample: dict | None = None  # stratified subsample for ad-hoc analysis


def _score(r: np.ndarray, cos_inc: np.ndarray | None = None) -> np.ndarray:
    s = 1.0 / (1.0 + (r / SCORE_R0) ** 2)
    if cos_inc is not None:
        s = s * np.abs(cos_inc)
    return s.astype(np.float32)


def colorize_tile(tile: TileInfo, store: CloudStore, fi: FrameIndex, opt: Options, log=print) -> TileResult:
    t0 = time.time()
    td = store.tile(tile.name)
    rows_all = np.arange(len(td)) if opt.subsample is None else np.arange(0, len(td), opt.subsample)
    n = len(rows_all)
    xyz = td.xyz_m(rows_all)
    gps = np.asarray(td.gps_time[rows_all])
    ref_rgb = np.asarray(td.rgb[rows_all])
    cls = np.asarray(td.classification[rows_all])
    psid = np.asarray(td.psid[rows_all])
    store.release()  # tile columns are copied above; drop the memmap pages from this worker's RSS

    nt_frame = fi.nearest_in_time(gps)  # designated frame per point
    frames = fi.frames_for_tile(tile, opt.r_max)
    if opt.frame_limit:
        frames = frames[: opt.frame_limit]
    acc_top = ColourTopK(n, opt.top_k)
    acc_nt = NearestInTime(n)
    acc_nt_noocc = NearestInTime(n)
    # per-point geometry from the nt frame for stratification
    el_nt = np.zeros(n, np.float32)
    az_nt = np.zeros(n, np.float32)
    n_sat = 0
    n_veh = 0
    vmask = None
    if opt.vehicle_mask:
        from .vehicle_mask import MASK_PATH, VehicleMask

        if MASK_PATH.exists():
            vmask = VehicleMask()
        else:
            log("WARNING: vehicle mask not built, continuing without it")

    for k in frames:
        R, C = fi.R[k], fi.C[k]
        d2 = (xyz[:, 0] - C[0]) ** 2 + (xyz[:, 1] - C[1]) ** 2
        sel = np.flatnonzero(d2 <= opt.r_max**2)
        is_nt_any = nt_frame == k
        if len(sel) == 0 and not is_nt_any.any():
            continue
        u, v, r, el = geometry.world_to_pano(xyz[sel], R, C)
        if opt.mirror:
            u = np.mod(-u, 8000.0)
        inr = zbuffer.range_filter(r, R_MIN, opt.r_max)
        sel, u, v, r, el = sel[inr], u[inr], v[inr], r[inr], el[inr]
        if len(sel) == 0:
            continue
        if opt.occlusion:
            fp = products.FrameProducts.load(k, opt.rig)
            vis = fp.visible(r, u, v)
        else:
            vis = zbuffer.range_filter(r, R_MIN, opt.r_max)
        if vmask is not None:
            on_vehicle = vmask(u, v)
            n_veh += int(on_vehicle.sum())
            # a sample on the vehicle body is unusable in every variant: drop it before sampling
            keep = ~on_vehicle
            sel, u, v, r, el, vis = sel[keep], u[keep], v[keep], r[keep], el[keep], vis[keep]
            if len(sel) == 0:
                continue

        ps = PanoSampler(load_pano_rgb(fi.poses.path(k)), footprint=(opt.sampling == "footprint"))
        # ---- product: visible, unsaturated, footprint-sampled, scored
        if opt.sampling == "nearest":
            rgb_p = ps.nearest(u, v)
        elif opt.sampling == "bilinear":
            rgb_p = ps.bilinear(u, v)
        else:
            rgb_p = ps.footprint(u, v)
        sat = ps.saturated(rgb_p)
        n_sat += int(sat.sum())
        good = vis & ~sat
        acc_top.update(sel[good], srgb_to_linear(rgb_p[good]), _score(r[good]), k)
        # ---- nearest-in-time variants (nearest pixel, like the pilot)
        is_nt = is_nt_any[sel]
        if is_nt.any():
            rgb_n = ps.nearest(u[is_nt], v[is_nt]) if opt.nt_sampling == "nearest" else rgb_p[is_nt]
            grad = ps.gradient(u[is_nt], v[is_nt])
            rows_nt = sel[is_nt]
            acc_nt_noocc.update(rows_nt, rgb_n, k, r[is_nt], grad)
            el_nt[rows_nt] = el[is_nt]
            az_nt[rows_nt] = np.mod(u[is_nt] / 8000.0 * 360.0, 360.0)
            v_nt = vis[is_nt]
            acc_nt.update(rows_nt[v_nt], rgb_n[v_nt], k, r[is_nt][v_nt], grad[v_nt])

    store.drop_cache((Path(products.FRAMES_DIR),))  # keep free memory high for the memory watchdog
    fused = acc_top.finalize()
    variants = {
        "med": (fused["rgb"], fused["n_views"] > 0),
        "nt": (acc_nt.rgb, acc_nt.filled),
        "nt_noocc": (acc_nt_noocc.rgb, acc_nt_noocc.filled),
    }
    de76 = {}
    de00 = {}
    hists = {}
    strata = metrics.strata_indices(acc_nt_noocc.cam_dist, acc_nt_noocc.img_grad, el_nt, az_nt, fused["n_views"], cls, acc_nt_noocc.frame, psid)
    for name, (rgb, valid) in variants.items():
        d76, d00 = metrics.delta_e(rgb, ref_rgb)
        de76[name], de00[name] = d76, d00
        h = metrics.StrataHist()
        h.add(d00, valid, **strata)
        # CIE76 total only, for comparison with the pilot
        h76 = metrics.StrataHist()
        h76.add(d76, valid, **{"class": cls})
        hists[name] = h
        hists[name + "_76"] = h76

    coverage = {k: float(v[1].mean()) for k, v in variants.items()}
    coverage["saturated_samples"] = n_sat
    coverage["vehicle_samples"] = n_veh

    if opt.write_las:
        extras = {
            "ref_r": ref_rgb[:, 0], "ref_g": ref_rgb[:, 1], "ref_b": ref_rgb[:, 2],
            "nt_r": acc_nt.rgb[:, 0], "nt_g": acc_nt.rgb[:, 1], "nt_b": acc_nt.rgb[:, 2],
            "dE00_med": np.where(variants["med"][1], de00["med"], np.nan).astype(np.float32),
            "dE00_nt": np.where(variants["nt"][1], de00["nt"], np.nan).astype(np.float32),
            "dE00_nt_noocc": np.where(variants["nt_noocc"][1], de00["nt_noocc"], np.nan).astype(np.float32),
            "src_image": fused["src_image"], "n_views": fused["n_views"], "col_conf": fused["col_conf"],
            "cam_dist": acc_nt_noocc.cam_dist,
            "inc_angle": np.full(n, 255, np.uint8),
            "img_grad": acc_nt_noocc.img_grad,
        }
        if opt.subsample is None:
            prov = {"rig": {"boresight_deg": list(opt.rig.boresight_deg), "lever_arm_m": list(opt.rig.lever_arm_m), "dt_s": opt.rig.dt_s, "hash": opt.rig.hash()},
                    "r_max": opt.r_max, "top_k": opt.top_k, "occlusion": opt.occlusion, "sampling": opt.sampling, "git": products.git_rev(),
                    "rgb_scaling": "8bit*256", "frames": [int(f) for f in frames]}
            las_out.write_tile(td, Path(opt.out_dir) / opt.tag / "tiles" / f"ID3432_000{tile.name}_colored.laz", fused["rgb"], extras, prov)

    # stratified random subsample for ad-hoc plots
    rng = np.random.default_rng(int(tile.name))
    take = rng.choice(n, size=min(n, 60_000), replace=False)
    sample = {
        "tile": np.full(len(take), int(tile.name), np.int16),
        "ref_rgb": ref_rgb[take], "med_rgb": fused["rgb"][take], "nt_rgb": acc_nt_noocc.rgb[take],
        "n_views": fused["n_views"][take], "cam_dist": acc_nt_noocc.cam_dist[take], "img_grad": acc_nt_noocc.img_grad[take],
        "el": el_nt[take], "az": az_nt[take], "cls": cls[take], "frame": acc_nt_noocc.frame[take], "psid": psid[take],
        "gps": gps[take].astype(np.float64), "xyz": xyz[take].astype(np.float32),
        **{f"de00_{k}": de00[k][take] for k in variants}, **{f"de76_{k}": de76[k][take] for k in variants},
        **{f"valid_{k}": variants[k][1][take] for k in variants},
    }
    res = TileResult(name=tile.name, n=n, seconds=time.time() - t0, n_frames=len(frames), coverage=coverage, hist=hists, sample=sample)
    log(f"tile {tile.name}: {n} pts, {len(frames)} frames, {res.seconds:.0f} s, coverage {json.dumps({k: round(v, 3) for k, v in coverage.items()})}")
    return res


# ----------------------------------------------------------------------------- multiprocessing driver
_G: dict = {}


def _init(opt: Options):
    _G["store"] = CloudStore()
    _G["poses"] = load_poses()
    _G["fi"] = FrameIndex(_G["poses"], opt.rig)
    _G["opt"] = opt


def _run_tile(name: str) -> TileResult:
    store = _G["store"]
    res = colorize_tile(store.by_name[name], store, _G["fi"], _G["opt"])
    # persist per-tile results so the parent does not hold them
    out = Path(_G["opt"].out_dir) / _G["opt"].tag / "stats"
    out.mkdir(parents=True, exist_ok=True)
    for k, h in res.hist.items():
        h.save(out / f"{name}_{k}.npz")
    np.savez_compressed(out / f"{name}_sample.npz", **res.sample)
    (out / f"{name}_meta.json").write_text(json.dumps({"n": res.n, "seconds": res.seconds, "n_frames": res.n_frames, "coverage": res.coverage}))
    res.hist = {}
    res.sample = None
    return res


def run(tiles: list[str] | None, opt: Options, workers: int = 8) -> list[TileResult]:
    from multiprocessing import Pool

    store = CloudStore()
    names = [t.name for t in store.tiles] if tiles is None else tiles
    names = sorted(names, key=lambda nm: -store.by_name[nm].n)  # largest first
    (Path(opt.out_dir) / opt.tag).mkdir(parents=True, exist_ok=True)
    # resume: tiles with finished stats are skipped
    done = {p.name[:3] for p in (Path(opt.out_dir) / opt.tag / "stats").glob("*_meta.json")}
    if done:
        print(f"resuming: {len(done)} tiles already done")
        names = [nm for nm in names if nm not in done]
    if workers <= 1:
        _init(opt)
        return [_run_tile(nm) for nm in names]
    with Pool(workers, initializer=_init, initargs=(opt,)) as pool:
        return list(pool.imap_unordered(_run_tile, names))
