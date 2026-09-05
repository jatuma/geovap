"""Per-frame quality manifest: which panoramas are well aligned with the cloud and safe to use.

For every frame:
  geometry   - silhouette / depth-edge points of the cloud (this pass only) vs nearest photo edge:
               median du, dv (full-res px), MAD, inlier fraction within 8 px
  pass_conflict - the same residual for the neighbouring pass's points scanned within the time
               window; a large disagreement between the two means the passes are not co-registered
               there (double surfaces) and multi-pass fusion is unsafe
  motion     - speed and yaw rate from the trajectory (slow turnarounds are where poses/passes fail)
  sharpness  - Laplacian variance of the photo (motion blur)
  de_nt      - median colour ΔE00 vs TerraScan from a colorization run (informational only:
               it measures which image TerraScan used, not alignment)

Classes: clean (all geometry + motion criteria), usable (geometry fine, slow/turning or minor
pass conflict), reject (geometry not verifiable or bad).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import geometry, zbuffer
from .cloud_store import CloudStore
from .config import OUT_DIR, PANO_H, PANO_W, R_MAX, R_MIN
from .frame_select import FrameIndex
from .poses import Poses, load_poses
from .products import TIME_WINDOW_S, gather_candidates
from .sample import load_pano_rgb

FINE_W, FINE_H = 4000, 2000

# thresholds
# Geometry thresholds. The per-frame median residual of well-aligned frames scatters with MAD ~11 px
# (edge detection on vegetation, lidar last-return vs photo silhouette); frames beyond 4 px show the same
# colour agreement as those within, so the tolerance is set at 6 px = 0.27 deg = 5 cm at 10 m.
GEO_MAX_MEDIAN_PX = 6.0  # |median du|, |median dv| (full-res px; 1 px = 0.045 deg)
GEO_MAX_MAD_PX = 16.0
GEO_MIN_INLIER = 0.08  # fraction of edge points within 8 CHAM px (16 full-res px) of a photo edge (vegetation-heavy scenes sit at 0.1-0.2)
GEO_MIN_POINTS = 300
CONFLICT_PX = 8.0  # neighbour-pass residual median beyond this = passes disagree
MIN_SPEED = 3.0  # m/s
MAX_YAW_RATE = 5.0  # deg/s
MIN_SHARPNESS = 40.0  # Laplacian variance on the 2000x1000 grey (empirical; blurred frames < 40)


@dataclass
class FrameQuality:
    frame: int
    filename: str
    pass_id: int
    t: float
    speed: float
    yaw_rate: float
    sharpness: float
    n_edge: int
    du: float
    dv: float
    du_mad: float
    dv_mad: float
    inlier8: float
    n_edge_other: int
    du_other: float
    dv_other: float
    de_nt: float
    geometry_ok: bool
    pass_conflict: bool
    motion_ok: bool
    sharp_ok: bool
    cls: str
    reasons: str


def yaw_rates(poses: Poses) -> np.ndarray:
    """|dyaw/dt| in deg/s, central difference inside a pass, one-sided at pass ends."""
    dy = np.full(len(poses), np.nan)
    n = len(poses)
    for k in range(n):
        a = k - 1 if k - 1 >= 0 and poses.pass_id[k - 1] == poses.pass_id[k] else k
        b = k + 1 if k + 1 < n and poses.pass_id[k + 1] == poses.pass_id[k] else k
        if a == b:
            continue
        dy[k] = abs(((poses.yaw[b] - poses.yaw[a] + 180) % 360 - 180) / (poses.t[b] - poses.t[a]))
    return dy


def _silhouette_points(xyz, R, C):
    """Exact-cell silhouette / depth-edge points from a fine z-buffer. Returns xyz subset."""
    u, v, r, el = geometry.world_to_pano(xyz, R, C)
    keep = zbuffer.range_filter(r, R_MIN, R_MAX)
    if keep.sum() < 1000:
        return xyz[:0]
    s = FINE_W / PANO_W
    x, y = u[keep] * s, v[keep] * s
    import mapping.zbuffer as zb

    old = zb.SPLAT_MAX_PX
    zb.SPLAT_MAX_PX = 3
    try:
        depth, _ = zbuffer.splat(x, y, r[keep], np.arange(keep.sum(), dtype=np.uint32), FINE_W, FINE_H)
    finally:
        zb.SPLAT_MAX_PX = old
    d = zbuffer.close_depth(depth)
    fin = np.isfinite(d)
    el_rows = 90.0 - (np.arange(FINE_H) + 0.5) / FINE_H * 180.0
    edge = np.zeros_like(fin)
    for dy, dx in ((0, 2), (0, -2), (2, 0), (-2, 0)):
        dn = np.roll(d, (-dy, -dx), axis=(0, 1))
        fn = np.isfinite(dn)
        edge |= fin & fn & (dn - d > np.maximum(0.5, 0.15 * d))
        edge |= fin & ~fn & (el_rows > 0)[:, None]
    edge &= (el_rows >= -40)[:, None]
    cx = np.mod(np.floor(x).astype(np.int64), FINE_W)
    cy = np.clip(np.floor(y).astype(np.int64), 0, FINE_H - 1)
    near = r[keep] <= d[cy, cx] + np.maximum(0.15, 0.03 * r[keep])
    sel = edge[cy, cx] & near
    return xyz[keep][sel]


def _residual(xyz_edge, R, C, dt_img, edge_idx, valid, window=20.0, n_max=20000):
    """Median du, dv (full-res px), MADs, inlier fraction within 8 CHAM px."""
    if len(xyz_edge) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    if len(xyz_edge) > n_max:
        xyz_edge = xyz_edge[:: len(xyz_edge) // n_max]
    s = FINE_W / PANO_W
    u, v, r, el = geometry.world_to_pano(xyz_edge, R, C, dtype=np.float64)
    x, y = u * s, v * s
    xi = np.mod(np.floor(x).astype(np.int64), FINE_W)
    yi = np.clip(np.floor(y).astype(np.int64), 0, FINE_H - 1)
    d = dt_img[yi, xi].astype(np.float32)
    ok = valid[yi, xi] & (d <= window)
    n = int(ok.sum())
    if n < 50:
        return n, np.nan, np.nan, np.nan, np.nan, np.nan
    ex = edge_idx[1, yi[ok], xi[ok]].astype(np.float64) + 0.5
    ey = edge_idx[0, yi[ok], xi[ok]].astype(np.float64) + 0.5
    du = ((ex - x[ok] + FINE_W / 2) % FINE_W - FINE_W / 2) / s
    dv = (ey - y[ok]) / s
    inl = float((d[ok] <= 8.0).mean() * ok.mean())  # fraction of all edge points within 8 CHAM px
    return n, float(np.median(du)), float(np.median(dv)), float(np.median(np.abs(du - np.median(du)))), float(np.median(np.abs(dv - np.median(dv)))), inl


def _photo_edges(poses: Poses, k: int, vmask):
    from scipy import ndimage

    from .calib import chamfer as Ch
    from .calib.icp import _photo_edges as pe

    edges, valid = pe(poses, k, vmask)
    pad = 64
    e = np.concatenate([edges[:, -pad:], edges, edges[:, :pad]], axis=1)
    dt, (ir, ic) = ndimage.distance_transform_edt(~e, return_indices=True)
    return dt[:, pad:-pad].astype(np.float16), np.stack([ir[:, pad:-pad], (ic[:, pad:-pad] - pad) % Ch.CHAM_W]).astype(np.int16), valid


def sharpness(poses: Poses, k: int) -> float:
    g = cv2.cvtColor(load_pano_rgb(poses.path(k)), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (2000, 1000), interpolation=cv2.INTER_AREA)
    band = g[int(1000 * 60 / 180) : int(1000 * 125 / 180)]
    return float(cv2.Laplacian(band, cv2.CV_32F).var())


_G: dict = {}


def _init():
    from .vehicle_mask import MASK_PATH, VehicleMask

    _G["store"] = CloudStore()
    _G["poses"] = load_poses()
    _G["fi"] = FrameIndex(_G["poses"])
    _G["vm"] = VehicleMask() if MASK_PATH.exists() else None
    _G["yr"] = yaw_rates(_G["poses"])
    p = _G["poses"]
    _G["pass_t0"] = np.array([p.t[np.flatnonzero(p.pass_id == q)].min() for q in range(p.pass_id.max() + 1)])
    _G["pass_t1"] = np.array([p.t[np.flatnonzero(p.pass_id == q)].max() for q in range(p.pass_id.max() + 1)])


def assess_frame(k: int, de_nt: float = np.nan) -> FrameQuality:
    store, poses, fi, vm = _G["store"], _G["poses"], _G["fi"], _G["vm"]
    R, C = fi.R[k], fi.C[k]
    xyz, pid = gather_candidates(store, C, R_MAX)
    # gps time per candidate (re-gather; cheap relative to the rest)
    parts = store.query_disc(float(C[0]), float(C[1]), R_MAX)
    gps = np.concatenate([np.asarray(store.tile(t.name).gps_time[rows]) for t, rows in parts])
    win = np.abs(gps - poses.t[k]) <= TIME_WINDOW_S
    p = int(poses.pass_id[k])
    own = win & (gps >= _G["pass_t0"][p] - 2) & (gps <= _G["pass_t1"][p] + 2)
    other = win & ~own
    dt_img, edge_idx, valid = _photo_edges(poses, k, vm)
    e_own = _silhouette_points(xyz[own], R, C) if own.sum() > 1000 else xyz[:0]
    n, du, dv, dum, dvm, inl = _residual(e_own, R, C, dt_img, edge_idx, valid)
    e_oth = _silhouette_points(xyz[other], R, C) if other.sum() > 20000 else xyz[:0]
    n2, du2, dv2, *_ = _residual(e_oth, R, C, dt_img, edge_idx, valid)
    sp = float(poses.speed[k])
    yr = float(_G["yr"][k]) if np.isfinite(_G["yr"][k]) else 0.0
    sh = sharpness(poses, k)
    geometry_ok = bool(n >= GEO_MIN_POINTS and np.isfinite(du) and abs(du) <= GEO_MAX_MEDIAN_PX and abs(dv) <= GEO_MAX_MEDIAN_PX and dum <= GEO_MAX_MAD_PX and dvm <= GEO_MAX_MAD_PX and inl >= GEO_MIN_INLIER)
    conflict = bool(n2 >= GEO_MIN_POINTS and np.isfinite(du2) and (abs(du2) > CONFLICT_PX or abs(dv2) > CONFLICT_PX))
    motion_ok = bool(sp >= MIN_SPEED and yr <= MAX_YAW_RATE)
    sharp_ok = bool(sh >= MIN_SHARPNESS)
    return _classify(k, str(poses.filename[k]), p, float(poses.t[k]), sp, yr, sh, n, du, dv, dum, dvm, inl, n2, du2, dv2, de_nt)


def _classify(k, filename, p, t, sp, yr, sh, n, du, dv, dum, dvm, inl, n2, du2, dv2, de_nt) -> FrameQuality:
    geometry_ok = bool(n >= GEO_MIN_POINTS and np.isfinite(du) and abs(du) <= GEO_MAX_MEDIAN_PX and abs(dv) <= GEO_MAX_MEDIAN_PX and dum <= GEO_MAX_MAD_PX and dvm <= GEO_MAX_MAD_PX and inl >= GEO_MIN_INLIER)
    conflict = bool(n2 >= GEO_MIN_POINTS and np.isfinite(du2) and (abs(du2) > CONFLICT_PX or abs(dv2) > CONFLICT_PX))
    motion_ok = bool(sp >= MIN_SPEED and yr <= MAX_YAW_RATE)
    sharp_ok = bool(sh >= MIN_SHARPNESS)
    few = n < GEO_MIN_POINTS
    reasons = []
    if few:
        reasons.append("few_edges")
    elif not geometry_ok:
        reasons.append("geometry")
    if conflict:
        reasons.append("pass_conflict")
    if sp < MIN_SPEED:
        reasons.append("slow")
    if yr > MAX_YAW_RATE:
        reasons.append("turning")
    if not sharp_ok:
        reasons.append("blur")
    if geometry_ok and motion_ok and sharp_ok and not conflict:
        cls = "clean"
    elif few and motion_ok and sharp_ok and not conflict:
        cls = "unverified"  # open field / no structure to check against; motion is benign
    elif (geometry_ok or few) and sharp_ok:
        cls = "usable"
    else:
        cls = "reject"
    rnd = lambda x, d=2: round(float(x), d) if np.isfinite(x) else np.nan
    return FrameQuality(int(k), filename, int(p), float(t), rnd(sp), rnd(yr), rnd(sh, 1), int(n), rnd(du), rnd(dv), rnd(dum), rnd(dvm), rnd(inl, 3), int(n2), rnd(du2), rnd(dv2), float(de_nt) if de_nt is not None else np.nan, geometry_ok, conflict, motion_ok, sharp_ok, cls, "+".join(reasons))


def write_outputs(res: list[FrameQuality], out_dir: Path) -> None:
    with open(out_dir / "frame_quality.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(res[0]).keys()))
        w.writeheader()
        for r in res:
            w.writerow(asdict(r))
    groups = {c: [r.frame for r in res if r.cls == c] for c in ("clean", "unverified", "usable", "reject")}
    (out_dir / "clean_frames.json").write_text(json.dumps({**groups, "n_total": len(res), "criteria": {"geo_max_median_px": GEO_MAX_MEDIAN_PX, "geo_max_mad_px": GEO_MAX_MAD_PX, "geo_min_inlier": GEO_MIN_INLIER, "geo_min_points": GEO_MIN_POINTS, "conflict_px": CONFLICT_PX, "min_speed": MIN_SPEED, "max_yaw_rate": MAX_YAW_RATE, "min_sharpness": MIN_SHARPNESS}}, indent=1))


def reclassify(out_dir: Path = OUT_DIR / "dataset") -> list[FrameQuality]:
    """Re-derive classes from the stored measurements with the current thresholds (no recompute)."""
    res = []
    for line in (out_dir / "frame_quality.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        g = lambda key: float(d[key]) if d[key] is not None else np.nan
        res.append(_classify(d["frame"], d["filename"], d["pass_id"], d["t"], g("speed"), g("yaw_rate"), g("sharpness"), int(d["n_edge"]), g("du"), g("dv"), g("du_mad"), g("dv_mad"), g("inlier8"), int(d["n_edge_other"]), g("du_other"), g("dv_other"), g("de_nt")))
    res.sort(key=lambda r: r.frame)
    write_outputs(res, out_dir)
    return res


def _job(args):
    k, de = args
    _G["n"] = _G.get("n", 0) + 1
    if _G["n"] % 60 == 0:  # keep the page cache small (see CloudStore.drop_cache)
        from .config import FRAMES_DIR

        _G["store"].drop_cache((FRAMES_DIR,))
    try:
        r = assess_frame(k, de)
        _G["store"].release()  # keep this worker's RSS small (memmap pages)
        return r
    except Exception as e:  # keep the manifest complete
        p = _G["poses"]
        return FrameQuality(k, str(p.filename[k]), int(p.pass_id[k]), float(p.t[k]), float(p.speed[k]), 0.0, 0.0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, de, False, False, False, False, "reject", f"error:{type(e).__name__}")


def run(workers: int = 10, run_tag: str | None = "tw45", out_dir: Path = OUT_DIR / "dataset", limit: int | None = None) -> list[FrameQuality]:
    from multiprocessing import Pool

    from tqdm import tqdm

    poses = load_poses()
    de = np.full(len(poses), np.nan)
    if run_tag:
        try:
            from . import metrics
            from .report import load_run

            _, hists, _ = load_run(run_tag)
            h = hists["nt"][0].h["frame"]
            for i in range(min(len(poses), h.shape[0])):
                if h[i].sum() >= 500:
                    de[i] = metrics.hist_stats(h[i])["median"]
        except Exception:
            pass
    out_dir.mkdir(parents=True, exist_ok=True)
    # resumable: results are appended to a JSONL as they arrive
    jl = out_dir / "frame_quality.jsonl"
    done: dict[int, FrameQuality] = {}
    if jl.exists():
        for line in jl.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["frame"]] = FrameQuality(**d)
    jobs = [(k, float(de[k])) for k in range(len(poses)) if k not in done]
    if limit is not None:
        jobs = jobs[:limit]
    if jobs:
        CloudStore().drop_cache((OUT_DIR.parent / "frames",))
        with Pool(workers, initializer=_init) as pool, open(jl, "a") as f:
            for r in tqdm(pool.imap_unordered(_job, jobs, chunksize=2), total=len(jobs), desc="frames"):
                done[r.frame] = r
                f.write(json.dumps(asdict(r), default=float) + "\n")
                f.flush()
    res = [done[k] for k in sorted(done)]
    write_outputs(res, out_dir)
    return res


if __name__ == "__main__":
    import sys

    run(workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10, limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
