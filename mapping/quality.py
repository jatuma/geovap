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

from dataclasses import replace
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from geovap.domain.math import edges as edges_math
from geovap.domain.model import geometry
from geovap.domain.math import depth as zbuffer
from .cloud_store import open_store
from .config import OUT_DIR, PANO_H, PANO_W, REPO_ROOT, R_MAX, R_MIN, SENSOR, source_dir
from .frame_select import FrameIndex
from .poses import Poses, load_poses
from .products import TIME_WINDOW_S, frames_dir, gather_candidates
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
    """`domain.math.edges.silhouette_points` at this module's fine z-buffer resolution."""
    return edges_math.silhouette_points(xyz, R, C, SENSOR, FINE_W, FINE_H)


def _residual(xyz_edge, R, C, dt_img, edge_idx, valid, window=20.0, n_max=20000):
    """`domain.math.edges.residual`; inlier threshold is this module's CONFLICT_PX convention."""
    return edges_math.residual(xyz_edge, R, C, dt_img, edge_idx, valid, SENSOR, FINE_W, FINE_H,
                               window=window, n_max=n_max, inlier_px=CONFLICT_PX)


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


def _init(poses_source=None):
    # explicit initarg (not just env inheritance via fork) so a poses_source given at call time is
    # honoured even if the pool start method is not "fork".
    from .vehicle_mask import MASK_PATH, VehicleMask

    _G["poses"] = load_poses(poses_source)
    _G["store"] = open_store(_G["poses"])  # registered cloud when poses carries a "registration"
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


def reclassify(out_dir: Path | None = None, poses_source: str | None = None) -> list[FrameQuality]:
    """Re-derive classes from the stored measurements with the current thresholds (no recompute).
    `out_dir` defaults to `source_dir(OUT_DIR / "dataset", load_poses(poses_source))` -- byte-identical
    to today for the default "export" source, a hash-suffixed sibling for a corrected one."""
    if out_dir is None:
        out_dir = source_dir(OUT_DIR / "dataset", load_poses(poses_source))
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
        _G["store"].drop_cache((frames_dir(_G["poses"]),))
    try:
        r = assess_frame(k, de)
        _G["store"].release()  # keep this worker's RSS small (memmap pages)
        return r
    except Exception as e:  # keep the manifest complete
        p = _G["poses"]
        return FrameQuality(k, str(p.filename[k]), int(p.pass_id[k]), float(p.t[k]), float(p.speed[k]), 0.0, 0.0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan, de, False, False, False, False, "reject", f"error:{type(e).__name__}")


def run(workers: int = 10, run_tag: str | None = "tw45", out_dir: Path | None = None, limit: int | None = None, poses_source: str | None = None) -> list[FrameQuality]:
    from multiprocessing import Pool

    from tqdm import tqdm

    poses = load_poses(poses_source)
    if out_dir is None:
        out_dir = source_dir(OUT_DIR / "dataset", poses)
    de = np.full(len(poses), np.nan)
    if run_tag:
        try:
            from geovap.domain.math import colour_metrics as metrics
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
        open_store(poses).drop_cache((frames_dir(poses),))
        with Pool(workers, initializer=_init, initargs=(poses_source,)) as pool, open(jl, "a") as f:
            for r in tqdm(pool.imap_unordered(_job, jobs, chunksize=2), total=len(jobs), desc="frames"):
                done[r.frame] = r
                f.write(json.dumps(asdict(r), default=float) + "\n")
                f.flush()
    res = [done[k] for k in sorted(done)]
    write_outputs(res, out_dir)
    return res


_TILE_NEAR_M = 20.0  # frame-to-tile-bbox distance cutoff for tile_summary.json (matches gen_quality_outputs.py)
_CLASS_ORDER = ["reject", "usable", "unverified", "clean"]  # z-order: clean drawn last (on top)
_CLASS_COLORS = {"reject": "#c0392b", "usable": "#f1a70a", "unverified": "#8fbf9f", "clean": "#2e7d4f"}


def _read_quality_csv(csv_path: Path) -> dict[str, np.ndarray]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    def col(key, cast):
        return np.array([cast(r[key]) if r[key] not in ("", "nan", "NaN") else np.nan for r in rows])
    d = {
        "frame": np.array([int(r["frame"]) for r in rows]),
        "cls": np.array([r["cls"] for r in rows]),
        "du": col("du", float), "dv": col("dv", float), "de_nt": col("de_nt", float),
        "speed": col("speed", float), "n_edge": np.array([int(r["n_edge"]) for r in rows]),
        "inlier8": col("inlier8", float),
    }
    order = np.argsort(d["frame"])
    return {k: v[order] for k, v in d.items()}


def _bbox_dist(px: np.ndarray, py: np.ndarray, xmin, xmax, ymin, ymax) -> np.ndarray:
    dx = np.maximum(np.maximum(xmin - px, 0.0), px - xmax)
    dy = np.maximum(np.maximum(ymin - py, 0.0), py - ymax)
    return np.sqrt(dx * dx + dy * dy)


def write_tile_summary(csv_path: Path, poses_source: str | None, out_path: Path) -> dict:
    """Per-tile frame-quality counts (LAZ tile bboxes, frames within `_TILE_NEAR_M`), written as
    tile_summary.json: {"tiles": [[name, n_near, n_clean, n_unverified, n_usable, n_reject], ...]}.
    Reconstructed from an ad hoc script (no such generator existed before); matches
    dataset/export_baseline/tile_summary.json byte-for-byte on the export csv."""
    from .cloud_store import _load_layout

    d = _read_quality_csv(csv_path)
    poses = load_poses(poses_source)
    if not np.array_equal(d["frame"], np.arange(len(poses))):
        raise ValueError(f"{csv_path}: frame column must be 0..N-1 matching poses order")
    E, N = poses.origin[:, 0], poses.origin[:, 1]
    layout = sorted(_load_layout(), key=lambda x: x[0])
    tiles = []
    for name, ring in layout:
        xmin, ymin = ring.min(axis=0)
        xmax, ymax = ring.max(axis=0)
        near = _bbox_dist(E, N, xmin, xmax, ymin, ymax) <= _TILE_NEAR_M
        counts = [int(np.sum(near & (d["cls"] == c))) for c in ("clean", "unverified", "usable", "reject")]
        tiles.append([name, int(near.sum())] + counts)
    out = {"tiles": tiles}
    out_path.write_text(json.dumps(out))
    return out


def plot_quality(csv_path: Path, poses_source: str | None, map_path: Path, stats_path: Path) -> None:
    """frame_quality_map.png (trajectory coloured by class over the tile layout) and
    frame_quality_stats.png (residual histogram + de_nt/speed + inlier8/n_edge scatter),
    matching the style of dataset/export_baseline/*.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .cloud_store import _load_layout

    d = _read_quality_csv(csv_path)
    poses = load_poses(poses_source)
    E, N = poses.origin[:, 0], poses.origin[:, 1]
    layout = _load_layout()

    fig, ax = plt.subplots(figsize=(14, 11))
    for name, ring in layout:
        ax.plot(ring[:, 0], ring[:, 1], color="0.6", lw=0.7, zorder=1)
        c = ring.mean(axis=0)
        ax.text(c[0], c[1], name, fontsize=7, color="0.35", ha="center", va="center", zorder=1)
    for c in _CLASS_ORDER:
        m = d["cls"] == c
        n = int(m.sum())
        ax.scatter(E[m], N[m], s=6, color=_CLASS_COLORS[c], label=f"{c} ({n})", zorder=2, alpha=0.85, edgecolors="none")
    ax.set_xlabel("E [m]")
    ax.set_ylabel("N [m]")
    ax.set_title("Dražkov — kvalita snímků podél trajektorie")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(map_path, dpi=110)
    plt.close(fig)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(17, 4.5))
    bins = np.linspace(-13, 13, 80)
    ax0.hist(d["du"][np.isfinite(d["du"])], bins=bins, color=_CLASS_COLORS["clean"], alpha=0.7, label="du")
    ax0.hist(d["dv"][np.isfinite(d["dv"])], bins=bins, color="#d99a3e", alpha=0.7, label="dv")
    ax0.axvline(-6, color="k", ls=":", lw=1)
    ax0.axvline(6, color="k", ls=":", lw=1)
    ax0.set_xlabel("medián rezidua siluet [px, 0,045°/px]")
    ax0.legend()

    for c in _CLASS_ORDER:
        m = d["cls"] == c
        ax1.scatter(d["speed"][m], d["de_nt"][m], s=8, color=_CLASS_COLORS[c], alpha=0.6, edgecolors="none")
    ax1.set_xlabel("rychlost [m/s]")
    ax1.set_ylabel("ΔE00 proti TerraScanu (nt)")

    for c in _CLASS_ORDER:
        m = d["cls"] == c
        ax2.scatter(d["n_edge"][m], d["inlier8"][m], s=8, color=_CLASS_COLORS[c], alpha=0.6, edgecolors="none")
    ax2.set_xscale("log")
    ax2.set_xlabel("počet hranových bodů")
    ax2.set_ylabel("podíl do 16 px od hrany fotky")

    fig.tight_layout()
    fig.savefig(stats_path, dpi=110)
    plt.close(fig)


PROMOTE_FILES = ["clean_frames.json", "frame_quality.csv", "tile_summary.json", "frame_quality_map.png", "frame_quality_stats.png"]


def promote(poses_source: str | None, backup_dir: Path | None = None) -> None:
    """Copy the quality outputs for `poses_source` (from `source_dir(OUT_DIR/"dataset", poses)`)
    into the repo's `dataset/` directory, after backing up whatever is there now.

    `clean_frames.json`, `frame_quality.csv` are required; `tile_summary.json` and the two PNGs are
    optional (warn, don't fail, if missing -- e.g. `tile-summary` was not run yet). Prints an added/
    removed summary of the clean-frame set vs. the previous `dataset/clean_frames.json`."""
    import shutil

    poses = load_poses(poses_source)
    source = source_dir(OUT_DIR / "dataset", poses)
    dest = REPO_ROOT / "dataset"
    if backup_dir is None:
        backup_dir = OUT_DIR / "pipeline" / "backup" / f"dataset_{poses.hash()[:6]}"
    backup_dir = Path(backup_dir)

    prev_clean: set[int] | None = None
    prev_path = dest / "clean_frames.json"
    if prev_path.exists():
        try:
            prev_clean = set(json.loads(prev_path.read_text())["clean"])
        except Exception:
            prev_clean = None

    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in PROMOTE_FILES:
        p = dest / name
        if p.exists():
            shutil.copy2(p, backup_dir / name)

    dest.mkdir(parents=True, exist_ok=True)
    required = {"clean_frames.json", "frame_quality.csv"}
    for name in PROMOTE_FILES:
        src = source / name
        if not src.exists():
            if name in required:
                raise FileNotFoundError(f"missing required promote source: {src}")
            print(f"warning: {src} missing, skipping {name}")
            continue
        shutil.copy2(src, dest / name)
    print(f"promoted {source} -> {dest} (backup: {backup_dir})")

    new_clean_path = dest / "clean_frames.json"
    new_clean = set(json.loads(new_clean_path.read_text())["clean"])
    if prev_clean is None:
        print(f"clean frames: {len(new_clean)} (no previous set to diff against)")
    else:
        added = new_clean - prev_clean
        removed = prev_clean - new_clean
        print(f"clean frames: {len(prev_clean)} -> {len(new_clean)} (+{len(added)} added, -{len(removed)} removed)")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="assess every frame, write frame_quality.{csv,jsonl}/clean_frames.json")
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--poses", default=None, help='pose table: "export" (default), "corrected", or a CSV path')

    t = sub.add_parser("tile-summary", help="write tile_summary.json + frame_quality_{map,stats}.png from an existing frame_quality.csv")
    t.add_argument("csv", type=Path, help="frame_quality.csv to read")
    t.add_argument("out_dir", type=Path, help="directory to write tile_summary.json + PNGs into")
    t.add_argument("--poses", default=None, help='pose table the csv was built against: "export" (default), "corrected", or a CSV path')

    rc = sub.add_parser("reclassify", help="re-derive classes from stored frame_quality.jsonl with current thresholds (no recompute)")
    rc.add_argument("--poses", default=None, help='pose table: "export" (default), "corrected", or a CSV path')

    pr = sub.add_parser("promote", help="copy quality outputs for --poses into repo dataset/ (with backup)")
    pr.add_argument("--poses", required=True, help='pose table: "export", "corrected", or a CSV path')
    pr.add_argument("--backup-dir", type=Path, default=None, help="where to back up the current dataset/ files (default OUT_DIR/pipeline/backup/dataset_<poses_hash6>)")

    args = ap.parse_args()
    if args.cmd == "run":
        run(workers=args.workers, limit=args.limit, poses_source=args.poses)
    elif args.cmd == "tile-summary":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_tile_summary(args.csv, args.poses, args.out_dir / "tile_summary.json")
        plot_quality(args.csv, args.poses, args.out_dir / "frame_quality_map.png", args.out_dir / "frame_quality_stats.png")
        print(f"wrote {args.out_dir}/tile_summary.json, frame_quality_map.png, frame_quality_stats.png")
    elif args.cmd == "reclassify":
        reclassify(poses_source=args.poses)
    elif args.cmd == "promote":
        promote(args.poses, backup_dir=args.backup_dir)


if __name__ == "__main__":
    main()
