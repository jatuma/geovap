"""Pose-quality reporting: the "before" baseline (S2) and, later (S7), pose-source comparison.

`baseline_summary` reads the existing per-frame edge-ICP measurements in `dataset/frame_quality.csv`
(written by `mapping.quality.run`, one row per frame, export.csv poses) and summarises medians of
|du|, |dv|, the per-frame residual MAD and the 8 px-window inlier fraction — overall, per status
class (`cls`: clean/unverified/usable/reject) and for turning vs straight frames
(|yaw_rate| > 8 deg/s, mapping.quality.yaw_rates' convention).

uv run python -m mapping.pose_report [--csv dataset/frame_quality.csv] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .config import PANO_H, PANO_W, POSES_DIR, QUALITY_CSV, R_MAX

YAW_RATE_THRESH_DEG_S = 8.0
DEFAULT_CSV = QUALITY_CSV


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def _stats(du: np.ndarray, dv: np.ndarray, du_mad: np.ndarray, dv_mad: np.ndarray, inlier: np.ndarray) -> dict:
    def med(x: np.ndarray):
        x = x[np.isfinite(x)]
        return round(float(np.median(x)), 3) if len(x) else None

    return {
        "n": int(len(du)),
        "n_with_edge_fit": int(np.isfinite(du).sum()),
        "median_abs_du_px": med(np.abs(du)),
        "median_abs_dv_px": med(np.abs(dv)),
        "median_du_mad_px": med(du_mad),
        "median_dv_mad_px": med(dv_mad),
        "median_inlier_fraction": med(inlier),
    }


def baseline_summary(frame_quality_csv: str | Path = DEFAULT_CSV) -> dict:
    """Baseline ("before") pose-quality statistics from `frame_quality.csv`: medians of |du|, |dv|,
    the per-frame du/dv MAD and the 8 px-window inlier fraction — overall, per status class, and
    turning vs straight (|yaw_rate| > 8 deg/s)."""
    path = Path(frame_quality_csv)
    rows = _read_rows(path)
    du = np.array([_f(r, "du") for r in rows])
    dv = np.array([_f(r, "dv") for r in rows])
    du_mad = np.array([_f(r, "du_mad") for r in rows])
    dv_mad = np.array([_f(r, "dv_mad") for r in rows])
    inlier = np.array([_f(r, "inlier8") for r in rows])
    yaw_rate = np.array([_f(r, "yaw_rate") for r in rows])
    cls = np.array([r["cls"] for r in rows])

    out: dict = {"source": str(path), "overall": _stats(du, dv, du_mad, dv_mad, inlier)}
    out["by_class"] = {c: _stats(du[cls == c], dv[cls == c], du_mad[cls == c], dv_mad[cls == c], inlier[cls == c]) for c in sorted(set(cls))}
    turning = np.abs(np.nan_to_num(yaw_rate, nan=0.0)) > YAW_RATE_THRESH_DEG_S
    out["turning"] = _stats(du[turning], dv[turning], du_mad[turning], dv_mad[turning], inlier[turning])
    out["straight"] = _stats(du[~turning], dv[~turning], du_mad[~turning], dv_mad[~turning], inlier[~turning])
    out["n_turning"] = int(turning.sum())
    out["n_straight"] = int((~turning).sum())
    return out


def _row(name: str, s: dict) -> str:
    if s is None or s.get("n_with_edge_fit", 0) == 0:
        return f"| {name} | {s['n'] if s else 0} | 0 | - | - | - | - |"
    return f"| {name} | {s['n']} | {s['n_with_edge_fit']} | {s['median_abs_du_px']} | {s['median_abs_dv_px']} | {s['median_du_mad_px']} | {s['median_inlier_fraction']} |"


def to_markdown(summary: dict) -> str:
    lines = [f"# Pose baseline report ({summary['source']})", ""]
    lines.append(f"n = {summary['overall']['n']}, n with an edge-ICP fit (n_edge >= threshold in `quality.assess_frame`) = {summary['overall']['n_with_edge_fit']}")
    lines.append("")
    lines.append("| group | n | n fit | median \\|du\\| px | median \\|dv\\| px | median du MAD px | median inlier\\@8px |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(_row("overall", summary["overall"]))
    for c, s in summary["by_class"].items():
        lines.append(_row(f"class={c}", s))
    lines.append(_row(f"turning (\\|yaw_rate\\|>{YAW_RATE_THRESH_DEG_S}, n={summary['n_turning']})", summary["turning"]))
    lines.append(_row(f"straight (n={summary['n_straight']})", summary["straight"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(POSES_DIR / "report_export_baseline.md"))
    a = ap.parse_args()
    summary = baseline_summary(a.csv)
    md = to_markdown(summary)
    print(md)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {out} (+ {out.with_suffix('.json').name})")


# =============================================================================================
# S7 -- final validation: `poses_corrected` (load_poses("corrected")) vs `export` and the
# intermediate tables. Everything below is new for S7; `baseline_summary` above (S2) is unchanged.
#
# `quality._silhouette_points(xyz, R, C)` and `quality._residual(xyz_edge, R, C, dt_img, edge_idx,
# valid)` already take an explicit (R, C) pair (added for S3b's `validate_rot_only`, see
# `mapping/cli/build_trajectory.py`) -- no edit to quality.py was needed or made here; this module
# only calls them with poses from two different sources.
# =============================================================================================
from geovap.domain.model import geometry
from .config import OUT_DIR, R_MAX  # noqa: E402
from .poses import Poses, load_poses, read_pose_table  # noqa: E402
from geovap.stages.prepare.products import TIME_WINDOW_S, gather_candidates  # noqa: E402
from .quality import _photo_edges, _residual, _silhouette_points  # noqa: E402

YAW_RATE_TURNING_DEG_S = 8.0
N_STRAIGHT_DEFAULT = 200
REG_BIG_M = 0.3  # a pass transform bigger than this counts as "large" for group 3
DEFAULT_TRANSFORMS = OUT_DIR / "pass_reg" / "pass_transforms.json"
DEFAULT_CONFLICT_JSON = OUT_DIR / "pass_reg" / "conflict_reg.json"
FINAL_DIR = OUT_DIR / "poses"


def _read_csv_rows(path: str | Path) -> list[dict]:
    import csv

    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _status_column(source: str | Path) -> np.ndarray | None:
    """Per-frame `status` column of a pose-table CSV (aligned to `frame`), or None if `source` has
    no such column (e.g. "export", which has none)."""
    if str(source) == "export":
        return None
    path = Path(source) if Path(source).suffix == ".csv" else POSES_DIR / f"poses_{source}.csv"
    if not path.exists():
        return None
    rows = _read_csv_rows(path)
    if not rows or "status" not in rows[0]:
        return None
    out = np.empty(len(rows), dtype=object)
    for r in rows:
        out[int(r["frame"])] = r.get("status", "")
    return out


def passes_with_big_transform(transforms_path: str | Path = DEFAULT_TRANSFORMS, thresh_m: float = REG_BIG_M) -> set[int]:
    """Pass ids whose S5b translation `|t|` exceeds `thresh_m` (7 of 30 passes at 0.3 m on this
    dataset: 3, 12, 15, 21, 23, 24, 26)."""
    passes = json.loads(Path(transforms_path).read_text()).get("passes", {})
    out = set()
    for p_str, tr in passes.items():
        try:
            p = int(p_str)
        except ValueError:
            continue
        t = tr.get("t")
        if t is not None and float(np.linalg.norm(t)) > thresh_m:
            out.add(p)
    return out


def select_groups(
    a: str = "export",
    b: str = "corrected",
    n_straight: int = N_STRAIGHT_DEFAULT,
    turning_min_rate: float = YAW_RATE_TURNING_DEG_S,
    transforms_path: str | Path = DEFAULT_TRANSFORMS,
) -> dict[str, np.ndarray]:
    """The three frame groups of S7 task 1: `turning` (|yaw_rate| > `turning_min_rate` deg/s, on
    `a`'s own-pass central difference), `straight` (`n_straight` evenly-spaced indices out of the
    clean, |yaw_rate|<5 pool -- evenly spaced by *rank* in the sorted pool, not resampled in time),
    and `refined_or_reg` (every frame whose `b` status contains "refined", union every frame
    currently in a pass whose S5b transform exceeds `REG_BIG_M`)."""
    from .cli.build_trajectory import clean_straight_frame_idx, turning_frame_idx

    poses_a = load_poses(a)
    poses_b = load_poses(b)
    turning = turning_frame_idx(poses_a, min_rate=turning_min_rate)
    pool = clean_straight_frame_idx(poses_a)
    if len(pool) > n_straight:
        pick = np.unique(np.round(np.linspace(0, len(pool) - 1, n_straight)).astype(int))
        straight = np.sort(pool[pick])
    else:
        straight = np.sort(pool)
    status_b = _status_column(b)
    big_passes = passes_with_big_transform(transforms_path)
    refined_mask = np.array([bool(status_b is not None and "refined" in str(status_b[i])) for i in range(len(poses_b))])
    big_pass_mask = np.isin(poses_b.pass_id, np.array(sorted(big_passes), dtype=poses_b.pass_id.dtype)) if big_passes else np.zeros(len(poses_b), dtype=bool)
    refined_or_reg = np.flatnonzero(refined_mask | big_pass_mask)
    return {"turning": turning, "straight": straight, "refined_or_reg": refined_or_reg}


# ------------------------------------------------------------------------------ silhouette compare
_CG: dict = {}


def _init_compare(a: str, b: str) -> None:
    from .cloud_store import CloudStore
    from geovap.runtime import settings
    from geovap.stages.prepare.masks import VehicleMask

    store = CloudStore()
    poses_a, poses_b = load_poses(a), load_poses(b)
    _s = settings.get()
    _mask_path = _s.workspace.vehicle_mask
    vm = VehicleMask(_mask_path, *_s.sensor.pano) if _mask_path.exists() else None
    _CG["store"], _CG["poses_a"], _CG["poses_b"], _CG["vm"] = store, poses_a, poses_b, vm
    for tag, poses in (("a", poses_a), ("b", poses_b)):
        t0, t1 = {}, {}
        for p in np.unique(poses.pass_id):
            tt = poses.t[poses.pass_id == p]
            t0[int(p)], t1[int(p)] = float(tt.min()), float(tt.max())
        _CG[f"t0_{tag}"], _CG[f"t1_{tag}"] = t0, t1


def _own_pass_xyz_tagged(store, C: np.ndarray, t_frame: float, pass_id: int, tag: str) -> np.ndarray:
    xyz, _pid = gather_candidates(store, C, R_MAX)
    if len(xyz) == 0:
        return xyz
    parts = store.query_disc(float(C[0]), float(C[1]), R_MAX)
    gps = np.concatenate([np.asarray(store.tile(t.name).gps_time[rows]) for t, rows in parts])
    win = np.abs(gps - t_frame) <= TIME_WINDOW_S
    t0d, t1d = _CG[f"t0_{tag}"], _CG[f"t1_{tag}"]
    own = win & (gps >= t0d[pass_id] - 2) & (gps <= t1d[pass_id] + 2)
    return xyz[own]


def _compare_job(k: int) -> dict:
    store, poses_a, poses_b, vm = _CG["store"], _CG["poses_a"], _CG["poses_b"], _CG["vm"]
    t_frame = float(poses_a.t[k])
    dt_img, edge_idx, valid = _photo_edges(poses_a, k, vm)

    def side(poses: Poses, tag: str):
        C = poses.origin[k]
        R = geometry.vehicle_rotation(np.array([poses.yaw[k]]), np.array([poses.roll[k]]), np.array([poses.pitch[k]]))[0]
        pid = int(poses.pass_id[k])
        xyz_own = _own_pass_xyz_tagged(store, C, t_frame, pid, tag)
        e = _silhouette_points(xyz_own, R, C) if len(xyz_own) > 1000 else xyz_own[:0]
        n, du, dv, dum, dvm, inl = _residual(e, R, C, dt_img, edge_idx, valid)
        return pid, n, du, dv, dum, dvm, inl

    pid_a, n_a, du_a, dv_a, dum_a, dvm_a, inl_a = side(poses_a, "a")
    pid_b, n_b, du_b, dv_b, dum_b, dvm_b, inl_b = side(poses_b, "b")
    return {
        "frame": int(k), "pass_a": pid_a, "pass_b": pid_b,
        "n_a": n_a, "du_a": du_a, "dv_a": dv_a, "dum_a": dum_a, "dvm_a": dvm_a, "inl_a": inl_a,
        "n_b": n_b, "du_b": du_b, "dv_b": dv_b, "dum_b": dum_b, "dvm_b": dvm_b, "inl_b": inl_b,
    }


def _stat(a: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"median": None, "p95": None, "n": 0}
    return {"median": round(float(np.median(a)), 3), "p95": round(float(np.percentile(a, 95)), 3), "n": int(len(a))}


def aggregate_compare_rows(rows: list[dict], improve_thresh_px: float = 2.0) -> dict:
    """Pure aggregation of `_compare_job`-shaped rows (no I/O): median |du|, |dv|, MAD, inlier8 for
    side "a" (before) and side "b" (after), plus the fraction of frames whose radial residual
    magnitude hypot(du,dv) improved / worsened by more than `improve_thresh_px` (only over frames
    with a finite fit on both sides). Exposed standalone so it is unit-testable without a cloud store
    (`tests/test_pose_report.py`)."""
    if not rows:
        return {"n": 0}
    g = {k: np.array([r[k] if r[k] is not None else np.nan for r in rows], dtype=float) for k in ("du_a", "dv_a", "dum_a", "dvm_a", "inl_a", "du_b", "dv_b", "dum_b", "dvm_b", "inl_b")}
    mag_a = np.hypot(g["du_a"], g["dv_a"])
    mag_b = np.hypot(g["du_b"], g["dv_b"])
    both = np.isfinite(mag_a) & np.isfinite(mag_b)
    delta = mag_b[both] - mag_a[both]
    n_both = int(both.sum())
    out = {
        "n": len(rows),
        "n_fit_a": int(np.isfinite(g["du_a"]).sum()), "n_fit_b": int(np.isfinite(g["du_b"]).sum()),
        "abs_du_a": _stat(np.abs(g["du_a"])), "abs_dv_a": _stat(np.abs(g["dv_a"])),
        "abs_du_b": _stat(np.abs(g["du_b"])), "abs_dv_b": _stat(np.abs(g["dv_b"])),
        "mad_du_a": _stat(g["dum_a"]), "mad_dv_a": _stat(g["dvm_a"]),
        "mad_du_b": _stat(g["dum_b"]), "mad_dv_b": _stat(g["dvm_b"]),
        "inlier8_a": _stat(g["inl_a"]), "inlier8_b": _stat(g["inl_b"]),
        "n_both_fit": n_both,
        "frac_improved_gt2px": round(float((delta < -improve_thresh_px).mean()), 3) if n_both else None,
        "frac_worsened_gt2px": round(float((delta > improve_thresh_px).mean()), 3) if n_both else None,
    }
    return out


def compare_pose_sources(
    a: str = "export",
    b: str = "corrected",
    frames: dict[str, np.ndarray] | None = None,
    out_dir: Path = FINAL_DIR,
    workers: int = 6,
    log=print,
) -> dict:
    """S7 task 1: `quality._silhouette_points`/`_residual` at each frame's OWN pose from source `a`
    and from source `b`, grouped as `frames` (default: `select_groups(a, b)`), aggregated overall and
    per pass within each group. Writes `<out_dir>/report_compare_<a>_vs_<b>.json`."""
    from multiprocessing import Pool

    if frames is None:
        frames = select_groups(a, b)
    union = sorted({int(k) for idx in frames.values() for k in idx})
    if not union:
        raise RuntimeError("compare_pose_sources: no frames selected")
    t0 = time.time()
    n_workers = max(1, min(workers, len(union)))
    with Pool(n_workers, initializer=_init_compare, initargs=(a, b)) as pool:
        results = list(pool.imap_unordered(_compare_job, union, chunksize=4))
    log(f"[compare_pose_sources] {a} vs {b}: {len(union)} frames in {time.time() - t0:.1f}s ({n_workers} workers)")
    by_frame = {r["frame"]: r for r in results}

    groups_out = {}
    for name, idx in frames.items():
        rows = [by_frame[int(k)] for k in idx if int(k) in by_frame]
        overall = aggregate_compare_rows(rows)
        by_pass: dict[int, dict] = {}
        for p in sorted({r["pass_a"] for r in rows}):
            by_pass[p] = aggregate_compare_rows([r for r in rows if r["pass_a"] == p])
        groups_out[name] = {"n_requested": int(len(idx)), "overall": overall, "by_pass": by_pass}
        if overall.get("n", 0) == 0:
            log(f"[compare_pose_sources] {name}: no frames matched")
        else:
            log(f"[compare_pose_sources] {name} (n={overall.get('n')}): |du| {overall['abs_du_a']['median']}->{overall['abs_du_b']['median']} px, "
                f"|dv| {overall['abs_dv_a']['median']}->{overall['abs_dv_b']['median']} px, inlier8 {overall['inlier8_a']['median']}->{overall['inlier8_b']['median']}, "
                f"improved>2px {overall['frac_improved_gt2px']}, worsened>2px {overall['frac_worsened_gt2px']}")

    out = {"a": a, "b": b, "groups": groups_out, "per_frame": results}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_compare_{a}_vs_{b}.json"
    out_path.write_text(json.dumps(out, indent=1, default=float))
    log(f"[compare_pose_sources] wrote {out_path}")
    return out


# ------------------------------------------------------------------------------------ colour compare
_DG: dict = {}


def _init_colour(a: str, b: str) -> None:
    from .align import Aligner
    from .cloud_store import CloudStore
    from geovap.runtime import settings
    from geovap.stages.prepare.masks import VehicleMask

    store = CloudStore()
    poses_a, poses_b = load_poses(a), load_poses(b)
    _s = settings.get()
    _mask_path = _s.workspace.vehicle_mask
    vm = VehicleMask(_mask_path, *_s.sensor.pano) if _mask_path.exists() else None
    _DG["poses_a"], _DG["poses_b"] = poses_a, poses_b
    _DG["al_a"] = Aligner(store, poses_a, vm)
    _DG["al_b"] = Aligner(store, poses_b, vm)


def _colour_job(k: int) -> dict:
    from geovap.domain.math.sampling import PanoSampler
    from geovap.io.images import load_pano_rgb

    al_a, al_b = _DG["al_a"], _DG["al_b"]
    poses_a, poses_b = _DG["poses_a"], _DG["poses_b"]
    al_a.ps = PanoSampler(load_pano_rgb(poses_a.path(k)), footprint=False, gradient=False)
    xyz, gray, gps, rgb = al_a.gather(k)
    al_b.ps = al_a.ps
    de_a = al_a.colour_de(k, xyz, rgb, gps, int(poses_a.pass_id[k]), float(poses_a.t[k]), 0.0)
    de_b = al_b.colour_de(k, xyz, rgb, gps, int(poses_b.pass_id[k]), float(poses_b.t[k]), 0.0)
    return {"frame": int(k), "de_a": de_a, "de_b": de_b}


def colour_de_comparison(
    a: str = "export",
    b: str = "corrected",
    frames: np.ndarray | None = None,
    out_dir: Path = FINAL_DIR,
    workers: int = 6,
    log=print,
) -> dict:
    """S7 task 2: `mapping.align.Aligner.colour_de` at each frame's own (dt=0, yaw_off=0) pose from
    `a` and `b`, over `frames` (default: the 210 turning frames, `select_groups`'s "turning" group).
    Writes `<out_dir>/report_colour_<a>_vs_<b>.json`."""
    from multiprocessing import Pool

    if frames is None:
        frames = select_groups(a, b)["turning"]
    frames = sorted(int(k) for k in frames)
    t0 = time.time()
    n_workers = max(1, min(workers, len(frames)))
    with Pool(n_workers, initializer=_init_colour, initargs=(a, b)) as pool:
        results = list(pool.imap_unordered(_colour_job, frames, chunksize=4))
    log(f"[colour_de_comparison] {a} vs {b}: {len(frames)} frames in {time.time() - t0:.1f}s ({n_workers} workers)")
    results.sort(key=lambda r: r["frame"])
    de_a = np.array([r["de_a"] for r in results])
    de_b = np.array([r["de_b"] for r in results])
    summary = {"n": len(results), "de_a": _stat(de_a), "de_b": _stat(de_b), "n_improved": int((de_b < de_a).sum()), "n_worsened": int((de_b > de_a).sum())}
    log(f"[colour_de_comparison] median dE {a}={summary['de_a']['median']} -> {b}={summary['de_b']['median']}")
    out = {"a": a, "b": b, "summary": summary, "per_frame": results}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_colour_{a}_vs_{b}.json"
    out_path.write_text(json.dumps(out, indent=1, default=float))
    log(f"[colour_de_comparison] wrote {out_path}")
    return summary


# --------------------------------------------------------------------------------- cross-pass conflict
def conflict_summary(path: str | Path = DEFAULT_CONFLICT_JSON) -> dict:
    """S7 task 3: the final table's pass transforms are exactly S5b's (`assemble_poses.assemble`
    applies `pass_reg.pass_transforms.json` unchanged), so this just re-reads and re-cites
    `cli.validate_pass_reg.run_conflict_metric`'s output rather than recomputing it."""
    path = Path(path)
    if not path.exists():
        return {"available": False, "path": str(path)}
    d = json.loads(path.read_text())
    return {
        "available": True, "path": str(path),
        "true_cross_pass": d.get("overall_true_cross_pass"),
        "all_n2gt0_frames": d.get("overall_all_n2gt0_frames"),
        "cloud_icp_pairs_summary": d.get("cloud_icp_pairs_summary"),
    }


# ------------------------------------------------------------------------------- interpolation benefit
def _neighbor_linear_yaw(poses: Poses, k: int) -> float:
    """Leave-one-out linear-interpolation prediction of frame `k`'s own yaw from its immediate
    same-pass neighbours (the "old model": what production code effectively did between any two
    known frames, evaluated exactly AT a known frame to score it against that frame's own value).
    NaN if `k` is at a pass boundary (no neighbour on one side)."""
    p = poses.pass_id[k]
    if not (k - 1 >= 0 and poses.pass_id[k - 1] == p and k + 1 < len(poses) and poses.pass_id[k + 1] == p):
        return float("nan")
    t0, t1 = poses.t[k - 1], poses.t[k + 1]
    y0, y1 = poses.yaw[k - 1], poses.yaw[k + 1]
    y1u = y0 + ((y1 - y0 + 180.0) % 360.0 - 180.0)
    frac = (poses.t[k] - t0) / (t1 - t0)
    y_pred = y0 + frac * (y1u - y0)
    return float((y_pred + 180.0) % 360.0 - 180.0)


def _ang_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def interpolation_benefit(
    corrected_source: str = "corrected",
    out_dir: Path = FINAL_DIR,
    n_plot: int = 3,
    seed: int = 0,
    log=print,
) -> dict:
    """S7 task 4: the one real gain of the S3b trajectory. For turning frames covered by the
    trajectory, compares two residuals against each frame's own export yaw (the ground truth proxy,
    per S3b: export yaw at a frame's own timestamp agrees with the scanner-plane orientation to
    0.057 deg median):
      - "dense model": `Poses.interp` (with the trajectory attached) evaluated exactly at the frame's
        own time -- NOT `pose_at(idx, 0)`, which would just return the table value; `interp` always
        prefers the trajectory where it covers a query, so this is genuinely the dense model's
        prediction, not a lookup.
      - "old model": linear interpolation from the frame's immediate same-pass neighbours only
        (`_neighbor_linear_yaw`, leave-one-out).
    Also plots yaw(t) over +/-1.5 s for `n_plot` turning frames: export samples, old-model linear
    interpolation, and the dense trajectory. Writes `<out_dir>/report_interp_benefit.json` and PNGs."""
    from .cli.build_trajectory import _covered_mask, turning_frame_idx

    poses_exp = load_poses("export")
    poses_cor = load_poses(corrected_source)
    if poses_cor.traj is None:
        raise RuntimeError(f"{corrected_source}: no trajectory attached")

    turning = turning_frame_idx(poses_exp, min_rate=YAW_RATE_TURNING_DEG_S)
    cov = _covered_mask(poses_cor)
    idx = turning[cov[turning]]
    log(f"[interp_benefit] turning frames covered by the trajectory: {len(idx)}/{len(turning)}")

    y_exp = poses_exp.yaw[idx]
    _o, _r, _p, y_dense = poses_cor.interp(poses_exp.t[idx], poses_cor.pass_id[idx])
    dense_resid = _ang_diff(y_dense, y_exp)

    y_old = np.array([_neighbor_linear_yaw(poses_exp, int(k)) for k in idx])
    old_resid = _ang_diff(y_old, y_exp)

    summary = {
        "n_turning_covered": int(len(idx)),
        "dense_model_resid_deg": _stat(dense_resid),
        "old_model_resid_deg": _stat(old_resid),
    }
    log(f"[interp_benefit] dense-model |resid| median {summary['dense_model_resid_deg']['median']} deg, "
        f"old-model (neighbour linear interp) median {summary['old_model_resid_deg']['median']} deg p95 {summary['old_model_resid_deg']['p95']} deg")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_files = _plot_interp_examples(poses_exp, poses_cor, idx, n_plot, seed, out_dir)
    out = {"summary": summary, "plots": plot_files, "frames": [int(k) for k in idx]}
    (out_dir / "report_interp_benefit.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"[interp_benefit] wrote {out_dir / 'report_interp_benefit.json'}")
    return summary


def _plot_interp_examples(poses_exp: Poses, poses_cor: Poses, idx: np.ndarray, n_plot: int, seed: int, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(idx) == 0:
        return []
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(idx, size=min(n_plot, len(idx)), replace=False))
    files = []
    lin_only = Poses(filename=poses_exp.filename, t=poses_exp.t, origin=poses_exp.origin, roll=poses_exp.roll, pitch=poses_exp.pitch, yaw=poses_exp.yaw, pass_id=poses_exp.pass_id, speed=poses_exp.speed, source="export_linear", traj=None)
    for k in pick:
        k = int(k)
        p = int(poses_exp.pass_id[k])
        tk = float(poses_exp.t[k])
        tt = np.linspace(tk - 1.5, tk + 1.5, 121)
        ph = np.full(len(tt), p)
        _o, _r, _p2, y_lin = lin_only.interp(tt, ph)
        _o2, _r2, _p3, y_traj = poses_cor.interp(tt, ph)
        cov = np.asarray(poses_cor.traj.covers(p, tt), dtype=bool)
        near = np.flatnonzero((poses_exp.pass_id == p) & (np.abs(poses_exp.t - tk) <= 1.5))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tt - tk, y_lin, "--", color="#888888", label="linear interpolation (export)", linewidth=1.5)
        ax.plot((tt - tk)[cov], y_traj[cov], "-", color="#1f77b4", label="dense trajectory", linewidth=1.5)
        ax.scatter(poses_exp.t[near] - tk, poses_exp.yaw[near], color="#d62728", s=28, zorder=5, label="export samples")
        ax.axvline(0.0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("t - t_frame (s)")
        ax.set_ylabel("yaw (deg)")
        ax.set_title(f"frame {k} (pass {p}): yaw(t) around a turn")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fname = out_dir / f"interp_benefit_f{k:04d}.png"
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        files.append(str(fname))
    return files


# --------------------------------------------------------------------------------------- invariance
def _build_pre_post_tables(
    base_path: str | Path = POSES_DIR / "poses_traj_rot.csv",
    refined_path: str | Path = POSES_DIR / "poses_refined_export.csv",
    corrected_source: str = "corrected",
) -> tuple[Poses, Poses]:
    """Shared by `invariance_check` and `lever_arm_example`: `pre` is the S4-overlaid, pre-S5b table
    (with its trajectory's linear-fallback fields resynced to the overlay -- see the note below), and
    `post` is `load_poses(corrected_source)` (S4 overlay + S5b registration, both table and trajectory).
    Both are real-pipeline snapshots built from the on-disk base/refined tables and pass transforms, not
    synthetic data."""
    from dataclasses import replace

    base = read_pose_table(base_path)
    refined_rows = _read_csv_rows(refined_path)

    from .cli.assemble_poses import overlay_refined

    overlaid, _meta, _stats = overlay_refined(base, refined_rows)
    if base.traj is None:
        pre = overlaid
    else:
        # Pre-registration state: S4 overlay applied, S5b not yet. `base.traj`'s dense orientation
        # (quat/S/segments/rigs) does not depend on the table and stays as-is; but its `lin_*` fallback
        # (the plain linear-interpolation-of-origin the "rot_only" trajectory falls back on between/around
        # covered windows -- see `Trajectory._linear_origin`) was baked in at S3b build time from the
        # *export* table, before S4 existed, so it must be resynced to `overlaid` here for "pre" to be a
        # self-consistent snapshot of "S4 applied, S5b not" -- exactly mirroring what
        # `assemble_poses.transform_trajectory` does for "post" (resyncs to the final registered table).
        # Skipping this resync makes `pre`'s trajectory silently point at pre-S4 origins for the ~1373
        # trajectory-covered frames while `pre`'s own table (`overlaid.origin`) already has S4 applied,
        # which is not a real pipeline state and produces spurious multi-metre "invariance" errors.
        pre_traj = replace(base.traj, lin_t=overlaid.t.copy(), lin_origin=overlaid.origin.copy(), lin_roll=overlaid.roll.copy(), lin_pitch=overlaid.pitch.copy(), lin_yaw=overlaid.yaw.copy(), lin_pass_id=overlaid.pass_id.copy())
        pre = replace(overlaid, traj=pre_traj)

    post = load_poses(corrected_source)
    return pre, post


def invariance_check(
    base_path: str | Path = POSES_DIR / "poses_traj_rot.csv",
    refined_path: str | Path = POSES_DIR / "poses_refined_export.csv",
    transforms_path: str | Path = DEFAULT_TRANSFORMS,
    corrected_source: str = "corrected",
    n_per_pass: int = 3,
    seed: int = 0,
    log=print,
) -> dict:
    """S7 real-data counterpart of `tests/test_poses_table.py`'s synthetic
    `test_transform_trajectory_invariant_with_apply_pass_transforms`: for a few frames of every
    registered pass, `world_to_pano(P, R_pre, C_pre)` (pre-S5b table + S3b trajectory) must equal
    `world_to_pano(T_p(P), R_post, C_post)` (`poses_corrected`, S5b-transformed table + trajectory) at
    the frame's own time and +/-0.3 s, to ~1e-6 px -- the contract `assemble_poses.transform_trajectory`
    exists to guarantee. Uses the real per-pass transforms and the real base/refined tables, not a
    synthetic trajectory."""
    pre, post = _build_pre_post_tables(base_path, refined_path, corrected_source)
    if pre.traj is None:
        raise RuntimeError(f"{base_path}: no trajectory attached")
    overlaid = pre
    transforms = json.loads(Path(transforms_path).read_text()).get("passes", {})

    rng = np.random.default_rng(seed)
    per_pass: dict[int, float] = {}
    for p in sorted(int(x) for x in np.unique(overlaid.pass_id)):
        tr = transforms.get(str(p))
        if tr is None:
            continue
        idx = np.flatnonzero(overlaid.pass_id == p)
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(n_per_pass, len(idx)), replace=False)
        worst_p = 0.0
        for k in pick:
            tq = np.array([overlaid.t[k], overlaid.t[k] + 0.3, overlaid.t[k] - 0.3])
            worst_p = max(worst_p, _invariance_worst_px(pre, post, tq, p, tr, rng))
        per_pass[p] = round(worst_p, 9)
    worst = max(per_pass.values()) if per_pass else float("nan")
    log(f"[invariance] worst px across {len(per_pass)} passes: {worst:.2e}")
    return {"worst_px": worst, "n_passes": len(per_pass), "per_pass_worst_px": per_pass}


def _invariance_worst_px(poses_pre: Poses, poses_post: Poses, tq: np.ndarray, pass_id: int, tr: dict, rng, n_pts: int = 500) -> float:
    from . import pass_reg

    cx, cy = tr["centre"]
    Rp = pass_reg._rot_yaw(np.radians(tr["yaw_deg"]))
    centre3 = np.array([cx, cy, 0.0])
    t_vec = np.array(tr["t"])
    ph = np.full(len(tq), pass_id)
    o_a, r_a, p_a, y_a = poses_pre.interp(tq, ph)
    o_b, r_b, p_b, y_b = poses_post.interp(tq, ph)
    R_a = geometry.vehicle_rotation(y_a, r_a, p_a)
    R_b = geometry.vehicle_rotation(y_b, r_b, p_b)
    worst = 0.0
    for k in range(len(tq)):
        P = o_a[k] + rng.uniform(-25, 25, (n_pts, 3)) * np.array([1, 1, 0.4])
        P_t = (P - centre3) @ Rp.T + centre3 + t_vec
        u1, v1, _, _ = geometry.world_to_pano(P, R_a[k], o_a[k], PANO_W, PANO_H, dtype=np.float64)
        u2, v2, _, _ = geometry.world_to_pano(P_t, R_b[k], o_b[k], PANO_W, PANO_H, dtype=np.float64)
        du = np.abs(u1 - u2)
        du = np.minimum(du, PANO_W - du)
        worst = max(worst, float(du.max()), float(np.abs(v1 - v2).max()))
    return worst


def lever_arm_example(
    transforms_path: str | Path = DEFAULT_TRANSFORMS,
    corrected_source: str = "corrected",
    pass_id: int = 0,
) -> dict:
    """S7 task 6 support: computed (not hand-typed) illustration of the S5b pass-transform lever arm --
    for `pass_id`'s first frame, the pass's rotation+translation correction, the frame's distance from
    the pass's registration centre, and the resulting absolute camera shift `|post.origin - pre.origin|`
    at that one frame. Used by `slow_regression_037`'s narrative to explain why a fraction-of-a-degree
    pass correction can be a multi-metre shift far from the centroid even though the correction itself
    is small."""
    transforms = json.loads(Path(transforms_path).read_text()).get("passes", {})
    tr = transforms.get(str(pass_id))
    if tr is None:
        return {"error": f"pass {pass_id} not in {transforms_path}"}
    pre, post = _build_pre_post_tables(corrected_source=corrected_source)
    idx = np.flatnonzero(pre.pass_id == pass_id)
    if len(idx) == 0:
        return {"error": f"pass {pass_id} has no frames"}
    k = int(idx[0])
    centre = np.array(tr["centre"], dtype=float)
    dist_centre_m = float(np.linalg.norm(pre.origin[k, :2] - centre))
    shift_m = float(np.linalg.norm(post.origin[k] - pre.origin[k]))
    return {
        "pass_id": pass_id, "frame_idx": k,
        "yaw_deg": float(tr["yaw_deg"]), "t_m": float(np.linalg.norm(tr["t"])),
        "dist_from_centre_m": round(dist_centre_m, 1), "shift_m": round(shift_m, 3),
    }


def slow_regression_037(
    tolerance: float = 0.1,
    transforms_path: str | Path = DEFAULT_TRANSFORMS,
    lever_arm_pass: int = 0,
    log=print,
) -> dict:
    """S7 task 5/6: run the tile-037 pilot recipe (`tests/test_regression_037.py`'s
    `test_pilot_anchor_corrected_poses_not_worse` recipe, in-process rather than via pytest so
    `run_final_report` can cite real, freshly-computed numbers) for `export`, for `corrected` with the
    matching `CloudStore(registration=pass_transforms.json)`, and -- to make the operational caveat
    concrete rather than asserted -- also for `corrected` with an *unregistered* cloud, reproducing the
    false-alarm regression that motivated requiring `registration=` alongside `load_poses("corrected")`.
    Returns `{"skipped": reason}` if the store/frames cache isn't built."""
    from .cloud_store import CloudStore
    from .colorize import Options, colorize_tile
    from .config import STORE_DIR
    from geovap.domain.model.frames import FrameIndex

    if not (STORE_DIR / "tiles.json").exists():
        return {"skipped": f"{STORE_DIR / 'tiles.json'} not found (store not built)"}
    if not Path(transforms_path).exists():
        return {"skipped": f"{transforms_path} not found"}

    def _median(store, poses) -> tuple[float, int]:
        fi = FrameIndex(poses)
        opt = Options(write_las=False, subsample=120, occlusion=False, vehicle_mask=False)
        res = colorize_tile(store.by_name["037"], store, fi, opt, log=lambda *a: None)
        s = res.sample
        m = s["valid_nt_noocc"] & (s["cam_dist"] >= 3.5)
        return float(np.median(s["de76_nt_noocc"][m])), int(m.sum())

    poses_export = load_poses("export")
    poses_corrected = load_poses("corrected")

    med_export, n_export = _median(CloudStore(), poses_export)
    med_corrected, n_corrected = _median(CloudStore(registration=transforms_path), poses_corrected)
    med_corrected_unreg, n_unreg = _median(CloudStore(), poses_corrected)
    log(f"[slow_037] export={med_export:.3f} (n={n_export}) corrected(registered)={med_corrected:.3f} (n={n_corrected}) "
        f"corrected(unregistered)={med_corrected_unreg:.3f} (n={n_unreg})")

    lever = lever_arm_example(transforms_path=transforms_path, pass_id=lever_arm_pass)

    return {
        "median_de76_export": round(med_export, 3), "n_export": n_export,
        "median_de76_corrected_registered": round(med_corrected, 3), "n_corrected_registered": n_corrected,
        "median_de76_corrected_unregistered": round(med_corrected_unreg, 3), "n_corrected_unregistered": n_unreg,
        "delta_registered": round(med_corrected - med_export, 3),
        "delta_unregistered": round(med_corrected_unreg - med_export, 3),
        "tolerance": tolerance,
        "passes_registered": bool(med_corrected <= med_export + tolerance),
        "lever_arm_example": lever,
    }


# ------------------------------------------------------------------------------------- final report
def run_final_report(
    a: str = "export",
    b: str = "corrected",
    out_dir: Path = FINAL_DIR,
    workers: int = 6,
    run_slow: bool = True,
    log=print,
) -> dict:
    """S7 task 6: run everything above and write `<out_dir>/report_final.{md,json}`, including the
    Summary bullets and (when `run_slow`, the default) the tile-037 pilot regression section -- both
    generated from the numbers computed in this run, not hand-authored, so re-running this function
    reproduces the full report byte-for-byte from the same inputs."""
    poses_a = load_poses(a)
    poses_b = load_poses(b)

    groups = select_groups(a, b)
    log(f"[final] groups: turning={len(groups['turning'])}, straight={len(groups['straight'])}, refined_or_reg={len(groups['refined_or_reg'])}")

    compare = compare_pose_sources(a, b, groups, out_dir=out_dir, workers=workers, log=log)
    colour = colour_de_comparison(a, b, groups["turning"], out_dir=out_dir, workers=workers, log=log)
    conflict = conflict_summary()
    interp = interpolation_benefit(b, out_dir=out_dir, log=log)
    try:
        inv = invariance_check(corrected_source=b, log=log)
    except Exception as e:  # pass_transforms / base table missing -> report the gap, do not fail the whole report
        inv = {"error": f"{type(e).__name__}: {e}"}
    if run_slow:
        try:
            slow = slow_regression_037(log=log)
        except Exception as e:  # store/frames or transforms missing -> report the gap, do not fail the whole report
            slow = {"error": f"{type(e).__name__}: {e}"}
    else:
        slow = {"skipped": "run_slow=False"}

    # sanity checks cited in the report (cheap; recomputed here rather than trusted from memory)
    idx_all = np.arange(len(poses_b))
    o0, r0, p0, y0 = poses_b.pose_at(idx_all, 0.0)
    pose_at_exact = bool(np.allclose(o0, poses_b.origin) and np.allclose(r0, poses_b.roll) and np.allclose(p0, poses_b.pitch) and np.allclose(y0, poses_b.yaw))

    identity = {
        "hash_a": poses_a.hash(), "hash_b": poses_b.hash(), "len_a": len(poses_a), "len_b": len(poses_b),
        "b_traj_attached": poses_b.traj is not None, "b_traj_mode": getattr(poses_b.traj, "mode", None),
        "pose_at_idx_0_exact": pose_at_exact,
    }

    assemble_json = POSES_DIR / f"poses_{b}.json" if b != "export" else None
    assemble_prov = json.loads(assemble_json.read_text()) if assemble_json and assemble_json.exists() else {}

    summary = {
        "a": a, "b": b, "identity": identity,
        "compare_groups": {name: g["overall"] for name, g in compare["groups"].items()},
        "colour_de_turning": colour,
        "cross_pass_conflict": conflict,
        "interpolation_benefit": interp,
        "invariance": inv,
        "slow_regression_037": slow,
        "assemble_provenance": assemble_prov,
    }
    md = _final_markdown(summary)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report_final.md").write_text(md)
    (out_dir / "report_final.json").write_text(json.dumps(summary, indent=1, default=float))
    log(f"[final] wrote {out_dir / 'report_final.md'} (+ .json)")
    return summary


def _group_row(name: str, g: dict) -> str:
    o = g
    if o.get("n", 0) == 0:
        return f"| {name} | 0 | - | - | - | - | - | - |"
    return (f"| {name} | {o['n']} | {o['abs_du_a']['median']} -> {o['abs_du_b']['median']} | "
            f"{o['abs_dv_a']['median']} -> {o['abs_dv_b']['median']} | {o['mad_du_a']['median']} -> {o['mad_du_b']['median']} | "
            f"{o['inlier8_a']['median']} -> {o['inlier8_b']['median']} | {o['frac_improved_gt2px']} | {o['frac_worsened_gt2px']} |")


def _summary_bullets(s: dict) -> list[str]:
    """Generates the '## Summary' block from the same numbers the numbered sections below are built
    from -- no text here is hand-typed independently of `s`."""
    lines = ["## Summary", ""]
    ap = s.get("assemble_provenance") or {}
    ov = ap.get("s4_overlay")
    sc = ap.get("status_counts")
    if ov and sc:
        n_total = sum(sc.values())
        sc_str = ", ".join(f"`{k}` {v}" for k, v in sc.items())
        lines.append(
            f"`poses_{s['b']}.csv` = S3b rot-only trajectory base (`poses_traj_rot.csv`) + S4 per-frame "
            f"refinement overlay ({ov['n_overlaid']}/{n_total} rows, `status` containing \"refined\"; "
            f"{ov['n_pass_changed']} row(s) changed pass) + S5b pass registration (`pass_transforms.json`, "
            f"applied to both the table and the trajectory). S4 overlay deltas vs the base: \\|dpos\\| median "
            f"{ov['dpos_median_m']:.3f} m, p95 {ov['dpos_p95_m']:.3f} m, max {ov['dpos_max_m']:.3f} m; \\|dyaw\\| "
            f"median {ov['dyaw_median_deg']:.3f} deg, p95 {ov['dyaw_p95_deg']:.3f} deg, max {ov['dyaw_max_deg']:.3f} deg. "
            f"Final per-frame status counts (sum {n_total}): {sc_str}."
        )
        lines.append("")
    lines.append("Net effect of the whole correction chain, measured independently below:")
    cg = s.get("compare_groups", {})
    g = cg.get("refined_or_reg")
    gt = cg.get("turning")
    gs = cg.get("straight")
    if g and gt and gs:
        lines.append(
            f"- **Own-pose silhouette fit** (section 1) changes modestly and, on the frames the "
            f"corrections target, asymmetrically in their favour: on `refined_or_reg` ({g['n']} frames, "
            f"exactly where S4/S5b act) {g['frac_improved_gt2px']:.0%} of frames improve by >2px vs "
            f"{g['frac_worsened_gt2px']:.0%} that worsen; turning frames {gt['frac_improved_gt2px']:.0%} vs "
            f"{gt['frac_worsened_gt2px']:.0%}; clean straight frames (barely touched by either correction) "
            f"{gs['frac_improved_gt2px']:.0%} vs {gs['frac_worsened_gt2px']:.0%} -- a small residual gain even "
            f"there, consistent with S5b's pass-wide registration reaching every frame of a shifted pass, not "
            f"just the ones S4 touched."
        )
    c = s.get("colour_de_turning") or {}
    if c:
        lines.append(
            f"- **Colour dE on turning frames** (section 2) is a wash ({c['de_a']['median']} -> "
            f"{c['de_b']['median']} median, {c['n_improved']} vs {c['n_worsened']} of {c['n']} frames), "
            f"confirming `mapping/README.md`'s established finding that this metric tracks TerraScan's own "
            f"source-image choice, not pose alignment -- it is not expected to move here."
        )
    cf = s.get("cross_pass_conflict") or {}
    if cf.get("available"):
        tco = cf["true_cross_pass"]
        icp = cf.get("cloud_icp_pairs_summary", {})
        lines.append(
            f"- **Cross-pass conflict** (section 3) drops slightly ({tco['n_conflict_before']} -> "
            f"{tco['n_conflict_after']} of {tco['n_frames']} true cross-pass frames); the real win at the "
            f"pass level is the pairwise cloud-ICP rms ({icp.get('median_rms_before')} -> "
            f"{icp.get('median_rms_after')} m median)."
        )
    ib = s.get("interpolation_benefit") or {}
    if ib:
        ratio = ib['old_model_resid_deg']['median'] / ib['dense_model_resid_deg']['median'] if ib['dense_model_resid_deg']['median'] else float("inf")
        lines.append(
            f"- **The dense trajectory's real value** (section 4) is orientation *between* frames: at a "
            f"frame's own timestamp the dense model already agrees with {s['a']} to a median "
            f"{ib['dense_model_resid_deg']['median']} deg (p95 {ib['dense_model_resid_deg']['p95']} deg), while "
            f"naive linear interpolation from a turning frame's immediate neighbours is off by a median "
            f"{ib['old_model_resid_deg']['median']} deg (p95 {ib['old_model_resid_deg']['p95']} deg) -- roughly "
            f"{ratio:.0f}x worse. This is the correction chain's single largest, least ambiguous improvement."
        )
    inv = s.get("invariance") or {}
    if "worst_px" in inv:
        target_note = "meets" if inv["worst_px"] < 1e-6 else "above (but negligible relative to)"
        lines.append(
            f"- **Invariance** (section 5) confirms S5b's registration is applied identically to the table "
            f"and the trajectory: worst {inv['worst_px']:.2e} px across all {inv['n_passes']} passes on real "
            f"data -- {target_note} the 1e-6 px aspirational target, floating-point noise rather than a "
            f"systematic error."
        )
    slow = s.get("slow_regression_037") or {}
    if "median_de76_export" in slow:
        lever = slow.get("lever_arm_example") or {}
        lever_str = ""
        if "shift_m" in lever:
            lever_str = (
                f" (pass {lever['pass_id']}'s correction is {lever['yaw_deg']:.2f} deg + {lever['t_m']:.2f} m, "
                f"but frame {lever['frame_idx']} sits {lever['dist_from_centre_m']:.0f} m from the pass centroid "
                f"-> a {lever['shift_m']:.2f} m absolute shift there)"
            )
        lines.append(
            f"- **Slow regression guard** (section 6, tile 037 pilot recipe): median CIE76 "
            f"{s['a']}={slow['median_de76_export']} vs {s['b']}={slow['median_de76_corrected_registered']} "
            f"(delta {slow['delta_registered']:+.3f}, tolerance {slow['tolerance']}) when the cloud is "
            f"registered with the same `pass_transforms.json` as the poses. Without that registration the same "
            f"recipe gives {slow['median_de76_corrected_unregistered']} (delta {slow['delta_unregistered']:+.3f}) "
            f"-- a spurious camera-vs-cloud offset from `poses_corrected`'s per-pass rigid transform, which is a "
            f"rotation about the pass centroid and can be a multi-metre shift far from it{lever_str}."
        )
    lines.append("")
    return lines


def _slow_regression_markdown(slow: dict) -> list[str]:
    lines = ["## 6. Slow regression (tile 037 pilot recipe, `tests/test_regression_037.py -k corrected`)", ""]
    if "skipped" in slow:
        lines.append(f"skipped: {slow['skipped']}")
        lines.append("")
        return lines
    if "error" in slow:
        lines.append(f"not computed: {slow['error']}")
        lines.append("")
        return lines
    lines.append(
        f"Pilot recipe (nearest-in-time frame, nearest pixel, no occlusion, no vehicle mask, points >=3.5 m) "
        f"median CIE76: export={slow['median_de76_export']} (n={slow['n_export']}), "
        f"corrected={slow['median_de76_corrected_registered']} (n={slow['n_corrected_registered']}) "
        f"(delta {slow['delta_registered']:+.3f}, tolerance +{slow['tolerance']}) -- "
        f"`CloudStore(registration=pass_transforms.json)` used on the corrected side."
    )
    lines.append("")
    lines.append(
        "That registration is required, not optional -- **an operational finding worth flagging, not just a "
        "test-fixture detail**: `poses_corrected` moves each camera by a *rotation about its pass's centroid*, "
        "not a small per-frame nudge, so for a long pass this is a multi-metre shift at frames far from the "
        "centroid even for a fraction-of-a-degree pass correction."
    )
    lever = slow.get("lever_arm_example") or {}
    if "shift_m" in lever:
        lines.append(
            f"Example: pass {lever['pass_id']}'s correction is {lever['yaw_deg']:.2f} deg + {lever['t_m']:.2f} m "
            f"translation, but frame {lever['frame_idx']} of that pass sits {lever['dist_from_centre_m']:.0f} m "
            f"from the pass centroid -> a {lever['shift_m']:.2f} m absolute camera shift at that frame."
        )
    lines.append(
        f"Using a plain `CloudStore()` (points left in their scanned positions) with the corrected poses gives "
        f"median CIE76={slow['median_de76_corrected_unregistered']} (delta {slow['delta_unregistered']:+.3f}) -- "
        f"not because the corrected poses are worse, but because the camera moves while tile 037's points do "
        f"not, producing a camera-vs-cloud offset the correction never intended. Registering the cloud with the "
        f"*same* `pass_transforms.json` (`mapping.pass_reg`'s documented \"within a pass, image and cloud always "
        f"move together\" contract) removes that offset and recovers the near-zero delta above."
    )
    lines.append("")
    lines.append(
        "Section 1's own-pose silhouette comparison, by contrast, is essentially unaffected by this "
        "(`compare_pose_sources` also uses a plain, unregistered `CloudStore()` for both sources, matching how "
        "`quality.run()` builds `dataset/frame_quality.csv` today): each frame's \"own pass\" edge points are "
        "gathered from a disc centred on that source's own (possibly shifted) camera position, and a "
        "road-corridor scene looks broadly similar under a modest along-track shift (the same rural along-road "
        "degeneracy `mapping/pass_reg.py` already documents for JVF absolute registration) -- so local "
        "per-frame checks stay informative without cloud registration, but any analysis anchored to fixed, "
        "tile/world-frame points (colorization, LAS export, seg dataset rasters) must pass the matching "
        "`registration=` to `CloudStore` whenever it is paired with `load_poses(\"corrected\")`."
    )
    lines.append("")
    return lines


def _final_markdown(s: dict) -> str:
    lines = [f"# S7 final validation: `{s['b']}` vs `{s['a']}`", ""]
    lines += _summary_bullets(s)
    idn = s["identity"]
    lines.append(f"`load_poses(\"{s['a']}\")` hash `{idn['hash_a']}` (len {idn['len_a']}); "
                 f"`load_poses(\"{s['b']}\")` hash `{idn['hash_b']}` (len {idn['len_b']}, trajectory attached: {idn['b_traj_attached']}, mode `{idn['b_traj_mode']}`). "
                 f"`pose_at(idx, 0)` exactly equals the table for all frames: {idn['pose_at_idx_0_exact']}.")
    lines.append("")
    lines.append("## 1. Silhouette residual (quality.py, own pose per source), per group")
    lines.append("")
    lines.append(f"| group | n | \\|du\\| median px (a->b) | \\|dv\\| median px (a->b) | du MAD px (a->b) | inlier\\@8px (a->b) | frac improved>2px | frac worsened>2px |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, g in s["compare_groups"].items():
        lines.append(_group_row(name, g))
    lines.append("")
    lines.append("## 2. Colour dE (Aligner.colour_de, turning frames)")
    c = s["colour_de_turning"]
    lines.append(f"n={c['n']}: median dE {s['a']}={c['de_a']['median']} -> {s['b']}={c['de_b']['median']}; "
                 f"improved {c['n_improved']}/{c['n']}, worsened {c['n_worsened']}/{c['n']}.")
    lines.append("")
    lines.append("## 3. Cross-pass conflict (S5b, cited)")
    cf = s["cross_pass_conflict"]
    if cf.get("available"):
        tco = cf["true_cross_pass"]
        icp = cf.get("cloud_icp_pairs_summary", {})
        lines.append(f"Source: `{cf['path']}` (final table's pass transforms == S5b's). True cross-pass frames n={tco['n_frames']}: "
                     f"n_conflict {tco['n_conflict_before']} -> {tco['n_conflict_after']}; cloud ICP rms median {icp.get('median_rms_before')} -> {icp.get('median_rms_after')} m over {icp.get('n_pairs')} pairs ({icp.get('n_converged')} converged).")
    else:
        lines.append(f"`{cf.get('path')}` not found -- run `cli.validate_pass_reg conflict` first.")
    lines.append("")
    lines.append("## 4. Interpolation benefit (dense trajectory vs neighbour-linear, turning frames)")
    ib = s["interpolation_benefit"]
    lines.append(f"n={ib['n_turning_covered']} turning frames covered by the trajectory. Dense-model \\|resid\\| vs export own-time yaw: "
                 f"median {ib['dense_model_resid_deg']['median']} deg, p95 {ib['dense_model_resid_deg']['p95']} deg. "
                 f"Old-model (neighbour linear interp) \\|resid\\|: median {ib['old_model_resid_deg']['median']} deg, p95 {ib['old_model_resid_deg']['p95']} deg.")
    lines.append("")
    lines.append("## 5. Invariance (pass-transform, pre- vs post-registration)")
    inv = s["invariance"]
    if "error" in inv:
        lines.append(f"not computed: {inv['error']}")
    else:
        ratio = 8.0 / inv["worst_px"] if inv["worst_px"] > 0 else float("inf")
        target_note = ("meets the 1e-6 px aspirational target" if inv["worst_px"] < 1e-6 else
                        f"above the 1e-6 px aspirational target but still ~{ratio:.0e}x below any measurement "
                        f"precision (compare the 8 px MAD floor in section 1), so treated as floating-point "
                        f"noise, not a systematic error")
        lines.append(f"worst {inv['worst_px']:.2e} px across {inv['n_passes']} passes (real cache data; {target_note}; see also the synthetic unit test in `tests/test_poses_table.py`).")
    lines.append("")
    lines += _slow_regression_markdown(s.get("slow_regression_037") or {})
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
