"""S5 validation / QA (task list in the plan, S5 section): sign-and-convention render check (task 1),
the near-field acceptance metric before/after registration (task 2/3), and the cross-pass silhouette
("double surface") conflict metric of `mapping.quality` before/after (task 2, the metric this whole
step is actually for).

Read-only user of `mapping.seg.nearfield` (the near-field metric definition), `mapping.quality` (the
pass-conflict silhouette residual `_silhouette_points`/`_residual`/`_photo_edges` and its
`GEO_MIN_POINTS`/`CONFLICT_PX` thresholds -- reused, not redefined, so "conflict" means the same thing
here as in `dataset/frame_quality.csv`), and `mapping.render` (`overlay_on_photo`); does not modify any
of them. Kept separate from `pass_reg.py` itself so that module stays free of QA/plotting concerns.

    uv run python -m mapping.cli.validate_pass_reg qa --passes 0,5 --n-frames 3
    uv run python -m mapping.cli.validate_pass_reg metric --frames 200 --out .../nearfield_reg.json
    uv run python -m mapping.cli.validate_pass_reg conflict --workers 6 --out .../conflict_reg.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from .. import compat, geometry, vectors
from ..config import OUT_DIR, R_MAX, ZB_H, ZB_W
from ..frame_select import FrameIndex
from ..pass_reg import PASS_REG_DIR, ROAD_BOUNDARY_CODE, CLS_CURB, Patches, apply_pass_transforms
from ..poses import Poses, load_poses
from ..products import FrameProducts
from ..render import overlay_on_photo
from ..seg.nearfield import ROWS, FLAG_PX

QA_DIR = PASS_REG_DIR / "qa"


def _road_objects():
    compat.ensure_experiments_on_path()
    from common import io_data

    objs = io_data.load_jvf_objects()
    return [o for o in objs if o.jvfcode == ROAD_BOUNDARY_CODE and o.geom_type == "LineString" and len(o.coords) >= 2]


def _band_mask(objs, R, C, fp, scale: float) -> np.ndarray:
    mask, _occ = vectors.render_objects(objs, R, C, fp, {ROAD_BOUNDARY_CODE: 1}, scale=scale)
    return mask


def _draw_band(photo: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = photo.copy()
    d = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    out[d > 0] = color
    return out


def _draw_curb_points(photo: np.ndarray, u: np.ndarray, v: np.ndarray, ok: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    out = photo.copy()
    for x, y in zip(u[ok], v[ok]):
        cv2.circle(out, (int(round(x)), int(round(y))), 2, color, -1)
    return out


def render_qa(pass_ids: list[int] = (0, 5), n_frames: int = 3, rows: tuple[int, int] = ROWS, out_dir: Path = QA_DIR) -> list[dict]:
    """For each pass, `n_frames` evenly spaced frames: three stacked crops (ground rows only) --
    (a) photo + JVF road boundary projected with the EXPORT pose, (b) photo + this pass' own curb
    points (also export pose -- curb points are raw store coordinates, never shifted), (c) photo +
    JVF projected with the pose AFTER the S5 transform (`apply_pass_transforms`); JVF itself is never
    moved -- see module docstring / pass_reg.apply_pass_transforms. If (c) sits closer to the photo
    edge than (a), the transform is corrective; if it sits further, the transform (or its sign) is
    wrong for that pass."""
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = load_poses()
    fi = FrameIndex(poses)
    transforms = json.loads((PASS_REG_DIR / "pass_transforms.json").read_text())
    poses_corr = apply_pass_transforms(poses, transforms)
    fi_corr = FrameIndex(poses_corr)
    objs = _road_objects()
    scale = ZB_W / 8000.0

    report = []
    for pid in pass_ids:
        idx_all = np.flatnonzero(poses.pass_id == pid)
        picks = idx_all[np.linspace(0, len(idx_all) - 1, n_frames).round().astype(int)]
        patches = Patches.load(PASS_REG_DIR / f"patches_p{pid:02d}.npz")
        curb = patches.select(patches.cls == CLS_CURB)
        tr = transforms.get("passes", transforms).get(str(pid))
        for k in picks.tolist():
            photo = cv2.imread(poses.path(k))
            photo = cv2.resize(photo, (ZB_W, ZB_H), interpolation=cv2.INTER_AREA)
            try:
                fp = FrameProducts.load(k)
            except Exception:
                fp = None

            mask_before = _band_mask(objs, fi.R[k], fi.C[k], fp, scale)
            mask_after = _band_mask(objs, fi_corr.R[k], fi_corr.C[k], fp, scale)

            near = curb.c[np.linalg.norm(curb.c[:, :2] - fi.C[k, :2], axis=1) < 25.0]
            u, v, r, el = geometry.world_to_pano(near, fi.R[k], fi.C[k], w=ZB_W, h=ZB_H)
            ok = (r > 0) & (u >= 0) & (u < ZB_W) & (v >= rows[0]) & (v < rows[1])

            row_a = _draw_band(photo, mask_before, (0, 255, 255))  # yellow = JVF, export pose
            row_b = _draw_curb_points(photo, u, v, ok, (0, 0, 255))  # red = own-pass curb pts
            row_c = _draw_band(photo, mask_after, (255, 0, 255))  # magenta = JVF, S5-corrected pose

            crop = np.concatenate([row_a[rows[0]:rows[1]], row_b[rows[0]:rows[1]], row_c[rows[0]:rows[1]]], axis=0)
            label = f"pass {pid} frame {k}  a=JVF/export(yellow) b=own-curb(red) c=JVF/S5-corrected(magenta) t={tr['t'] if tr else None}"
            cv2.putText(crop, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            path = out_dir / f"p{pid:02d}_f{k:04d}.png"
            cv2.imwrite(str(path), crop)
            report.append({"pass_id": pid, "frame": k, "path": str(path), "n_band_before": int((mask_before > 0).sum()), "n_band_after": int((mask_after > 0).sum()), "n_curb_drawn": int(ok.sum())})
    (out_dir / "qa_report.json").write_text(json.dumps(report, indent=1))
    return report


# ------------------------------------------------------------------------------- near-field metric
def _band_edge_distance_from_mask(photo_bgr: np.ndarray, mask: np.ndarray, rows=ROWS) -> dict:
    """Same statistic as `mapping.seg.nearfield.band_edge_distance`, but taking an already-rendered
    band mask (so it can be recomputed for the S5-corrected pose) instead of reading `bands_erp/*.png`
    off disk (those were rendered once, with the export pose only)."""
    g = cv2.GaussianBlur(cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(g, 40, 100) > 0
    dist = ndimage.distance_transform_edt(~edges)
    sel = np.zeros(mask.shape, bool)
    sel[rows[0]:rows[1]] = mask[rows[0]:rows[1]] > 0
    n = int(sel.sum())
    if n < 200:
        return {"n": n, "median_px": None}
    inside = ndimage.distance_transform_edt(sel)
    centre = sel & (inside >= np.maximum(1, ndimage.maximum_filter(inside, 5) - 0.5))
    d = dist[centre]
    return {"n": int(centre.sum()), "median_px": round(float(np.median(d)), 1)}


def run_metric(frames: list[int] | None = None, n_subset: int = 200, out: Path = PASS_REG_DIR / "nearfield_reg.json") -> dict:
    """The acceptance test (task 2): band-to-photo-edge distance (`seg.nearfield`'s own statistic),
    computed per frame with (i) export pose vs the static JVF band [-> should equal `segds/nearfield.json`
    where MIN_PX/rows agree] and (ii) the S5-corrected pose vs the same static JVF band. Frames are a
    fixed, pass-stratified `n_subset` sample of the clean set unless `frames` is given explicitly."""
    poses = load_poses()
    from ..seg.render_labels import clean_frames

    if frames is None:
        clean = np.array(clean_frames())
        rng = np.random.default_rng(0)
        pass_of = poses.pass_id[clean]
        frames = []
        for pid in np.unique(pass_of):
            pf = clean[pass_of == pid]
            take = max(1, round(n_subset * len(pf) / len(clean)))
            frames.extend(rng.choice(pf, size=min(take, len(pf)), replace=False).tolist())
        frames = sorted(set(frames))

    transforms = json.loads((PASS_REG_DIR / "pass_transforms.json").read_text())
    poses_corr = apply_pass_transforms(poses, transforms)
    fi = FrameIndex(poses)
    fi_corr = FrameIndex(poses_corr)
    objs = _road_objects()
    scale = ZB_W / 8000.0

    per_frame = {}
    for k in frames:
        photo = cv2.imread(poses.path(k))
        photo = cv2.resize(photo, (ZB_W, ZB_H), interpolation=cv2.INTER_AREA)
        try:
            fp = FrameProducts.load(k)
        except Exception:
            fp = None
        mask_before = _band_mask(objs, fi.R[k], fi.C[k], fp, scale)
        mask_after = _band_mask(objs, fi_corr.R[k], fi_corr.C[k], fp, scale)
        r_before = _band_edge_distance_from_mask(photo, mask_before)
        r_after = _band_edge_distance_from_mask(photo, mask_after)
        per_frame[int(k)] = {"pass_id": int(poses.pass_id[k]), "before": r_before, "after": r_after}

    by_pass: dict[int, dict] = {}
    for k, r in per_frame.items():
        pid = r["pass_id"]
        by_pass.setdefault(pid, {"before": [], "after": [], "flag_before": 0, "flag_after": 0, "n": 0})
        agg = by_pass[pid]
        agg["n"] += 1
        mb, ma = r["before"]["median_px"], r["after"]["median_px"]
        if mb is not None:
            agg["before"].append(mb)
            agg["flag_before"] += mb > FLAG_PX
        if ma is not None:
            agg["after"].append(ma)
            agg["flag_after"] += ma > FLAG_PX
    summary = {}
    for pid, agg in sorted(by_pass.items()):
        summary[str(pid)] = {
            "n": agg["n"],
            "n_measured_before": len(agg["before"]),
            "n_measured_after": len(agg["after"]),
            "median_px_before": round(float(np.median(agg["before"])), 1) if agg["before"] else None,
            "median_px_after": round(float(np.median(agg["after"])), 1) if agg["after"] else None,
            "n_flagged_before": agg["flag_before"],
            "n_flagged_after": agg["flag_after"],
        }
    all_before = [v["median_px"] for v in [r["before"] for r in per_frame.values()] if v["median_px"] is not None]
    all_after = [v["median_px"] for v in [r["after"] for r in per_frame.values()] if v["median_px"] is not None]
    overall = {
        "n_frames": len(frames),
        "median_px_before": round(float(np.median(all_before)), 1) if all_before else None,
        "median_px_after": round(float(np.median(all_after)), 1) if all_after else None,
        "n_flagged_before": int(sum(1 for v in all_before if v > FLAG_PX)),
        "n_flagged_after": int(sum(1 for v in all_after if v > FLAG_PX)),
        "n_measured_before": len(all_before),
        "n_measured_after": len(all_after),
    }
    out.write_text(json.dumps({"rows": ROWS, "flag_px": FLAG_PX, "overall": overall, "by_pass": summary, "frames": per_frame}, indent=1))
    print(json.dumps(overall, indent=1))
    print(json.dumps(summary, indent=1))
    return {"overall": overall, "by_pass": summary}


# --------------------------------------------------------------------- cross-pass conflict metric
# Task 2: `mapping.quality.assess_frame`'s own "pass_conflict" residual (du2/dv2 -- the second
# silhouette residual, computed from OTHER-pass points gathered in the same +-45 s window as the
# frame's own geometry check, see `quality.py` module docstring) is the metric pairwise registration
# is actually meant to fix (double surfaces), unlike the near-field metric above (which never touches
# a second pass at all). Every frame with `n_edge_other > 0` in `frame_quality.csv` is recomputed here
# with the OTHER-pass points and the frame's own camera pose transformed by the S5 solve (points via
# `PassRegistration.apply` = `T[poses.pass_of_time(gps_time)]`, camera via `apply_pass_transforms`);
# "before" is recomputed the same way with the identity transform (T=0) rather than read back from the
# CSV, so before/after use byte-identical candidate gathering / edge images and differ only in the
# transform applied -- a fair paired comparison.
#
# Caveat found while building this (see `pass_reg.py` module docstring and this file's report): the
# frame-level own/other split in `quality.assess_frame` is a fixed +-2 s pad around each PHOTO pass's
# own [t0, t1], not `poses.pass_of_time`'s gap-midpoint pass boundary. Near a pass' start/end (vehicle
# slowing/turning, camera not yet firing but the scanner still running) points quality.py calls
# "other" can still resolve to the SAME pass under `pass_of_time` -- for those frames pairwise
# registration moves the camera and those points by the identical transform, so du2/dv2 are invariant
# by construction (see the geometry invariance test) and correctly do not change. `frac_true_other`
# records, per frame, the fraction of its "other" candidate points that resolve to a genuinely
# different pass under `pass_of_time`; the headline before/after comparison is restricted to frames
# with `frac_true_other >= MIN_FRAC_TRUE_OTHER`, with the full (unfiltered) numbers reported alongside
# for transparency.
MIN_FRAC_TRUE_OTHER = 0.5

_CG: dict = {}


def _conflict_init(transforms_path: str) -> None:
    from ..cloud_store import CloudStore, PassRegistration
    from ..poses import load_poses as _lp
    from ..vehicle_mask import MASK_PATH, VehicleMask

    poses = _lp()
    transforms = json.loads(Path(transforms_path).read_text())
    _CG["poses"] = poses
    _CG["fi"] = FrameIndex(poses)
    poses_corr = apply_pass_transforms(poses, transforms)
    _CG["fi_corr"] = FrameIndex(poses_corr)
    _CG["store"] = CloudStore()
    _CG["reg"] = PassRegistration(transforms, poses=poses)
    _CG["vm"] = VehicleMask() if MASK_PATH.exists() else None
    p = poses.pass_id
    _CG["pass_t0"] = np.array([poses.t[np.flatnonzero(p == q)].min() for q in range(int(p.max()) + 1)])
    _CG["pass_t1"] = np.array([poses.t[np.flatnonzero(p == q)].max() for q in range(int(p.max()) + 1)])


def _conflict_job(k: int) -> dict:
    from .. import quality as qmod
    from ..products import TIME_WINDOW_S, gather_candidates

    poses, fi, fi_corr = _CG["poses"], _CG["fi"], _CG["fi_corr"]
    store, reg, vm = _CG["store"], _CG["reg"], _CG["vm"]
    pass_t0, pass_t1 = _CG["pass_t0"], _CG["pass_t1"]
    p = int(poses.pass_id[k])
    R, C = fi.R[k], fi.C[k]
    dominant, frac_true_other = None, 0.0
    try:
        xyz, _pid = gather_candidates(store, C, R_MAX)
        parts = store.query_disc(float(C[0]), float(C[1]), R_MAX)
        gps = np.concatenate([np.asarray(store.tile(t.name).gps_time[rows]) for t, rows in parts]) if parts else np.empty(0)
        win = np.abs(gps - poses.t[k]) <= TIME_WINDOW_S
        own = win & (gps >= pass_t0[p] - 2) & (gps <= pass_t1[p] + 2)
        other = win & ~own
        n_other = int(other.sum())
        dt_img, edge_idx, valid = qmod._photo_edges(poses, k, vm)

        e_before = qmod._silhouette_points(xyz[other], R, C) if n_other > 20000 else xyz[:0]
        n2b, du2b, dv2b, *_ = qmod._residual(e_before, R, C, dt_img, edge_idx, valid)

        if n_other:
            gps_other = gps[other]
            other_pass = poses.pass_of_time(gps_other)
            uniq, counts = np.unique(other_pass, return_counts=True)
            dominant = int(uniq[np.argmax(counts)])
            same = int(counts[uniq == p].sum()) if (uniq == p).any() else 0
            frac_true_other = float((counts.sum() - same) / counts.sum())
            xyz_after = reg.apply(xyz[other], gps_other)
        else:
            xyz_after = xyz[:0]

        Rc, Cc = fi_corr.R[k], fi_corr.C[k]
        e_after = qmod._silhouette_points(xyz_after, Rc, Cc) if n_other > 20000 else xyz_after[:0]
        n2a, du2a, dv2a, *_ = qmod._residual(e_after, Rc, Cc, dt_img, edge_idx, valid)
        store.release()
        err = None
    except Exception as e:  # keep the manifest complete, mirrors quality._job
        n_other = 0
        n2b = n2a = 0
        du2b = dv2b = du2a = dv2a = np.nan
        err = f"{type(e).__name__}: {e}"

    conflict_b = bool(n2b >= qmod.GEO_MIN_POINTS and np.isfinite(du2b) and (abs(du2b) > qmod.CONFLICT_PX or abs(dv2b) > qmod.CONFLICT_PX))
    conflict_a = bool(n2a >= qmod.GEO_MIN_POINTS and np.isfinite(du2a) and (abs(du2a) > qmod.CONFLICT_PX or abs(dv2a) > qmod.CONFLICT_PX))
    return {
        "frame": k,
        "own_pass": p,
        "other_pass_dominant": dominant,
        "frac_true_other": round(frac_true_other, 3),
        "n_other_pts": n_other,
        "n2_before": int(n2b),
        "du2_before": None if not np.isfinite(du2b) else round(float(du2b), 3),
        "dv2_before": None if not np.isfinite(dv2b) else round(float(dv2b), 3),
        "conflict_before": conflict_b,
        "n2_after": int(n2a),
        "du2_after": None if not np.isfinite(du2a) else round(float(du2a), 3),
        "dv2_after": None if not np.isfinite(dv2a) else round(float(dv2a), 3),
        "conflict_after": conflict_a,
        "error": err,
    }


def run_conflict_metric(
    frames: list[int] | None = None,
    workers: int = 6,
    out: Path = PASS_REG_DIR / "conflict_reg.json",
    dataset_dir: Path = OUT_DIR / "dataset",
) -> dict:
    """Task 2: cross-pass silhouette conflict (`mapping.quality`'s `pass_conflict` residual) before vs
    after the S5 pass-registration transform, over every frame `dataset/frame_quality.csv` records with
    `n_edge_other > 0`. See the module-docstring-adjacent comment above for the before/after definition
    and the `frac_true_other` caveat. Also folds in the pairwise-ICP cloud-only point-to-plane rms
    already recorded per overlap pair in `pairs.json` (task 2's third leg)."""
    from multiprocessing import Pool

    if frames is None:
        rows = list(csv.DictReader(open(dataset_dir / "frame_quality.csv")))
        frames = sorted(int(r["frame"]) for r in rows if int(r["n_edge_other"]) > 0)

    transforms_path = str(PASS_REG_DIR / "pass_transforms.json")
    results: list[dict] = []
    with Pool(min(workers, 6), initializer=_conflict_init, initargs=(transforms_path,)) as pool:
        for r in pool.imap_unordered(_conflict_job, frames, chunksize=2):
            results.append(r)
    results.sort(key=lambda r: r["frame"])

    def _agg(rows: list[dict]) -> dict:
        du_b = [abs(r["du2_before"]) for r in rows if r["du2_before"] is not None]
        dv_b = [abs(r["dv2_before"]) for r in rows if r["dv2_before"] is not None]
        du_a = [abs(r["du2_after"]) for r in rows if r["du2_after"] is not None]
        dv_a = [abs(r["dv2_after"]) for r in rows if r["dv2_after"] is not None]
        return {
            "n_frames": len(rows),
            "n_measured_before": len(du_b),
            "n_measured_after": len(du_a),
            "median_abs_du2_before": round(float(np.median(du_b)), 2) if du_b else None,
            "median_abs_dv2_before": round(float(np.median(dv_b)), 2) if dv_b else None,
            "median_abs_du2_after": round(float(np.median(du_a)), 2) if du_a else None,
            "median_abs_dv2_after": round(float(np.median(dv_a)), 2) if dv_a else None,
            "n_conflict_before": sum(1 for r in rows if r["conflict_before"]),
            "n_conflict_after": sum(1 for r in rows if r["conflict_after"]),
        }

    true_other = [r for r in results if r["frac_true_other"] >= MIN_FRAC_TRUE_OTHER]
    same_pass_only = [r for r in results if r["frac_true_other"] < MIN_FRAC_TRUE_OTHER]
    overall_all = _agg(results)
    overall_true_other = _agg(true_other)
    overall_same_pass = _agg(same_pass_only)

    by_pair: dict[str, list[dict]] = {}
    for r in true_other:
        if r["other_pass_dominant"] is None:
            continue
        a, b = sorted((r["own_pass"], r["other_pass_dominant"]))
        by_pair.setdefault(f"{a}_{b}", []).append(r)
    pair_summary = {k: _agg(v) for k, v in sorted(by_pair.items())}

    cloud_icp = {}
    pairs_path = PASS_REG_DIR / "pairs.json"
    if pairs_path.exists():
        raw = json.loads(pairs_path.read_text())
        for k, v in raw.items():
            cloud_icp[k] = {"n": v.get("n"), "rms_before": v.get("rms_before"), "rms_after": v.get("rms_after"), "converged": v.get("converged")}
        rb = [v["rms_before"] for v in cloud_icp.values() if v["rms_before"] is not None]
        ra = [v["rms_after"] for v in cloud_icp.values() if v["rms_after"] is not None and v["converged"]]
        cloud_icp_summary = {
            "n_pairs": len(cloud_icp),
            "n_converged": sum(1 for v in cloud_icp.values() if v["converged"]),
            "median_rms_before": round(float(np.median(rb)), 4) if rb else None,
            "median_rms_after": round(float(np.median(ra)), 4) if ra else None,
        }
    else:
        cloud_icp_summary = {}

    report = {
        "min_frac_true_other": MIN_FRAC_TRUE_OTHER,
        "conflict_px": qmod_thresholds(),
        "overall_all_n2gt0_frames": overall_all,
        "overall_true_cross_pass": overall_true_other,
        "overall_same_pass_only": overall_same_pass,
        "by_pass_pair": pair_summary,
        "cloud_icp_pairs_summary": cloud_icp_summary,
        "cloud_icp_pairs": cloud_icp,
        "frames": {r["frame"]: r for r in results},
    }
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"overall_true_cross_pass": overall_true_other, "overall_same_pass_only": overall_same_pass, "cloud_icp_pairs_summary": cloud_icp_summary}, indent=1))
    return report


def qmod_thresholds() -> dict:
    from .. import quality as qmod

    return {"conflict_px": qmod.CONFLICT_PX, "geo_min_points": qmod.GEO_MIN_POINTS}


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "qa"
    if cmd == "qa":
        passes = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "5"])]
        render_qa(passes)
    elif cmd == "metric":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        run_metric(n_subset=n)
    elif cmd == "conflict":
        w = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        run_conflict_metric(workers=w)
