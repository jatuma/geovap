"""S3: dense trajectory from scanner planes, camera rig, and the resulting pose table.

uv run python -m mapping.cli.build_trajectory --passes 5 --workers 1
uv run python -m mapping.cli.build_trajectory --passes all --workers 8

Runs `mapping.trajectory.build_pass_trajectory` per pass (one pass per worker, `Pool` capped at
`--workers`, `store.release()` after each), fits the camera rig from clean straight frames
(`dataset/clean_frames.json`, own-pass |yaw_rate| < 5), saves `trajectory.npz`/`.json`, and writes
the S3 pose table `poses_traj.csv` (+ provenance json naming the trajectory): every frame covered by
the trajectory gets `traj.camera_pose(t + dt, pass_id)` (status "traj"), everything else keeps its
export pose (status "kept"). Also writes per-pass/per-head diagnostics (JSON + PNGs) under
`Geovap_cache/out/poses/traj_diag/`.

S3b -- orientation-only salvage (S3's scan-CENTRE fit S(t) is degenerate on this dataset, see
`mapping/trajectory.py`'s module docstring; the plane-fit ORIENTATION is not):

uv run python -m mapping.cli.build_trajectory --rot-only --passes 5 --workers 1
uv run python -m mapping.cli.build_trajectory --rot-only --passes all --workers 6
uv run python -m mapping.cli.build_trajectory --validate-rot

`--rot-only` runs `mapping.trajectory.build_pass_orientation` per pass (never calls
`fit_scan_centre`, so it is much cheaper than the default path), fits `(R_cs, dt_s)` on rotations
only (`fit_rotation_rig`: R_cs from clean straight frames, dt_s from the yaw residual on turning
frames -- the position-based dt scan was flat), and writes `trajectory_rot.npz`/`.json` +
`poses_traj_rot.csv`/`.json` (status "traj_rot"). `--validate-rot` compares the export and rot-only
pose tables on turning vs. straight frames with two independent metrics (`Aligner.colour_de`,
`quality._silhouette_points`/`_residual`) and writes `traj_diag/validate_rot.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..config import CLEAN_FRAMES_JSON, POSES_DIR, R_MAX
from ..poses import Poses, load_poses, write_pose_table
from ..products import TIME_WINDOW_S, gather_candidates
from ..quality import _photo_edges, _residual, _silhouette_points, yaw_rates
from ..trajectory import (
    CamSensorRig,
    Trajectory,
    build_pass_orientation,
    build_pass_trajectory,
    fit_camera_from_trajectory,
    fit_rotation_rig,
)

DIAG_DIR = POSES_DIR / "traj_diag"
N_WORKERS_DEFAULT = 8
ROT_ONLY_WORKERS_DEFAULT = 6  # a background job (refine_poses) already runs 8 workers; cap ours lower
YAW_RATE_STRAIGHT = 5.0
YAW_RATE_TURNING_FIT = 3.0  # S3b step 2: dt is observable where yaw changes fast
YAW_RATE_TURNING_VALIDATE = 8.0  # S3b step 4: the validation set (matches quality.py's "turning" cut)


def _run_one(args) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, list, dict]:
    pass_id, stride = args
    from ..cloud_store import CloudStore  # re-imported per worker (fork-safe)
    from ..poses import load_poses as _load_poses

    store = CloudStore()
    poses = _load_poses("export")
    t0 = time.time()
    t_out, S_out, R_out, segments, diag = build_pass_trajectory(store, poses, pass_id, stride=stride)
    diag["wall_s"] = time.time() - t0
    store.release()
    return pass_id, t_out, S_out, R_out, segments, diag


def _run_one_rot(args) -> tuple[int, np.ndarray, np.ndarray, list, dict]:
    (pass_id,) = args
    from ..cloud_store import CloudStore  # re-imported per worker (fork-safe)
    from ..poses import load_poses as _load_poses

    store = CloudStore()
    poses = _load_poses("export")
    t0 = time.time()
    t_out, quat_out, segments, diag = build_pass_orientation(store, poses, pass_id)
    diag["wall_s"] = time.time() - t0
    store.release()
    return pass_id, t_out, quat_out, segments, diag


def clean_straight_frame_idx(poses: Poses, pass_id: int | None = None) -> np.ndarray:
    """Indices into `poses` of clean, straight (|yaw_rate| < 5 deg/s) frames, optionally restricted
    to one pass. `dataset/clean_frames.json`'s "clean" class already enforces this yaw-rate cut, so
    the check here is redundant defence, not a second filter."""
    clean = set(json.loads(CLEAN_FRAMES_JSON.read_text())["clean"])
    yr = np.nan_to_num(yaw_rates(poses), nan=0.0)
    idx = np.array(sorted(clean), dtype=np.int64)
    idx = idx[np.abs(yr[idx]) < YAW_RATE_STRAIGHT]
    if pass_id is not None:
        idx = idx[poses.pass_id[idx] == pass_id]
    return idx


def turning_frame_idx(poses: Poses, pass_id: int | None = None, min_rate: float = YAW_RATE_TURNING_VALIDATE) -> np.ndarray:
    """Indices with |yaw_rate| > `min_rate` deg/s (own-pass central difference), optionally restricted
    to one pass. Deliberately NOT filtered by `dataset/clean_frames.json`'s "clean" class: that class
    enforces |yaw_rate| < 5 deg/s (`YAW_RATE_STRAIGHT`), so filtering by it would remove almost
    exactly the turning frames this is meant to find (S3b step 2's dt fit, step 4's validation set)."""
    yr = np.nan_to_num(yaw_rates(poses), nan=0.0)
    idx = np.flatnonzero(np.abs(yr) > min_rate)
    if pass_id is not None:
        idx = idx[poses.pass_id[idx] == pass_id]
    return idx


def _covered_mask(poses: Poses) -> np.ndarray:
    """Bool[len(poses)]: frames `poses.traj` actually covers (per-pass, via `Trajectory.covers`)."""
    out = np.zeros(len(poses), dtype=bool)
    if poses.traj is None:
        return out
    for p in np.unique(poses.pass_id):
        sel = np.flatnonzero(poses.pass_id == p)
        out[sel] = np.asarray(poses.traj.covers(int(p), poses.t[sel]))
    return out


def build(passes: list[int] | None = None, workers: int = N_WORKERS_DEFAULT, stride: int = 5, out_csv: Path = POSES_DIR / "poses_traj.csv", log=print) -> Path:
    poses = load_poses("export")
    all_passes = sorted(int(p) for p in np.unique(poses.pass_id))
    passes = all_passes if passes is None else passes
    workers = min(workers, 8, len(passes))
    log(f"passes: {passes}, workers: {workers}, stride: {stride}")

    per_pass = {}
    diags = {}
    jobs = [(p, stride) for p in passes]
    t0 = time.time()
    if workers <= 1:
        results = [_run_one(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            results = pool.map(_run_one, jobs)
    for pass_id, t_out, S_out, R_out, segments, diag in results:
        per_pass[pass_id] = (t_out, S_out, R_out, segments)
        diags[pass_id] = diag
        log(f"pass {pass_id}: {diag['n_points']} pts, {diag['n_samples_200hz']} samples @ 200Hz, "
            f"{diag['n_segments']} segments, head1 a={diag['head1']['a_deg_per_unit']:.4f} "
            f"resid={diag['head1']['centre_resid_deg']:.3f} deg, head_offset_std={diag['head_offset_mm_std']:.1f} mm, "
            f"{diag['wall_s']:.1f} s")
    log(f"trajectory fit: {time.time() - t0:.1f} s total")

    # -------------------------------------------------------------- concatenate + rig
    from scipy.spatial.transform import Rotation

    t_all, S_all, quat_all, passid_all = [], [], [], []
    segments_all = {}
    for pass_id, (t_out, S_out, R_out, segments) in per_pass.items():
        if len(t_out) == 0:
            segments_all[pass_id] = []
            continue
        t_all.append(t_out)
        S_all.append(S_out)
        quat_all.append(Rotation.from_matrix(R_out).as_quat())
        passid_all.append(np.full(len(t_out), pass_id, dtype=np.int32))
        segments_all[pass_id] = segments
    if not t_all:
        raise RuntimeError("no pass produced any trajectory samples")
    t_all = np.concatenate(t_all)
    S_all = np.concatenate(S_all, axis=0)
    quat_all = np.concatenate(quat_all, axis=0)
    passid_all = np.concatenate(passid_all)

    clean_idx = clean_straight_frame_idx(poses)
    log(f"clean straight frames (all passes): {len(clean_idx)}")
    global_rig = fit_camera_from_trajectory(t_all, S_all, Rotation.from_quat(quat_all).as_matrix(), poses, clean_idx)
    log(f"global rig: dt_s={global_rig.dt_s:.4f} R_cs~I-dev(deg)={_rot_dev_deg(global_rig.R_cs):.3f} "
        f"l_cs={global_rig.l_cs} rms_m={global_rig.stats['rms_m']:.3f} n={global_rig.stats['n_frames']}")

    # per-pass rig spread check (re-anchor per pass only if disagreement > 0.2 m)
    per_pass_rig = {}
    for pass_id, (t_out, S_out, R_out, segments) in per_pass.items():
        idx = clean_straight_frame_idx(poses, pass_id)
        if len(idx) < 8 or len(t_out) == 0:
            continue
        try:
            per_pass_rig[pass_id] = fit_camera_from_trajectory(t_out, S_out, R_out, poses, idx, dt_scan=(global_rig.dt_s - 0.05, global_rig.dt_s + 0.05, 0.005))
        except RuntimeError:
            continue
    spread_m = 0.0
    if len(per_pass_rig) >= 2:
        l_cs_all = np.stack([r.l_cs for r in per_pass_rig.values()])
        spread_m = float(np.max(np.linalg.norm(l_cs_all - l_cs_all.mean(0), axis=1)))
    log(f"per-pass l_cs spread: {spread_m:.3f} m over {len(per_pass_rig)} passes with >=8 clean straight frames")
    use_per_pass = spread_m > 0.2 and len(per_pass_rig) >= 2
    if use_per_pass:
        log("per-pass l_cs disagree by > 0.2 m: re-anchoring the rig per pass")
        rigs = {p: per_pass_rig.get(p, global_rig) for p in per_pass.keys()}
    else:
        rigs = {p: global_rig for p in per_pass.keys()}

    traj = Trajectory(t=t_all, pass_id=passid_all, S=S_all, quat=quat_all, segments=segments_all, rigs=rigs,
                       meta={"passes": passes, "stride": stride, "per_pass_rig_spread_m": spread_m, "used_per_pass_rig": use_per_pass})
    npz_path = POSES_DIR / "trajectory.npz"
    traj.save(npz_path)
    log(f"saved {npz_path}")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    (DIAG_DIR / "diag.json").write_text(json.dumps(_jsonable_diag(diags, global_rig, per_pass_rig, spread_m), indent=1))

    # -------------------------------------------------------------- S3 pose table
    origin = poses.origin.copy()
    roll = poses.roll.copy()
    pitch = poses.pitch.copy()
    yaw = poses.yaw.copy()
    status = np.array(["kept"] * len(poses), dtype=object)
    dt_col = np.zeros(len(poses))
    for pass_id in per_pass.keys():
        rig = rigs[pass_id]
        sel = np.flatnonzero(poses.pass_id == pass_id)
        if len(sel) == 0:
            continue
        tq = poses.t[sel] + rig.dt_s
        cov = np.asarray(traj.covers(pass_id, tq))
        if not cov.any():
            continue
        idx_cov = sel[cov]
        C, r, p, y = traj.camera_pose(tq[cov], pass_id)
        origin[idx_cov] = C
        roll[idx_cov] = r
        pitch[idx_cov] = p
        yaw[idx_cov] = y
        status[idx_cov] = "traj"
        dt_col[idx_cov] = rig.dt_s
    poses_out = replace(poses, origin=origin, roll=roll, pitch=pitch, yaw=yaw, source="traj")
    n_traj = int((status == "traj").sum())
    log(f"pose table: {n_traj}/{len(poses)} frames get the trajectory pose, {len(poses) - n_traj} keep export")

    provenance = {
        "stage": "S3",
        "trajectory": "trajectory.npz",
        "passes": passes,
        "stride": stride,
        "n_traj_frames": n_traj,
        "n_kept_frames": int(len(poses) - n_traj),
        "global_rig": {"dt_s": global_rig.dt_s, "l_cs": global_rig.l_cs.tolist(), "rms_m": global_rig.stats["rms_m"]},
        "per_pass_rig_spread_m": spread_m,
        "used_per_pass_rig": use_per_pass,
    }
    write_pose_table(poses_out, {"status": status, "dt_s": dt_col}, provenance, out_csv)
    log(f"wrote {out_csv}")

    diag_numbers = diagnostics(poses, poses_out, status, diags, global_rig, DIAG_DIR, log=log)
    (DIAG_DIR / "diag.json").write_text(json.dumps({**_jsonable_diag(diags, global_rig, per_pass_rig, spread_m), "comparison": diag_numbers}, indent=1))
    return out_csv


def diagnostics(poses: Poses, poses_out: Poses, status: np.ndarray, diags: dict, global_rig: CamSensorRig, out_dir: Path, log=print) -> dict:
    """Trajectory-vs-export comparison at the frame times covered by the trajectory: position (E, N,
    H) and yaw differences, split into turning (|yaw_rate| > 8 deg/s) and straight, plus the dt-scan
    curve and the plane-RMS histograms. Returns the summary numbers; also writes PNGs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cov = status == "traj"
    yr = np.nan_to_num(yaw_rates(poses), nan=0.0)
    turning = cov & (np.abs(yr) > 8.0)
    straight = cov & ~turning

    dE = poses_out.origin[:, 0] - poses.origin[:, 0]
    dN = poses_out.origin[:, 1] - poses.origin[:, 1]
    dH = poses_out.origin[:, 2] - poses.origin[:, 2]
    dpos = np.linalg.norm(np.stack([dE, dN, dH], axis=1), axis=1)
    dyaw = ((poses_out.yaw - poses.yaw + 180.0) % 360.0) - 180.0

    def stats(mask):
        if not mask.any():
            return {"n": 0}
        return {
            "n": int(mask.sum()),
            "dE_median_m": float(np.median(dE[mask])), "dE_p95_m": float(np.percentile(np.abs(dE[mask]), 95)),
            "dN_median_m": float(np.median(dN[mask])), "dN_p95_m": float(np.percentile(np.abs(dN[mask]), 95)),
            "dH_median_m": float(np.median(dH[mask])), "dH_p95_m": float(np.percentile(np.abs(dH[mask]), 95)),
            "dpos_median_m": float(np.median(dpos[mask])), "dpos_p95_m": float(np.percentile(dpos[mask], 95)),
            "dyaw_median_deg": float(np.median(dyaw[mask])), "dyaw_p95_deg": float(np.percentile(np.abs(dyaw[mask]), 95)),
        }

    summary = {"straight": stats(straight), "turning": stats(turning), "n_covered": int(cov.sum())}
    log(f"KEY QUESTION -- export vs trajectory at turning frames (n={summary['turning'].get('n', 0)}): "
        f"median |dpos|={summary['turning'].get('dpos_median_m', float('nan')):.3f} m, "
        f"p95={summary['turning'].get('dpos_p95_m', float('nan')):.3f} m, "
        f"median |dyaw|={abs(summary['turning'].get('dyaw_median_deg', float('nan'))):.3f} deg, "
        f"p95={summary['turning'].get('dyaw_p95_deg', float('nan')):.3f} deg")
    log(f"straight frames (n={summary['straight'].get('n', 0)}): median |dpos|={summary['straight'].get('dpos_median_m', float('nan')):.3f} m, "
        f"median |dyaw|={abs(summary['straight'].get('dyaw_median_deg', float('nan'))):.3f} deg")

    # correlation vs Aligner's yaw_offset_deg on turning frames (align.json)
    align_path = POSES_DIR / "align.json"
    corr = None
    if align_path.exists() and turning.any():
        align = json.loads(align_path.read_text())
        by_frame = {int(r["frame"]): r for r in align if not r.get("suspicious", False)}
        idx = np.flatnonzero(turning)
        a_dyaw, t_dyaw = [], []
        for i in idx:
            r = by_frame.get(int(i))
            if r is None:
                continue
            a_dyaw.append(r["yaw_offset_deg"])
            t_dyaw.append(dyaw[i])
        if len(a_dyaw) >= 5:
            corr = float(np.corrcoef(a_dyaw, t_dyaw)[0, 1])
            log(f"traj-minus-export yaw vs Aligner yaw_offset_deg on turning frames: r={corr:.3f} (n={len(a_dyaw)})")
    summary["corr_dyaw_vs_align_yaw_offset"] = corr

    # -------------------------------------------------------------- plots
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist([dpos[straight], dpos[turning]] if turning.any() else [dpos[straight]], bins=30, label=["straight", "turning"] if turning.any() else ["straight"], stacked=False, alpha=0.7)
    axes[0].set_xlabel("|export - trajectory| position (m)")
    axes[0].legend()
    axes[1].hist([np.abs(dyaw[straight]), np.abs(dyaw[turning])] if turning.any() else [np.abs(dyaw[straight])], bins=30, label=["straight", "turning"] if turning.any() else ["straight"], alpha=0.7)
    axes[1].set_xlabel("|export - trajectory| yaw (deg)")
    axes[1].legend()
    curve = global_rig.stats.get("dt_scan_curve", [])
    if curve:
        dts, costs = zip(*curve)
        axes[2].plot(dts, costs, ".-")
        axes[2].axvline(global_rig.dt_s, color="r", ls="--", label=f"dt={global_rig.dt_s:.3f}s")
        axes[2].set_xlabel("dt (s)")
        axes[2].set_ylabel("rig fit residual (m rms)")
        axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pos_yaw_dt.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for h, ax in zip((1, 2), axes):
        counts = np.zeros(20)
        edges = np.linspace(0, 10, 21)
        for d in diags.values():
            hd = d.get(f"head{h}", {})
            if "rms_mm_hist_counts" in hd:
                counts = counts + np.array(hd["rms_mm_hist_counts"])
                edges = np.array(hd["rms_mm_hist_edges"])
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge")
        ax.set_xlabel(f"head {h} plane RMS (mm)")
    fig.tight_layout()
    fig.savefig(out_dir / "plane_rms_hist.png", dpi=110)
    plt.close(fig)

    return summary


def _rot_dev_deg(R: np.ndarray) -> float:
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _jsonable_diag(diags, global_rig, per_pass_rig, spread_m) -> dict:
    def j(x):
        if isinstance(x, dict):
            return {k: j(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [j(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    return {
        "passes": j(diags),
        "global_rig": {"dt_s": global_rig.dt_s, "l_cs": j(global_rig.l_cs), "stats": j(global_rig.stats)},
        "per_pass_rig": {str(p): {"dt_s": r.dt_s, "l_cs": j(r.l_cs)} for p, r in per_pass_rig.items()},
        "per_pass_l_cs_spread_m": spread_m,
    }


# =============================================================================== S3b: rot-only build
def build_rot_only(passes: list[int] | None = None, workers: int = ROT_ONLY_WORKERS_DEFAULT, out_csv: Path = POSES_DIR / "poses_traj_rot.csv", log=print) -> Path:
    """S3b end-to-end: `build_pass_orientation` per pass (never `fit_scan_centre`) -> `fit_rotation_rig`
    (R_cs on clean straight frames, dt_s on the yaw residual of turning frames) -> `trajectory_rot.npz`
    + the S3b pose table (status "traj_rot" for covered frames, export "kept" otherwise)."""
    poses = load_poses("export")
    all_passes = sorted(int(p) for p in np.unique(poses.pass_id))
    passes = all_passes if passes is None else passes
    workers = max(1, min(workers, ROT_ONLY_WORKERS_DEFAULT, len(passes)))
    log(f"[rot-only] passes: {passes}, workers: {workers}")

    per_pass, diags = {}, {}
    jobs = [(p,) for p in passes]
    t0 = time.time()
    if workers <= 1:
        results = [_run_one_rot(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            results = pool.map(_run_one_rot, jobs)
    for pass_id, t_out, quat_out, segments, diag in results:
        per_pass[pass_id] = (t_out, quat_out, segments)
        diags[pass_id] = diag
        h1, h2 = diag.get("head1", {}), diag.get("head2", {})
        log(f"[rot-only] pass {pass_id}: {diag['n_points']} pts, {diag.get('n_samples_200hz', 0)} samples @ 200Hz, "
            f"{diag.get('n_segments', 0)} segments, head1 rms={h1.get('rms_mm_median', float('nan')):.3f} mm "
            f"cont={h1.get('normal_continuity_dot_median', float('nan')):.5f}, head2 rms={h2.get('rms_mm_median', float('nan')):.3f} mm "
            f"cont={h2.get('normal_continuity_dot_median', float('nan')):.5f}, {diag['wall_s']:.1f} s")
    log(f"[rot-only] orientation fit: {time.time() - t0:.1f} s total")

    # -------------------------------------------------------------- per-pass rig (R_cs, dt_s)
    # `fit_orientation`'s sensor frame is defined PER PASS: `R_s(t_ref) = I` at that pass's own first
    # common (head-1/head-2) window, an arbitrary rotation with no relation to any other pass's
    # reference window. Confirmed on real data (all 30 passes, see the honest-negative record this
    # produced the first time it was tried): pooling `t_all`/`quat_all` across passes into ONE
    # `fit_rotation_rig` call gave a catastrophic 68 deg MEDIAN rotation residual (p95 145 deg) on the
    # very straight frames it was fit on, and turning-frame yaw differences up to 167 deg p95 -- not
    # noise, a real cross-pass reference-frame mismatch (the R_cs euler angles' first two components,
    # which correspond to the rig's genuine physical mounting, came out similar to a single-pass fit;
    # only the third -- the yaw-like rotation about the near-vertical axis -- differed wildly, exactly
    # what an arbitrary per-pass heading reference would do). So R_cs (and, since evaluating dt's yaw
    # -residual cost needs a *consistent* R_cs, dt_s too) is fit independently per pass here, never
    # pooled; `dt_s` is a genuine physical camera<->scanner clock offset with no reference-frame
    # ambiguity, so its per-pass values are logged as a consistency check (they should cluster).
    rigs = {}
    per_pass_dt = []
    for pass_id, (t_out, quat_out, segments) in per_pass.items():
        straight_idx_p = clean_straight_frame_idx(poses, pass_id=pass_id)
        turning_idx_p = turning_frame_idx(poses, pass_id=pass_id, min_rate=YAW_RATE_TURNING_FIT)
        if len(t_out) == 0 or len(straight_idx_p) < 8 or len(turning_idx_p) < 5:
            log(f"[rot-only] pass {pass_id}: too few clean frames for its own rig (straight={len(straight_idx_p)}, "
                f"turning={len(turning_idx_p)}) -- excluded, its frames keep the export pose")
            continue
        try:
            rig_p = fit_rotation_rig(t_out, quat_out, poses, straight_idx_p, turning_idx_p)
        except RuntimeError as exc:
            log(f"[rot-only] pass {pass_id}: rig fit failed ({exc}) -- excluded, its frames keep the export pose")
            continue
        rigs[pass_id] = rig_p
        per_pass_dt.append(rig_p.dt_s)
        ex_p = rig_p.stats["euler_R_cs_deg"]
        log(f"[rot-only] pass {pass_id} rig: dt_s={rig_p.dt_s:.4f} R_cs euler(xyz,deg)=({ex_p[0]:.3f},{ex_p[1]:.3f},{ex_p[2]:.3f}) "
            f"resid_median={rig_p.stats['resid_deg_median']:.4f} deg p95={rig_p.stats['resid_deg_p95']:.4f} deg "
            f"n_straight={rig_p.stats['n_straight']} n_turning={rig_p.stats['n_turning']}")

    if not rigs:
        raise RuntimeError("[rot-only] no pass had enough clean frames for its own rig fit")
    dt_arr = np.array(per_pass_dt)
    log(f"[rot-only] per-pass dt_s consistency check ({len(dt_arr)} passes): median={np.median(dt_arr):.4f} "
        f"mean={dt_arr.mean():.4f} std={dt_arr.std():.4f} min={dt_arr.min():.4f} max={dt_arr.max():.4f} "
        f"(dt_s is reference-frame-independent, so a tight cluster is the expected physical signature; "
        f"R_cs is NOT pooled across passes -- see above)")

    segments_all = {p: (segs if p in rigs else []) for p, (t_out, quat_out, segs) in per_pass.items()}
    t_all = np.concatenate([t_out for t_out, _, _ in per_pass.values() if len(t_out)]) if per_pass else np.empty(0)
    quat_all = np.concatenate([quat_out for _, quat_out, _ in per_pass.values() if len(quat_out)], axis=0) if per_pass else np.empty((0, 4))
    passid_all = np.concatenate([np.full(len(t_out), p, dtype=np.int32) for p, (t_out, _, _) in per_pass.items() if len(t_out)]) if per_pass else np.empty(0, dtype=np.int32)

    traj = Trajectory(
        t=t_all, pass_id=passid_all, S=np.zeros((len(t_all), 3)), quat=quat_all,
        segments=segments_all, rigs=rigs, mode="rot_only",
        lin_t=poses.t.copy(), lin_origin=poses.origin.copy(), lin_roll=poses.roll.copy(),
        lin_pitch=poses.pitch.copy(), lin_yaw=poses.yaw.copy(), lin_pass_id=poses.pass_id.copy(),
        meta={
            "passes": passes, "note": "S3b orientation-only trajectory; fit_scan_centre never called; rig fit PER PASS (see build_rot_only)",
            "passes_with_rig": sorted(rigs.keys()), "per_pass_dt_s_std": float(dt_arr.std()), "per_pass_dt_s_median": float(np.median(dt_arr)),
        },
    )
    npz_path = POSES_DIR / "trajectory_rot.npz"
    traj.save(npz_path)
    log(f"[rot-only] saved {npz_path}")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    (DIAG_DIR / "diag_rot.json").write_text(json.dumps(_jsonable_diag_rot(diags, rigs), indent=1))

    # -------------------------------------------------------------- S3b pose table
    origin = poses.origin.copy()
    roll = poses.roll.copy()
    pitch = poses.pitch.copy()
    yaw = poses.yaw.copy()
    status = np.array(["kept"] * len(poses), dtype=object)
    dt_col = np.zeros(len(poses))
    for pass_id, rig in rigs.items():
        sel = np.flatnonzero(poses.pass_id == pass_id)
        if len(sel) == 0:
            continue
        tq = poses.t[sel]
        cov = np.asarray(traj.covers(pass_id, tq))
        if not cov.any():
            continue
        idx_cov = sel[cov]
        C, r, p, y = traj.camera_pose(tq[cov], pass_id)
        origin[idx_cov] = C
        roll[idx_cov] = r
        pitch[idx_cov] = p
        yaw[idx_cov] = y
        status[idx_cov] = "traj_rot"
        dt_col[idx_cov] = rig.dt_s
    poses_out = replace(poses, origin=origin, roll=roll, pitch=pitch, yaw=yaw, source="traj_rot")
    n_traj = int((status == "traj_rot").sum())
    log(f"[rot-only] pose table: {n_traj}/{len(poses)} frames get the rot-only orientation, {len(poses) - n_traj} keep export "
        f"({len(rigs)}/{len(per_pass)} passes got their own rig)")

    provenance = {
        "stage": "S3b",
        "trajectory": "trajectory_rot.npz",
        "passes": passes,
        "passes_with_rig": sorted(rigs.keys()),
        "n_traj_frames": n_traj,
        "n_kept_frames": int(len(poses) - n_traj),
        "per_pass_dt_s": {str(p): r.dt_s for p, r in rigs.items()},
        "per_pass_dt_s_median": float(np.median(dt_arr)),
        "per_pass_dt_s_std": float(dt_arr.std()),
        "per_pass_R_cs_euler_xyz_deg": {str(p): r.stats["euler_R_cs_deg"] for p, r in rigs.items()},
        "per_pass_resid_deg_median": {str(p): r.stats["resid_deg_median"] for p, r in rigs.items()},
    }
    write_pose_table(poses_out, {"status": status, "dt_s": dt_col}, provenance, out_csv)
    log(f"[rot-only] wrote {out_csv}")

    diag_numbers = diagnostics_rot(poses, poses_out, status, diags, rigs, DIAG_DIR, log=log)
    (DIAG_DIR / "diag_rot.json").write_text(json.dumps({**_jsonable_diag_rot(diags, rigs), "comparison": diag_numbers}, indent=1))
    return out_csv


def diagnostics_rot(poses: Poses, poses_out: Poses, status: np.ndarray, diags: dict, rigs: dict, out_dir: Path, log=print) -> dict:
    """S3b: export-vs-rot-only comparison at frame times the rot-only trajectory covers. Position is
    unchanged by construction (`camera_pose`'s origin is the export table's own linear interpolation)
    -- reported anyway as a sanity check that it really is ~0, not a claimed improvement. Yaw is the
    real comparison, split turning (|yaw_rate| > `YAW_RATE_TURNING_VALIDATE`) vs. straight. Also plots
    the rotation-only dt-scan cost curve (RMS yaw residual on turning frames vs. dt) and its correlation
    with `align.json`'s independently-fit `yaw_offset_deg` on non-suspicious turning frames. Mirrors
    `diagnostics()` (S3) but for the orientation-only rig -- no position/dt-position-RMS numbers here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cov = status == "traj_rot"
    yr = np.nan_to_num(yaw_rates(poses), nan=0.0)
    turning = cov & (np.abs(yr) > YAW_RATE_TURNING_VALIDATE)
    straight = cov & ~turning

    dE = poses_out.origin[:, 0] - poses.origin[:, 0]
    dN = poses_out.origin[:, 1] - poses.origin[:, 1]
    dH = poses_out.origin[:, 2] - poses.origin[:, 2]
    dpos = np.linalg.norm(np.stack([dE, dN, dH], axis=1), axis=1)
    dyaw = ((poses_out.yaw - poses.yaw + 180.0) % 360.0) - 180.0

    def stats(mask):
        if not mask.any():
            return {"n": 0}
        return {
            "n": int(mask.sum()),
            "dpos_median_m": float(np.median(dpos[mask])), "dpos_p95_m": float(np.percentile(dpos[mask], 95)),
            "dyaw_median_deg": float(np.median(dyaw[mask])), "dyaw_p95_deg": float(np.percentile(np.abs(dyaw[mask]), 95)),
        }

    summary = {"straight": stats(straight), "turning": stats(turning), "n_covered": int(cov.sum())}
    log(f"[rot-only] KEY QUESTION -- export vs rot-only orientation at turning frames (n={summary['turning'].get('n', 0)}): "
        f"median |dpos|={summary['turning'].get('dpos_median_m', float('nan')):.4f} m (sanity check, should be ~0), "
        f"median |dyaw|={abs(summary['turning'].get('dyaw_median_deg', float('nan'))):.3f} deg, "
        f"p95={summary['turning'].get('dyaw_p95_deg', float('nan')):.3f} deg")
    log(f"[rot-only] straight frames (n={summary['straight'].get('n', 0)}): median |dyaw|={abs(summary['straight'].get('dyaw_median_deg', float('nan'))):.3f} deg")

    align_path = POSES_DIR / "align.json"
    corr = None
    if align_path.exists() and turning.any():
        align = json.loads(align_path.read_text())
        by_frame = {int(r["frame"]): r for r in align if not r.get("suspicious", False)}
        idx = np.flatnonzero(turning)
        a_dyaw, t_dyaw = [], []
        for i in idx:
            r = by_frame.get(int(i))
            if r is None:
                continue
            a_dyaw.append(r["yaw_offset_deg"])
            t_dyaw.append(dyaw[i])
        if len(a_dyaw) >= 5:
            corr = float(np.corrcoef(a_dyaw, t_dyaw)[0, 1])
            log(f"[rot-only] (rot_only - export) yaw vs Aligner yaw_offset_deg on turning frames: r={corr:.3f} (n={len(a_dyaw)})")
    summary["corr_dyaw_vs_align_yaw_offset"] = corr

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist([np.abs(dyaw[straight]), np.abs(dyaw[turning])] if turning.any() else [np.abs(dyaw[straight])],
                 bins=30, label=["straight", "turning"] if turning.any() else ["straight"], alpha=0.7)
    axes[0].set_xlabel("|export - rot_only| yaw (deg)")
    axes[0].legend()
    # rigs are fit PER PASS (see build_rot_only): overlay every pass' own dt-scan curve (thin lines,
    # its own chosen dt marked) to show each has a clear minimum, plus the spread of those minima.
    dt_mins = []
    for p, rig in sorted(rigs.items()):
        curve = rig.stats.get("dt_scan_curve", [])
        if not curve:
            continue
        dts, costs = zip(*curve)
        axes[1].plot(dts, costs, "-", lw=0.8, alpha=0.6)
        axes[1].axvline(rig.dt_s, color="k", lw=0.5, alpha=0.3)
        dt_mins.append(rig.dt_s)
    axes[1].set_xlabel("dt (s)")
    axes[1].set_ylabel("RMS yaw residual on that pass' turning frames (deg)")
    axes[1].set_title(f"{len(dt_mins)} per-pass dt-scan curves")
    if dt_mins:
        axes[2].hist(dt_mins, bins=min(15, max(3, len(dt_mins) // 2)))
        axes[2].axvline(float(np.median(dt_mins)), color="r", ls="--", label=f"median={np.median(dt_mins):.4f}s")
        axes[2].set_xlabel("per-pass fitted dt_s (s)")
        axes[2].set_ylabel("passes")
        axes[2].legend()
        axes[2].set_title("dt_s consistency across passes (reference-frame independent)")
    fig.tight_layout()
    fig.savefig(out_dir / "pos_yaw_dt_rot.png", dpi=110)
    plt.close(fig)
    return summary


def _jsonable_diag_rot(diags: dict, rigs: dict) -> dict:
    def j(x):
        if isinstance(x, dict):
            return {k: j(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [j(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x

    return {"passes": j(diags), "per_pass_rig": {str(p): {"dt_s": r.dt_s, "stats": j(r.stats)} for p, r in rigs.items()}}


# =============================================================================== S3b: validation
_VG: dict = {}


def _init_validate(rot_csv_str: str) -> None:
    """Pool worker init: build the per-worker global state once (CloudStore, both pose tables, the
    vehicle mask, two `Aligner`s, own-pass time bounds) rather than per frame."""
    from ..align import Aligner
    from ..cloud_store import CloudStore
    from ..poses import load_poses as _load_poses
    from ..vehicle_mask import MASK_PATH, VehicleMask

    store = CloudStore()
    poses_exp = _load_poses("export")
    poses_rot = _load_poses(rot_csv_str)
    vm = VehicleMask() if MASK_PATH.exists() else None
    _VG["store"] = store
    _VG["poses_exp"] = poses_exp
    _VG["poses_rot"] = poses_rot
    _VG["vm"] = vm
    _VG["al_exp"] = Aligner(store, poses_exp, vm)
    _VG["al_rot"] = Aligner(store, poses_rot, vm)
    pass_t0, pass_t1 = {}, {}
    for p in np.unique(poses_exp.pass_id):
        tt = poses_exp.t[poses_exp.pass_id == p]
        pass_t0[int(p)], pass_t1[int(p)] = float(tt.min()), float(tt.max())
    _VG["pass_t0"], _VG["pass_t1"] = pass_t0, pass_t1


def _own_pass_xyz(store, C: np.ndarray, t_frame: float, pass_id: int, pass_t0: dict, pass_t1: dict) -> np.ndarray:
    """Points within `R_MAX` of `C` scanned within `TIME_WINDOW_S` of `t_frame` AND belonging to
    `pass_id`'s own time range (reproduces `quality.assess_frame`'s "own" split, standalone)."""
    xyz, _pid = gather_candidates(store, C, R_MAX)
    if len(xyz) == 0:
        return xyz
    parts = store.query_disc(float(C[0]), float(C[1]), R_MAX)
    gps = np.concatenate([np.asarray(store.tile(t.name).gps_time[rows]) for t, rows in parts])
    win = np.abs(gps - t_frame) <= TIME_WINDOW_S
    own = win & (gps >= pass_t0[pass_id] - 2) & (gps <= pass_t1[pass_id] + 2)
    return xyz[own]


def _validate_job(k: int) -> dict:
    """Both validation metrics (S3b step 4) for one frame, comparing the export and rot-only pose
    tables: (a) `Aligner.colour_de` at dt=0, yaw_off=0; (b) `quality._silhouette_points`/`_residual`
    with explicit (R, C) from each table. Both tables share the same `C` by construction (rot-only
    origin == the export table's own linear interpolation), so the candidate cloud is gathered once."""
    from geovap.domain.model import geometry
    from ..sample import PanoSampler, load_pano_rgb

    store = _VG["store"]
    poses_exp, poses_rot = _VG["poses_exp"], _VG["poses_rot"]
    al_exp, al_rot = _VG["al_exp"], _VG["al_rot"]
    vm = _VG["vm"]
    pass_t0, pass_t1 = _VG["pass_t0"], _VG["pass_t1"]

    pass_id = int(poses_exp.pass_id[k])
    t_frame = float(poses_exp.t[k])

    al_exp.ps = PanoSampler(load_pano_rgb(poses_exp.path(k)), footprint=False, gradient=False)
    xyz, gray, gps, rgb = al_exp.gather(k)
    al_rot.ps = al_exp.ps
    de_export = al_exp.colour_de(k, xyz, rgb, gps, pass_id, t_frame, 0.0)
    de_rot = al_rot.colour_de(k, xyz, rgb, gps, int(poses_rot.pass_id[k]), float(poses_rot.t[k]), 0.0)

    o_e, r_e, p_e, y_e = poses_exp.origin[k], poses_exp.roll[k], poses_exp.pitch[k], poses_exp.yaw[k]
    o_r, r_r, p_r, y_r = poses_rot.origin[k], poses_rot.roll[k], poses_rot.pitch[k], poses_rot.yaw[k]
    R_e = geometry.vehicle_rotation(np.array([y_e]), np.array([r_e]), np.array([p_e]))[0]
    R_r = geometry.vehicle_rotation(np.array([y_r]), np.array([r_r]), np.array([p_r]))[0]
    C = o_e  # by construction == o_r for a covered frame (rot-only origin is the export linear interp)

    dt_img, edge_idx, valid = _photo_edges(poses_exp, k, vm)
    xyz_own = _own_pass_xyz(store, C, t_frame, pass_id, pass_t0, pass_t1)
    e_e = _silhouette_points(xyz_own, R_e, C) if len(xyz_own) > 1000 else xyz_own[:0]
    n_e, du_e, dv_e, dum_e, dvm_e, inl_e = _residual(e_e, R_e, C, dt_img, edge_idx, valid)
    e_r = _silhouette_points(xyz_own, R_r, C) if len(xyz_own) > 1000 else xyz_own[:0]
    n_r, du_r, dv_r, dum_r, dvm_r, inl_r = _residual(e_r, R_r, C, dt_img, edge_idx, valid)

    return {
        "frame": int(k), "pass_id": pass_id, "origin_match_m": float(np.linalg.norm(o_e - o_r)),
        "de_export": de_export, "de_rot": de_rot,
        "n_export": n_e, "du_export": du_e, "dv_export": dv_e, "dum_export": dum_e, "dvm_export": dvm_e, "inl_export": inl_e,
        "n_rot": n_r, "du_rot": du_r, "dv_rot": dv_r, "dum_rot": dum_r, "dvm_rot": dvm_r, "inl_rot": inl_r,
    }


def validate_rot_only(
    rot_csv: Path = POSES_DIR / "poses_traj_rot.csv",
    n_straight: int = 100,
    turning_min_rate: float = YAW_RATE_TURNING_VALIDATE,
    seed: int = 0,
    workers: int = ROT_ONLY_WORKERS_DEFAULT,
    out_path: Path | None = None,
    log=print,
) -> dict:
    """S3b step 4: validate `rot_csv` against export.csv on turning (|yaw_rate| > `turning_min_rate`
    deg/s, covered by the trajectory) and a random sample of `n_straight` clean straight (also
    covered) frames, with `Aligner.colour_de` and the `quality.py` silhouette residual. Writes
    `traj_diag/validate_rot.json` ({"summary": {...}, "per_frame": [...]}) and returns the summary."""
    out_path = out_path or (DIAG_DIR / "validate_rot.json")
    poses_exp = load_poses("export")
    poses_rot = load_poses(str(rot_csv))
    if poses_rot.traj is None:
        raise RuntimeError(f"{rot_csv}: no trajectory attached (missing/invalid sidecar json or npz) -- run --rot-only first")

    cov = _covered_mask(poses_rot)
    turning_all = turning_frame_idx(poses_exp, min_rate=turning_min_rate)
    turning_idx = turning_all[cov[turning_all]]
    straight_all = clean_straight_frame_idx(poses_exp)
    straight_covered = straight_all[cov[straight_all]]
    rng = np.random.default_rng(seed)
    straight_idx = (
        np.sort(rng.choice(straight_covered, n_straight, replace=False))
        if len(straight_covered) > n_straight
        else straight_covered
    )
    log(f"[validate-rot] turning frames (|yaw_rate|>{turning_min_rate} deg/s, covered): {len(turning_idx)}/{len(turning_all)}; "
        f"straight clean covered frames sampled: {len(straight_idx)}/{len(straight_covered)}")

    frames = np.concatenate([turning_idx, straight_idx]).astype(int).tolist()
    if not frames:
        raise RuntimeError("validate_rot_only: no covered turning or straight frames to validate")

    t0 = time.time()
    n_workers = max(1, min(workers, ROT_ONLY_WORKERS_DEFAULT, len(frames)))
    with Pool(n_workers, initializer=_init_validate, initargs=(str(rot_csv),)) as pool:
        results = pool.map(_validate_job, frames)
    log(f"[validate-rot] {len(frames)} frames validated in {time.time() - t0:.1f} s ({n_workers} workers)")

    by_frame = {r["frame"]: r for r in results}

    def _stat(a: np.ndarray) -> dict:
        a = a[np.isfinite(a)]
        return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "n": int(len(a))} if len(a) else {"median": float("nan"), "p95": float("nan"), "n": 0}

    def agg(idx: np.ndarray) -> dict:
        rows = [by_frame[int(k)] for k in idx if int(k) in by_frame]
        if not rows:
            return {"n": 0}
        g = {key: np.array([r[key] for r in rows]) for key in ("de_export", "de_rot", "du_export", "dv_export", "dum_export", "dvm_export", "inl_export", "du_rot", "dv_rot", "dum_rot", "dvm_rot", "inl_rot")}
        return {
            "n": len(rows),
            "colour_de_export": _stat(g["de_export"]), "colour_de_rot": _stat(g["de_rot"]),
            "du_export": _stat(np.abs(g["du_export"])), "dv_export": _stat(np.abs(g["dv_export"])),
            "du_rot": _stat(np.abs(g["du_rot"])), "dv_rot": _stat(np.abs(g["dv_rot"])),
            "mad_du_export": _stat(g["dum_export"]), "mad_dv_export": _stat(g["dvm_export"]),
            "mad_du_rot": _stat(g["dum_rot"]), "mad_dv_rot": _stat(g["dvm_rot"]),
            "inlier8_export": _stat(g["inl_export"]), "inlier8_rot": _stat(g["inl_rot"]),
        }

    summary = {"turning": agg(turning_idx), "straight": agg(straight_idx), "n_turning": len(turning_idx), "n_straight": len(straight_idx)}
    for label, key in (("TURNING", "turning"), ("STRAIGHT", "straight")):
        s = summary[key]
        if s.get("n", 0) == 0:
            log(f"[validate-rot] {label}: no covered frames")
            continue
        log(f"[validate-rot] {label} (n={s['n']}): colour dE median export={s['colour_de_export']['median']:.2f} -> rot={s['colour_de_rot']['median']:.2f}; "
            f"|du| median export={s['du_export']['median']:.2f}px -> rot={s['du_rot']['median']:.2f}px; "
            f"|dv| median export={s['dv_export']['median']:.2f}px -> rot={s['dv_rot']['median']:.2f}px; "
            f"inlier8 median export={s['inlier8_export']['median']:.3f} -> rot={s['inlier8_rot']['median']:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "per_frame": results}, indent=1))
    log(f"[validate-rot] wrote {out_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passes", nargs="+", default=["all"], help="pass ids, or 'all'")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--stride", type=int, default=5, help="fit_scan_centre stage-2 window stride (ignored by --rot-only: it never fits the centre)")
    ap.add_argument("--out", type=Path, default=None, help="pose-table CSV path (default: poses_traj.csv, or poses_traj_rot.csv with --rot-only/--validate-rot)")
    ap.add_argument("--rot-only", action="store_true", help="S3b: orientation-only build, skips fit_scan_centre entirely")
    ap.add_argument("--validate-rot", action="store_true", help="S3b: validate an existing --out (or poses_traj_rot.csv) pose table against export.csv; does not build")
    args = ap.parse_args()
    passes = None if args.passes == ["all"] else [int(p) for p in args.passes]
    if args.validate_rot:
        out_csv = args.out or (POSES_DIR / "poses_traj_rot.csv")
        validate_rot_only(rot_csv=out_csv, workers=args.workers or ROT_ONLY_WORKERS_DEFAULT)
    elif args.rot_only:
        out_csv = args.out or (POSES_DIR / "poses_traj_rot.csv")
        build_rot_only(passes=passes, workers=args.workers or ROT_ONLY_WORKERS_DEFAULT, out_csv=out_csv)
    else:
        out_csv = args.out or (POSES_DIR / "poses_traj.csv")
        build(passes=passes, workers=args.workers or N_WORKERS_DEFAULT, stride=args.stride, out_csv=out_csv)


if __name__ == "__main__":
    main()
