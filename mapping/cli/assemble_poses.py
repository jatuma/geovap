"""S_assemble: compose the validated correction layers into the final `poses_corrected` pose table
-- the one `mapping.poses.load_poses("corrected")` returns.

Order (`07_revize_geometrie_a_data.md`, plan step "ASSEMBLE"):

1. Base = `out/poses/poses_traj_rot.csv` (S3b rot-only trajectory attached; frame-time values equal
   export, orientation *between* frames comes from the dense scanner-plane trajectory).
2. Overlay S4 (`out/poses/poses_refined_export.csv`): rows with `status == "refined"` replace their
   own (E, N, H, roll, pitch, yaw, pass_id) and carry their theta/rms columns; every other row is
   left exactly as the base table has it.
3. Apply S5b (`out/pass_reg/pass_transforms.json`) with `pass_reg.apply_pass_transforms` to the
   overlaid table, and transform the attached rot-only trajectory identically: `R_p^T` is
   right-composed into every per-sample scanner quaternion (`R_s'(t) := R_s(t) @ R_p^T`, `R_cs`
   untouched -- see `transform_trajectory` for why folding it into `R_cs` would be wrong), so
   `camera_pose`'s orientation becomes `R_v' = R_v @ R_p^T` at every query time, and the
   trajectory's `lin_*` origin (what `Trajectory._linear_origin` linearly interpolates *between*
   frames) is resynced to this same transformed table -- not the frozen, pre-registration table
   `poses_traj_rot` was built from. A rotation-about-a-fixed-centre-plus-translation is affine, so
   `camera_pose` stays exactly consistent with the transformed table at every query time (see
   `tests/test_poses_table.py::test_transform_trajectory_invariant_with_apply_pass_transforms`
   and `::test_assemble_end_to_end`).
4. Write `poses_corrected.csv` (+ `.json` provenance chaining every input's `poses_hash`/file sha1,
   per-frame `status`, and a counts summary) and `trajectory_corrected.npz/.json`.

Honest limitation carried from S4/S3b (do not blur this in the output): a refined frame's pose is
only used AT its own timestamp (`Poses.pose_at(idx, 0.0)`, and any query that happens to land exactly
there). `Poses.interp` with the trajectory attached still gets its orientation *between* frames from
the rot-only trajectory (S3b), never from S4's per-frame refinement -- the two are not merged; S4 and
the trajectory are just two independent, non-conflicting sources for two different things (one frame's
own-time pose vs. the interpolant between frames).

    uv run python -m mapping.cli.assemble_poses run
        [--base out/poses/poses_traj_rot.csv] [--refined out/poses/poses_refined_export.csv]
        [--transforms out/pass_reg/pass_transforms.json]
        [--out out/poses/poses_corrected.csv] [--traj-out out/poses/trajectory_corrected.npz]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import pass_reg
from ..config import OUT_DIR, POSES_DIR
from ..poses import Poses, read_pose_table, write_pose_table
from ..trajectory import CamSensorRig, Trajectory

DEFAULT_BASE = POSES_DIR / "poses_traj_rot.csv"
DEFAULT_REFINED = POSES_DIR / "poses_refined_export.csv"
DEFAULT_TRANSFORMS = OUT_DIR / "pass_reg" / "pass_transforms.json"
DEFAULT_OUT = POSES_DIR / "poses_corrected.csv"
DEFAULT_TRAJ_OUT = POSES_DIR / "trajectory_corrected.npz"

META_COLS = [
    "status", "src_pass_changed", "dt_s", "dyaw", "droll", "dpitch",
    "n_edge", "rms_before", "rms_after", "reg_dE", "reg_dN", "reg_dH", "reg_dyaw",
]


def _sha1_file(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_rows(path: Path) -> list[dict]:
    import csv

    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _sidecar_json(csv_path: Path) -> dict:
    p = Path(csv_path).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else {}


# ------------------------------------------------------------------------------ step 2: S4 overlay
def overlay_refined(base: Poses, refined_rows: list[dict]) -> tuple[Poses, dict[str, np.ndarray], dict]:
    """Overlay rows with `status == "refined"` in `refined_rows` (one row per frame, same order as
    `base`) onto `base`'s (E, N, H, roll, pitch, yaw, pass_id). Returns
    (overlaid Poses [traj=None], per-frame meta arrays for the refinement columns [NaN / False where
    not overlaid], stats dict: n_overlaid, n_pass_changed, |delta pos|/|delta yaw| distribution)."""
    n = len(base)
    if len(refined_rows) != n:
        raise ValueError(f"poses_refined_export has {len(refined_rows)} rows, base has {n}")

    origin = base.origin.copy()
    roll = base.roll.copy()
    pitch = base.pitch.copy()
    yaw = base.yaw.copy()
    pass_id = base.pass_id.copy()

    refined_mask = np.zeros(n, dtype=bool)
    src_pass_changed = np.zeros(n, dtype=bool)
    dt_s = np.full(n, np.nan)
    dyaw = np.full(n, np.nan)
    droll = np.full(n, np.nan)
    dpitch = np.full(n, np.nan)
    n_edge = np.full(n, np.nan)
    rms_before = np.full(n, np.nan)
    rms_after = np.full(n, np.nan)

    dpos_list, dyaw_abs_list = [], []
    n_pass_changed = 0
    for i, row in enumerate(refined_rows):
        if row["status"] != "refined":
            continue
        if row["filename"] != str(base.filename[i]):
            raise ValueError(f"row {i}: filename mismatch {row['filename']!r} != {base.filename[i]!r} -- tables out of sync")
        old_o, old_yaw = origin[i].copy(), yaw[i]
        new_o = np.array([float(row["E"]), float(row["N"]), float(row["H"])])
        new_pid = int(row["pass_id"])
        dpos_list.append(float(np.linalg.norm(new_o - old_o)))
        dyaw_abs_list.append(float(abs((float(row["yaw"]) - old_yaw + 180.0) % 360.0 - 180.0)))

        origin[i], roll[i], pitch[i], yaw[i] = new_o, float(row["roll"]), float(row["pitch"]), float(row["yaw"])
        if new_pid != int(pass_id[i]):
            src_pass_changed[i] = True
            n_pass_changed += 1
        pass_id[i] = new_pid
        refined_mask[i] = True

        def _f(key):
            v = row[key]
            return float(v) if v not in ("", "nan", None) else np.nan

        dt_s[i], dyaw[i], droll[i], dpitch[i] = _f("dt_s"), _f("dyaw"), _f("droll"), _f("dpitch")
        n_edge[i], rms_before[i], rms_after[i] = _f("n_edge"), _f("rms_before"), _f("rms_after")

    overlaid = Poses(
        filename=base.filename.copy(), t=base.t.copy(), origin=origin, roll=roll, pitch=pitch,
        yaw=yaw, pass_id=pass_id, speed=base.speed.copy(), source=base.source, traj=None,
    )
    stats = {
        "n_overlaid": int(refined_mask.sum()),
        "n_pass_changed": n_pass_changed,
        "dpos_median_m": float(np.median(dpos_list)) if dpos_list else None,
        "dpos_p95_m": float(np.percentile(dpos_list, 95)) if dpos_list else None,
        "dpos_max_m": float(np.max(dpos_list)) if dpos_list else None,
        "dyaw_median_deg": float(np.median(dyaw_abs_list)) if dyaw_abs_list else None,
        "dyaw_p95_deg": float(np.percentile(dyaw_abs_list, 95)) if dyaw_abs_list else None,
        "dyaw_max_deg": float(np.max(dyaw_abs_list)) if dyaw_abs_list else None,
    }
    meta = {
        "refined_mask": refined_mask, "src_pass_changed": src_pass_changed,
        "dt_s": dt_s, "dyaw": dyaw, "droll": droll, "dpitch": dpitch,
        "n_edge": n_edge, "rms_before": rms_before, "rms_after": rms_after,
    }
    return overlaid, meta, stats


# ------------------------------------------------------------------------- step 3: S5b + trajectory
def _pass_rotation_yaw(transforms: dict) -> dict[int, float]:
    passes = transforms.get("passes", transforms)
    out = {}
    for p_str, tr in passes.items():
        try:
            out[int(p_str)] = float(tr["yaw_deg"])
        except (KeyError, ValueError):
            continue
    return out


def transform_trajectory(traj: Trajectory, transforms: dict, transformed_table: Poses) -> Trajectory:
    """rot-only only. `camera_pose` computes `R_cam(t) = R_cs @ R_s(t)` (`R_cs` constant per pass,
    `R_s(t)` the interpolated per-sample scanner orientation) and we need
    `R_cam'(t) = R_cam(t) @ R_p^T` for *every* t in the pass (matching
    `pass_reg.apply_pass_transforms`'s `R_v' = R_v @ R_p^T` on the table, so the two stay exactly
    consistent -- see the invariance tests). `R_p^T` sits at the *right* end of that product, after
    the time-varying `R_s(t)`; folding it into `R_cs` instead (`R_cs' = R_cs @ R_p^T`) would insert
    it in the *middle* (`R_cs @ R_p^T @ R_s(t)`) which only agrees with the wanted
    `R_cs @ R_s(t) @ R_p^T` when `R_p` and `R_s(t)` commute -- false in general (checked: up to ~8 px
    at 25 m for this dataset's transforms, not the needed 1e-6). The fix is to fold `R_p^T` into the
    *sample* orientations instead: `R_s'(t) := R_s(t) @ R_p^T` (`R_cs` untouched). Quaternion
    composition is right-linear (`(a+b)*c = a*c + b*c` and right-multiplication by a unit quaternion
    preserves norm), so right-composing every per-sample quaternion with the *same* `R_p^T` commutes
    exactly with `camera_pose`'s linear-interpolate-then-renormalise step -- so this reproduces
    `R_cam(t) @ R_p^T` for every interpolated t, not only at the samples themselves.

    Also resyncs `lin_*` (the origin `Trajectory._linear_origin` linearly interpolates between
    frames) to `transformed_table` -- every frame, same order, so `covers`/`camera_pose` and the
    plain-linear fallback (frames/passes the trajectory does not cover) agree on the same table."""
    if traj.mode != "rot_only":
        raise NotImplementedError("transform_trajectory only implements the rot_only path (S3b); pos-mode trajectories are not in production use (see trajectory.py)")
    from scipy.spatial.transform import Rotation

    yaw_by_pass = _pass_rotation_yaw(transforms)
    quat2 = traj.quat.copy()
    for p in np.unique(traj.pass_id):
        Rp = pass_reg._rot_yaw(np.radians(yaw_by_pass.get(int(p), 0.0)))
        m = traj.pass_id == p
        quat2[m] = (Rotation.from_quat(traj.quat[m]) * Rotation.from_matrix(Rp.T)).as_quat()
    new_rigs: dict[int, CamSensorRig] = {p: CamSensorRig(R_cs=rig.R_cs.copy(), l_cs=rig.l_cs.copy(), dt_s=rig.dt_s, stats=dict(rig.stats)) for p, rig in traj.rigs.items()}
    return Trajectory(
        t=traj.t.copy(), pass_id=traj.pass_id.copy(), S=traj.S.copy(), quat=quat2,
        segments={k: list(v) for k, v in traj.segments.items()}, rigs=new_rigs, sample_hz=traj.sample_hz,
        meta={**traj.meta, "pass_reg_applied": True},
        mode="rot_only",
        lin_t=transformed_table.t.copy(), lin_origin=transformed_table.origin.copy(),
        lin_roll=transformed_table.roll.copy(), lin_pitch=transformed_table.pitch.copy(),
        lin_yaw=transformed_table.yaw.copy(), lin_pass_id=transformed_table.pass_id.copy(),
    )


def _reg_deltas(pass_id: np.ndarray, transforms: dict) -> dict[str, np.ndarray]:
    """Broadcast each frame's *current* pass' (dE, dN, dH, dyaw_deg) transform onto per-frame arrays
    -- constant within a pass, informational (the table already carries the applied pose)."""
    passes = transforms.get("passes", transforms)
    n = len(pass_id)
    reg_dE, reg_dN, reg_dH, reg_dyaw = (np.full(n, np.nan) for _ in range(4))
    for p_str, tr in passes.items():
        try:
            p = int(p_str)
        except ValueError:
            continue
        sel = pass_id == p
        if not sel.any():
            continue
        dE, dN, dH = tr["t"]
        reg_dE[sel], reg_dN[sel], reg_dH[sel], reg_dyaw[sel] = dE, dN, dH, tr["yaw_deg"]
    return {"reg_dE": reg_dE, "reg_dN": reg_dN, "reg_dH": reg_dH, "reg_dyaw": reg_dyaw}


# ---------------------------------------------------------------------------------------- assemble
def assemble(
    base_path: Path = DEFAULT_BASE,
    refined_path: Path = DEFAULT_REFINED,
    transforms_path: Path = DEFAULT_TRANSFORMS,
    out_path: Path = DEFAULT_OUT,
    traj_out_path: Path = DEFAULT_TRAJ_OUT,
    log=print,
) -> tuple[Path, dict]:
    base_path, refined_path, transforms_path = Path(base_path), Path(refined_path), Path(transforms_path)
    out_path, traj_out_path = Path(out_path), Path(traj_out_path)

    base = read_pose_table(base_path)
    if base.traj is None:
        raise RuntimeError(f"{base_path} has no attached trajectory (sidecar missing 'trajectory' or npz not found) -- assemble step 1 requires it")
    base_status = _read_csv_rows(base_path)
    if len(base_status) != len(base):
        raise RuntimeError(f"{base_path}: row count mismatch with its own reader")
    base_status_col = [r["status"] for r in base_status]

    refined_rows = _read_csv_rows(refined_path)
    overlaid, refine_meta, refine_stats = overlay_refined(base, refined_rows)
    log(f"S4 overlay: {refine_stats['n_overlaid']} / {len(overlaid)} rows refined "
        f"({refine_stats['n_pass_changed']} with a changed pass_id); "
        f"|dpos| median {refine_stats['dpos_median_m']:.3f} m p95 {refine_stats['dpos_p95_m']:.3f} m; "
        f"|dyaw| median {refine_stats['dyaw_median_deg']:.3f} deg p95 {refine_stats['dyaw_p95_deg']:.3f} deg")

    transforms = json.loads(transforms_path.read_text())
    registered = pass_reg.apply_pass_transforms(overlaid, transforms)
    traj_corrected = transform_trajectory(base.traj, transforms, registered)
    registered.traj = traj_corrected

    status = [f"{b}+refined+reg" if m else f"{b}+reg" for b, m in zip(base_status_col, refine_meta["refined_mask"])]
    reg_meta = _reg_deltas(registered.pass_id, transforms)

    per_frame_meta = {
        "status": np.array(status, dtype=object),
        "src_pass_changed": refine_meta["src_pass_changed"],
        "dt_s": refine_meta["dt_s"], "dyaw": refine_meta["dyaw"],
        "droll": refine_meta["droll"], "dpitch": refine_meta["dpitch"],
        "n_edge": refine_meta["n_edge"], "rms_before": refine_meta["rms_before"], "rms_after": refine_meta["rms_after"],
        **reg_meta,
    }
    assert list(per_frame_meta.keys()) == META_COLS

    from collections import Counter

    status_counts = dict(Counter(status))
    log(f"status counts: {status_counts}")

    traj_out_path.parent.mkdir(parents=True, exist_ok=True)
    traj_corrected.save(traj_out_path)
    log(f"wrote {traj_out_path}")

    base_prov = _sidecar_json(base_path)
    refined_prov = _sidecar_json(refined_path)
    provenance = {
        "stage": "assemble (S0-S5b composition)",
        "inputs": {
            "base": {"path": str(base_path), "stage": "S3b traj_rot", "poses_hash": base.hash(), "sha1": _sha1_file(base_path), "trajectory_npz_sha1": _sha1_file(base_path.parent / base_prov.get("trajectory", "trajectory_rot.npz"))},
            "refined": {"path": str(refined_path), "stage": "S4 pose_refine", "poses_hash": refined_prov.get("poses_hash"), "sha1": _sha1_file(refined_path)},
            "pass_transforms": {"path": str(transforms_path), "stage": "S5b pass_reg", "sha1": _sha1_file(transforms_path), "datum": transforms.get("summary", {}).get("datum")},
        },
        "order": ["poses_traj_rot (S3b)", "overlay S4 refined rows", "apply S5b pass_transforms to table and trajectory (per-sample orientation + lin_origin)"],
        "trajectory": traj_out_path.name,
        # the pass_transforms this table's points are registered against -- `Poses.registration`
        # (mapping.poses.read_pose_table) and `mapping.cloud_store.open_store` read this back so a
        # `CloudStore` opened for this pose table applies the same S5b transform the poses assume.
        "registration": {"path": str(transforms_path), "sha1": _sha1_file(transforms_path)},
        "s4_overlay": refine_stats,
        "status_counts": status_counts,
        "note": "S4's refined pose is used only at the frame's own timestamp; Poses.interp between "
                "frames still gets orientation from the S3b rot-only trajectory (never merged with S4).",
    }
    write_pose_table(registered, per_frame_meta, provenance, out_path)
    log(f"wrote {out_path} ({len(registered)} frames)")
    return out_path, {"s4_overlay": refine_stats, "status_counts": status_counts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run"])
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--refined", default=str(DEFAULT_REFINED))
    ap.add_argument("--transforms", default=str(DEFAULT_TRANSFORMS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--traj-out", default=str(DEFAULT_TRAJ_OUT))
    a = ap.parse_args()
    assemble(Path(a.base), Path(a.refined), Path(a.transforms), Path(a.out), Path(a.traj_out))


if __name__ == "__main__":
    main()
