"""S4: per-frame pose refinement (edge-ICP per pass), writing a full pose table.

uv run python -m mapping.cli.refine_poses --poses export --passes all --workers 8 \
    --out Geovap_cache/out/poses/poses_refined.csv

`--poses` accepts "export", "corrected", or a path (any `mapping.poses.load_poses` source) so the
same CLI runs on the dense S3 trajectory once it exists. Frames not covered by `--passes` (or
rejected by acceptance) keep their input pose with status "kept".
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..cloud_store import CloudStore
from ..config import POSES_DIR
from ..frame_select import FrameIndex
from ..pose_refine import DEFAULT_FREE, default_prior, group_by_pass, load_align, plan_frames, refine_pass
from ..poses import Poses, load_poses, write_pose_table
from ..rig import IDENTITY
from ..vehicle_mask import MASK_PATH, VehicleMask

_G: dict = {}


def _init_worker(poses_source, n_points, free, prior_kind, own_pass_only):
    _G["store"] = CloudStore()
    _G["poses"] = load_poses(poses_source)
    _G["fi"] = FrameIndex(_G["poses"], IDENTITY)
    _G["vmask"] = VehicleMask() if MASK_PATH.exists() else None
    _G["n_points"] = n_points
    _G["free"] = free
    _G["prior"] = default_prior(prior_kind)
    _G["own_pass_only"] = own_pass_only


def _run_one(args) -> tuple[int, list[dict], dict]:
    pass_id, plans = args
    results, summary = refine_pass(pass_id, plans, _G["poses"], _G["store"], _G["fi"], _G["vmask"], n_points=_G["n_points"], free=_G["free"], prior=_G["prior"], own_pass_only=_G["own_pass_only"], log=lambda *a: None)
    return pass_id, [r.__dict__ for r in results], summary


def run(poses_source: str = "export", passes: list[int] | None = None, workers: int = 8, out: Path = POSES_DIR / "poses_refined.csv", align_path: Path = POSES_DIR / "align.json", n_points: int = 20_000, free: tuple[str, ...] = DEFAULT_FREE, turning_prior: str | None = None, own_pass_only: bool = True, log=print) -> Path:
    poses = load_poses(poses_source)
    turning_prior = turning_prior or ("wide" if poses.source == "export" else "narrow")
    align = load_align(align_path) if Path(align_path).exists() else None
    plans = plan_frames(poses, align)
    groups = group_by_pass(plans)
    if passes is not None:
        groups = {p: g for p, g in groups.items() if p in passes}
    log(f"refining {len(groups)} pass(es), {sum(len(g) for g in groups.values())} frames, turning_prior={turning_prior}, free={free}")

    # per-frame output: default is the input table, unchanged, status "kept"
    n = len(poses)
    out_rows: dict[int, dict] = {
        k: {
            "E": float(poses.origin[k, 0]), "N": float(poses.origin[k, 1]), "H": float(poses.origin[k, 2]),
            "roll": float(poses.roll[k]), "pitch": float(poses.pitch[k]), "yaw": float(poses.yaw[k]),
            "pass_id": int(poses.pass_id[k]), "status": "kept", "src": poses.source,
            "dt_s": 0.0, "dyaw": 0.0, "droll": 0.0, "dpitch": 0.0, "dlat": 0.0, "dh": 0.0,
            "n_edge": 0, "rms_before": float("nan"), "rms_after": float("nan"),
            "du_median_before": float("nan"), "dv_median_before": float("nan"),
            "du_median_after": float("nan"), "dv_median_after": float("nan"),
            "inlier8_before": float("nan"), "inlier8_after": float("nan"), "flags": "",
        }
        for k in range(n)
    }

    t0 = time.time()
    summaries = []
    tasks = list(groups.items())
    with Pool(min(workers, max(1, len(tasks))), initializer=_init_worker, initargs=(poses_source, n_points, free, turning_prior, own_pass_only)) as pool:
        for pass_id, results, summary in pool.imap_unordered(_run_one, tasks):
            log(f"pass {pass_id}: refined {summary['n_refined']}, interpolated {summary['n_interpolated']}, kept {summary['n_kept']} ({summary['s']:.0f} s)")
            summaries.append(summary)
            for r in results:
                out_rows[r["frame"]] = {k: v for k, v in r.items() if k != "frame"}
    log(f"total {time.time()-t0:.0f} s")

    meta_cols = ["status", "src", "dt_s", "dyaw", "droll", "dpitch", "dlat", "dh", "n_edge", "rms_before", "rms_after", "du_median_before", "dv_median_before", "du_median_after", "dv_median_after", "inlier8_before", "inlier8_after", "flags"]
    per_frame_meta = {c: np.array([out_rows[k][c] for k in range(n)], dtype=object if c in ("status", "src", "flags") else np.float64) for c in meta_cols}
    new_pass_id = np.array([out_rows[k]["pass_id"] for k in range(n)], dtype=np.int32)
    new_origin = np.array([[out_rows[k]["E"], out_rows[k]["N"], out_rows[k]["H"]] for k in range(n)])
    new_roll = np.array([out_rows[k]["roll"] for k in range(n)])
    new_pitch = np.array([out_rows[k]["pitch"] for k in range(n)])
    new_yaw = np.array([out_rows[k]["yaw"] for k in range(n)])
    out_poses = Poses(filename=poses.filename, t=poses.t, origin=new_origin, roll=new_roll, pitch=new_pitch, yaw=new_yaw, pass_id=new_pass_id, speed=poses.speed, source=poses.source)

    provenance = {
        "stage": "S4 pose_refine", "input_poses_source": poses_source, "input_poses_hash": poses.hash(),
        "align_json": str(align_path), "passes": sorted(groups.keys()), "n_points": n_points,
        "free": list(free), "turning_prior": turning_prior, "own_pass_only": own_pass_only, "summaries": summaries,
    }
    return write_pose_table(out_poses, per_frame_meta, provenance, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default="export")
    ap.add_argument("--passes", nargs="*", default=["all"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(POSES_DIR / "poses_refined.csv"))
    ap.add_argument("--align", default=str(POSES_DIR / "align.json"))
    ap.add_argument("--n-points", type=int, default=20_000)
    ap.add_argument("--free", nargs="*", default=list(DEFAULT_FREE))
    ap.add_argument("--turning-prior", choices=["wide", "narrow"], default=None)
    ap.add_argument("--no-own-pass-only", dest="own_pass_only", action="store_false", default=True, help="disable the own-pass candidate restriction (S4 step 2); default on")
    a = ap.parse_args()
    passes = None if a.passes == ["all"] else [int(p) for p in a.passes]
    path = run(poses_source=a.poses, passes=passes, workers=a.workers, out=Path(a.out), align_path=Path(a.align), n_points=a.n_points, free=tuple(a.free), turning_prior=a.turning_prior, own_pass_only=a.own_pass_only)
    print("wrote", path)


if __name__ == "__main__":
    main()
