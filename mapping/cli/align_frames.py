"""Re-registers frames against the point cloud (mapping.align.Aligner) for the turning frames
(|yaw_rate| > 8 deg/s, mapping.quality.yaw_rates) plus a set of straight, clean frames evenly
spaced from dataset/clean_frames.json's "clean" list; writes Geovap_cache/out/poses/align.json
(mapping.align.save format: a list of Alignment dicts).

uv run python -m mapping.cli.align_frames [--workers 8] [--straight 100] [--out PATH] [--no-cap]

Runs a single-frame probe first to estimate the total wall-clock at `--workers` workers; if that
estimate exceeds ~2.5 h the straight-frame count is capped to 40 and the cap is logged (skip with
--no-cap).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .. import align as A
from ..config import POSES_DIR, REPO_ROOT
from ..poses import Poses, load_poses
from ..quality import yaw_rates

YAW_RATE_THRESH_DEG_S = 8.0
TIME_BUDGET_S = 2.5 * 3600.0  # wall-clock budget for the whole run, at --workers workers
N_WORKERS_DEFAULT = 8
N_STRAIGHT_DEFAULT = 100
N_STRAIGHT_CAPPED = 40

_G: dict = {}


def _init() -> None:
    from ..cloud_store import CloudStore
    from ..vehicle_mask import MASK_PATH, VehicleMask

    _G["store"] = CloudStore()
    _G["poses"] = load_poses()
    vmask = VehicleMask() if MASK_PATH.exists() else None
    _G["aligner"] = A.Aligner(_G["store"], _G["poses"], vmask)


def _job(k: int) -> A.Alignment:
    r = _G["aligner"].align(int(k))
    _G["store"].release()  # keep this worker's RSS small (memmap pages), like quality.py / colorize.py
    return r


def straight_frames(n: int) -> np.ndarray:
    """n frames evenly spaced (by index into the sorted list) from dataset/clean_frames.json's "clean" class."""
    clean = json.loads((REPO_ROOT / "dataset" / "clean_frames.json").read_text())["clean"]
    clean = np.array(sorted(clean))
    if n >= len(clean):
        return clean
    idx = np.unique(np.linspace(0, len(clean) - 1, n).round().astype(int))
    return clean[idx]


def build_frame_list(poses: Poses, n_straight: int) -> tuple[np.ndarray, np.ndarray]:
    """(turning, straight) frame index arrays, disjoint."""
    yr = np.nan_to_num(yaw_rates(poses), nan=0.0)
    turning = np.flatnonzero(np.abs(yr) > YAW_RATE_THRESH_DEG_S)
    straight = np.setdiff1d(straight_frames(n_straight), turning)
    return turning, straight


def _pctile_summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "median": round(float(med), 3), "iqr": round(float(q3 - q1), 3), "min": round(float(x.min()), 3), "max": round(float(x.max()), 3)}


def summarise(results: list[A.Alignment], poses: Poses, kinds: dict[int, str]) -> dict:
    out: dict = {"n_total": len(results)}
    for kind in ("turning", "straight"):
        dt = [r.dt_s for r in results if kinds.get(r.frame) == kind]
        out[f"dt_s_{kind}"] = _pctile_summary(np.array(dt))
    out["yaw_offset_deg"] = _pctile_summary(np.array([r.yaw_offset_deg for r in results]))
    out["n_suspicious"] = int(sum(r.suspicious for r in results))
    out["n_src_pass_mismatch"] = int(sum(1 for r in results if r.src_pass != int(poses.pass_id[r.frame])))
    out["n_turning"] = sum(1 for r in results if kinds.get(r.frame) == "turning")
    out["n_straight"] = sum(1 for r in results if kinds.get(r.frame) == "straight")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
    ap.add_argument("--straight", type=int, default=N_STRAIGHT_DEFAULT)
    ap.add_argument("--out", default=str(POSES_DIR / "align.json"))
    ap.add_argument("--no-cap", action="store_true", help="skip the runtime probe / budget cap")
    a = ap.parse_args()

    from multiprocessing import Pool

    poses = load_poses()
    turning, straight = build_frame_list(poses, a.straight)
    print(f"turning frames (|yaw_rate| > {YAW_RATE_THRESH_DEG_S} deg/s): {len(turning)}; straight (clean, evenly spaced): {len(straight)}")

    if not a.no_cap and len(turning) + len(straight) > 0:
        _init()
        probe_frame = int(turning[0]) if len(turning) else int(straight[0])
        t0 = time.time()
        _job(probe_frame)
        per_frame_s = time.time() - t0
        n_total = len(turning) + len(straight)
        est_h = per_frame_s * n_total / a.workers / 3600.0
        print(f"probe: frame {probe_frame} took {per_frame_s:.1f} s single-process -> est. {est_h:.2f} h total with {a.workers} workers")
        if per_frame_s * n_total / a.workers > TIME_BUDGET_S and a.straight > N_STRAIGHT_CAPPED:
            straight = np.setdiff1d(straight_frames(N_STRAIGHT_CAPPED), turning)
            n_total = len(turning) + len(straight)
            print(f"CAPPED: estimated runtime {est_h:.2f} h exceeds the {TIME_BUDGET_S/3600:.1f} h budget -> "
                  f"reduced to turning frames + {N_STRAIGHT_CAPPED} straight frames ({n_total} total, "
                  f"est. {per_frame_s * n_total / a.workers / 3600:.2f} h)")

    frames = np.sort(np.concatenate([turning, straight])).astype(int)
    kinds = {int(k): "turning" for k in turning} | {int(k): "straight" for k in straight}
    print(f"aligning {len(frames)} frames with {a.workers} workers")

    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        results = list(pool.imap_unordered(_job, [int(k) for k in frames], chunksize=1))
    results.sort(key=lambda r: r.frame)
    print(f"aligned {len(results)} frames in {(time.time()-t0)/60:.1f} min")

    out = Path(a.out)
    A.save(results, out)
    print(f"wrote {out}")

    summary = summarise(results, poses, kinds)
    (out.parent / "align_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("summary: " + json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
