"""Rig calibration driver: edge-ICP fit on a stratified frame subsample, hold-out and time-block validation.

(An earlier attempt with a rendered-intensity NGF/NMI objective and with chamfer scans is kept in
objective.py / chamfer.py for reference; both were too flat on this rural dataset. The edge-ICP in
icp.py is the working method.)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..cloud_store import CloudStore
from ..config import DEG_PER_PX, OUT_DIR
from ..frame_select import FrameIndex
from ..poses import Poses, load_poses
from ..rig import IDENTITY, RigModel
from . import icp as I


def select_frames(poses: Poses, n: int = 120, seed: int = 0, min_speed: float = 1.0) -> np.ndarray:
    """Stratified over 5 time blocks x 3 speed terciles, moving frames only."""
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(poses.speed >= min_speed)
    blocks = np.array_split(ok, 5)
    per_block = n // 5
    out = []
    for blk in blocks:
        sp = poses.speed[blk]
        terc = np.digitize(sp, np.percentile(sp, [33.3, 66.7]))
        for t in range(3):
            cand = blk[terc == t]
            k = min(len(cand), per_block // 3 + (1 if t < per_block % 3 else 0))
            out.append(rng.choice(cand, size=k, replace=False))
    return np.sort(np.concatenate(out))


def _summ(icp: I.EdgeICP, theta) -> dict:
    s = icp.structure(np.asarray(theta), window=20.0)
    return {k: v for k, v in s.items() if not isinstance(v, dict)} | {"by_el": s["by_el"], "by_r": s["by_r"], "by_az": s["by_az"]}


def run(n_frames: int = 120, n_points: int = 30_000, free=("omega", "phi", "kappa", "dt", "lx", "ly", "lz"), do_cv: bool = True, out_dir: Path = OUT_DIR, tag: str = "calib", log=print) -> RigModel:
    from ..vehicle_mask import MASK_PATH, VehicleMask

    out = Path(out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    store = CloudStore()
    poses = load_poses()
    fi = FrameIndex(poses, IDENTITY)
    vmask = VehicleMask() if MASK_PATH.exists() else None
    frames_idx = select_frames(poses, n_frames)
    t = time.time()
    frames = [I.prepare_frame(int(f), store, fi, vmask, n_points) for f in frames_idx]
    frames = [f for f in frames if len(f.xyz) >= 200]
    log(f"prepared {len(frames)} calibration frames in {time.time()-t:.0f} s ({sum(len(f.xyz) for f in frames)} edge points)")

    icp = I.EdgeICP(frames, poses, free=free)
    before = _summ(icp, np.zeros(len(free)))
    log("identity residual structure: " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in before.items() if not isinstance(v, dict)}))
    sol = icp.solve(log=log)
    after = _summ(icp, sol["theta_free"])
    rig = RigModel(boresight_deg=tuple(sol["rig"]["boresight_deg"]), lever_arm_m=tuple(sol["rig"]["lever_arm_m"]), dt_s=sol["rig"]["dt_s"], note=f"edge-ICP on {len(frames)} frames")
    rig.to_json(out / "rig_fit.json")
    result = {"free": list(free), "rig": sol["rig"], "history": sol["history"], "before": before, "after": after, "frames": [int(f) for f in frames_idx]}

    if do_cv:
        rng = np.random.default_rng(1)
        idx = np.arange(len(frames))
        rng.shuffle(idx)
        n_train = int(0.6 * len(frames))
        tr = I.EdgeICP([frames[i] for i in idx[:n_train]], poses, free=free)
        te = I.EdgeICP([frames[i] for i in idx[n_train:]], poses, free=free)
        log("hold-out fit (60 %)")
        s_tr = tr.solve(log=log)
        th = np.asarray(s_tr["theta_free"])
        r_id = te.structure(np.zeros(len(free)), 8.0)["rms_px"]
        r_fit = te.structure(th, 8.0)["rms_px"]
        result["holdout"] = {"theta": s_tr["theta_free"], "test_rms_identity_px": r_id, "test_rms_fit_px": r_fit}
        log(f"  hold-out 40 %: rms(8px window) identity {r_id:.2f} -> fit {r_fit:.2f} full-res px")
        blocks = []
        order = np.argsort([f.frame for f in frames])
        for b, part in enumerate(np.array_split(order, 5)):
            sub = [frames[i] for i in part]
            log(f"time block {b}: frames {sub[0].frame}-{sub[-1].frame}")
            s_b = I.EdgeICP(sub, poses, free=free).solve(log=lambda *a: None)
            blocks.append({"block": b, "theta": s_b["theta_free"], "rig": s_b["rig"]})
            log("  " + json.dumps({k: round(v, 4) for k, v in zip(free, s_b["theta_free"])}))
        thetas = np.array([b["theta"] for b in blocks])
        result["blocks"] = blocks
        result["block_spread_std"] = dict(zip(free, thetas.std(0).round(4).tolist()))
        log("block spread (std): " + json.dumps(result["block_spread_std"]))
    (out / "fit.json").write_text(json.dumps(result, indent=1, default=float))
    return rig
