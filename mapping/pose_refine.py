"""S4: per-frame pose refinement (edge-ICP per pass).

`export.csv` / a corrected trajectory table gives one pose per frame at ~5 m spacing; between
frames `Poses.interp` is linear, which is wrong on turning frames (5-10 deg / 0.6 s) and leaves a
per-frame residual even on straight ones. `PassRefiner` (a `mapping.calib.icp.EdgeICP` subclass)
fits SIX small corrections per frame of one pass -- theta[k] = (dt, dyaw, droll, dpitch, dlat, dh)
-- against the same photo-edge objective used for rig calibration, regularised by a per-parameter
Gaussian prior and a smoothness term along the pass (so frames with too few edge points are carried
by their neighbours instead of drifting).

Works on ANY `Poses` (export or a dense-trajectory table) via `Poses.interp`; only the turning-frame
prior width and the align.json-based initialisation assume export-table conventions.

    uv run python -m mapping.cli.refine_poses --poses export --passes 24 15 29 --workers 3 \
        --out Geovap_cache/out/poses/refine_dev/poses_refined_dev.csv
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from . import geometry
from .calib import chamfer as Ch
from .calib import icp as I
from .cloud_store import CloudStore
from .config import PANO_W, R_MAX, R_MIN
from .frame_select import FrameIndex
from .poses import Poses
from .quality import yaw_rates

PARAM_NAMES = ("dt", "dyaw", "droll", "dpitch", "dlat", "dh")
DEFAULT_FREE = ("dt", "dyaw", "droll", "dpitch")
WINDOWS = (40.0, 30.0, 20.0, 12.0, 8.0, 6.0)
N_EDGE_MIN = 300  # mapping.quality.GEO_MIN_POINTS convention
MEDIAN_PX_ACCEPT = 6.0  # mapping.quality.GEO_MAX_MEDIAN_PX convention
INLIER_PX = 8.0  # mapping.quality.CONFLICT_PX / GEO_MIN_INLIER convention (CHAM px)
INLIER_SLACK = 0.02  # tolerance on the after-vs-before inlier-fraction check (resampling noise)
SIGMA_ACCEPT = 3.0
YAW_RATE_THRESH_DEG_S = 8.0  # mapping.quality.yaw_rates convention
ALIGN_GAIN_MIN = 1.5  # score0 - score >= this: trust the Aligner's candidate over the recorded pose
STRUCTURE_WINDOW = 20.0  # CHAM px, association window used for before/after diagnostics

# Acceptance used to be `rms_after <= 0.85 * rms_before` (or median <= 6px). On 20 clean, straight
# (|yaw_rate| < 3 deg/s) frames spread over 13 passes, at their AS-RECORDED (unrefined) pose, median
# rms_px = 22.5 -- matching the dev-run "before" numbers almost exactly -- while median |du|, |dv|
# are only 0.46/1.16 px and MAD is 7.9/11.5 px, matching dataset/frame_quality.csv's clean-frame
# baseline (0.38/0.78 px median, 8.1/11.2 px MAD) to within sampling noise. The edge metric's own
# floor (vegetation/occlusion-boundary noise inflating a soft-L1 RMS through its tails) *is* ~20 px
# even for well-aligned frames, so an RMS-ratio target chases noise instead of pose error. Acceptance
# below instead mirrors quality.py directly: robust median offset + inlier fraction.


# --------------------------------------------------------------------------------------- priors
@dataclass(frozen=True)
class Prior:
    """1-sigma widths per parameter (units: seconds, degrees, degrees, degrees, metres, metres).
    `turning_sigma` overrides a subset (typically dt, dyaw) for frames flagged as turning."""

    sigma: dict[str, float] = field(default_factory=lambda: {"dt": 0.05, "dyaw": 0.5, "droll": 0.3, "dpitch": 0.3, "dlat": 0.15, "dh": 0.15})
    turning_sigma: dict[str, float] = field(default_factory=lambda: {"dt": 0.3, "dyaw": 2.0})

    def sigma_for(self, name: str, turning: bool) -> float:
        if turning and name in self.turning_sigma:
            return self.turning_sigma[name]
        return self.sigma[name]


def default_prior(turning_prior: str = "wide") -> Prior:
    """`turning_prior`: "wide" widens dt/dyaw sigma on turning frames (export-table default,
    §07 5.1/S4); "narrow" uses the base sigma everywhere (a dense trajectory already resolves
    turning, S3-backed tables)."""
    p = Prior()
    if turning_prior == "narrow":
        return Prior(sigma=p.sigma, turning_sigma={})
    return p


# --------------------------------------------------------------------------------------- init from align.json
@dataclass
class FramePlan:
    frame: int
    t_base: float  # trajectory time the frame's theta is relative to (src_time if reassigned, else its own t)
    pass_target: int  # pass_id this frame is refined (and finally reported) under
    dt0: float
    dyaw0: float
    reassigned: bool


def load_align(path: str | Path) -> dict[int, dict]:
    return {int(r["frame"]): r for r in json.loads(Path(path).read_text())}


def plan_frames(poses: Poses, align: dict[int, dict] | None) -> list[FramePlan]:
    """Per-frame base time / target pass / theta0 seed from the Aligner's results (S2's align.json).
    A frame is only moved off its own initial guess when the Aligner's chosen pose clearly beats
    the recorded one (score0 - score >= ALIGN_GAIN_MIN); frames with `src_pass` != the recorded
    pass are reassigned there and use the Aligner's `src_time` as their base time (its `dt_s` is
    already 0 in that case -- the offset is baked into `src_time`)."""
    out = []
    align = align or {}
    for k in range(len(poses)):
        row = align.get(k)
        if row is not None and (row["score0"] - row["score"]) >= ALIGN_GAIN_MIN:
            src_pass = int(row["src_pass"])
            reassigned = src_pass != int(poses.pass_id[k])
            if reassigned:
                out.append(FramePlan(k, float(row["src_time"]), src_pass, 0.0, float(row["yaw_offset_deg"]), True))
            else:
                out.append(FramePlan(k, float(poses.t[k]), src_pass, float(row["dt_s"]), float(row["yaw_offset_deg"]), False))
        else:
            out.append(FramePlan(k, float(poses.t[k]), int(poses.pass_id[k]), 0.0, 0.0, False))
    return out


def group_by_pass(plans: list[FramePlan]) -> dict[int, list[FramePlan]]:
    """Plans grouped by `pass_target`, each group sorted by `t_base` (the order `PassRefiner`'s
    smoothness term needs)."""
    groups: dict[int, list[FramePlan]] = {}
    for pl in plans:
        groups.setdefault(pl.pass_target, []).append(pl)
    for g in groups.values():
        g.sort(key=lambda pl: pl.t_base)
    return groups


# --------------------------------------------------------------------------------------- the refiner
class PassRefiner(I.EdgeICP):
    """theta[k] = (dt, dyaw, droll, dpitch, dlat, dh) per frame of ONE pass (K frames -> up to 6K
    unknowns; `free` selects which columns are estimated, the rest stay frozen at 0).

    `poses_for`: o, r, p, y = poses.interp(t_base_k + dt_k, pass_target_k); R_v = vehicle_rotation
    (y+dyaw_k, r+droll_k, p+dpitch_k); C = o + R_v^T (0, dlat_k, dh_k) -- the same lever-arm
    convention as `geometry.frame_rotations` (R_v^T applied to a body-axes offset)."""

    def __init__(self, frames: list[I.IcpFrame], poses: Poses, t_base: np.ndarray, pass_target: np.ndarray, free: tuple[str, ...] = DEFAULT_FREE, prior: Prior | None = None, turning: np.ndarray | None = None, smooth_lambda: float = 1.0, r_max: float = R_MAX):
        self.frames = frames
        self.poses = poses
        self.idx = np.array([f.frame for f in frames])
        self.t_base = np.asarray(t_base, dtype=np.float64)
        self.pass_target = np.asarray(pass_target, dtype=np.int64)
        self.free_names = tuple(free)
        self.free_idx = [PARAM_NAMES.index(n) for n in free]
        self.K = len(frames)
        self.n_free = len(self.free_idx)
        self.r_max = r_max
        self.s = Ch.CHAM_W / PANO_W
        self.prior = prior or default_prior()
        self.turning = np.zeros(self.K, dtype=bool) if turning is None else np.asarray(turning, dtype=bool)
        self.smooth_lambda = smooth_lambda
        assert len(self.t_base) == self.K == len(self.pass_target)
        self.sigma = np.array([[self.prior.sigma_for(n, self.turning[k]) for n in self.free_names] for k in range(self.K)])  # [K, n_free]

    # ------------------------------------------------------------------ theta <-> pose
    def full_theta(self, theta_free) -> np.ndarray:
        """[K,6] with frozen columns at 0."""
        full = np.zeros((self.K, 6))
        full[:, self.free_idx] = np.asarray(theta_free, dtype=np.float64).reshape(self.K, self.n_free)
        return full

    def poses_for(self, theta_free) -> tuple[np.ndarray, np.ndarray]:
        full = self.full_theta(theta_free)
        dt, dyaw, droll, dpitch, dlat, dh = full.T
        o, r, p, y = self.poses.interp(self.t_base + dt, self.pass_target)
        R_v = geometry.vehicle_rotation(y + dyaw, r + droll, p + dpitch)  # [K,3,3]
        lever = np.stack([np.zeros(self.K), dlat, dh], axis=1)  # [K,3] body axes
        C = o + np.einsum("kji,kj->ki", R_v, lever)  # R_v^T @ lever, per frame
        return R_v, C

    # ------------------------------------------------------------------ extra residual blocks
    def prior_residuals(self, theta_free) -> np.ndarray:
        x = np.asarray(theta_free, dtype=np.float64).reshape(self.K, self.n_free)
        return (x / self.sigma).ravel()

    def smoothness_residuals(self, theta_free) -> np.ndarray:
        if self.K < 3:
            return np.zeros(0)
        x = np.asarray(theta_free, dtype=np.float64).reshape(self.K, self.n_free)
        lap = x[2:] - 2 * x[1:-1] + x[:-2]  # [K-2, n_free]
        return (self.smooth_lambda * lap / self.sigma[1:-1]).ravel()

    def residuals_fixed(self, theta_free, assoc) -> np.ndarray:
        edge = super().residuals_fixed(theta_free, assoc)
        return np.concatenate([edge, self.prior_residuals(theta_free), self.smoothness_residuals(theta_free)])

    # ------------------------------------------------------------------ per-frame diagnostics
    def robust_stats(self, theta_free, window: float = STRUCTURE_WINDOW, inlier_px: float = INLIER_PX) -> list[dict]:
        """Per frame (order of `self.frames`), full-res px, mirroring `mapping.quality._residual`
        exactly (independent of `EdgeICP.residuals`'s meta so `inlier8` can see the full in-range
        sample, not just the window-matched subset): n_edge (matched within `window`), n_total (all
        in-range edge points), rms_px, signed median du/dv (bias), MAD du/dv (spread), and inlier8 =
        fraction of ALL in-range points within `inlier_px` CHAM px of a photo edge (quality.py's
        GEO_MIN_INLIER convention) -- comparable before/after even though the association `window`
        shrinks across outer-loop stages."""
        R, C = self.poses_for(theta_free)
        out = []
        for i, f in enumerate(self.frames):
            u, v, r, el = geometry.world_to_pano(f.xyz, R[i], C[i], dtype=np.float64)
            m = (r >= R_MIN) & (r <= self.r_max)
            x, y = u[m] * self.s, v[m] * self.s
            n_total = int(m.sum())
            if n_total == 0:
                out.append({"frame": f.frame, "n_edge": 0, "n_total": 0, "rms_px": float("nan"), "du_median": float("nan"), "dv_median": float("nan"), "du_mad": float("nan"), "dv_mad": float("nan"), "inlier8": float("nan")})
                continue
            xi = np.mod(np.floor(x).astype(np.int64), Ch.CHAM_W)
            yi = np.clip(np.floor(y).astype(np.int64), 0, Ch.CHAM_H - 1)
            d = f.dt[yi, xi].astype(np.float32)
            validm = f.valid[yi, xi]
            inlier8 = float((validm & (d <= inlier_px)).mean())
            ok = validm & (d <= window)
            n = int(ok.sum())
            if n < 50:
                out.append({"frame": f.frame, "n_edge": n, "n_total": n_total, "rms_px": float("nan"), "du_median": float("nan"), "dv_median": float("nan"), "du_mad": float("nan"), "dv_mad": float("nan"), "inlier8": inlier8})
                continue
            er = f.edge_idx[0, yi[ok], xi[ok]].astype(np.float64) + 0.5
            ec = f.edge_idx[1, yi[ok], xi[ok]].astype(np.float64) + 0.5
            du = ((ec - x[ok] + Ch.CHAM_W / 2) % Ch.CHAM_W - Ch.CHAM_W / 2) / self.s
            dv = (er - y[ok]) / self.s
            du_med, dv_med = float(np.median(du)), float(np.median(dv))
            out.append({
                "frame": f.frame, "n_edge": n, "n_total": n_total,
                "rms_px": float(np.sqrt(np.mean(du**2 + dv**2))),
                "du_median": du_med, "dv_median": dv_med,
                "du_mad": float(np.median(np.abs(du - du_med))), "dv_mad": float(np.median(np.abs(dv - dv_med))),
                "inlier8": inlier8,
            })
        return out

    # ------------------------------------------------------------------ sparse jacobian pattern
    def _jac_sparsity(self, edge_ns: list[int]) -> sparse.csr_matrix:
        """Rows: per-frame edge blocks (2*n_i rows, dense in that frame's `n_free` columns) then
        K*n_free prior rows (one column each) then (K-2)*n_free smoothness rows (three columns:
        k-1, k, k+1 of the same parameter)."""
        ncols = self.K * self.n_free
        rows, cols = [], []
        r0 = 0
        for k, n in enumerate(edge_ns):
            if n == 0:
                continue
            block_rows = np.repeat(np.arange(r0, r0 + 2 * n), self.n_free)
            block_cols = np.tile(np.arange(k * self.n_free, (k + 1) * self.n_free), 2 * n)
            rows.append(block_rows)
            cols.append(block_cols)
            r0 += 2 * n
        for k in range(self.K):
            for j in range(self.n_free):
                rows.append(np.array([r0]))
                cols.append(np.array([k * self.n_free + j]))
                r0 += 1
        for k in range(1, self.K - 1):
            for j in range(self.n_free):
                rows.append(np.full(3, r0))
                cols.append(np.array([(k - 1) * self.n_free + j, k * self.n_free + j, (k + 1) * self.n_free + j]))
                r0 += 1
        rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        data = np.ones(len(rows))
        return sparse.csr_matrix((data, (rows, cols)), shape=(r0, ncols))

    def solve(self, theta0=None, windows=WINDOWS, f_scale: float = 2.0, log=print) -> dict:
        x = np.zeros(self.K * self.n_free) if theta0 is None else np.asarray(theta0, dtype=np.float64).copy()
        x_scale = np.tile(self.sigma.mean(axis=0), self.K)  # per-parameter scale (mean over frames, sigma widths)
        x_scale = np.where(x_scale > 0, x_scale, 1.0)
        history = []
        for w in windows:
            t = time.time()
            assoc = self.associate(x, w)
            edge_ns = [int(ok.sum()) for ok, _, _ in assoc]
            spars = self._jac_sparsity(edge_ns)
            r0 = self.residuals_fixed(x, assoc)
            sol = least_squares(self.residuals_fixed, x, args=(assoc,), loss="soft_l1", f_scale=f_scale, x_scale=x_scale, jac_sparsity=spars, max_nfev=40)
            x = sol.x
            r1 = self.residuals_fixed(x, assoc)
            # per-parameter RMS of the free theta (diagnostic: confirms the solve actually moves --
            # a Jacobian-sparsity or x_scale bug that froze a parameter shows up as ~0 here)
            xg = x.reshape(self.K, self.n_free) if self.n_free else x.reshape(self.K, 0)
            theta_rms = {n: float(np.sqrt(np.mean(xg[:, j] ** 2))) for j, n in enumerate(self.free_names)}
            rec = {"window": w, "n_edge": int(sum(edge_ns)), "rms_before": float(np.sqrt(np.mean(r0**2))), "rms_after": float(np.sqrt(np.mean(r1**2))), "nfev": int(sol.nfev), "s": round(time.time() - t, 1), "theta_rms": theta_rms}
            history.append(rec)
            log(f"    window {w:4.0f}px: n_edge={rec['n_edge']:6d} rms {rec['rms_before']:.2f} -> {rec['rms_after']:.2f}  theta_rms={ {k: round(v, 4) for k, v in theta_rms.items()} }  ({sol.nfev} fev, {rec['s']} s)")
        return {"theta": x, "history": history}


# --------------------------------------------------------------------------------------- driver
@dataclass
class FrameResult:
    frame: int
    pass_id: int  # possibly reassigned
    E: float
    N: float
    H: float
    roll: float
    pitch: float
    yaw: float
    status: str  # refined | interpolated | kept
    src: str
    dt_s: float
    dyaw: float
    droll: float
    dpitch: float
    dlat: float
    dh: float
    n_edge: int
    rms_before: float
    rms_after: float
    du_median_before: float
    dv_median_before: float
    du_median_after: float
    dv_median_after: float
    inlier8_before: float
    inlier8_after: float
    flags: str


def _initial_pose(poses: Poses, t_base: float, pass_target: int, dt0: float, dyaw0: float) -> tuple[np.ndarray, np.ndarray, float]:
    """(R[3,3], C[3], t_query) for the seed hypothesis (dt0, dyaw0; droll=dpitch=0, no lever)."""
    t_q = t_base + dt0
    o, r, p, y = poses.interp(np.array([t_q]), np.array([pass_target]))
    R = geometry.vehicle_rotation(y + dyaw0, r, p)[0]
    return R, o[0], t_q


def refine_pass(pass_id: int, plans: list[FramePlan], poses: Poses, store: CloudStore, fi: FrameIndex, vmask, n_points: int = 20_000, free: tuple[str, ...] = DEFAULT_FREE, prior: Prior | None = None, own_pass_only: bool = True, log=print) -> tuple[list[FrameResult], dict]:
    """Prepare edge points at the seed pose, solve, re-prepare under the refined pose (outer loop
    x2), solve again; return per-frame `FrameResult`s plus a summary dict.

    `own_pass_only` (default True) restricts the edge-point candidate gather to the frame's own
    pass (`calib.icp.prepare_frame`'s `own_pass_only`): the plain +-45s time window also admits a
    neighbouring pass whenever two passes are <~90s apart, and S5 found 0.1-0.6 m offsets between
    passes -- a leaked double silhouette biases the fit there (measured: pass 24's first frames
    pull 39-43% of candidates from pass 23; restricting to own-pass moves that frame's median |du|
    from 4.7 to 2.1 px). Off by default only for `own_pass_only=False` callers that want the old
    (rig-calibration-identical) candidate gather for comparison."""
    prior = prior or default_prior()
    K = len(plans)
    t_base = np.array([pl.t_base for pl in plans])
    pass_target = np.full(K, pass_id, dtype=np.int64)
    dt0 = np.array([pl.dt0 for pl in plans])
    dyaw0 = np.array([pl.dyaw0 for pl in plans])
    yr = np.nan_to_num(yaw_rates(poses), nan=0.0)
    turning = np.abs(yr[[pl.frame for pl in plans]]) > YAW_RATE_THRESH_DEG_S

    def prepare(R, C, t):
        out = []
        for i, pl in enumerate(plans):
            out.append(I.prepare_frame(pl.frame, store, fi, vmask, n_points, R=R[i], C=C[i], t=float(t[i]), own_pass_only=own_pass_only))
        return out

    t0 = time.time()
    R0 = np.empty((K, 3, 3))
    C0 = np.empty((K, 3))
    for i, pl in enumerate(plans):
        R0[i], C0[i], _ = _initial_pose(poses, pl.t_base, pass_id, pl.dt0, pl.dyaw0)
    icp_frames = prepare(R0, C0, t_base + dt0)
    log(f"  pass {pass_id}: {K} frames, prepared seed points in {time.time()-t0:.0f} s")

    x0 = np.zeros((K, len(free)))
    if "dt" in free:
        x0[:, free.index("dt")] = dt0
    if "dyaw" in free:
        x0[:, free.index("dyaw")] = dyaw0
    x0 = x0.ravel()

    r1 = PassRefiner(icp_frames, poses, t_base, pass_target, free=free, prior=prior, turning=turning)
    stats_before = r1.robust_stats(x0)
    sol1 = r1.solve(x0, log=log)
    theta1 = sol1["theta"]

    t1 = time.time()
    R1, C1 = r1.poses_for(theta1)
    full1 = r1.full_theta(theta1)
    icp_frames2 = prepare(R1, C1, t_base + full1[:, 0])
    log(f"  pass {pass_id}: re-prepared under refined pose in {time.time()-t1:.0f} s")

    r2 = PassRefiner(icp_frames2, poses, t_base, pass_target, free=free, prior=prior, turning=turning)
    sol2 = r2.solve(theta1, log=log)
    theta_final = sol2["theta"]
    stats_after = r2.robust_stats(theta_final)
    R_final, C_final = r2.poses_for(theta_final)
    full_final = r2.full_theta(theta_final)
    sigma3 = SIGMA_ACCEPT * r2.sigma

    # Acceptance (mapping.quality convention, not an RMS ratio -- see the module docstring note
    # above the constants: this edge metric's own noise floor is ~20 px RMS / ~8-11 px MAD even on
    # verified-clean frames, so "rms_after <= 0.85 rms_before" mostly measures resampling noise).
    results = []
    for i, pl in enumerate(plans):
        yaw, roll, pitch = geometry.euler_from_vehicle_rotation(R_final[i])
        nb, na = stats_before[i], stats_after[i]
        n_edge = na["n_edge"]
        row_free = np.asarray(theta_final).reshape(K, len(free))[i]
        beyond_3s = bool(np.any(np.abs(row_free) > sigma3[i]))
        median_ok = np.isfinite(na["du_median"]) and abs(na["du_median"]) <= MEDIAN_PX_ACCEPT and abs(na["dv_median"]) <= MEDIAN_PX_ACCEPT
        inlier_ok = np.isfinite(na["inlier8"]) and np.isfinite(nb["inlier8"]) and na["inlier8"] >= nb["inlier8"] - INLIER_SLACK
        flags = []
        if n_edge < N_EDGE_MIN:
            status = "interpolated"
        elif median_ok and inlier_ok and not beyond_3s:
            status = "refined"
        else:
            status = "kept"
            if beyond_3s:
                flags.append("beyond_3sigma")
            if not median_ok:
                flags.append("median_high")
            if not inlier_ok:
                flags.append("inlier_dropped")
        if pl.reassigned and status == "kept":
            flags.append("reassign_rejected")

        diag = (n_edge, nb["rms_px"], na["rms_px"], nb["du_median"], nb["dv_median"], na["du_median"], na["dv_median"], nb["inlier8"], na["inlier8"])
        if status == "kept":
            # fall back to the frame's own recorded pose (never the reassigned/refined hypothesis)
            k = pl.frame
            results.append(FrameResult(k, int(poses.pass_id[k]), *poses.origin[k], float(poses.roll[k]), float(poses.pitch[k]), float(poses.yaw[k]), "kept", "export", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *diag, ";".join(flags)))
        else:
            src = "align+icp" if (pl.dt0 or pl.dyaw0 or pl.reassigned) else "icp"
            results.append(FrameResult(pl.frame, pass_id, float(C_final[i, 0]), float(C_final[i, 1]), float(C_final[i, 2]), float(roll), float(pitch), float(yaw), status, src, *full_final[i].tolist(), *diag, ";".join(flags)))
    # raw (un-gated) fitted theta per frame, for diagnostics: unlike FrameResult.dt_s/dyaw/... (which
    # are zeroed for status=="kept"), this reports what the solve actually found for every frame,
    # regardless of acceptance -- used by the dev-run report to sanity-check against align.json and
    # to summarise the fitted-parameter distribution (S4 step 7).
    raw_theta = [
        {"frame": pl.frame, "status": results[i].status, "turning": bool(turning[i]),
         **{n: float(full_final[i, j]) for j, n in enumerate(PARAM_NAMES)}}
        for i, pl in enumerate(plans)
    ]
    summary = {"pass_id": pass_id, "K": K, "n_refined": sum(r.status == "refined" for r in results), "n_interpolated": sum(r.status == "interpolated" for r in results), "n_kept": sum(r.status == "kept" for r in results), "history1": sol1["history"], "history2": sol2["history"], "raw_theta": raw_theta, "s": round(time.time() - t0, 1)}
    log(f"  pass {pass_id} done in {summary['s']:.0f} s: refined {summary['n_refined']}, interpolated {summary['n_interpolated']}, kept {summary['n_kept']}")
    return results, summary
