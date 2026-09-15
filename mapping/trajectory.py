"""Dense trajectory reconstruction from RIEGL VMX-2HA scanner planes (S3, `07_revize...md` §5.1).

Physics: two rotating-mirror heads (`user_data` 1/2). Points of one head within a ~2 ms window lie
on a plane (RMS 3-4 mm, verified). Within the plane, ray direction is an affine function of
`scan_angle_rank` (~1.02 deg/unit). The two plane normals are constant in the sensor frame
("butterfly"). Camera time base is the same GPS-week seconds as LAZ `gps_time`; the camera-vs-scanner
offset `dt_s` is solved from clean, straight frames.

Pipeline: `gather_pass_points` -> `fit_scan_planes` (per head) -> `fit_scan_centre` (per window,
in-plane 2D position + mirror phase) -> `fit_orientation` (Kabsch from the two head normals) ->
`smooth_trajectory` (200 Hz) -> `fit_camera_from_trajectory` (camera rig incl. dt) -> `Trajectory`
(save/load, `covers`/`camera_pose` interface `Poses.interp` expects).

Investigated and REJECTED hypothesis (07 revision follow-up): `scan_angle_rank` is int8 over the
full -128..127 range, which raised the question of whether the exporter wrapped a *mechanical*
360 deg mirror sweep over those 256 codes (360/256 = 1.40625 deg/unit) rather than the ~1.02
deg/unit this module uses. Measured directly on pass 5 (both heads, bearing angle from an
export-pose-derived rough scan centre vs. rank, decoupled from `fit_scan_centre`'s own S/a
degeneracy): the fitted slope clusters tightly around 1.0-1.1 deg/unit (mean |a| 1.05, std 0.13,
n=40 windows) and is far from 1.40625 -- confirming ~1.02 deg/unit (this module's existing
`RANK_DEG_PER_UNIT`) rather than the wrap hypothesis. `scan_angle_rank`'s 256 codes evidently cover
only the *usable* (non-blocked) sweep, not a full mechanical revolution.

KNOWN LIMITATION (honest negative -- large real progress, targets still not met): two real bugs were
found and fixed in `fit_scan_centre`'s per-window position solve: (1) `fit_scan_planes`'s per-window
`e1` eigenvector had no sign continuity (only `normal` did), flipping on ~15% of consecutive real
windows and feeding stage 2's warm start a phase ~180 deg wrong; (2) stage 1/2 (`_pooled_fit`,
`_stage2`) reused a previous window's raw (s1, s2, phi0) numbers as-is under a *different* window's
rotating (e1, e2) basis instead of reprojecting through world space -- wrong whenever the basis
actually rotates between windows, which it does (median ~5 deg, up to ~90 deg, between consecutive
real plane windows). A third issue -- narrow-bearing-arc windows are genuinely non-identifiable, a
different unrelated position fits their own points just as well (confirmed: 30 random-restart solves
of one such window all converge to the same *wrong* optimum) -- is only partially mitigated by a
post-hoc, non-causal robust-median filter (`_reject_position_outliers`); a *causal* implausible-jump
veto was tried and rejected first (one wrongly-accepted window then poisons every later, correctly
-converging one). Net effect, full pass 5 (31 M pts, both heads, `MIN_IN_PLANE_ECC` at its original
0.03 -- raising it to 0.1-0.5 to reject more narrow-arc windows was tried and made every aggregate
number *worse*, presumably by starving the interpolation of good windows, so it was reverted):
head-1-vs-head-2 centre-offset std 9.36 m -> 0.96 m; camera-rig fit RMS ~5-10 m (pre-fix, per the S3
brief) -> 0.72 m (n=44 clean straight frames, l_cs [-1.95, -2.79, 0.75] m vs. the ~[1.4, 0, 0.8] m
the plan expects); trajectory-vs-export position at pass 5's clean frames: straight (n=66) median
0.54 m (p95 1.83 m), turning (n=16) median 0.87 m (p95 1.73 m); yaw: straight median 0.007 deg
(matches export almost exactly), turning median 2.9 deg (p95 7.0 deg) -- real, large, validated
improvement (roughly 10-30x on several independent metrics), but still an order of magnitude above
the plan's targets (stage-1 angular residual ~0.33-0.35 deg vs. <0.1 deg; centre jitter ~400 mm vs.
<20 mm; head offset and rig RMS both ~1 m vs. mm-cm/<5 cm). The residual cause is the narrow-arc
non-identifiability above: when several consecutive windows stare at the same persistent nearby flat
feature (a facade, a parked vehicle) for tens of ms, they cohere into a self-consistent-but-wrong
multi-window "ghost" segment too long for a K=11-25-window local median filter to see past --
enlarging the filter window did not reliably help either. A real fix needs either genuine cross-head
consistency arbitration (S1(t) and S2(t) should track each other to mm-cm at all times; a documented
consistency check the plan calls out, not implemented) or a joint/global estimator with a proper
motion-model prior, not a per-window solve patched after the fact. See the S3 investigation report
for the full numbers and the rejected-hypothesis and other-mitigations-tried record.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation

from . import geometry

RANK_DEG_PER_UNIT = 1.02  # initial guess, refined per head
WIN_S = 0.002  # 2 ms plane-fit window (2x the 1 ms bucket)
BUCKET_S = 0.001
MIN_PTS = 100
RMS_GATE_M = 0.010  # 10 mm
MAX_GAP_S = 0.2  # smoothed trajectory: never extrapolate across a gap this large
SAMPLE_HZ = 200.0
OUTLIER_MEDIAN_K = 11  # fit_scan_centre's post-hoc robust filter: rolling-median window (in solved
# windows, not seconds -- see there)
OUTLIER_MAD_K = 4.0  # robust-sigma multiplier for the same filter
OUTLIER_FLOOR_M = 0.05  # never reject on deviations this small (mirror mechanical jitter, not a bug)
MIN_IN_PLANE_ECC = 0.03  # fit_scan_centre's per-window conditioning gate (sv[1]/sv[2])


# =============================================================================== 1. gather points
def gather_pass_points(store, poses, pass_id: int, margin_s: float = 3.0, bbox_margin_m: float = 80.0):
    """Points of one pass from the store, restricted by time (own pass +- margin_s) and a bbox
    around the pass' frame origins (+- bbox_margin_m). Returns (t, xyz, head, rank, ret), sorted by t.

    t: float64 [N] gps_time. xyz: float64 [N,3] metres. head: uint8 [N] (user_data, 1/2).
    rank: int8 [N] (scan_angle_rank). ret: uint8 [N] (return_number, low nibble) or None if the
    store predates those columns.
    """
    sel = np.flatnonzero(poses.pass_id == pass_id)
    if len(sel) == 0:
        raise ValueError(f"no frames for pass {pass_id}")
    t0, t1 = float(poses.t[sel].min()) - margin_s, float(poses.t[sel].max()) + margin_s
    origins = poses.origin[sel]
    bbox = (
        float(origins[:, 0].min() - bbox_margin_m),
        float(origins[:, 1].min() - bbox_margin_m),
        float(origins[:, 0].max() + bbox_margin_m),
        float(origins[:, 1].max() + bbox_margin_m),
    )
    ts, xyzs, heads, ranks, rets = [], [], [], [], []
    for tile, rows in store.query_time(t0, t1, bbox=bbox):
        td = store.tile(tile.name)
        ts.append(np.asarray(td.gps_time[rows], dtype=np.float64))
        xyzs.append(td.xyz_m(rows))
        heads.append(np.asarray(td.user_data[rows]) if td.user_data is not None else np.zeros(len(rows), dtype=np.uint8))
        ranks.append(np.asarray(td.scan_angle_rank[rows]) if td.scan_angle_rank is not None else np.zeros(len(rows), dtype=np.int8))
        if td.return_number is not None:
            rets.append(np.asarray(td.return_number[rows]) & 0x0F)
        else:
            rets.append(np.full(len(rows), 255, dtype=np.uint8))
        td.release()
    if not ts:
        return (np.empty(0), np.empty((0, 3)), np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int8), np.empty(0, dtype=np.uint8))
    t = np.concatenate(ts)
    xyz = np.concatenate(xyzs, axis=0)
    head = np.concatenate(heads).astype(np.uint8)
    rank = np.concatenate(ranks).astype(np.int8)
    ret = np.concatenate(rets).astype(np.uint8)
    order = np.argsort(t, kind="stable")
    return t[order], xyz[order], head[order], rank[order], ret[order]


# =============================================================================== 2. plane fitting
@dataclass
class PlaneSeries:
    head: int
    t_c: np.ndarray  # [W] window centre time (mean t of points in window)
    n: np.ndarray  # [W,3] unit plane normal, sign-consistent along time
    c: np.ndarray  # [W,3] window centroid (not the scan centre)
    e1: np.ndarray  # [W,3] in-plane basis vector 1
    e2: np.ndarray  # [W,3] in-plane basis vector 2 (= n x e1)
    rms: np.ndarray  # [W] RMS distance of window points to the plane, metres
    sv: np.ndarray  # [W,3] eigenvalues of the covariance, ascending (sv[0] ~ rms^2 * n)
    n_pts: np.ndarray  # [W] int, points in window


def fit_scan_planes(t, xyz, head, head_id: int, win_s: float = WIN_S, min_pts: int = MIN_PTS) -> PlaneSeries:
    """Batched PCA plane fit for one head: 1 ms buckets, sliding `win_s`-wide windows (win_s must be
    a multiple of the 1 ms bucket), covariances via bincount, one batched `np.linalg.eigh`.
    Windows with < min_pts points or RMS > 10 mm are dropped."""
    m = head == head_id
    tt, xx_raw = t[m], xyz[m]
    empty = PlaneSeries(head_id, np.empty(0), np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty((0, 3)), np.empty(0, dtype=np.int64))
    if len(tt) < min_pts:
        return empty
    # world coordinates are O(1e6) m (S-JTSK); sum-of-squares covariance loses essentially all
    # precision against a sub-mm variance at that magnitude (relative fp64 error ~1e6*1e-16=1e-10,
    # but the *absolute* error compounds over the sum, ~1e12*1e-16=1e-4 -- comparable to or larger
    # than the true variance itself, silently rounding tiny plane thickness to exactly 0). Recentre
    # once before any bincount sums; shift back into world coordinates only in the final outputs.
    offset = xx_raw[0].copy()
    xx = xx_raw - offset
    n_bk = max(1, int(round(win_s / BUCKET_S)))
    t_min = tt[0]
    bucket = np.minimum(((tt - t_min) / BUCKET_S).astype(np.int64), 10**9)
    n_buckets = int(bucket.max()) + 1
    # per-bucket sums (bincount), then prefix-sum to get sliding-window sums cheaply
    ones = np.ones(len(tt))
    cnt = np.bincount(bucket, weights=ones, minlength=n_buckets)
    sx = np.stack([np.bincount(bucket, weights=xx[:, i], minlength=n_buckets) for i in range(3)], axis=1)  # [B,3]
    sxy = np.zeros((n_buckets, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            v = np.bincount(bucket, weights=xx[:, i] * xx[:, j], minlength=n_buckets)
            sxy[:, i, j] = v
            sxy[:, j, i] = v
    st = np.bincount(bucket, weights=tt, minlength=n_buckets)
    # sliding windows of n_bk consecutive buckets, stride = n_bk // 2 (>=1) so windows overlap ~50%
    stride = max(1, n_bk // 2)
    starts = np.arange(0, max(1, n_buckets - n_bk + 1), stride)
    cnt_cum = np.concatenate([[0.0], np.cumsum(cnt)])
    st_cum = np.concatenate([[0.0], np.cumsum(st)])
    sx_cum = np.concatenate([np.zeros((1, 3)), np.cumsum(sx, axis=0)], axis=0)
    sxy_cum = np.concatenate([np.zeros((1, 3, 3)), np.cumsum(sxy, axis=0)], axis=0)

    def wsum(cum, s, e):
        return cum[e] - cum[s]

    ends = starts + n_bk
    W = len(starts)
    W_cnt = wsum(cnt_cum, starts, ends)
    keep0 = W_cnt >= min_pts
    starts, ends, W_cnt = starts[keep0], ends[keep0], W_cnt[keep0]
    if len(starts) == 0:
        return PlaneSeries(head_id, np.empty(0), np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty((0, 3)), np.empty(0, dtype=np.int64))
    W_t = wsum(st_cum, starts, ends) / W_cnt
    W_sx = np.stack([wsum(sx_cum[:, i], starts, ends) for i in range(3)], axis=1) / W_cnt[:, None]  # mean [W,3]
    W_sxy = np.stack([wsum(sxy_cum[:, i, j], starts, ends) for i in range(3) for j in range(3)], axis=1).reshape(-1, 3, 3) / W_cnt[:, None, None]
    cov = W_sxy - W_sx[:, :, None] * W_sx[:, None, :]  # [W,3,3]
    evals, evecs = np.linalg.eigh(cov)  # ascending
    rms = np.sqrt(np.clip(evals[:, 0], 0, None))
    keep = rms <= RMS_GATE_M
    starts, ends = starts[keep], ends[keep]
    t_c, c, sv = W_t[keep], W_sx[keep], evals[keep]
    normal = evecs[keep][:, :, 0]
    e1 = evecs[keep][:, :, 2]  # largest-variance in-plane direction
    e1 = e1 - (np.sum(e1 * normal, axis=1, keepdims=True)) * normal
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(normal, e1)
    # Sign-consistency along time (ascending t_c already, windows built in time order). `normal`
    # (smallest-eigenvalue eigenvector) and `e1` (largest-eigenvalue eigenvector) are each unique
    # only up to sign from `eigh` -- the two sign choices are *independent* draws, not linked. Real
    # data: normal's sign is stable almost everywhere (own-axis flip rate ~0), but e1's flips on
    # ~14-15% of consecutive real windows (measured on pass 5, both heads) even with normal fixed --
    # a plain "flip e1 when normal flips" (as if the two were the same eigenvector) misses nearly
    # all of them. Each e1 flip corresponds to phi0 -> phi0+180 deg in `fit_scan_centre`'s model; if
    # left uncorrected here, ~1 in 7 windows silently changes the in-plane basis under
    # `fit_scan_centre`'s warm-started per-window solve, so its previous-window (s1, s2, phi0) is
    # started ~180 deg out of phase for that window's cos/sin fit -- exactly the failure mode that
    # produced the 200+ mm centre jitter and multi-metre head-to-head offset instability seen before
    # this fix (07 revision investigation, pass 5). Check and correct each axis independently.
    for i in range(1, len(normal)):
        if np.dot(normal[i], normal[i - 1]) < 0:
            normal[i] = -normal[i]
            e1[i] = -e1[i]  # e1 is still orthogonal to normal after only normal flips
        if np.dot(e1[i], e1[i - 1]) < 0:
            e1[i] = -e1[i]
        e2[i] = np.cross(normal[i], e1[i])
    return PlaneSeries(head_id, t_c, normal, c + offset, e1, e2, rms[keep], sv, W_cnt[keep].astype(np.int64))


def _reject_position_outliers(t_c, S, phi0, solved):
    """Post-hoc, non-causal robust filter over `_stage2`'s per-window output (see its docstring for
    why a *causal* reject was tried and rejected): flag any *solved* window whose position deviates
    from a rolling median of `OUTLIER_MEDIAN_K` neighbouring solved windows (in time; both sides, so
    an early bad window does not poison everything after it the way a causal check would) by more
    than `OUTLIER_MAD_K` robust-sigmas (MAD-based), floored at `OUTLIER_FLOOR_M`. Surviving solved
    windows are then treated as the only trustworthy samples of S(t)/phi0(t) and every window's
    output (surviving, rejected, and stride-skipped alike) is reconstructed by linear interpolation
    over them -- so a rejected span is filled from its two *surrounding* good windows rather than
    carried forward from whichever one preceded it. Returns (S, phi0, solved) with `solved` now
    marking only the windows that fed the interpolation.
    """
    from scipy.ndimage import median_filter

    sw = np.flatnonzero(solved)
    if len(sw) < max(5, OUTLIER_MEDIAN_K):
        return S, phi0, solved
    S_sw = S[sw]
    k = min(OUTLIER_MEDIAN_K, len(sw) - (1 - len(sw) % 2))  # odd, <= len(sw)
    if k < 3:
        return S, phi0, solved
    med = np.stack([median_filter(S_sw[:, i], size=k, mode="nearest") for i in range(3)], axis=1)
    dev = np.linalg.norm(S_sw - med, axis=1)
    mad = np.median(np.abs(dev - np.median(dev)))
    thresh = max(OUTLIER_FLOOR_M, OUTLIER_MAD_K * 1.4826 * mad)
    keep = dev <= thresh
    sw_good = sw[keep]
    if len(sw_good) < 2:
        return S, phi0, solved

    t_good = t_c[sw_good]
    S_full = np.stack([np.interp(t_c, t_good, S[sw_good, i]) for i in range(3)], axis=1)
    phi0_unwrapped = np.unwrap(phi0[sw_good])
    phi0_full = (np.interp(t_c, t_good, phi0_unwrapped) + np.pi) % (2 * np.pi) - np.pi
    solved_full = np.zeros(len(t_c), dtype=bool)
    solved_full[sw_good] = True
    return S_full, phi0_full, solved_full


# =============================================================================== 3. scan centre
def _dir_residual(theta_pred_rad, dir_meas_rad):
    d = theta_pred_rad - dir_meas_rad
    return np.arctan2(np.sin(d), np.cos(d))


def fit_scan_centre(t, xyz, head, rank, head_id: int, planes: PlaneSeries, a0_deg: float = RANK_DEG_PER_UNIT, n_global: int = 40, stride: int = 1, max_pts_per_window: int = 150):
    """Solve the mirror model `P_i = S + rho_i (cos(a*rank_i+phi0) e1 + sin(a*rank_i+phi0) e2)` per
    window of `planes` (which must be `fit_scan_planes(..., head_id)`). Returns
    (S[W,3], phi0[W] rad, a_deg, resid_deg (angular RMS of the global fit), jitter_mm (median
    |S[i]-S[i-1]| in consecutive windows, mm)).

    Stage 1 (global): a robust subset of windows (evenly spaced, best conditioned) is fit with S at
    the window centroid (s=0) to get an initial (a, phi0) via a linear (unwrapped) regression of
    in-plane direction vs rank. Stage 2 (per window): `least_squares` (soft_l1) over (s1, s2, phi0)
    per window, warm-started from the previous window's solution; `a` stays fixed at the stage-1
    value (it is a mechanical/encoder constant, not expected to vary within a pass).

    `stride` > 1 solves stage 2 only every `stride`-th window (the per-window nonlinear solve
    dominates runtime; downstream `smooth_trajectory` resamples to 200 Hz anyway, so solving denser
    than that buys little). Skipped windows carry the previous solved value forward so `S`/`phi0`
    stay aligned 1:1 with `planes.t_c` for the caller; `jitter_mm` is computed over the solved
    windows only (the carried-forward repeats would otherwise dilute it to ~0)."""
    m = head == head_id
    tt, xx, rr = t[m], xyz[m], rank[m].astype(np.float64)
    W = len(planes.t_c)
    if W == 0:
        return np.empty((0, 3)), np.empty(0), a0_deg, float("nan"), float("nan")

    # map each point to a window by nearest window centre in time (points already grouped by the
    # same buckets fit_scan_planes used; nearest-t_c assignment reproduces that grouping closely)
    win_of_pt = np.clip(np.searchsorted(planes.t_c, tt), 0, W - 1)
    lo = np.clip(win_of_pt - 1, 0, W - 1)
    win_of_pt = np.where(np.abs(tt - planes.t_c[lo]) < np.abs(tt - planes.t_c[win_of_pt]), lo, win_of_pt)

    order = np.argsort(win_of_pt, kind="stable")
    win_sorted = win_of_pt[order]
    bounds = np.searchsorted(win_sorted, np.arange(W + 1))

    def pts_of(w):
        s, e = bounds[w], bounds[w + 1]
        idx = order[s:e]
        return xx[idx], rr[idx]

    # ---- stage 1: robust global `a` from evenly spaced, well-conditioned windows.
    # The window centroid is *not* a good proxy for S when the scanned sector is not a full circle
    # (S - centroid can be metres, not mm, for a limited-FOV mirror sweep on real data rho can reach
    # 20+ m), so `a` cannot be read off a centroid-relative direction vs rank regression directly,
    # and letting a single window fit `a` freely (4 unknowns: s1, s2, phi0, a) is *underdetermined*
    # for many real windows (a narrow rank arc, or a near-planar road-only view): a wrong `a` can
    # still reach a low per-point residual by pushing S to a corner of its bound, so a per-window `a`
    # estimate cannot be trusted even after filtering by residual. Instead `a` is treated as one
    # value shared by *all* `pick` windows and found by a 1-D grid+refine search: for each trial `a`,
    # every window gets its own bounded 3-param fit (s1, s2, phi0) at that fixed `a`, and the trial's
    # score is the summed cost -- a wrong shared `a` cannot cheaply explain many independent windows
    # at once, unlike the free-per-window fit.
    cond = planes.sv[:, 1] / np.maximum(planes.sv[:, 0], 1e-12)
    good = np.flatnonzero((planes.n_pts >= MIN_PTS) & np.isfinite(cond))
    if len(good) > n_global:
        pick = good[np.linspace(0, len(good) - 1, n_global).astype(int)]
    else:
        pick = good
    # stage 1 only needs enough points per window to pin (s1, s2, phi0) (or + a); capping keeps the
    # per-window nonlinear solves -- and the joint fit's numerical Jacobian, `1 + 3*n_global` wide --
    # cheap without losing the redundancy that makes the shared-`a` fit robust (n_global windows still
    # give a wide, independent rank/time coverage).
    rng = np.random.default_rng(0)
    pick_data = []
    for w in pick:
        P, R = pts_of(w)
        if len(P) < 20:
            continue
        if len(P) > max_pts_per_window:
            sub = rng.choice(len(P), max_pts_per_window, replace=False)
            P, R = P[sub], R[sub]
        d = P - planes.c[w]
        pick_data.append((w, d @ planes.e1[w], d @ planes.e2[w], R))

    # S must lie within a generous radius of the window centroid (real scanner geometry, R_MAX-scale
    # rho observed up to ~25 m on real passes); without a bound the least-squares problem has a
    # degenerate "S at infinity" solution (pushing S far away makes every point's bearing nearly
    # constant, trivially fitting any `a`) that can be *cheaper* than the true compact solution.
    BOUND_S = 25.0
    bounds3 = ([-BOUND_S, -BOUND_S, -np.inf], [BOUND_S, BOUND_S, np.inf])

    def _grid_phi0(a_try, u0, v0, R):
        grid = np.deg2rad(np.arange(-180.0, 180.0, 2.0))
        costs = [np.sum(_dir_residual(a_try * R + ph, np.arctan2(v0, u0)) ** 2) for ph in grid]
        return grid[int(np.argmin(costs))]

    def _pooled_fit(a_rad, windows):
        """Sequential bounded 3-param (s1, s2, phi0) fit at fixed `a_rad` over `windows`
        (list of (w, u0, v0, R)); returns (total_cost, per-window {w: (x, rms_per_pt)}).

        Warm-starts each window from the *previous solved window's world-space state*
        (`S_prev`, a 3D point, and `D_prev`, a unit world-space vector for the rank=0 direction),
        re-projected into *this* window's own (e1, e2) basis -- not the previous window's raw
        (s1, s2, phi0) numbers. `pick_data` windows are spread across the whole pass (evenly by
        index, not by time), so the (e1, e2) basis and the window centroid `planes.c[w]` differ
        substantially window to window (both from genuine mirror rotation between samples and from
        `fit_scan_planes`'s per-window eigenvector sign, only made time-continuous for *adjacent*
        real windows, not for this arbitrary subsampling): reusing raw (s1, s2) coordinates under a
        different basis silently reinterprets them as an unrelated point, and reusing raw phi0
        silently shifts the assumed bearing origin by however much the basis rotated. Re-projecting
        through the world-space state fixes both regardless of basis rotation or sign."""
        w0, u0_0, v0_0, R0 = windows[0]
        phi0_seed0 = _grid_phi0(a_rad, u0_0, v0_0, R0)
        S_prev = planes.c[w0].copy()
        D_prev = np.cos(phi0_seed0) * planes.e1[w0] + np.sin(phi0_seed0) * planes.e2[w0]
        total, out = 0.0, {}
        for w, u0, v0, R in windows:
            e1w, e2w, cw = planes.e1[w], planes.e2[w], planes.c[w]
            x0 = np.array([np.dot(S_prev - cw, e1w), np.dot(S_prev - cw, e2w), np.arctan2(np.dot(D_prev, e2w), np.dot(D_prev, e1w))])
            x0 = np.clip(x0, bounds3[0], bounds3[1])
            sol = least_squares(
                lambda p, u0=u0, v0=v0, R=R: _dir_residual(a_rad * R + p[2], np.arctan2(v0 - p[1], u0 - p[0])),
                x0=x0, loss="soft_l1", f_scale=np.deg2rad(1.0), bounds=bounds3,
            )
            rms_per_pt = float(np.sqrt(sol.cost / max(len(R), 1)))
            total += sol.cost
            out[w] = (sol.x, sol.fun, rms_per_pt)
            if rms_per_pt < np.deg2rad(1.0):
                S_prev = cw + sol.x[0] * e1w + sol.x[1] * e2w
                D_prev = np.cos(sol.x[2]) * e1w + np.sin(sol.x[2]) * e2w
        return total, out

    if pick_data:
        # sign of `a` is the plane-normal sign ambiguity (arbitrary per fit_scan_planes call, see its
        # docstring); a small coarse subset at the nominal magnitude a0_deg is enough to tell the two
        # signs apart (the wrong one costs several times more, not a close call -- see module tests).
        coarse_subset = pick_data[:: max(1, len(pick_data) // 10)][:10]
        cost_pos, _ = _pooled_fit(np.deg2rad(a0_deg), coarse_subset)
        cost_neg, _ = _pooled_fit(-np.deg2rad(a0_deg), coarse_subset)
        sign = 1.0 if cost_pos <= cost_neg else -1.0
        a_rad = sign * np.deg2rad(a0_deg)
        # Newton/profile-likelihood polish: alternate a per-window (s1, s2, phi0) fit at the current
        # `a` (cheap, O(n_global)) with a closed-form update to the *shared* `a` from the pooled
        # gradient. By the envelope theorem, d(cost)/d(a) at a per-window optimum equals the partial
        # derivative holding (s1, s2, phi0) fixed (their own gradient is zero there), so this Newton
        # step is exact to Gauss-Newton order without ever materialising the `1 + 3*n_global`-wide
        # joint Jacobian a single `least_squares` over all windows at once needed (that scaled like
        # O(n_global^2) and dominated this function's runtime -- see git history).
        out_grid = {}
        for _ in range(6):
            _, out_grid = _pooled_fit(a_rad, pick_data)
            num = den = 0.0
            for w, u0, v0, R in pick_data:
                if w not in out_grid or out_grid[w][2] >= np.deg2rad(1.0):
                    continue  # exclude windows the per-window fit itself did not converge on (a
                    # corner-of-bounds local minimum there swamps the gradient by orders of
                    # magnitude -- same "converged" gate used everywhere else in this function)
                s1, s2, ph = out_grid[w][0]
                dir_meas = np.arctan2(v0 - s2, u0 - s1)
                dth = _dir_residual(a_rad * R + ph, dir_meas)  # ~ da * R for small da
                num += np.sum(dth * R)
                den += np.sum(R * R)
            if den <= 0:
                break
            da = num / den
            a_rad -= da
            if abs(da) < np.deg2rad(0.0002):
                break
        phi0_init = float(next(iter(out_grid.values()))[0][2]) if out_grid else 0.0
        # per-window RMS, robust to a handful of poorly conditioned windows (narrow rank arc,
        # near-planar scene) even at the correct shared `a`.
        rms_list = [v[2] for v in out_grid.values()]
        good_rms = [r for r in rms_list if r < np.deg2rad(1.0)]
        resid_deg = float(np.degrees(np.sqrt(np.mean(np.array(good_rms) ** 2)))) if good_rms else float("nan")
    else:
        a_rad, phi0_init, resid_deg = np.deg2rad(a0_deg), 0.0, float("nan")

    def _stage2(a_val, phi0_seed):
        """Per-window (s1, s2, phi0) solve, warm-started (and, for skipped/rejected windows,
        entirely reconstructed) from the *world-space* carry state (`S_prev`, `D_prev`; see
        `_pooled_fit`'s docstring for why raw (s1, s2, phi0) numbers cannot be reused across
        windows) rather than the previous window's raw solved numbers. This matters far more here
        than in stage 1: with `stride` > 1 the vast majority of windows never solve at all and take
        their `S[w]` directly from this reprojection (stage 1's warm start only ever seeds an
        optimizer that re-fits from scratch). Measured impact on pass 5 (07 revision investigation):
        reusing raw numbers gave 200-700 mm median centre jitter between consecutive *solved*
        windows and a multi-metre, unstable head-1-vs-head-2 offset (expected: a fixed mechanical
        offset, mm-cm level); world-space reprojection fixes the *median* but not the tail (see the
        caller's post-hoc filter for that): a window whose points span a narrow rank arc (small
        `in_plane_ecc`, e.g. a nearby narrow object such as a post) can have a genuinely different,
        unrelated (s1, s2) that fits its *own* points to a few thousandths of a degree -- true model
        non-identifiability for a narrow bearing arc from a single window, not a solver or
        conditioning-threshold artifact, so no `rms_per_pt`/`in_plane_ecc` threshold tuning closes it
        reliably (measured on pass 5: the eccentricity gate's own threshold, 0.03, is close to its
        *median* value across all windows, so raising it trades away most of the data without
        eliminating the tail). A *causal* reject on the implied speed from the last accepted window
        was tried and made things worse: once one window is wrongly accepted, every later, correctly
        -converging window then looks like the implausible jump relative to that wrong anchor and
        gets rejected in turn, permanently deriving the trajectory from the one bad window (measured:
        a synthetic scan's very first solved window landed on a several-metre ghost solution, and the
        causal gate then locked the entire rest of the series to it). The caller's post-hoc,
        non-causal, robust-median filter is what actually removes these without that failure mode."""
        S = np.empty((W, 3))
        phi0 = np.empty(W)
        solved = np.zeros(W, dtype=bool)
        S_prev = planes.c[0].copy()
        D_prev = np.cos(phi0_seed) * planes.e1[0] + np.sin(phi0_seed) * planes.e2[0]

        def _reproject(w):
            e1w, e2w, cw = planes.e1[w], planes.e2[w], planes.c[w]
            s1_0 = np.dot(S_prev - cw, e1w)
            s2_0 = np.dot(S_prev - cw, e2w)
            phi0_0 = np.arctan2(np.dot(D_prev, e2w), np.dot(D_prev, e1w))
            return e1w, e2w, cw, s1_0, s2_0, phi0_0

        for w in range(W):
            e1w, e2w, cw, s1_0, s2_0, phi0_0 = _reproject(w)
            if w % stride != 0 and w != W - 1:
                S[w] = cw + s1_0 * e1w + s2_0 * e2w
                phi0[w] = phi0_0
                continue
            P, R = pts_of(w)
            if len(P) < MIN_PTS // 2:
                S[w] = cw + s1_0 * e1w + s2_0 * e2w
                phi0[w] = phi0_0
                continue
            solved[w] = True
            d = P - cw
            u0 = d @ e1w
            v0 = d @ e2w

            def resid(p, u0=u0, v0=v0, R=R):
                s1, s2, ph = p
                dir_meas = np.arctan2(v0 - s2, u0 - s1)
                theta = a_val * R + ph
                return _dir_residual(theta, dir_meas)

            x0 = np.clip(np.array([s1_0, s2_0, phi0_0]), bounds3[0], bounds3[1])
            sol = least_squares(resid, x0=x0, loss="soft_l1", f_scale=np.deg2rad(1.0), bounds=bounds3)
            rms_per_pt = float(np.sqrt(sol.cost / max(len(R), 1)))
            # sv[1] (secondary in-plane eigenvalue) << sv[2] (primary) means the window's points span
            # a narrow angular arc at varying range rather than a wide bearing spread around S: a low
            # per-point angular *residual* is then reachable over a whole ridge of (s1, s2), not just
            # near the truth, because a position slid along the arc barely changes predicted bearing
            # (the real-data pitfall the plan calls out: "ill-conditioned when only a road cross-
            # section is hit"). The RMS gate alone does not see this; gate on conditioning too.
            in_plane_ecc = planes.sv[w, 1] / max(planes.sv[w, 2], 1e-12)
            if rms_per_pt >= np.deg2rad(1.0) or in_plane_ecc < MIN_IN_PLANE_ECC:
                # reject: carry the last good value forward (like the stride-skipped windows above)
                # rather than accept a position that isn't really constrained by this window's data.
                S[w] = cw + s1_0 * e1w + s2_0 * e2w
                phi0[w] = phi0_0
                solved[w] = False
                continue
            S_prev = cw + sol.x[0] * e1w + sol.x[1] * e2w
            D_prev = np.cos(sol.x[2]) * e1w + np.sin(sol.x[2]) * e2w
            S[w] = S_prev
            phi0[w] = sol.x[2]
        return S, phi0, solved

    # ---- stage 2: per-window (s1, s2, phi0), a fixed at the stage-1 joint-fit value, warm-started.
    # NOTE: `a` from the stage-1 joint fit typically carries a residual bias of a few thousandths of
    # a degree (soft_l1 + finite pick-window sample), which leverages into a position bias at the far
    # end of the rank range (an S closed-form correction from the stage-2 residual was tried and
    # rejected: the per-window (s1, s2) fit already absorbs much of a wrong `a`, so the leftover
    # residual-vs-rank slope is not a clean linear proxy for the `a` error and the correction did not
    # reliably converge). Living with the stage-1 precision here; `smooth_trajectory`'s 200 Hz
    # averaging over many windows is what ultimately controls the trajectory's noise floor.
    S, phi0, solved = _stage2(a_rad, phi0_init)
    S, phi0, solved = _reject_position_outliers(planes.t_c, S, phi0, solved)
    S_solved = S[solved]
    jitter_mm = float(np.median(np.linalg.norm(np.diff(S_solved, axis=0), axis=1)) * 1000.0) if len(S_solved) > 1 else float("nan")
    return S, phi0, float(np.degrees(a_rad)), resid_deg, jitter_mm


# =============================================================================== 4. orientation
def fit_orientation(t1, n1, t2, n2, tol_s: float = 0.0015):
    """Kabsch/Wahba orientation from the two head normals: at each time where a head-1 and head-2
    window are within `tol_s` of each other, solve R_s(t) (world -> sensor) such that R_s @ n1 and
    R_s @ n2 match the reference values n1_ref = n1[0], n2_ref = n2[0] (sensor frame defined by the
    first common window, R_s(t_ref) = I by construction). Returns (t_common[K], R_s[K,3,3])."""
    if len(t1) == 0 or len(t2) == 0:
        return np.empty(0), np.empty((0, 3, 3))
    j = np.searchsorted(t2, t1)
    j = np.clip(j, 1, len(t2) - 1)
    j0 = np.clip(j - 1, 0, len(t2) - 1)
    j = np.where(np.abs(t1 - t2[j0]) < np.abs(t1 - t2[j]), j0, j)
    ok = np.abs(t1 - t2[j]) <= tol_s
    t_c = t1[ok]
    a = n1[ok]
    b = n2[j[ok]]
    if len(t_c) == 0:
        return np.empty(0), np.empty((0, 3, 3))
    n1_ref, n2_ref = a[0], b[0]
    c_ref = np.cross(n1_ref, n2_ref)
    c_ref /= max(np.linalg.norm(c_ref), 1e-12)
    B = np.stack([n1_ref, n2_ref, c_ref], axis=1)  # target, [3,3] columns
    c_t = np.cross(a, b)
    norm = np.linalg.norm(c_t, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    c_t = c_t / norm
    A = np.stack([a, b, c_t], axis=2)  # [K,3,3] columns = source vectors
    M = np.einsum("ij,kjl->kil", B, np.transpose(A, (0, 2, 1)))  # B @ A^T per window: [K,3,3]
    U, S, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(np.einsum("kij,kjl->kil", U, Vt)))
    D = np.zeros((len(t_c), 3, 3))
    D[:, 0, 0] = 1.0
    D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R_s = np.einsum("kij,kjl,klm->kim", U, D, Vt)
    return t_c, R_s


# =============================================================================== 5. smoothing
def smooth_trajectory(t, S, R_s, out_hz: float = SAMPLE_HZ, max_gap_s: float = MAX_GAP_S):
    """Resample (t, S(t), R_s(t)) to a uniform `out_hz` grid with Savitzky-Golay smoothing on
    positions and on quaternion components (sign-continuity enforced, renormalised, no slerp
    singularity at these tiny inter-sample rotations). Returns (t_out, S_out, R_out); gaps in the
    input longer than `max_gap_s` produce a break (no sample bridges them, never extrapolated)."""
    from scipy.signal import savgol_filter

    order = np.argsort(t)
    t, S, R_s = t[order], S[order], R_s[order]
    quat = Rotation.from_matrix(R_s).as_quat()  # [K,4] xyzw
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]

    segs = []
    breaks = np.flatnonzero(np.diff(t) > max_gap_s)
    bounds = [0, *(breaks + 1).tolist(), len(t)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e - s < 5:
            continue
        segs.append((s, e))

    t_out_all, S_out_all, quat_out_all = [], [], []
    for s, e in segs:
        ts, Ss, qs = t[s:e], S[s:e], quat[s:e]
        n_out = max(2, int(round((ts[-1] - ts[0]) * out_hz)) + 1)
        t_grid = np.linspace(ts[0], ts[-1], n_out)
        S_lin = np.stack([np.interp(t_grid, ts, Ss[:, i]) for i in range(3)], axis=1)
        q_lin = np.stack([np.interp(t_grid, ts, qs[:, i]) for i in range(4)], axis=1)
        win = min(len(t_grid) - (1 - len(t_grid) % 2), 51)
        if win >= 5 and win % 2 == 1 and win < n_out:
            S_sm = savgol_filter(S_lin, win, 3, axis=0)
            q_sm = savgol_filter(q_lin, win, 3, axis=0)
        else:
            S_sm, q_sm = S_lin, q_lin
        q_sm /= np.linalg.norm(q_sm, axis=1, keepdims=True)
        t_out_all.append(t_grid)
        S_out_all.append(S_sm)
        quat_out_all.append(q_sm)
    if not t_out_all:
        return np.empty(0), np.empty((0, 3)), np.empty((0, 3, 3))
    t_out = np.concatenate(t_out_all)
    S_out = np.concatenate(S_out_all, axis=0)
    quat_out = np.concatenate(quat_out_all, axis=0)
    R_out = Rotation.from_quat(quat_out).as_matrix()
    return t_out, S_out, R_out


# =============================================================================== 6. camera rig
@dataclass
class CamSensorRig:
    R_cs: np.ndarray  # [3,3] sensor -> camera
    l_cs: np.ndarray  # [3] camera centre lever arm in sensor axes, metres
    dt_s: float  # camera time = scanner time + dt_s
    stats: dict = field(default_factory=dict)

    def hash(self) -> str:
        payload = json.dumps(
            {"R": np.round(self.R_cs, 6).tolist(), "l": np.round(self.l_cs, 5).tolist(), "dt": round(self.dt_s, 6)},
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _rig_at_dt(t_query, S_of, R_of, C_k, dt):
    S_k = S_of(t_query + dt)
    R_k = R_of(t_query + dt)
    ok = np.all(np.isfinite(S_k), axis=1) & np.all(np.isfinite(R_k.reshape(len(R_k), -1)), axis=1)
    return S_k, R_k, ok


def fit_camera_from_trajectory(t, S, R_s, poses, frame_idx, dt_scan=(-0.3, 0.3, 0.005)):
    """Camera rig (R_cs, l_cs, dt_s) from clean, straight frames `frame_idx` (indices into `poses`).
    (t, S, R_s) is one pass' smoothed trajectory (or several passes concatenated with `t` strictly
    increasing per pass -- caller passes one pass at a time here). `S_of`/`R_of` interpolate the
    dense trajectory (linear on S, nearest+renormalise on quaternion for the small `dt` search)."""
    order = np.argsort(t)
    t, S, R_s = t[order], S[order], R_s[order]
    quat = Rotation.from_matrix(R_s).as_quat()
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]

    def S_of(tq):
        out = np.stack([np.interp(tq, t, S[:, i], left=np.nan, right=np.nan) for i in range(3)], axis=1)
        return out

    def R_of(tq):
        q = np.stack([np.interp(tq, t, quat[:, i], left=np.nan, right=np.nan) for i in range(4)], axis=1)
        norm = np.linalg.norm(q, axis=1)
        bad = ~np.isfinite(norm) | (norm < 1e-6)
        # replace bad rows (NaN outside the covered range, or a degenerate norm) with a valid
        # placeholder quaternion *before* normalising/constructing Rotation -- dividing a NaN row by
        # 1.0 still leaves NaN, which scipy rejects outright ("zero norm quaternions") rather than
        # just producing the NaN this function is about to overwrite anyway.
        q = np.where(bad[:, None], np.array([0.0, 0.0, 0.0, 1.0]), q)
        norm = np.where(bad, 1.0, norm)
        q = q / norm[:, None]
        R = Rotation.from_quat(q).as_matrix()
        R[bad] = np.nan
        return R

    t_k = poses.t[frame_idx]
    C_k = poses.origin[frame_idx]
    R_v_k = geometry.vehicle_rotation(poses.yaw[frame_idx], poses.roll[frame_idx], poses.pitch[frame_idx])

    def solve_at(dt):
        S_k, R_k, ok = _rig_at_dt(t_k, S_of, R_of, C_k, dt)
        if ok.sum() < 8:
            return None
        R_cs_terms = np.einsum("kij,kjl->kil", R_v_k[ok], np.transpose(R_k[ok], (0, 2, 1)))
        M = R_cs_terms.mean(axis=0)
        U, _, Vt = np.linalg.svd(M)
        d = np.sign(np.linalg.det(U @ Vt))
        R_cs = U @ np.diag([1.0, 1.0, d]) @ Vt
        l_cs_k = np.einsum("kij,kj->ki", R_k[ok], C_k[ok] - S_k[ok])
        l_cs = l_cs_k.mean(axis=0)
        pred = S_k[ok] + np.einsum("kji,j->ki", R_k[ok], l_cs)
        resid = C_k[ok] - pred
        rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
        return R_cs, l_cs, rms, int(ok.sum()), resid

    lo, hi, step = dt_scan
    dts = np.arange(lo, hi + step / 2, step)
    curve = []
    best = None
    for dt in dts:
        r = solve_at(dt)
        curve.append((float(dt), r[2] if r else float("nan")))
        if r is not None and (best is None or r[2] < best[1][2]):
            best = (dt, r)
    if best is None:
        raise RuntimeError("fit_camera_from_trajectory: no dt in scan range had 8+ covered clean frames")
    dt0 = best[0]
    fine = minimize_scalar(lambda dt: (solve_at(dt) or (None, None, 1e9))[2], bounds=(dt0 - step, dt0 + step), method="bounded")
    dt_best = float(fine.x)
    r = solve_at(dt_best) or best[1]
    R_cs, l_cs, rms, n, resid = r
    stats = {
        "dt_s": dt_best,
        "rms_m": rms,
        "n_frames": n,
        "dt_scan_curve": curve,
        "l_cs_spread_by_frame_mm": float(np.std(resid, axis=0).mean() * 1000.0) if n else float("nan"),
    }
    return CamSensorRig(R_cs=R_cs, l_cs=l_cs, dt_s=dt_best, stats=stats)


# =============================================================================== 7. Trajectory
@dataclass
class Trajectory:
    """`mode="pos"` (default, S3): the original position+orientation trajectory, `camera_pose`
    unchanged from before S3b -- every code path below that only touches `S`/`l_cs` is gated on
    `mode != "rot_only"` and reproduces the exact prior arithmetic (no behaviour change, no dt shift).

    `mode="rot_only"` (S3b, `07_revize...md` follow-up): salvages the orientation half of S3 without
    the degenerate scan-CENTRE fit `S(t)` (see module KNOWN LIMITATION: per-window position jitter
    ~0.4 m, head offset ~1 m -- the plane-fit orientation itself is unaffected, RMS 0.6 mm, normal
    continuity dot > 0.998 window-to-window). `camera_pose` then returns:
      - orientation: `R_cs @ R_s(t + dt_s)`, `R_s` purely from `fit_orientation`'s two-head-normal
        Kabsch fit and its Savitzky-Golay smoothing (`build_pass_orientation` never calls
        `fit_scan_centre` -- grep it to confirm S(t) truly does not enter this file's rot_only path).
      - origin: the *export* pose table's plain piecewise-linear interpolation at `t` (unshifted; see
        `_linear_origin` below) -- i.e. exactly `Poses.interp`'s non-`traj` branch, never `S(t)`.
    The `lin_*` fields hold the export table's own per-frame (t, origin, roll, pitch, yaw, pass_id)
    (every frame, not just covered ones) so `_linear_origin` can rebuild a `traj=None` `Poses` on
    demand after a save/load round trip, without this module depending on `mapping.poses` at import
    time (mirrors `Poses.traj`'s loose typing the other way) or `poses.py` needing to know about
    `Trajectory` internals."""

    t: np.ndarray  # [N] concatenated over passes, ascending within each pass (not globally)
    pass_id: np.ndarray  # [N] int32
    S: np.ndarray  # [N,3] -- "pos" mode: fitted scan centre. "rot_only" mode: unused zeros.
    quat: np.ndarray  # [N,4] xyzw, R_s(t) = Rotation.from_quat(quat[i]).as_matrix()
    segments: dict  # {pass_id: [(t0,t1), ...]} contiguous (no >max_gap_s hole) coverage runs
    rigs: dict  # {pass_id: CamSensorRig} (may share the same rig object across passes)
    sample_hz: float = SAMPLE_HZ
    meta: dict = field(default_factory=dict)
    mode: str = "pos"  # "pos" (S3, default) or "rot_only" (S3b)
    # rot_only only (see class docstring); None in "pos" mode.
    lin_t: np.ndarray | None = None
    lin_origin: np.ndarray | None = None
    lin_roll: np.ndarray | None = None
    lin_pitch: np.ndarray | None = None
    lin_yaw: np.ndarray | None = None
    lin_pass_id: np.ndarray | None = None

    # -------------------------------------------------------------- Poses interface
    def covers(self, pass_id: int, t_query: np.ndarray) -> np.ndarray:
        t_query = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
        segs = self.segments.get(int(pass_id), [])
        shift = 0.0
        if self.mode == "rot_only":
            rig = self.rigs.get(int(pass_id))
            shift = rig.dt_s if rig is not None else 0.0
        tq = t_query + shift
        out = np.zeros(len(t_query), dtype=bool)
        for t0, t1 in segs:
            out |= (tq >= t0) & (tq <= t1)
        return out

    def _linear_origin(self, t_query: np.ndarray, pass_id: int) -> np.ndarray:
        """rot_only only: the export table's plain linear interpolation of origin at `t_query`, via a
        fresh, minimal `traj=None` Poses built from `lin_*` (see class docstring) -- reproduces
        `Poses.interp`'s non-traj branch exactly (same code, not a re-derivation) without an
        import-time dependency on `mapping.poses` or a live reference to the original Poses object
        (which would not survive a save/load round trip)."""
        from .poses import Poses  # local: avoid any import-time coupling with mapping.poses

        n = len(self.lin_t)
        plain = Poses(
            filename=np.full(n, "", dtype=object), t=self.lin_t, origin=self.lin_origin,
            roll=self.lin_roll, pitch=self.lin_pitch, yaw=self.lin_yaw, pass_id=self.lin_pass_id,
            speed=np.zeros(n), source="export_linear", traj=None,
        )
        origin, _roll, _pitch, _yaw = plain.interp(t_query, np.full(len(t_query), int(pass_id)))
        return origin

    def camera_pose(self, t_query: np.ndarray, pass_id: int):
        t_query = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
        rig = self.rigs.get(int(pass_id)) or next(iter(self.rigs.values()))
        m = self.pass_id == pass_id
        tt, Ss, qq = self.t[m], self.S[m], self.quat[m]
        order = np.argsort(tt)
        tt, Ss, qq = tt[order], Ss[order], qq[order]
        q_query = t_query + rig.dt_s if self.mode == "rot_only" else t_query
        q_q = np.stack([np.interp(q_query, tt, qq[:, i]) for i in range(4)], axis=1)
        q_q /= np.linalg.norm(q_q, axis=1, keepdims=True)
        R_s = Rotation.from_quat(q_q).as_matrix()
        R_cam = np.einsum("ij,kjl->kil", rig.R_cs, R_s)
        yaw, roll, pitch = geometry.euler_from_vehicle_rotation(R_cam)
        if self.mode == "rot_only":
            C = self._linear_origin(t_query, pass_id)
        else:
            S_q = np.stack([np.interp(t_query, tt, Ss[:, i]) for i in range(3)], axis=1)
            C = S_q + np.einsum("kji,j->ki", R_s, rig.l_cs)
        return C, roll, pitch, yaw

    # -------------------------------------------------------------- persistence
    def save(self, npz_path: str | Path, json_path: str | Path | None = None) -> None:
        npz_path = Path(npz_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(t=self.t, pass_id=self.pass_id, S=self.S, quat=self.quat)
        if self.mode == "rot_only":
            arrays.update(
                lin_t=self.lin_t, lin_origin=self.lin_origin, lin_roll=self.lin_roll,
                lin_pitch=self.lin_pitch, lin_yaw=self.lin_yaw, lin_pass_id=self.lin_pass_id,
            )
        np.savez_compressed(npz_path, **arrays)
        if json_path is None:
            json_path = npz_path.with_suffix(".json")
        rig_ids = {str(k): v.hash() for k, v in self.rigs.items()}
        rigs_by_hash = {}
        for k, v in self.rigs.items():
            rigs_by_hash[v.hash()] = {"R_cs": v.R_cs.tolist(), "l_cs": v.l_cs.tolist(), "dt_s": v.dt_s, "stats": _jsonable(v.stats)}
        payload = {
            "mode": self.mode,
            "sample_hz": self.sample_hz,
            "passes_covered": sorted(int(k) for k in self.segments if self.segments[k]),
            "segments": {str(k): v for k, v in self.segments.items()},
            "rig_by_pass": rig_ids,
            "rigs": rigs_by_hash,
            "meta": _jsonable(self.meta),
        }
        Path(json_path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, npz_path: str | Path, json_path: str | Path | None = None) -> "Trajectory":
        npz_path = Path(npz_path)
        if json_path is None:
            json_path = npz_path.with_suffix(".json")
        d = np.load(npz_path)
        payload = json.loads(Path(json_path).read_text())
        rigs_by_hash = {
            h: CamSensorRig(R_cs=np.asarray(v["R_cs"]), l_cs=np.asarray(v["l_cs"]), dt_s=v["dt_s"], stats=v.get("stats", {}))
            for h, v in payload["rigs"].items()
        }
        rigs = {int(k): rigs_by_hash[h] for k, h in payload["rig_by_pass"].items()}
        segments = {int(k): [tuple(seg) for seg in v] for k, v in payload["segments"].items()}
        mode = payload.get("mode", "pos")
        lin_kwargs = {}
        if mode == "rot_only":
            lin_kwargs = dict(
                lin_t=d["lin_t"], lin_origin=d["lin_origin"], lin_roll=d["lin_roll"],
                lin_pitch=d["lin_pitch"], lin_yaw=d["lin_yaw"], lin_pass_id=d["lin_pass_id"],
            )
        return cls(
            t=d["t"],
            pass_id=d["pass_id"],
            S=d["S"],
            quat=d["quat"],
            segments=segments,
            rigs=rigs,
            sample_hz=payload.get("sample_hz", SAMPLE_HZ),
            meta=payload.get("meta", {}),
            mode=mode,
            **lin_kwargs,
        )


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_pass_trajectory(store, poses, pass_id: int, margin_s: float = 3.0, stride: int = 5):
    """End-to-end for one pass: gather -> per-head plane+centre fit -> orientation -> smoothing.
    `stride` is passed to `fit_scan_centre` (see there): the per-window nonlinear solve dominates
    runtime, and denser than that buys nothing once `smooth_trajectory` resamples to 200 Hz.
    Returns (t_out, S_out, R_out, segments, diag) where `segments` is the list of (t0, t1) gap-free
    coverage runs in the output (from `smooth_trajectory`'s uniform-grid gap detection) and `diag`
    carries per-stage numbers for reporting."""
    t, xyz, head, rank, ret = gather_pass_points(store, poses, pass_id, margin_s=margin_s)
    diag = {"pass_id": int(pass_id), "n_points": int(len(t))}
    planes = {}
    centres = {}
    for h in (1, 2):
        pl = fit_scan_planes(t, xyz, head, h)
        S_h, phi0_h, a_h, resid_h, jitter_h = fit_scan_centre(t, xyz, head, rank, h, pl, stride=stride)
        planes[h] = pl
        centres[h] = S_h
        rms_hist_counts, rms_hist_edges = (np.histogram(pl.rms * 1000.0, bins=20, range=(0.0, 10.0)) if len(pl.rms) else (np.zeros(20), np.linspace(0, 10, 21)))
        diag[f"head{h}"] = {
            "n_windows": int(len(pl.t_c)),
            "rms_mm_median": float(np.median(pl.rms) * 1000.0) if len(pl.rms) else float("nan"),
            "a_deg_per_unit": a_h,
            "centre_resid_deg": resid_h,
            "centre_jitter_mm": jitter_h,
            "rms_mm_hist_counts": rms_hist_counts.tolist(),
            "rms_mm_hist_edges": rms_hist_edges.tolist(),
        }
    t1, R_s = fit_orientation(planes[1].t_c, planes[1].n, planes[2].t_c, planes[2].n)
    # position: head-1 centre at the common (orientation) times; head-2 offset as a consistency check
    S1 = np.stack([np.interp(t1, planes[1].t_c, centres[1][:, i]) for i in range(3)], axis=1)
    S2 = np.stack([np.interp(t1, planes[2].t_c, centres[2][:, i]) for i in range(3)], axis=1)
    head_offset_mm_std = float(np.std(np.linalg.norm(S2 - S1, axis=1)) * 1000.0) if len(t1) else float("nan")
    diag["head_offset_mm_std"] = head_offset_mm_std
    diag["n_common_windows"] = int(len(t1))
    t_out, S_out, R_out = smooth_trajectory(t1, S1, R_s)
    diag["n_samples_200hz"] = int(len(t_out))
    segments = []
    if len(t_out):
        brk = np.flatnonzero(np.diff(t_out) > 1.5 / SAMPLE_HZ)
        bounds_ = [0, *(brk + 1).tolist(), len(t_out)]
        segments = [(float(t_out[s]), float(t_out[e - 1])) for s, e in zip(bounds_[:-1], bounds_[1:]) if e > s]
    diag["n_segments"] = len(segments)
    return t_out, S_out, R_out, segments, diag


# =============================================================================== 8. rot_only (S3b)
def build_pass_orientation(store, poses, pass_id: int, margin_s: float = 3.0):
    """S3b: orientation-only per-pass build -- gather -> per-head plane fit (normals only) -> Kabsch
    orientation `R_s(t)` (`fit_orientation`) -> smoothed quaternion series. Deliberately never calls
    `fit_scan_centre` (the degenerate scan-CENTRE / position solve -- see module KNOWN LIMITATION):
    this path is strictly cheaper than `build_pass_trajectory` (no per-window nonlinear nonlinear
    least_squares) and, since it never touches `S(t)` at all, cannot leak the position fit's jitter
    into the orientation. `smooth_trajectory` is reused for its quaternion Savitzky-Golay path with a
    dummy all-zero `S` (that function smooths S and quat as two fully independent arrays -- see its
    body -- so the dummy has no numerical effect on `R_out`; this is the "does not involve S(t)"
    contract, verified by `tests/test_trajectory.py::test_build_pass_orientation_ignores_position`).

    Returns (t_out[K], quat_out[K,4], segments, diag)."""
    t, xyz, head, rank, ret = gather_pass_points(store, poses, pass_id, margin_s=margin_s)
    diag = {"pass_id": int(pass_id), "n_points": int(len(t))}
    planes = {}
    for h in (1, 2):
        pl = fit_scan_planes(t, xyz, head, h)
        planes[h] = pl
        # window-to-window normal continuity (dot product, ~1 = smooth turn, this module's own
        # honest-negative record on real pass 5: > 0.998 -- reported here so a caller can see it
        # without re-deriving it from the raw PlaneSeries).
        cont = float(np.median(np.sum(pl.n[1:] * pl.n[:-1], axis=1))) if len(pl.n) > 1 else float("nan")
        diag[f"head{h}"] = {
            "n_windows": int(len(pl.t_c)),
            "rms_mm_median": float(np.median(pl.rms) * 1000.0) if len(pl.rms) else float("nan"),
            "normal_continuity_dot_median": cont,
        }
    t1, R_s = fit_orientation(planes[1].t_c, planes[1].n, planes[2].t_c, planes[2].n)
    diag["n_common_windows"] = int(len(t1))
    if len(t1) < 5:
        diag["n_samples_200hz"] = 0
        diag["n_segments"] = 0
        return np.empty(0), np.empty((0, 4)), [], diag
    # smooth_trajectory smooths S and quat as two independent Savitzky-Golay passes (see its body):
    # the all-zero S below is never read again and cannot influence R_out.
    S_dummy = np.zeros((len(t1), 3))
    t_out, _S_unused, R_out = smooth_trajectory(t1, S_dummy, R_s)
    diag["n_samples_200hz"] = int(len(t_out))
    segments = []
    if len(t_out):
        brk = np.flatnonzero(np.diff(t_out) > 1.5 / SAMPLE_HZ)
        bounds_ = [0, *(brk + 1).tolist(), len(t_out)]
        segments = [(float(t_out[s]), float(t_out[e - 1])) for s, e in zip(bounds_[:-1], bounds_[1:]) if e > s]
    diag["n_segments"] = len(segments)
    quat_out = Rotation.from_matrix(R_out).as_quat()
    return t_out, quat_out, segments, diag


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Angle (deg) of `Ra @ Rb^T` per row, for `Ra`, `Rb` both `[K,3,3]`."""
    Rd = np.einsum("kij,kjl->kil", Ra, np.transpose(Rb, (0, 2, 1)))
    cos_a = np.clip((np.trace(Rd, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def _euler_xyz_deg(R: np.ndarray) -> list:
    return Rotation.from_matrix(R).as_euler("xyz", degrees=True).tolist()


def fit_rotation_rig(t: np.ndarray, quat: np.ndarray, poses, straight_idx: np.ndarray, turning_idx: np.ndarray, dt_scan=(-0.3, 0.3, 0.005)) -> "CamSensorRig":
    """S3b rotation-only camera rig. `R_cs` = projected mean of `R_v(k) R_s(t_k+dt)^T` over
    `straight_idx` (clean, |yaw_rate| < 5 deg/s -- accurate orientation, position-independent).
    `dt_s` is chosen by scanning `dt_scan` and minimising the RMS yaw residual on `turning_idx`
    (large |yaw_rate|, e.g. > 3 deg/s): the position-based dt scan in `fit_camera_from_trajectory`
    was flat on this dataset (S3 finding -- the scan-centre fit is too noisy to see dt in), but where
    yaw changes fast a wrong dt shows up directly as a yaw error, so dt *is* observable from rotation
    alone. `l_cs` is left 0 (rot_only mode never fits a lever arm -- origin comes from the export
    table's linear interpolation, not `S(t) + R_s^T l_cs`); `CamSensorRig` is reused only for its
    `R_cs`/`dt_s`/`stats` fields and the existing save/load plumbing.

    Returns a `CamSensorRig` whose `.stats` carries `dt_scan_curve`, `resid_deg_median/p95` (rotation
    residual, degrees, on `straight_idx` at the fitted dt) and `euler_R_cs_deg`."""
    order = np.argsort(t)
    t, quat = t[order], quat[order]
    quat = quat.copy()
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]

    def R_of(tq):
        q = np.stack([np.interp(tq, t, quat[:, i], left=np.nan, right=np.nan) for i in range(4)], axis=1)
        norm = np.linalg.norm(q, axis=1)
        bad = ~np.isfinite(norm) | (norm < 1e-6)
        q = np.where(bad[:, None], np.array([0.0, 0.0, 0.0, 1.0]), q)
        norm = np.where(bad, 1.0, norm)
        q = q / norm[:, None]
        R = Rotation.from_quat(q).as_matrix()
        R[bad] = np.nan
        return R, ~bad

    t_s = poses.t[straight_idx]
    R_v_s = geometry.vehicle_rotation(poses.yaw[straight_idx], poses.roll[straight_idx], poses.pitch[straight_idx])
    t_tn = poses.t[turning_idx]
    yaw_tn = poses.yaw[turning_idx]

    def rcs_at(dt):
        R_s_k, ok = R_of(t_s + dt)
        if ok.sum() < 8:
            return None
        M = np.einsum("kij,kjl->kil", R_v_s[ok], np.transpose(R_s_k[ok], (0, 2, 1))).mean(axis=0)
        U, _, Vt = np.linalg.svd(M)
        d = np.sign(np.linalg.det(U @ Vt))
        R_cs = U @ np.diag([1.0, 1.0, d]) @ Vt
        return R_cs, int(ok.sum())

    def cost_at(dt):
        r = rcs_at(dt)
        if r is None:
            return float("nan"), 0
        R_cs, n_s = r
        R_s_t, ok_t = R_of(t_tn + dt)
        if ok_t.sum() < 5:
            return float("nan"), 0
        R_cam = np.einsum("ij,kjl->kil", R_cs, R_s_t[ok_t])
        yaw_pred, _roll_pred, _pitch_pred = geometry.euler_from_vehicle_rotation(R_cam)
        dyaw = ((yaw_pred - yaw_tn[ok_t] + 180.0) % 360.0) - 180.0
        return float(np.sqrt(np.mean(dyaw**2))), int(ok_t.sum())

    lo, hi, step = dt_scan
    dts = np.arange(lo, hi + step / 2, step)
    curve = []
    best = None
    for dt in dts:
        c, n = cost_at(dt)
        curve.append((float(dt), c))
        if np.isfinite(c) and (best is None or c < best[1]):
            best = (dt, c)
    if best is None:
        raise RuntimeError("fit_rotation_rig: no dt in scan range had 8+ straight and 5+ turning covered frames")
    dt0 = best[0]

    def _bounded_cost(dt):
        c, _n = cost_at(dt)
        return c if np.isfinite(c) else 1e9

    fine = minimize_scalar(_bounded_cost, bounds=(dt0 - step, dt0 + step), method="bounded")
    dt_best = float(fine.x)
    r = rcs_at(dt_best)
    if r is None:
        dt_best = dt0
        r = rcs_at(dt0)
    R_cs, n_straight = r
    _cost_best, n_turn = cost_at(dt_best)

    R_s_s, ok_s = R_of(t_s + dt_best)
    R_cam_s = np.einsum("ij,kjl->kil", R_cs, R_s_s[ok_s])
    resid_deg = rotation_angle_deg(R_cam_s, R_v_s[ok_s])
    stats = {
        "dt_s": dt_best,
        "n_straight": n_straight,
        "n_turning": n_turn,
        "resid_deg_median": float(np.median(resid_deg)) if len(resid_deg) else float("nan"),
        "resid_deg_p95": float(np.percentile(resid_deg, 95)) if len(resid_deg) else float("nan"),
        "dt_scan_curve": curve,
        "euler_R_cs_deg": _euler_xyz_deg(R_cs),
    }
    return CamSensorRig(R_cs=R_cs, l_cs=np.zeros(3), dt_s=dt_best, stats=stats)
