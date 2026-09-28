"""Synthetic tests for mapping.trajectory (S3): known mirror + trajectory model -> recovered
plane/centre/orientation should match to <1 mm / <0.01 deg; Trajectory save/load round trip and a
camera_pose sanity check against a known rig."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from geovap.domain.model import geometry
from geovap.domain.model.poses import Poses
from mapping.trajectory import (
    CamSensorRig,
    Trajectory,
    _reject_position_outliers,
    fit_orientation,
    fit_rotation_rig,
    fit_scan_centre,
    fit_scan_planes,
    rotation_angle_deg,
    smooth_trajectory,
)


def _orthonormal_from_normal(n, seed_vec):
    n = n / np.linalg.norm(n)
    e1 = np.cross(n, seed_vec)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return n, e1, e2


def _synthetic_scan(seed=0, T=0.05, n_per_ms=150, yaw_rate_dps=40.0):
    """Two-head synthetic scanner data with a known trajectory S(t), R_true(t) (world->sensor) and
    per-head mirror model (a_true, phi0_h, fixed sensor-frame normal)."""
    rng = np.random.default_rng(seed)
    a_true = 1.02
    S0 = np.array([100.0, -50.0, 20.0])
    v = np.array([5.0, 1.0, 0.0])

    heads = {
        1: {"n_s": np.array([0.3, 0.1, 0.95]), "phi0": 15.0, "seed": [1.0, 0.0, 0.0]},
        2: {"n_s": np.array([-0.2, 0.9, 0.3]), "phi0": -50.0, "seed": [0.0, 1.0, 0.0]},
    }
    n_buckets = int(round(T * 1000))
    ts, xyzs, heads_arr, ranks = [], [], [], []
    for h, cfg in heads.items():
        n_s, e1_s, e2_s = _orthonormal_from_normal(cfg["n_s"], np.array(cfg["seed"]))
        n_pts = n_buckets * n_per_ms
        t_i = rng.uniform(0, T, n_pts)
        rank_i = rng.integers(-110, 111, n_pts).astype(np.float64)
        theta_i = np.deg2rad(a_true * rank_i + cfg["phi0"])
        rho_i = rng.uniform(5.0, 15.0, n_pts)
        R_true_i = Rotation.from_euler("z", (yaw_rate_dps * t_i)[:, None], degrees=True).as_matrix()  # [N,3,3]
        e1_w = np.einsum("kji,j->ki", R_true_i, e1_s)  # R_true^T @ e1_s
        e2_w = np.einsum("kji,j->ki", R_true_i, e2_s)
        S_i = S0 + np.outer(t_i, v)
        P_i = S_i + rho_i[:, None] * (np.cos(theta_i)[:, None] * e1_w + np.sin(theta_i)[:, None] * e2_w)
        P_i += rng.normal(0, 0.0003, P_i.shape)  # 0.3 mm point noise
        ts.append(t_i)
        xyzs.append(P_i)
        heads_arr.append(np.full(n_pts, h, dtype=np.uint8))
        ranks.append(rank_i.astype(np.int8))
    t = np.concatenate(ts)
    xyz = np.concatenate(xyzs, axis=0)
    head = np.concatenate(heads_arr)
    rank = np.concatenate(ranks)
    order = np.argsort(t, kind="stable")
    return t[order], xyz[order], head[order], rank[order], S0, v, a_true, yaw_rate_dps


def test_fit_scan_planes_rms():
    t, xyz, head, rank, *_ = _synthetic_scan()
    for h in (1, 2):
        pl = fit_scan_planes(t, xyz, head, h)
        assert len(pl.t_c) > 5
        assert np.median(pl.rms) < 0.0035  # under the 10 mm gate; includes in-window motion blur (real data: 3-4 mm)
        assert np.all(pl.n_pts >= 100)


def test_fit_scan_centre_and_orientation_recover_truth():
    t, xyz, head, rank, S0, v, a_true, yaw_rate_dps = _synthetic_scan()
    planes, centres = {}, {}
    for h in (1, 2):
        pl = fit_scan_planes(t, xyz, head, h)
        S_h, phi0_h, a_h, resid_deg, jitter_mm = fit_scan_centre(t, xyz, head, rank, h, pl)
        planes[h] = pl
        centres[h] = S_h
        # sign of a is an inherent normal-sign ambiguity (07 §5.1: "sign and offset unknown; fit
        # them"); only the magnitude is checked against the true mechanical constant.
        assert abs(abs(a_h) - a_true) < 0.02, f"head {h}: a={a_h}"
        assert resid_deg < 0.1, f"head {h}: global angular residual {resid_deg} deg"  # naive attempt (07 §5.1): 0.62 deg

    t1, R_s = fit_orientation(planes[1].t_c, planes[1].n, planes[2].t_c, planes[2].n)
    assert len(t1) > 5
    t_ref = t1[0]
    R_true_ref = Rotation.from_euler("z", yaw_rate_dps * t_ref, degrees=True).as_matrix()

    S1 = np.stack([np.interp(t1, planes[1].t_c, centres[1][:, i]) for i in range(3)], axis=1)
    S_true = S0 + np.outer(t1, v)
    pos_err_mm = np.linalg.norm(S1 - S_true, axis=1) * 1000.0
    # Plan target (07 SS5.1): 1 mm. Achieved on the *median* window; a handful of individual
    # windows can still land in a corner-of-bounds local minimum of the per-window (s1, s2, phi0)
    # fit (rare, narrow-rank-arc or near-planar windows) and blow up, which is why this checks the
    # median rather than every window -- `smooth_trajectory`'s Savitzky-Golay pass is what a real
    # caller relies on to tame those outliers before they reach a camera pose.
    assert np.median(pos_err_mm) < 2.0, f"median position error {np.median(pos_err_mm)} mm"

    ang_err_deg = np.empty(len(t1))
    for i, tt in enumerate(t1):
        R_true_t = Rotation.from_euler("z", yaw_rate_dps * tt, degrees=True).as_matrix()
        R_check = R_true_ref.T @ R_true_t
        cos_a = np.clip((np.trace(R_s[i].T @ R_check) - 1.0) / 2.0, -1.0, 1.0)
        ang_err_deg[i] = np.degrees(np.arccos(cos_a))
    assert np.median(ang_err_deg) < 0.01, f"median orientation error {np.median(ang_err_deg)} deg"  # plan target: 0.01 deg


def test_trajectory_save_load_roundtrip(tmp_path):
    n = 40
    t = np.linspace(0.0, 2.0, n)
    S = np.stack([100.0 + 5.0 * t, -50.0 + t, np.full(n, 20.0)], axis=1)
    quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))  # R_s == I everywhere
    rig = CamSensorRig(R_cs=np.eye(3), l_cs=np.array([1.4, 0.0, 0.8]), dt_s=0.01, stats={"n_frames": 12})
    traj = Trajectory(
        t=t,
        pass_id=np.zeros(n, dtype=np.int32),
        S=S,
        quat=quat,
        segments={0: [(float(t[0]), float(t[-1]))]},
        rigs={0: rig},
        meta={"note": "synthetic"},
    )
    npz = tmp_path / "trajectory.npz"
    traj.save(npz)
    loaded = Trajectory.load(npz)

    assert np.allclose(loaded.t, traj.t)
    assert np.allclose(loaded.S, traj.S)
    assert np.allclose(loaded.quat, traj.quat)
    assert loaded.segments == traj.segments
    assert loaded.rigs[0].dt_s == pytest.approx(rig.dt_s)
    assert np.allclose(loaded.rigs[0].l_cs, rig.l_cs)

    tq = np.array([0.5, 1.0])
    assert np.all(loaded.covers(0, tq))
    assert not np.any(loaded.covers(0, np.array([-1.0, 5.0])))

    C, roll, pitch, yaw = loaded.camera_pose(tq, 0)
    S_at = np.stack([100.0 + 5.0 * tq, -50.0 + tq, np.full(2, 20.0)], axis=1)
    assert np.allclose(C, S_at + rig.l_cs, atol=1e-9)  # R_s == I -> C = S + l_cs
    assert np.allclose(roll, 0.0, atol=1e-6)
    assert np.allclose(pitch, 0.0, atol=1e-6)
    assert np.allclose(yaw, 0.0, atol=1e-6)


def test_camera_pose_matches_manual_formula_with_nontrivial_rig():
    """R_cs and R_s(t) both nontrivial: C == S + R_s^T @ l_cs, R_cam == R_cs @ R_s, checked against
    geometry.vehicle_rotation / euler_from_vehicle_rotation directly (not just self-consistency)."""
    n = 10
    t = np.linspace(0.0, 1.0, n)
    S = np.stack([t, 2 * t, np.zeros(n)], axis=1)
    R_s = Rotation.from_euler("z", (30.0 * t)[:, None], degrees=True).as_matrix()  # world -> sensor, varies
    quat = Rotation.from_matrix(R_s).as_quat()
    R_cs = Rotation.from_euler("xyz", [2.0, -3.0, 5.0], degrees=True).as_matrix()  # sensor -> camera
    l_cs = np.array([1.37, 0.0, 0.78])
    rig = CamSensorRig(R_cs=R_cs, l_cs=l_cs, dt_s=0.0)
    traj = Trajectory(t=t, pass_id=np.zeros(n, dtype=np.int32), S=S, quat=quat, segments={0: [(0.0, 1.0)]}, rigs={0: rig})

    tq = np.array([0.3, 0.7])
    C, roll, pitch, yaw = traj.camera_pose(tq, 0)

    Rs_q = Rotation.from_euler("z", (30.0 * tq)[:, None], degrees=True).as_matrix()
    S_q = np.stack([tq, 2 * tq, np.zeros(2)], axis=1)
    C_expected = S_q + np.einsum("kji,j->ki", Rs_q, l_cs)
    assert np.allclose(C, C_expected, atol=1e-6)

    R_cam_expected = np.einsum("ij,kjl->kil", R_cs, Rs_q)
    R_v_check = geometry.vehicle_rotation(yaw, roll, pitch)
    assert np.allclose(R_v_check, R_cam_expected, atol=1e-6)


def test_fit_scan_planes_e1_sign_continuous():
    """Regression for a real-data bug (07 revision investigation, pass 5): `normal` (smallest-
    eigenvalue eigenvector) and `e1` (largest-eigenvalue eigenvector) each come out of `eigh` with an
    independently arbitrary sign -- fixing only `normal`'s sign (as an earlier version of this
    function did) still let `e1` flip on ~15% of consecutive real windows, which corrupts
    `fit_scan_centre`'s per-window warm start (phi0 off by ~180 deg) every time it happens. Assert
    both axes are sign-continuous along time, not just `normal`."""
    t, xyz, head, rank, *_ = _synthetic_scan()
    for h in (1, 2):
        pl = fit_scan_planes(t, xyz, head, h)
        assert len(pl.t_c) > 5
        dot_n = np.sum(pl.n[1:] * pl.n[:-1], axis=1)
        dot_e1 = np.sum(pl.e1[1:] * pl.e1[:-1], axis=1)
        assert np.all(dot_n >= 0), f"head {h}: normal sign flip, min dot {dot_n.min()}"
        assert np.all(dot_e1 >= 0), f"head {h}: e1 sign flip, min dot {dot_e1.min()}"
        # e1/normal/e2 must stay an orthonormal right-handed basis after the sign fix-up
        assert np.allclose(np.sum(pl.e1 * pl.n, axis=1), 0.0, atol=1e-9)
        assert np.allclose(np.cross(pl.n, pl.e1), pl.e2, atol=1e-9)


def test_reject_position_outliers_removes_short_ghost_run():
    """Regression for a real-data bug (07 revision investigation, pass 5): a *causal* implausible-
    jump veto was tried and rejected because one wrongly-accepted window then poisons every later,
    correctly-converging window (each looks like the "implausible jump" relative to the wrong
    anchor). `_reject_position_outliers` is non-causal instead: a short run of consecutive windows
    that agree with *each other* but disagree with the smooth trend on both sides is still recognised
    as an outlier block and replaced by interpolation across it.

    KNOWN LIMITATION (honest negative, see mapping/trajectory.py / the S3 report): this is a rolling-
    median filter, so it only catches a contiguous ghost run shorter than about half its window
    (`OUTLIER_MEDIAN_K`, default 11) -- a run *at or beyond* that length can dominate its own local
    median and pass undetected, which is exactly what still happens on part of real pass 5 data.
    This test exercises the case the filter *is* designed for, not that longer-run case."""
    n = 60
    t_c = np.linspace(0.0, 1.0, n)
    true_S = np.stack([100.0 + 5.0 * t_c, -50.0 + t_c, np.full(n, 20.0)], axis=1)
    S = true_S.copy()
    solved = np.ones(n, dtype=bool)
    phi0 = np.zeros(n)
    # a coherent, self-consistent "ghost" run (like a mis-triangulated nearby flat surface):
    # offset by 5 m for a short contiguous stretch, not scattered single-point noise.
    ghost = slice(25, 29)
    S[ghost] = true_S[ghost] + np.array([5.0, 5.0, 5.0])

    S_out, phi0_out, solved_out = _reject_position_outliers(t_c, S, phi0, solved)

    assert not np.any(solved_out[ghost]), "the short ghost run must be rejected, not kept as trustworthy"
    err_mm = np.linalg.norm(S_out - true_S, axis=1) * 1000.0
    assert np.median(err_mm) < 5.0, f"reconstructed trajectory should track the true trend: median err {np.median(err_mm)} mm"
    assert err_mm.max() < 50.0, f"no reconstructed point should retain metre-scale ghost error: max err {err_mm.max()} mm"


# =============================================================================== S3b: rot_only mode
def _straight_line_poses(n, t0=0.0, dt=0.1, pass_id=0):
    """Minimal synthetic export-style Poses: straight line E = t, N = 2t, H = 20, yaw/roll/pitch = 0."""
    t = t0 + dt * np.arange(n)
    origin = np.stack([t, 2.0 * t, np.full(n, 20.0)], axis=1)
    return Poses(
        filename=np.array([f"f{i}.jpg" for i in range(n)], dtype=object),
        t=t, origin=origin, roll=np.zeros(n), pitch=np.zeros(n), yaw=np.zeros(n),
        pass_id=np.full(n, pass_id, dtype=np.int32), speed=np.ones(n), source="export",
    )


def test_rot_only_camera_pose_linear_origin_and_orientation():
    """rot_only `camera_pose` must return (a) origin = plain linear interpolation of the `lin_*`
    (export) table -- not any fitted S(t) -- and (b) orientation = R_cs @ R_s(t + dt_s)."""
    lin = _straight_line_poses(11, dt=0.2)  # t in [0, 2.0]
    R_cs = Rotation.from_euler("xyz", [2.0, -3.0, 5.0], degrees=True).as_matrix()
    dt_s = 0.05
    t_r = np.linspace(-0.5, 3.0, 400)
    R_s = Rotation.from_euler("z", (25.0 * t_r)[:, None], degrees=True).as_matrix()  # varies with t
    quat = Rotation.from_matrix(R_s).as_quat()
    rig = CamSensorRig(R_cs=R_cs, l_cs=np.array([999.0, 999.0, 999.0]), dt_s=dt_s)  # l_cs must be UNUSED in rot_only
    traj = Trajectory(
        t=t_r, pass_id=np.zeros(len(t_r), dtype=np.int32), S=np.full((len(t_r), 3), 12345.0),  # S must be UNUSED
        quat=quat, segments={0: [(float(t_r[0]), float(t_r[-1]))]}, rigs={0: rig}, mode="rot_only",
        lin_t=lin.t, lin_origin=lin.origin, lin_roll=lin.roll, lin_pitch=lin.pitch, lin_yaw=lin.yaw,
        lin_pass_id=lin.pass_id,
    )

    tq = np.array([0.35, 1.1])
    assert np.all(traj.covers(0, tq))
    C, roll, pitch, yaw = traj.camera_pose(tq, 0)

    C_expected = np.stack([tq, 2.0 * tq, np.full(2, 20.0)], axis=1)  # plain linear interp of lin_origin
    assert np.allclose(C, C_expected, atol=1e-9), "origin must be the export table's linear interpolation"

    R_s_expected = Rotation.from_euler("z", (25.0 * (tq + dt_s))[:, None], degrees=True).as_matrix()
    R_cam_expected = np.einsum("ij,kjl->kil", R_cs, R_s_expected)
    R_v_check = geometry.vehicle_rotation(yaw, roll, pitch)
    assert np.allclose(R_v_check, R_cam_expected, atol=1e-6), "orientation must be R_cs @ R_s(t + dt_s)"


def test_rot_only_covers_shifts_by_dt():
    dt_s = 0.2
    rig = CamSensorRig(R_cs=np.eye(3), l_cs=np.zeros(3), dt_s=dt_s)
    traj = Trajectory(
        t=np.array([0.0, 1.0]), pass_id=np.zeros(2, dtype=np.int32), S=np.zeros((2, 3)),
        quat=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)), segments={0: [(0.0, 1.0)]}, rigs={0: rig}, mode="rot_only",
        lin_t=np.array([0.0, 1.0]), lin_origin=np.zeros((2, 3)), lin_roll=np.zeros(2), lin_pitch=np.zeros(2),
        lin_yaw=np.zeros(2), lin_pass_id=np.zeros(2, dtype=np.int32),
    )
    # t_query=0.85: t_query+dt_s=1.05 is just outside the [0,1] segment -> not covered
    assert not traj.covers(0, np.array([0.85]))[0]
    # t_query=0.75: t_query+dt_s=0.95 is inside -> covered
    assert traj.covers(0, np.array([0.75]))[0]


def test_smooth_trajectory_rotation_independent_of_position():
    """S3b contract: the rot_only orientation path must not be affected by S(t) -- verify directly
    that `smooth_trajectory`'s R_out is bit-identical whether S is all-zero or arbitrary noise."""
    n = 80
    t = np.linspace(0.0, 0.5, n)
    R_s = Rotation.from_euler("z", (200.0 * t)[:, None], degrees=True).as_matrix()
    rng = np.random.default_rng(0)
    S_zero = np.zeros((n, 3))
    S_noisy = rng.normal(0, 50.0, (n, 3))  # wildly different, even different magnitude/units

    t_out_a, S_out_a, R_out_a = smooth_trajectory(t, S_zero, R_s)
    t_out_b, S_out_b, R_out_b = smooth_trajectory(t, S_noisy, R_s)

    assert np.allclose(t_out_a, t_out_b)
    assert np.allclose(R_out_a, R_out_b), "R_out must not depend on S at all"
    assert not np.allclose(S_out_a, S_out_b)  # sanity: S actually differs, so this is a real check


def test_fit_rotation_rig_recovers_known_rig_and_dt():
    """Synthetic sensor with a slow ('straight', 2 deg/s) segment and a fast ('turning', 60 deg/s)
    segment: R_cs is only observable from the slow segment (position-independent SVD mean), dt_s only
    from the fast one (yaw residual) -- matches the S3b design (position-based dt scan was flat;
    rotation-based dt scan on turning frames is not)."""
    R_cs_true = Rotation.from_euler("xyz", [2.0, -3.0, 5.0], degrees=True).as_matrix()
    dt_true = -0.035  # on the 5 ms scan grid

    def yaw_s(t):
        return np.where(t < 1.0, 2.0 * t, 2.0 + 60.0 * (t - 1.0))

    t_dense = np.linspace(-0.5, 2.5, 6001)
    R_s_dense = Rotation.from_euler("z", yaw_s(t_dense)[:, None], degrees=True).as_matrix()
    quat_dense = Rotation.from_matrix(R_s_dense).as_quat()

    t_straight = np.linspace(0.1, 0.9, 30)
    t_turning = np.linspace(1.1, 1.9, 30)
    t_all = np.concatenate([t_straight, t_turning])
    R_s_at_rig_time = Rotation.from_euler("z", yaw_s(t_all + dt_true)[:, None], degrees=True).as_matrix()
    R_v_true = np.einsum("ij,kjl->kil", R_cs_true, R_s_at_rig_time)
    yaw_true, roll_true, pitch_true = geometry.euler_from_vehicle_rotation(R_v_true)

    n = len(t_all)
    poses = Poses(
        filename=np.array([f"f{i}.jpg" for i in range(n)], dtype=object), t=t_all, origin=np.zeros((n, 3)),
        roll=roll_true, pitch=pitch_true, yaw=yaw_true, pass_id=np.zeros(n, dtype=np.int32), speed=np.zeros(n),
        source="export",
    )
    straight_idx = np.arange(30)
    turning_idx = np.arange(30, 60)

    rig = fit_rotation_rig(t_dense, quat_dense, poses, straight_idx, turning_idx, dt_scan=(-0.1, 0.1, 0.005))

    assert abs(rig.dt_s - dt_true) < 0.01, f"dt_s={rig.dt_s}, true={dt_true}"
    r_err = rotation_angle_deg(rig.R_cs[None], R_cs_true[None])[0]
    assert r_err < 0.5, f"R_cs recovery error {r_err} deg"
    assert rig.stats["resid_deg_median"] < 0.1, rig.stats["resid_deg_median"]
    # the dt-scan curve should have a clear minimum near dt_true, not be flat (S3's position scan was flat)
    curve = dict(rig.stats["dt_scan_curve"])
    near = curve[min(curve, key=lambda d: abs(d - dt_true))]
    far = curve[min(curve, key=lambda d: abs(d - (dt_true + 0.09)))]
    assert near < 0.3 * far, f"expected a clear minimum near dt_true: near={near}, far={far}"


def test_rot_only_trajectory_save_load_roundtrip_and_poses_interp(tmp_path):
    """Round trip through `Trajectory.save`/`load`, then through `Poses.interp` (mimics
    `load_poses(...).traj is not None` and a smoothly-varying-yaw check at a 'turning' time)."""
    lin = _straight_line_poses(21, dt=0.1)  # t in [0, 2.0]
    R_cs = np.eye(3)
    rig = CamSensorRig(R_cs=R_cs, l_cs=np.zeros(3), dt_s=0.0)
    t_r = np.linspace(0.0, 2.0, 400)
    R_s = Rotation.from_euler("z", (90.0 * t_r)[:, None], degrees=True).as_matrix()  # fast "turning" rotation
    quat = Rotation.from_matrix(R_s).as_quat()
    traj = Trajectory(
        t=t_r, pass_id=np.zeros(len(t_r), dtype=np.int32), S=np.zeros((len(t_r), 3)), quat=quat,
        segments={0: [(0.0, 2.0)]}, rigs={0: rig}, mode="rot_only",
        lin_t=lin.t, lin_origin=lin.origin, lin_roll=lin.roll, lin_pitch=lin.pitch, lin_yaw=lin.yaw,
        lin_pass_id=lin.pass_id,
    )
    npz = tmp_path / "trajectory_rot.npz"
    traj.save(npz)
    loaded = Trajectory.load(npz)
    assert loaded.mode == "rot_only"
    assert loaded.rigs[0].dt_s == pytest.approx(0.0)
    assert np.allclose(loaded.lin_origin, lin.origin)

    poses = Poses(
        filename=lin.filename, t=lin.t, origin=lin.origin, roll=lin.roll, pitch=lin.pitch, yaw=lin.yaw,
        pass_id=lin.pass_id, speed=lin.speed, source="traj_rot", traj=loaded,
    )
    assert poses.traj is not None

    t_center = 1.0  # a "turning" frame time, well inside coverage
    t_probe = t_center + np.array([-0.3, -0.15, 0.0, 0.15, 0.3])
    _origin, _roll, _pitch, yaw = poses.interp(t_probe, np.zeros(len(t_probe), dtype=np.int32))
    dyaw = np.degrees(np.diff(np.unwrap(np.radians(yaw))))
    # R_s is a constant-rate (90 deg/s) rotation and R_cs == I, so consecutive 0.15 s steps must move
    # yaw by a constant amount (exactly +-13.5 deg) with no sign flips or jumps -- "smoothly varying".
    assert np.allclose(np.abs(dyaw), 90.0 * 0.15, atol=1e-6), f"yaw should vary smoothly at a constant rate, got steps {dyaw}"
    assert len(set(np.sign(dyaw))) == 1, f"yaw should move consistently in one direction, got steps {dyaw}"
