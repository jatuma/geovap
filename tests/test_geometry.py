"""Geometry: equivalence with the verified torch oracle, inverse consistency, rig parametrisation."""
import numpy as np
import pytest

from mapping import compat, geometry
from mapping.config import PANO_H, PANO_W
from mapping.poses import Poses
from mapping.rig import IDENTITY, RigModel


def _synthetic_poses(m: int, seed: int = 0) -> Poses:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(300_000, 302_000, m))
    origin = np.stack([rng.uniform(-642_500, -642_000, m), rng.uniform(-1_055_800, -1_055_000, m), rng.uniform(220, 230, m)], 1)
    return Poses(
        filename=np.array([f"f{i}.jpg" for i in range(m)], dtype=object),
        t=t,
        origin=origin,
        roll=rng.uniform(-9, 9, m),
        pitch=rng.uniform(-6, 6, m),
        yaw=rng.uniform(-180, 180, m),
        pass_id=np.zeros(m, np.int32),
        speed=np.full(m, 8.0),
    )


def test_forward_matches_camera_py_oracle():
    compat.ensure_experiments_on_path()
    torch = pytest.importorskip("torch")
    from common.camera import world_to_panorama_px

    rng = np.random.default_rng(1)
    poses = _synthetic_poses(50)
    R, C = geometry.frame_rotations(poses, IDENTITY)
    worst = {np.float64: 0.0, np.float32: 0.0}
    for k in range(len(poses)):
        P = poses.origin[k] + rng.uniform(-40, 40, (10_000, 3)) * np.array([1, 1, 0.5])
        uo, vo, elo = world_to_panorama_px(
            torch.tensor(P), torch.tensor(poses.origin[k]), torch.tensor(poses.roll[k]), torch.tensor(poses.pitch[k]), torch.tensor(poses.yaw[k])
        )
        for dt in worst:
            u, v, r, el = geometry.world_to_pano(P, R[k], C[k], dtype=dt)
            du = np.abs(u.astype(np.float64) - uo.numpy())
            du = np.minimum(du, PANO_W - du)  # seam wrap
            dv = np.abs(v.astype(np.float64) - vo.numpy())
            worst[dt] = max(worst[dt], du.max(), dv.max())
    assert worst[np.float64] < 1e-6, worst  # same formula, matrix form
    assert worst[np.float32] < 5e-3, worst  # float32 working precision (~1e-3 px at 40 m)


def test_inverse_ray_matches_forward():
    rng = np.random.default_rng(2)
    poses = _synthetic_poses(20)
    rig = RigModel(boresight_deg=(0.3, -0.2, 0.7), lever_arm_m=(0.1, -0.05, 0.4), dt_s=0.0)
    R, C = geometry.frame_rotations(poses, rig)
    for k in range(len(poses)):
        P = C[k] + rng.uniform(-30, 30, (5000, 3))
        # use float64 forward for a tight check
        x_cam = (P - C[k]) @ R[k].T
        u, v, r, el = geometry.cam_to_pano(x_cam)
        d = geometry.pano_to_world_ray(u, v, R[k])
        d_true = (P - C[k]) / np.linalg.norm(P - C[k], axis=1, keepdims=True)
        assert np.abs(d - d_true).max() < 1e-9
        P2 = geometry.pano_to_world(u, v, r, R[k], C[k])
        assert np.abs(P2 - P).max() < 1e-7


def test_round_trip_float32_forward_under_1cm():
    rng = np.random.default_rng(3)
    poses = _synthetic_poses(5)
    R, C = geometry.frame_rotations(poses)
    P = C[0] + rng.uniform(-40, 40, (20000, 3))
    u, v, r, el = geometry.world_to_pano(P, R[0], C[0])
    P2 = geometry.pano_to_world(u.astype(np.float64), v.astype(np.float64), r.astype(np.float64), R[0], C[0])
    assert np.linalg.norm(P2 - P, axis=1).max() < 0.01


def test_identity_rig_is_pure_vehicle_rotation():
    poses = _synthetic_poses(3)
    R, C = geometry.frame_rotations(poses, IDENTITY)
    Rv = geometry.vehicle_rotation(poses.yaw, poses.roll, poses.pitch)
    assert np.allclose(R, Rv) and np.allclose(C, poses.origin)
    assert np.allclose(R @ np.swapaxes(R, 1, 2), np.eye(3)[None])


def test_boresight_kappa_is_constant_u_shift():
    poses = _synthetic_poses(2)
    rig = RigModel(boresight_deg=(0.0, 0.0, 1.0))
    R0, C0 = geometry.frame_rotations(poses, IDENTITY)
    R1, C1 = geometry.frame_rotations(poses, rig)
    P = C0[0] + np.array([[10.0, 3.0, 1.0], [-5.0, 8.0, -2.0], [2.0, -12.0, 4.0]])
    u0, v0, *_ = geometry.world_to_pano(P, R0[0], C0[0])
    u1, v1, *_ = geometry.world_to_pano(P, R1[0], C1[0])
    du = (u1 - u0 + PANO_W / 2) % PANO_W - PANO_W / 2
    # +1 deg kappa rotates the camera; every point moves by the same |1 deg| in u, v unchanged
    assert np.allclose(np.abs(du), PANO_W / 360.0, atol=1e-2) and np.allclose(v1, v0, atol=1e-3)


def test_pose_at_dt_zero_is_exact_and_dt_interpolates():
    poses = _synthetic_poses(10)
    o, r, p, y = poses.pose_at(np.arange(10), 0.0)
    assert np.array_equal(o, poses.origin) and np.array_equal(y, poses.yaw)
    dt = 0.5 * (poses.t[1] - poses.t[0])
    o, r, p, y = poses.pose_at(np.array([0]), dt)
    assert np.allclose(o[0], 0.5 * (poses.origin[0] + poses.origin[1]))


# ------------------------------------------------------------------ 02_obarveni §7.5 quick checks on real data
def test_trajectory_projects_near_horizon(poses):
    R, C = geometry.frame_rotations(poses)
    for k in (100, 400, 800, 1200):
        for direction, expect_az in ((+1, 0.0), (-1, 180.0)):
            nb = k + direction * np.arange(1, 7)
            nb = nb[poses.pass_id[nb] == poses.pass_id[k]]
            assert len(nb) >= 2
            u, v, r, el = geometry.world_to_pano(poses.origin[nb], R[k], C[k])
            assert np.all(np.abs(el) < 6.0), el  # neighbouring camera centres lie near the horizon
            az = u / PANO_W * 360.0
            daz = np.abs((az - expect_az + 180.0) % 360.0 - 180.0)
            assert np.median(daz) < 20.0, (k, direction, az)  # ahead -> seam (az 0), behind -> az 180


def test_nadir_projects_to_bottom_row(poses):
    R, C = geometry.frame_rotations(poses)
    nadir = C[:5] + np.array([0.0, 0.0, -2.0])
    for k in range(5):
        u, v, r, el = geometry.world_to_pano(nadir[k : k + 1], R[k], C[k])
        assert v[0] > PANO_H * (1 - 10 / 180), v  # within 10 deg of nadir (roll/pitch <= 9 deg)
