"""Panorama spheres: the angles written to coordinates.txt must reproduce our own projection.

The reference below is a transcription of what the viewer does -- three.js r124 `SphereGeometry`
plus `Potree.Images360Loader` -- and is deliberately independent of `mapping.panos`.
"""
from pathlib import Path

import numpy as np
import pytest

from geovap.domain.model import geometry
from mapping import panos
from mapping.config import PANO_H, PANO_W
from geovap.domain.model.poses import Poses


def _synthetic_poses(m: int = 7, seed: int = 3) -> Poses:
    rng = np.random.default_rng(seed)
    return Poses(
        filename=np.array([f"f{i}.jpg" for i in range(m)], dtype=object),
        t=np.sort(rng.uniform(300_000, 302_000, m)),
        origin=np.stack([rng.uniform(-642_500, -642_000, m), rng.uniform(-1_055_800, -1_055_000, m), rng.uniform(220, 230, m)], 1),
        roll=rng.uniform(-9, 9, m),
        pitch=rng.uniform(-6, 6, m),
        yaw=rng.uniform(-180, 180, m),
        pass_id=np.zeros(m, np.int32),
        speed=np.full(m, 8.0),
    )


# ------------------------------------------------------------------ viewer-side reference
def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def three_euler_zyx(x: float, y: float, z: float) -> np.ndarray:
    """three.js `Euler(x, y, z, "ZYX")` as a matrix: R = Rz(z) Ry(y) Rx(x) (radians)."""
    return _rot("z", z) @ _rot("y", y) @ _rot("x", x)


def potree_mesh_rotation(course: float, pitch: float, roll: float) -> np.ndarray:
    """Images360Loader.createSceneNodes: mesh.rotation.set(roll+90, -pitch, -course+90, "ZYX")."""
    d = np.deg2rad
    return three_euler_zyx(d(roll + 90.0), d(-pitch), d(-course + 90.0))


def sphere_local_dir(u: float, v: float, w: int = PANO_W, h: int = PANO_H) -> np.ndarray:
    """Direction, in mesh-local axes, of the vertex whose texture is panorama pixel (u, v).

    three.js r124 SphereGeometry: position = (-cos(phi) sin(theta), cos(theta), sin(phi) sin(theta)),
    uv = (phi / 2pi, 1 - theta / pi). Potree's loader sets `texture.repeat.x = -1`; that is the
    correct display and the pages keep it, so texel s = 1 - u / w and t = 1 - v / h.
    """
    phi = 2.0 * np.pi * (1.0 - u / w)
    theta = np.pi * (v / h)
    return np.array([-np.cos(phi) * np.sin(theta), np.cos(theta), np.sin(phi) * np.sin(theta)])


# ------------------------------------------------------------------ tests
def test_sphere_local_dir_matches_potree_m():
    """POTREE_M is exactly the camera -> sphere-local map of the reference sphere."""
    for u, v in [(0, 0.5 * PANO_H), (1234.5, 900.0), (PANO_W * 0.75, PANO_H * 0.9), (7999.0, 10.0)]:
        d_cam = geometry.pano_rays(u, v, PANO_W, PANO_H)
        assert np.allclose(panos.POTREE_M @ d_cam, sphere_local_dir(u, v), atol=1e-12)


@pytest.mark.parametrize("az_offset", [0.0, 180.0])
def test_written_angles_reproduce_the_projection(az_offset):
    """A pixel put on the sphere by the viewer points where geometry says it points.

    With az_offset = 0 this is the pure convention check; the production value turns the sphere
    about the camera vertical, so the expected ray is the geometry ray for the shifted column.
    The offset only re-derotates `pano_rays` by `rz_cam(az_offset)^T` before POTREE_M is applied,
    and POTREE_M cancels out of that step regardless of its value (it's orthogonal), so the sign
    of the column shift (u + az_offset/360*W) is unchanged by the repeat.x = -1 convention fix --
    verified numerically against the production `potree_angles` code path for az_offset in
    {0, 37, 180} before writing this test.
    """
    poses = _synthetic_poses()
    R, _C = geometry.frame_rotations(poses)
    course, pitch, roll = panos.potree_angles(R, az_offset)
    rng = np.random.default_rng(0)
    uv = np.stack([rng.uniform(0, PANO_W, 40), rng.uniform(1, PANO_H - 1, 40)], 1)
    for k in range(len(poses)):
        mesh = potree_mesh_rotation(course[k], pitch[k], roll[k])
        for u, v in uv:
            want = geometry.pano_to_world_ray((u + az_offset / 360.0 * PANO_W) % PANO_W, v, R[k], PANO_W, PANO_H)
            got = mesh @ sphere_local_dir(u, v)
            # arccos near 1 is float-noisy; 1e-5 deg is 7e-4 px at 8000 px width
            assert np.degrees(np.arccos(np.clip(want @ got, -1, 1))) < 1e-5


@pytest.mark.parametrize("az_offset", [0.0, 90.0, 180.0])
def test_potree_angles_roundtrip_through_sphere_rotation(az_offset):
    poses = _synthetic_poses(11, seed=5)
    R, _C = geometry.frame_rotations(poses)
    course, pitch, roll = panos.potree_angles(R, az_offset)
    for k in range(len(poses)):
        assert np.allclose(potree_mesh_rotation(course[k], pitch[k], roll[k]), panos.sphere_rotation(R[k], az_offset), atol=1e-12)


def test_az_offset_is_a_pure_yaw_about_the_camera_vertical():
    """The offset must not tilt the sphere: only the azimuth of the texture may move."""
    poses = _synthetic_poses(4, seed=7)
    R, _C = geometry.frame_rotations(poses)
    for k in range(len(poses)):
        rel = panos.sphere_rotation(R[k], 0.0).T @ panos.sphere_rotation(R[k], panos.AZ_OFFSET_DEG)
        # in sphere-local axes the camera vertical is +y, which the offset must leave alone
        assert np.allclose(rel @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_coordinates_text_is_potree_parseable():
    poses = _synthetic_poses(3)
    R, C = geometry.frame_rotations(poses)
    names = [f"f{k:04d}.jpg" for k in range(len(poses))]
    text = panos.coordinates_text(names, poses.t, C, R)
    lines = text.strip().split("\n")
    assert lines[0].split("\t") == ["file", "time", "longitude", "latitude", "altitude", "course", "pitch", "roll"]
    assert len(lines) == 1 + len(poses)
    for i, line in enumerate(lines[1:]):
        tok = line.split("\t")
        assert len(tok) == 8 and tok[0] == names[i]
        assert np.allclose([float(x) for x in tok[2:5]], C[i], atol=1e-4)


def test_select_frames_bbox_and_stride():
    poses = _synthetic_poses(20, seed=1)
    e, n = poses.origin[:, 0], poses.origin[:, 1]
    bbox = (e.min() - 1, n.min() - 1, np.median(e), n.max() + 1)
    idx = panos.select_frames(poses, None, bbox, 1)
    assert idx == [int(i) for i in np.flatnonzero(e <= np.median(e))]
    assert panos.select_frames(poses, None, None, 5) == list(range(0, 20, 5))
    assert panos.select_frames(poses, [3, 1, 1], None, 1) == [1, 3]


@pytest.mark.parametrize("k", [0, 119])
def test_real_poses_angles_are_finite(poses, k):
    R, _C = geometry.frame_rotations(poses, idx=np.array([k]))
    course, pitch, roll = panos.potree_angles(R)
    assert np.isfinite([course[0], pitch[0], roll[0]]).all()
    assert abs(pitch[0]) < 30 and abs(roll[0]) < 30


def test_new_export_reproduces_user_confirmed_panos_corr180():
    """The user visually confirmed the sphere export made by the OLD camera model with
    `az_offset_deg = 180` and `texture.repeat.x = -1` (fixture: `panos_corr180_coordinates.txt`,
    built from `poses_corrected_34bca9.csv`, rig IDENTITY, all 1503 frames, old POTREE_M, AZ 180).
    The new camera model + new POTREE_M, at the new default az offset of 0, must write the exact
    same numbers -- that is the evidence that the new derivation is correct, not just self-consistent.
    """
    from geovap.runtime.pose_tables import read as read_pose_table
    from geovap.domain.model.rig import IDENTITY

    fixtures = Path(__file__).parent / "fixtures"
    poses = read_pose_table(fixtures / "poses_corrected_34bca9.csv")
    R, C = geometry.frame_rotations(poses, IDENTITY)
    names = [f"f{k:04d}.jpg" for k in range(len(poses))]
    lines = panos.coordinates_text(names, poses.t, C, R).strip().split("\n")

    ref_lines = (fixtures / "panos_corr180_coordinates.txt").read_text().strip().split("\n")
    assert lines[0] == ref_lines[0]
    assert len(lines) == len(ref_lines)
    for got_line, ref_line in zip(lines[1:], ref_lines[1:]):
        got = got_line.split("\t")
        ref = ref_line.split("\t")
        assert got[0] == ref[0]
        pos_got = np.array([float(x) for x in got[2:5]])
        pos_ref = np.array([float(x) for x in ref[2:5]])
        assert np.max(np.abs(pos_got - pos_ref)) < 1e-4
        ang_got = np.array([float(x) for x in got[5:8]])
        ang_ref = np.array([float(x) for x in ref[5:8]])
        d = np.abs(ang_got - ang_ref)
        d[0] = min(d[0], 360.0 - d[0])  # course wraps at +-180
        assert np.max(d) < 1e-6


def test_potree_angles_at_az_0_matches_the_old_formula_at_az_180():
    """Documents *why* the numbers above are identical: with the camera model reflected about the
    lateral axis, POTREE_M_new = POTREE_M_old @ Rz(180 deg), and az_offset folds a Rz(az) into the
    same product -- so (POTREE_M_new, az=0) and (POTREE_M_old, az=180) give the same R_mesh.
    """
    m_old = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    def old_potree_angles(R):
        az = np.deg2rad(180.0)
        c, s = np.cos(az), np.sin(az)
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        m = np.swapaxes(m_old @ rz @ np.asarray(R, dtype=np.float64), -1, -2)
        b = np.arcsin(np.clip(-m[..., 2, 0], -1.0, 1.0))
        c2 = np.arctan2(m[..., 1, 0], m[..., 0, 0])
        a = np.arctan2(m[..., 2, 1], m[..., 2, 2])
        course = 90.0 - np.degrees(c2)
        pitch = -np.degrees(b)
        roll = np.degrees(a) - 90.0
        return (course + 180.0) % 360.0 - 180.0, pitch, roll

    poses = _synthetic_poses(9, seed=13)
    R, _C = geometry.frame_rotations(poses)
    course_new, pitch_new, roll_new = panos.potree_angles(R, 0.0)
    course_old, pitch_old, roll_old = old_potree_angles(R)
    d_course = np.abs(course_new - course_old)
    d_course = np.minimum(d_course, 360.0 - d_course)
    assert np.max(d_course) < 1e-9
    assert np.allclose(pitch_new, pitch_old, atol=1e-9)
    assert np.allclose(roll_new, roll_old, atol=1e-9)
