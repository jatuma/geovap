"""Panorama spheres: the angles written to coordinates.txt must reproduce our own projection.

The reference below is a transcription of what the viewer does -- three.js r124 `SphereGeometry`
plus `Potree.Images360Loader` -- and is deliberately independent of `mapping.panos`.
"""
import numpy as np
import pytest

from mapping import geometry, panos
from mapping.config import PANO_H, PANO_W
from mapping.poses import Poses


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
    uv = (phi / 2pi, 1 - theta / pi); with `texture.repeat.x = 1` (view.html resets Potree's -1)
    texel s = u / w and t = 1 - v / h.
    """
    phi = 2.0 * np.pi * (u / w)
    theta = np.pi * (v / h)
    return np.array([-np.cos(phi) * np.sin(theta), np.cos(theta), np.sin(phi) * np.sin(theta)])


# ------------------------------------------------------------------ tests
def test_sphere_local_dir_matches_potree_m():
    """POTREE_M is exactly the camera -> sphere-local map of the reference sphere."""
    for u, v in [(0, 0.5 * PANO_H), (1234.5, 900.0), (PANO_W * 0.75, PANO_H * 0.9), (7999.0, 10.0)]:
        d_cam = geometry.pano_rays(u, v)
        assert np.allclose(panos.POTREE_M @ d_cam, sphere_local_dir(u, v), atol=1e-12)


@pytest.mark.parametrize("az_offset", [0.0, panos.AZ_OFFSET_DEG])
def test_written_angles_reproduce_the_projection(az_offset):
    """A pixel put on the sphere by the viewer points where geometry says it points.

    With az_offset = 0 this is the pure convention check; the production value turns the sphere
    about the camera vertical, so the expected ray is the geometry ray for the shifted column.
    """
    poses = _synthetic_poses()
    R, _C = geometry.frame_rotations(poses)
    course, pitch, roll = panos.potree_angles(R, az_offset)
    rng = np.random.default_rng(0)
    uv = np.stack([rng.uniform(0, PANO_W, 40), rng.uniform(1, PANO_H - 1, 40)], 1)
    for k in range(len(poses)):
        mesh = potree_mesh_rotation(course[k], pitch[k], roll[k])
        for u, v in uv:
            want = geometry.pano_to_world_ray((u + az_offset / 360.0 * PANO_W) % PANO_W, v, R[k])
            got = mesh @ sphere_local_dir(u, v)
            # arccos near 1 is float-noisy; 1e-5 deg is 7e-4 px at 8000 px width
            assert np.degrees(np.arccos(np.clip(want @ got, -1, 1))) < 1e-5


@pytest.mark.parametrize("az_offset", [0.0, 90.0, panos.AZ_OFFSET_DEG])
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
