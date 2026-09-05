import numpy as np

from mapping import vectors
from mapping.config import PANO_W


def test_subdivision_limits_angular_step():
    C = np.zeros(3)
    line = np.array([[5.0, -5.0, 0.0], [5.0, 5.0, 0.0]])  # 90 deg of azimuth at 5-7 m
    pts = vectors.subdivide_3d(line, C, 0.25)
    d = pts - C
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(1), -1, 1)))
    assert ang.max() <= 0.25 + 1e-6 and len(pts) > 300


def test_seam_split_produces_two_pieces_reaching_the_seam():
    u = np.array([7900.0, 7990.0, 20.0, 100.0])
    v = np.array([2000.0, 2001.0, 2002.0, 2003.0])
    pieces = vectors.split_at_seam(u, v)
    assert len(pieces) == 2
    (u0, v0), (u1, v1) = pieces
    assert u0[-1] == PANO_W and u1[0] == 0.0
    assert abs(v0[-1] - v1[0]) < 1e-9 and 2001.0 < v0[-1] < 2002.0
    assert np.all(np.abs(np.diff(u0)) < PANO_W / 2) and np.all(np.abs(np.diff(u1)) < PANO_W / 2)


def test_project_polyline_without_depth_is_visible_in_range():
    C = np.array([0.0, 0.0, 2.0])
    R = np.eye(3)
    line = np.array([[10.0, -2.0, 0.0], [10.0, 2.0, 0.0], [60.0, 2.0, 0.0]])  # last vertex beyond R_MAX
    pieces = vectors.project_polyline(line, R, C, None)
    allv = np.concatenate([p.visible for p in pieces])
    r = np.concatenate([p.r for p in pieces])
    assert allv[r <= 40].all() and not allv[r > 40].any()
    hw = np.concatenate([p.half_width_px for p in pieces])
    assert np.all(np.diff(hw[r > 10.5]) <= 1e-9)  # band narrows with range
