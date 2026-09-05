"""Synthetic occlusion: a near wall must hide a far wall behind it; seam wrap must work."""
import numpy as np

from mapping import geometry, zbuffer
from mapping.config import NO_POINT, PANO_W, ZB_W


def _wall(x0, y_range, z_range, spacing=0.05):
    ys = np.arange(*y_range, spacing)
    zs = np.arange(*z_range, spacing)
    Y, Z = np.meshgrid(ys, zs)
    return np.stack([np.full(Y.size, x0), Y.ravel(), Z.ravel()], 1)


def test_near_wall_occludes_far_wall():
    C = np.zeros(3)
    R = np.eye(3)
    near = _wall(5.0, (-3, 3), (-1, 2))
    far = _wall(12.0, (-3, 3), (-1, 2))
    P = np.concatenate([near, far])
    pid = np.arange(len(P), dtype=np.uint32)
    u, v, r, el = geometry.world_to_pano(P, R, C)
    s = ZB_W / PANO_W
    depth, ids = zbuffer.splat(u * s, v * s, r, pid)
    closed = zbuffer.close_depth(depth)
    vis = zbuffer.visible(r, u * s, v * s, closed)
    n_near = len(near)
    assert vis[:n_near].mean() > 0.99
    assert vis[n_near:].mean() < 0.02
    # the buffer holds the near wall's ids where both overlap
    assert np.all(ids[np.isfinite(depth)] < n_near)


def test_seam_wrap_and_empty_cells():
    C = np.zeros(3)
    R = np.eye(3)
    # points straight ahead (az ~ 0) straddle the seam u=0/W
    P = np.array([[10.0, 0.001, 0.0], [10.0, -0.001, 0.0]])
    u, v, r, el = geometry.world_to_pano(P, R, C)
    assert u[0] < 1 and u[1] > PANO_W - 1
    s = ZB_W / PANO_W
    depth, ids = zbuffer.splat(u * s, v * s, r, np.array([7, 8], np.uint32))
    row = int(v[0] * s)
    assert np.isfinite(depth[row, 0]) and np.isfinite(depth[row, ZB_W - 1])
    assert ids[0, 0] == NO_POINT and depth[0, 0] == np.inf
    mm = zbuffer.depth_to_mm(depth)
    back = zbuffer.depth_from_mm(mm)
    assert mm[0, 0] == 0 and back[0, 0] == np.inf
    assert abs(back[row, 0] - depth[row, 0]) <= 0.0005 + 1e-6
