import numpy as np
import pytest

from mapping import geometry
from mapping.config import PANO_H, PANO_W
from mapping.seg import views


def _frame(yaw=37.0, roll=6.0, pitch=-4.0):
    R_cam = geometry.vehicle_rotation(yaw, roll, pitch)
    R_lev = geometry.vehicle_rotation(yaw, 0.0, 0.0)
    return R_cam, R_lev


@pytest.mark.parametrize("view", [(0.0, 0.0, 90.0), (135.0, 0.0, 90.0), (270.0, -45.0, 90.0)])
def test_round_trip_off_centre_both_axes(view):
    R_cam, R_lev = _frame()
    size = 512
    mu, mv = views.view_to_pano_maps(R_cam, R_lev, view, size)
    # pick off-centre pixels in both axes and map them back
    ys = np.array([50, 100, 400, 460])
    xs = np.array([60, 450, 120, 500])
    d_cam = geometry.pano_rays(mu[ys, xs], mv[ys, xs])
    x, y, ok = views.pano_to_view_xy(d_cam, R_cam, R_lev, view)
    assert ok.all()
    px = (x + 0.5) * size - 0.5
    py = (y + 0.5) * size - 0.5
    assert np.allclose(px, xs, atol=1e-6) and np.allclose(py, ys, atol=1e-6)


def test_levelled_view_keeps_horizon_horizontal():
    """World-horizontal rays must land on y = 0 of a pitch-0 view whatever the frame roll/pitch."""
    R_cam, R_lev = _frame(yaw=200.0, roll=7.0, pitch=-5.0)
    az = np.deg2rad(np.linspace(-40, 40, 9)) + np.deg2rad(200.0 + 90.0)  # around the view direction (yaw 90 -> left)
    d_world = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], -1)
    d_cam = d_world @ R_cam.T
    x, y, ok = views.pano_to_view_xy(d_cam, R_cam, R_lev, (90.0, 0.0, 90.0))
    assert ok.all() and np.allclose(y, 0.0, atol=1e-9)
    assert x.min() < -0.3 and x.max() > 0.3  # the rays really spread across the view


def test_view_centre_direction():
    """Centre pixel of view (yaw, pitch) looks along yaw_frame+yaw at elevation pitch in the world."""
    R_cam, R_lev = _frame(yaw=30.0, roll=3.0, pitch=2.0)
    for yaw, pitch in [(0.0, 0.0), (90.0, 0.0), (180.0, -45.0)]:
        mu, mv = views.view_to_pano_maps(R_cam, R_lev, (yaw, pitch, 90.0), 64)
        d_cam = geometry.pano_rays(mu[31:33, 31:33].ravel(), mv[31:33, 31:33].ravel()).mean(0)  # average directions, not seam-crossing pixels
        d_world = d_cam @ R_cam
        az = np.degrees(np.arctan2(d_world[1], d_world[0])) % 360
        el = np.degrees(np.arcsin(np.clip(d_world[2], -1, 1)))
        assert abs(((az - (30.0 + yaw)) + 180) % 360 - 180) < 0.5
        assert abs(el - pitch) < 0.5


def test_operational_band_coverage():
    """Every ERP pixel with elevation in [-55, +45] deg is inside at least one of the 12 views."""
    R_cam, R_lev = _frame(yaw=10.0, roll=2.0, pitch=-1.0)
    v = np.linspace((90 - 45) / 180 * PANO_H, (90 + 55) / 180 * PANO_H, 60)
    u = np.linspace(0, PANO_W, 240, endpoint=False)
    uu, vv = np.meshgrid(u, v)
    d_cam = geometry.pano_rays(uu.ravel(), vv.ravel())
    covered = np.zeros(d_cam.shape[0], bool)
    for view in views.VIEWS:
        x, y, ok = views.pano_to_view_xy(d_cam, R_cam, R_lev, view)
        covered |= ok & (np.abs(x) <= 0.5) & (np.abs(y) <= 0.5)
    assert covered.mean() > 0.99


def test_extract_image_wraps_seam():
    pano = np.zeros((100, 200, 3), np.uint8)
    pano[:, :5] = 255  # bright stripe at the seam
    mu = np.array([[199.5, 0.5]], np.float32) * (8000 / 200)
    mv = np.array([[50.0, 50.0]], np.float32) * (4000 / 100)
    out = views.extract_image(pano, mu, mv, views.cv2.INTER_NEAREST)
    assert out[0, 1, 0] == 255 and out[0, 0, 0] == 0
