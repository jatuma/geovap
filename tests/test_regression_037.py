"""Regression anchor from the pilot (02_obarveni_pointcloudu.md SS2.3): tile 037, nearest-in-time frame,
nearest pixel, no occlusion -> median CIE76 in 6.1-6.3; mirrored azimuth ~17.  Slow: needs store + frames."""
import numpy as np
import pytest

from mapping import config, metrics
from mapping.cloud_store import CloudStore
from mapping.colorize import Options, colorize_tile
from mapping.frame_select import FrameIndex

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def env():
    if not (config.STORE_DIR / "tiles.json").exists() or not (config.FRAMES_DIR / "f0367.npz").exists():
        pytest.skip("store/frames not built")
    from mapping.poses import load_poses

    store = CloudStore()
    poses = load_poses()
    return store, FrameIndex(poses)


def _run(store, fi, **kw):
    opt = Options(write_las=False, subsample=120, **kw)
    return colorize_tile(store.by_name["037"], store, fi, opt, log=lambda *a: None)


def test_pilot_anchor_reproduced(env):
    """The pilot's recipe turned out to be: nearest-in-time frame, nearest pixel, no occlusion, no vehicle
    mask, and points closer than 3.5 m to the camera dropped -> median CIE76 6.08 (n 43k of the 1/120 subsample)."""
    store, fi = env
    res = _run(store, fi, occlusion=False, vehicle_mask=False)
    s = res.sample
    m = s["valid_nt_noocc"] & (s["cam_dist"] >= 3.5)
    med = float(np.median(s["de76_nt_noocc"][m]))
    print(f"pilot recipe: n={m.sum()} median CIE76={med:.3f}")
    assert 5.9 <= med <= 6.3, med
    # without the range cut the vehicle body dominates the tail
    med_all = metrics.hist_stats(res.hist["nt_noocc_76"].total_hist())["median"]
    assert med_all > med


def test_vehicle_mask_and_occlusion(env):
    store, fi = env
    res = _run(store, fi, occlusion=True, vehicle_mask=True)
    s76_noocc = metrics.hist_stats(res.hist["nt_noocc_76"].total_hist())
    s76_occ = metrics.hist_stats(res.hist["nt_76"].total_hist())
    s00_noocc = metrics.hist_stats(res.hist["nt_noocc"].total_hist())
    s00_occ = metrics.hist_stats(res.hist["nt"].total_hist())
    print("nt_noocc CIE76", s76_noocc, "\nnt CIE76", s76_occ, "\nnt_noocc dE00", s00_noocc, "\nnt dE00", s00_occ)
    assert 5.3 <= s76_noocc["median"] <= 6.1, s76_noocc
    # A1 instrumentation rule: occlusion cuts the gross-error share / upper percentiles, barely moves the median
    assert s00_occ["p90"] < s00_noocc["p90"] and s00_occ["pct_gt_20"] <= s00_noocc["pct_gt_20"]
    assert abs(s76_occ["median"] - s76_noocc["median"]) < 1.0


def test_mirrored_is_much_worse(env):
    store, fi = env
    res = _run(store, fi, mirror=True)
    s76 = metrics.hist_stats(res.hist["nt_noocc_76"].total_hist())
    print("mirrored CIE76", s76)
    assert s76["median"] > 14.0
