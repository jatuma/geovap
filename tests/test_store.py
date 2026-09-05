"""Cloud store integrity: counts, bit-exact coordinates, cell query vs brute force."""
import numpy as np
import pytest

from mapping import config
from mapping.cloud_store import SCALE, CloudStore


@pytest.fixture(scope="module")
def store():
    if not (config.STORE_DIR / "tiles.json").exists():
        pytest.skip("store not built")
    return CloudStore()


def test_total_count(store):
    assert store.total == config.EXPECTED_TOTAL_POINTS
    assert len(store.tiles) == 38
    offs = [t.row_offset for t in store.tiles]
    assert offs == sorted(offs) and offs[0] == 0


def test_tile_037_roundtrip_against_laz(store):
    laspy = pytest.importorskip("laspy")
    t = store.by_name["037"]
    td = store.tile("037")
    las = laspy.read(t.laz)
    assert len(td) == las.header.point_count
    inv = np.asarray(td.orig_index)
    xyz_back = np.empty((len(td), 3), dtype=np.int32)
    xyz_back[inv] = td.xyz  # undo the cell sort
    assert np.array_equal(xyz_back[:, 0], las.X) and np.array_equal(xyz_back[:, 1], las.Y) and np.array_equal(xyz_back[:, 2], las.Z)
    cls_back = np.empty(len(td), np.uint8)
    cls_back[inv] = td.classification
    assert np.array_equal(cls_back, np.asarray(las.classification))
    rgb_back = np.empty((len(td), 3), np.uint8)
    rgb_back[inv] = td.rgb
    assert np.array_equal(rgb_back[:, 0].astype(np.uint16) * 256, np.asarray(las.red))


def test_gps_time_overlaps_frames(store, poses):
    td = store.tile("037")
    g = np.asarray(td.gps_time)
    # same time base: the tile's scan interval overlaps the camera timestamps and starts within a
    # couple of seconds of a frame (scanner runs continuously, camera fires every ~5 m)
    assert g.min() < poses.t.max() and g.max() > poses.t.min()
    assert np.abs(poses.t - g.min()).min() < 5.0


def test_query_disc_matches_brute_force(store):
    t = store.by_name["037"]
    td = store.tile("037")
    rng = np.random.default_rng(0)
    xy_all = td.xyz_m()[:, :2]
    for _ in range(3):
        c = xy_all[rng.integers(len(td))]
        R = 25.0
        res = store.query_disc(float(c[0]), float(c[1]), R)
        got = np.concatenate([store.global_ids(ti, rows) for ti, rows in res])
        d2 = ((xy_all - c) ** 2).sum(1)
        brute_local = np.flatnonzero(d2 <= R * R)
        brute = t.row_offset + brute_local
        # points from neighbouring tiles may legitimately be included; the 037 subset must match exactly
        got_037 = np.sort(got[(got >= t.row_offset) & (got < t.row_offset + t.n)])
        assert np.array_equal(got_037, np.sort(brute))


def test_locate_inverse_of_global_ids(store):
    t = store.tiles[5]
    rows = np.array([0, 10, t.n - 1])
    gid = store.global_ids(t, rows)
    ti, local = store.locate(gid)
    assert np.all(ti == 5) and np.array_equal(local, rows)
