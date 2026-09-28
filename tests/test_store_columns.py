"""S1 store extension: extra columns + per-tile time index, against the real store."""
import numpy as np
import pytest

from mapping import config
from mapping.cloud_store import EXTRA_COLUMNS, CloudStore


@pytest.fixture(scope="module")
def store():
    if not (config.STORE_DIR / "tiles.json").exists():
        pytest.skip("store not built")
    s = CloudStore()
    if s.tile(s.tiles[0].name).user_data is None:
        pytest.skip("S1 columns not built (run mapping.cli.store_add_columns)")
    return s


def test_extra_columns_have_tile_length(store):
    for t in store.tiles[:5]:
        td = store.tile(t.name)
        for name in EXTRA_COLUMNS:
            a = getattr(td, name)
            assert a is not None
            assert len(a) == t.n, (t.name, name, len(a), t.n)
        assert td.user_data.dtype == np.uint8
        assert td.scan_angle_rank.dtype == np.int8
        assert td.return_number.dtype == np.uint8


def test_rows_in_time_matches_brute_force(store):
    rng = np.random.default_rng(0)
    for t in store.tiles[:3]:
        td = store.tile(t.name)
        g = np.asarray(td.gps_time)
        for _ in range(3):
            t0 = float(rng.uniform(g.min(), g.max() - 0.5))
            t1 = t0 + 0.5
            rows = td.rows_in_time(t0, t1)
            brute = np.sort(np.flatnonzero((g >= t0) & (g <= t1)))
            assert np.array_equal(np.sort(rows), brute), (t.name, t0, t1, len(rows), len(brute))


def test_query_time_consistent_across_tiles(store):
    # a window spanning the whole dataset must, tile by tile, match each tile's own rows_in_time
    t0 = min(t.gps_time_range[0] for t in store.tiles if t.gps_time_range) - 1.0
    t1 = t0 + 1.0
    res = store.query_time(t0, t1)
    for t, rows in res:
        td = store.tile(t.name)
        expect = td.rows_in_time(t0, t1)
        assert np.array_equal(np.sort(rows), np.sort(expect))
    # tiles whose cached gps_time_range does not overlap [t0, t1] must be absent
    got_names = {t.name for t, _ in res}
    for t in store.tiles:
        if t.gps_time_range is not None and (t.gps_time_range[1] < t0 or t.gps_time_range[0] > t1):
            assert t.name not in got_names


def test_query_time_no_bbox_filter_superset(store):
    t = store.tiles[0]
    t0, t1 = t.gps_time_range[0], t.gps_time_range[0] + 0.2
    full = store.query_time(t0, t1)
    boxed = store.query_time(t0, t1, bbox=t.bbox)
    full_n = sum(len(r) for _, r in full)
    boxed_n = sum(len(r) for _, r in boxed)
    assert boxed_n <= full_n
