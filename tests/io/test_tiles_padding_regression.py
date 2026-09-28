"""Regression tests for two tile-identity bugs that produce no error, only wrong output.

Both were found by diffing the new adapter against the legacy loaders on the real dataset, not by
any test -- which is the point of pinning them here. The synthetic fixture in `conftest.py` uses
unpadded tile ids (`tile_007.laz`) and no grid, so neither bug can reproduce there.

1. Zero padding. The legacy pipeline derived a tile's name as `stem.split("_")[1][-3:]` and wrote
   its products as `ID3432_000<name>.laz`. An `id_regex` of `0*(?P<id>\\d+)` strips the padding, so
   every one of the 38 tiles would have been written as `ID3432_0001.laz` instead of
   `ID3432_000001.laz` -- a silent, total rename of the product.

2. Id normalisation asymmetry. The grid reader keys its rings by a normalised (bare-integer) id
   while `tiles()` looked them up by the raw captured id. That happens to work only while the regex
   already strips padding; fixing (1) made every lookup miss and every `ring`/`bbox` silently None.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geovap.io.descriptor import Descriptor
from geovap.io.registry import build_tile_source

# A skewed quadrilateral, as the real tile grid uses -- a bbox is not a substitute for the ring.
RING = [[0.0, 0.0], [10.0, 2.0], [12.0, 12.0], [2.0, 10.0], [0.0, 0.0]]

DESCRIPTOR = """
name = "padded"
[paths]
data_root = "{root}"
workspace = "{root}/w"
publish   = "{root}/p"
[crs]
epsg = 5514
[poses]
adapter = "ladybug_export_csv"
file = "export.csv"
columns = {{ t = 0, file = 1, e = 2, n = 3, h = 4, roll = 5, pitch = 6, yaw = 7 }}
[panos]
adapter = "equirect_dir"
dir = "."
width = 100
height = 50
[tiles]
adapter = "laz_dir"
dir = "tiles"
glob = "*.laz"
id_regex = 'ID\\d+_\\d*(?P<id>\\d{{3}})_JTSK\\.laz'
out_name = "ID3432_000{{id}}{{kind}}.laz"
{grid_line}
[sensor]
pano_w = 100
pano_h = 50
zb_w = 25
zb_h = 12
r_min = 1.0
r_max = 10.0
point_spacing = 0.05
cell_size = 1.0
tol_abs = 0.1
tol_rel = 0.02
splat_k = 1.0
splat_min_px = 1
splat_max_px = 4
[tuning]
score_r0 = 4.0
incidence_max_deg = 70.0
top_k = 3
mad_cutoff = 2.0
"""


@pytest.fixture
def padded(tmp_path: Path):
    """Two zero-padded tiles plus a grid whose label is padded too ("001")."""
    (tmp_path / "tiles").mkdir()
    for name in ("ID3432_000001_JTSK.laz", "ID3432_000037_JTSK.laz"):
        (tmp_path / "tiles" / name).write_bytes(b"")
    grid = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": RING}},
            {"type": "Feature", "properties": {"name": "001"}, "geometry": {"type": "Point", "coordinates": [5.0, 5.0]}},
        ],
    }
    (tmp_path / "grid.geojson").write_text(json.dumps(grid), encoding="utf-8")

    def _make(with_grid: bool) -> Descriptor:
        text = DESCRIPTOR.format(root=tmp_path, grid_line='grid = "grid.geojson"' if with_grid else "")
        path = tmp_path / "padded.toml"
        path.write_text(text, encoding="utf-8")
        return Descriptor.load(path)

    return _make


def _legacy_name(path: Path) -> str:
    """`mapping/cloud_store.py:_tile_name_from_laz` -- the behaviour the products were written under."""
    return path.stem.split("_")[1][-3:]


def test_tile_id_keeps_zero_padding(padded):
    ts = build_tile_source(padded(with_grid=False))
    ids = {t.path.name: t.id.value for t in ts.tiles()}
    assert ids == {"ID3432_000001_JTSK.laz": "001", "ID3432_000037_JTSK.laz": "037"}
    for t in ts.tiles():
        assert t.id.value == _legacy_name(t.path)


def test_out_name_matches_the_legacy_product_filenames(padded):
    ts = build_tile_source(padded(with_grid=False))
    for t in ts.tiles():
        legacy = _legacy_name(t.path)
        assert ts.out_name(t.id) == f"ID3432_000{legacy}.laz"
        assert ts.out_name(t.id, "_colored") == f"ID3432_000{legacy}_colored.laz"


def test_grid_ring_is_found_for_a_padded_id(padded):
    """The lookup must normalise both sides; a raw-id lookup silently yields ring=None."""
    tiles = {t.id.value: t for t in build_tile_source(padded(with_grid=True)).tiles()}
    assert tiles["001"].ring is not None, "padded id failed to match its grid label"
    assert np.allclose(tiles["001"].ring, np.asarray(RING)[:, :2])
    assert tiles["037"].ring is None  # no label for it in the grid


def test_bbox_is_derived_from_the_ring_not_substituted_for_it(padded):
    """Tiles are skewed quadrilaterals: the ring must survive, with bbox as a derived convenience."""
    t = {x.id.value: x for x in build_tile_source(padded(with_grid=True)).tiles()}["001"]
    assert t.ring.shape == (5, 2)
    assert t.bbox == (0.0, 0.0, 12.0, 12.0)
    # the bbox corner is outside the actual tile, which is why the ring cannot be dropped
    from matplotlib.path import Path as MplPath

    assert not MplPath(t.ring).contains_point((11.5, 0.5))
