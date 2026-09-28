"""`laz_dir`: tile id parsing from the descriptor's `id_regex`, `out_name` for both product kinds,
and the error for a file that doesn't match the pattern."""
from __future__ import annotations

import pytest

from geovap.domain.model.tiles import TileId
from geovap.io.descriptor import Descriptor
from geovap.io.registry import build_tile_source


def test_tiles_parsed_by_id_regex(make_descriptor_file, env_paths):
    d = Descriptor.load(make_descriptor_file())
    source = build_tile_source(d)
    tiles = {t.id.value: t for t in source.tiles()}
    assert set(tiles) == {"007", "012"}
    assert tiles["007"].path.name == "tile_007.laz"


def test_out_name_for_both_kinds(make_descriptor_file, env_paths):
    d = Descriptor.load(make_descriptor_file())
    source = build_tile_source(d)
    tile = TileId("007")
    assert source.out_name(tile) == "OUT_007.laz"
    assert source.out_name(tile, "_colored") == "OUT_007_colored.laz"


def test_describe_reports_count(make_descriptor_file, env_paths):
    d = Descriptor.load(make_descriptor_file())
    info = build_tile_source(d).describe()
    assert info["exists"] is True
    assert info["count"] == 2
    assert set(info["ids"]) == {"007", "012"}


def test_non_matching_filename_raises(make_descriptor_file, env_paths, dataset_dir):
    (dataset_dir / "tiles" / "not_a_tile.laz").write_bytes(b"")
    d = Descriptor.load(make_descriptor_file())
    source = build_tile_source(d)
    with pytest.raises(ValueError) as exc_info:
        source.tiles()
    message = str(exc_info.value)
    assert "not_a_tile.laz" in message
    assert "id_regex" in message or r"tile_(?P<id>\d+)" in message
