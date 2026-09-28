"""`io.registry`: unknown adapter names fail loudly, and an absent `[reference]` table is `None`
rather than a stub adapter."""
from __future__ import annotations

import pytest

from geovap.io.descriptor import Descriptor
from geovap.io.registry import (
    UnknownAdapterError,
    build_pano_source,
    build_pose_source,
    build_reference,
    build_tile_source,
)


def test_absent_reference_table_returns_none(make_descriptor_file, env_paths):
    d = Descriptor.load(make_descriptor_file(with_reference=False))
    assert d.reference is None
    assert build_reference(d) is None


def test_present_reference_table_builds_adapter(make_descriptor_file, env_paths, dataset_dir):
    (dataset_dir / "ref.geojson").write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    d = Descriptor.load(make_descriptor_file(with_reference=True))
    ref = build_reference(d)
    assert ref is not None
    assert ref.objects() == []


def test_unknown_adapter_name_lists_known_ones(make_descriptor_file, env_paths):
    d = Descriptor.load(make_descriptor_file())
    broken = d.tiles | {"adapter": "not_a_real_adapter"}
    d = Descriptor(
        name=d.name, file=d.file, data_root=d.data_root, workspace=d.workspace, publish=d.publish,
        crs=d.crs, sensor=d.sensor, tuning=d.tuning, poses=d.poses, panos=d.panos,
        tiles=broken, reference=d.reference,
    )
    with pytest.raises(UnknownAdapterError) as exc_info:
        build_tile_source(d)
    message = str(exc_info.value)
    assert "not_a_real_adapter" in message
    assert "laz_dir" in message  # the registered name it should have used instead


def test_all_builtin_adapters_are_registered(make_descriptor_file, env_paths):
    d = Descriptor.load(make_descriptor_file())
    assert build_pose_source(d).describe()["exists"] is True
    assert build_tile_source(d).describe()["exists"] is True
    assert build_pano_source(d).describe()["exists"] is True
