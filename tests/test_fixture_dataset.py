"""The generated fixture dataset must stay usable, because it is what every stream tests against.

Without it, "run your stage against a dataset" means mounting 600 GB of Dražkov, which no CI and no
new team member has. The fixture is generated from nothing in under a second, and its panoramas are
rendered from its own point cloud -- so a stage run against it produces a checkable answer, not
noise.
"""
from __future__ import annotations

import numpy as np
import pytest

from geovap.io.datasets import synthetic


@pytest.fixture(scope="module")
def fixture_dataset(tmp_path_factory, ):
    root = tmp_path_factory.mktemp("fixture-data")
    synthetic.build(root)
    return root


@pytest.fixture
def fixture_settings(fixture_dataset, tmp_path, monkeypatch):
    from geovap.runtime import settings

    monkeypatch.setenv("GEOVAP_SYNTHETIC_ROOT", str(fixture_dataset))
    monkeypatch.setenv("GEOVAP_SYNTHETIC_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("GEOVAP_SYNTHETIC_PUBLISH", str(tmp_path / "pub"))
    return settings.build(dataset="synthetic")


def test_all_four_adapters_read_it(fixture_settings):
    s = fixture_settings
    poses = s.poses.load()
    assert len(poses) >= 8 and np.all(np.diff(poses.t) > 0)
    tiles = s.tiles.tiles()
    assert len(tiles) >= 3, "fewer than three tiles cannot exercise a cross-tile query"
    assert all(t.ring is not None for t in tiles), "every tile must get a ring from the grid"
    assert all(s.panos.path(str(f)).exists() for f in poses.filename)
    assert s.reference is not None and len(s.reference.objects()) > 0


def test_its_tile_ids_are_not_three_characters(fixture_settings):
    """Chosen deliberately: Dražkov's ids are three digits, and the old code sliced `[-3:]` off
    filenames. A fixture with the same width would let that assumption survive unnoticed."""
    widths = {len(t.id.value) for t in fixture_settings.tiles.tiles()}
    assert widths and 3 not in widths


def test_a_store_builds_from_it_and_queries_across_tiles(fixture_settings):
    from geovap.runtime import store as store_mod

    s = fixture_settings
    store_mod.build_store(s, workers=2)
    cs = store_mod.CloudStore(s.workspace.store)
    assert cs.total > 10_000 and len(cs.tiles) == len(s.tiles.tiles())

    poses = s.poses.load()
    mid = poses.origin[len(poses) // 2]
    parts = cs.query_disc(float(mid[0]), float(mid[1]), 15.0)
    assert len(parts) > 1, "the tile grid does not actually split the scene"


def test_the_panoramas_agree_with_the_point_cloud(fixture_settings):
    """The property that makes this fixture worth having: project the points a pose should see and
    they must land on painted pixels. A registration stage tested here gets a real signal."""
    from geovap.domain.math import depth
    from geovap.domain.model import geometry
    from geovap.io.images import load_pano_rgb
    from geovap.runtime import store as store_mod

    s = fixture_settings
    store_mod.build_store(s, workers=2)
    cs = store_mod.CloudStore(s.workspace.store)
    poses = s.poses.load()
    k = len(poses) // 2
    R, C = geometry.frame_rotations(poses)

    parts = cs.query_disc(float(C[k, 0]), float(C[k, 1]), s.sensor.r_max)
    xyz = np.concatenate([cs.tile(t.name).xyz_m(rows) for t, rows in parts])
    u, v, r, _el = geometry.world_to_pano(xyz, R[k], C[k], s.sensor.pano_w, s.sensor.pano_h)
    keep = depth.range_filter(r, s.sensor.r_min, s.sensor.r_max)
    assert keep.sum() > 1000

    img = load_pano_rgb(str(s.panos.path(str(poses.filename[k]))))
    iu = np.mod(np.floor(u[keep]).astype(np.int64), s.sensor.pano_w)
    iv = np.clip(np.floor(v[keep]).astype(np.int64), 0, s.sensor.pano_h - 1)
    painted = (img[iv, iu] > 0).any(axis=-1).mean()
    assert painted > 0.95, f"only {painted:.1%} of visible points land on a painted pixel"


def test_it_is_deterministic(tmp_path):
    """Two developers must get the same fixture, or a failure is not reproducible."""
    import hashlib

    def digest(root):
        return {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()
        }

    a, b = tmp_path / "a", tmp_path / "b"
    synthetic.build(a)
    synthetic.build(b)
    assert digest(a) == digest(b)
