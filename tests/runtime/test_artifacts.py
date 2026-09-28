"""The inter-stream contracts. A stream may change its own code freely; changing one of these
breaks another team's build, so they are pinned."""
from __future__ import annotations

from geovap.runtime import artifacts, settings


def test_every_artifact_resolves_under_the_workspace(make_descriptor_file, env_paths):
    s = settings.build(dataset=make_descriptor_file())
    for a in artifacts.ARTIFACTS:
        p = a.path(s)
        assert s.workspace.root in p.parents or p == s.workspace.root, (a.name, p)


def test_the_product_is_one_laz_per_tile_and_nothing_else(make_descriptor_file, env_paths):
    """The consolidation this restructuring exists to deliver: `out/consolidated/` holds `tiles/`
    only. The former `objects/` and `vendor/` sets stored the same 585 M XYZ triples again just to
    hand the viewer a different RGB."""
    s = settings.build(dataset=make_descriptor_file())
    a = artifacts.get("consolidated_tiles")
    assert a.path(s) == s.workspace.consolidated / "tiles"
    assert a.per_tile and a.producer == "merge"


def test_optional_artifacts_are_the_detachable_streams(make_descriptor_file, env_paths):
    """Clustering and semantics must be droppable without failing a run."""
    optional = {a.name for a in artifacts.ARTIFACTS if a.optional}
    assert optional == {"object_clusters", "point_labels"}


def test_pose_table_columns_are_pinned():
    """Four streams parse this CSV. Renaming a column is a cross-team break."""
    assert artifacts.POSE_TABLE_COLUMNS == (
        "frame", "filename", "t", "E", "N", "H", "roll", "pitch", "yaw", "pass_id")
    assert "poses_hash" in artifacts.POSE_TABLE_META


def test_docs_table_is_generated_from_this_module():
    md = artifacts.as_markdown()
    for a in artifacts.ARTIFACTS:
        assert f"`{a.name}`" in md
