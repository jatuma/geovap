"""Stream C: clustering. Two things are pinned here.

First that it works -- on the generated fixture, end to end, including the cross-tile object merge,
which is the part no single-tile test can reach.

Second, and more importantly for the team split, that the ALGORITHMS import no project code. That
property is what lets this stream be handed to someone with no Geovap context, run before anything
else exists, or be dropped from a run entirely. It is easy to break by accident -- one convenient
`from geovap.runtime import settings` inside `cluster_laz` and the stream is welded to the rest --
and nothing but a test notices.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from geovap.io.datasets import synthetic

ALGORITHMS = ("cluster_laz.py", "merge_tiles.py")
OBJECTS_DIR = Path(__file__).resolve().parents[3] / "src/geovap/stages/objects"


def _top_level_imports(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


@pytest.mark.parametrize("name", ALGORITHMS)
def test_the_algorithms_import_no_project_code(name):
    mods = _top_level_imports(OBJECTS_DIR / name)
    assert not (mods & {"geovap", "mapping"}), f"{name} now depends on the project: {sorted(mods)}"
    assert mods <= {"argparse", "glob", "os", "sys", "time", "concurrent", "multiprocessing",
                    "numpy", "scipy", "laspy", "skimage"}, sorted(mods)


def test_only_the_driver_knows_about_datasets():
    """`cluster.py` is allowed -- and required -- to read the descriptor; the algorithms are not."""
    assert "geovap" in _top_level_imports(OBJECTS_DIR / "cluster.py")


@pytest.fixture(scope="module")
def clustered(tmp_path_factory):
    """The fixture dataset, clustered, with the cross-tile merge applied."""
    from geovap.runtime import settings
    from geovap.stages.objects.cluster import STAGE, out_dir

    root = tmp_path_factory.mktemp("cluster-data")
    synthetic.build(root)
    ws = tmp_path_factory.mktemp("cluster-ws")
    s = settings.build(dataset="synthetic", data_root=root, workspace=ws, publish=ws / "pub")
    STAGE.run(s, workers=2)
    return s, out_dir(s)


def test_it_writes_a_cluster_id_dimension_per_tile(clustered):
    import laspy

    s, d = clustered
    for tile in s.tiles.tiles():
        p = d / s.tiles.out_name(tile.id, variant="cluster")
        assert p.exists(), p
        cid = np.asarray(laspy.read(str(p)).cluster_id)
        assert len(cid) > 0
        assert (cid >= 0).any(), f"tile {tile.id} has no clustered points at all"


def test_objects_are_merged_across_tile_borders(clustered):
    """The fixture's facades and street run through all four tiles, so a correct merge must produce
    objects whose ids appear in more than one tile. Without it, every tile border cuts an object in
    two and the delivered product has a seam."""
    import laspy

    s, d = clustered
    per_tile = {}
    for tile in s.tiles.tiles():
        cid = np.asarray(laspy.read(str(d / s.tiles.out_name(tile.id, variant="cluster"))).cluster_id)
        per_tile[tile.id.value] = set(int(x) for x in np.unique(cid) if x >= 0)

    shared = [i for i in set().union(*per_tile.values())
              if sum(i in ids for ids in per_tile.values()) > 1]
    assert shared, f"no object spans a tile border; per-tile ids: {per_tile}"

    labels = np.load(d / "global_labels.npy")
    assert len(np.unique(labels)) < sum(len(v) for v in per_tile.values()), (
        "the merge produced as many global objects as tile-local ones -- nothing was joined"
    )


def test_the_stage_is_optional(clustered):
    """A run must complete with clustering skipped entirely."""
    from geovap.stages.objects.cluster import STAGE

    assert STAGE.spec.optional
    assert STAGE.spec.after == (), "clustering must not wait for any other stage"
