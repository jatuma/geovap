"""The real test of the restructuring: a dataset that is not Dražkov.

The point was never to tidy one repository — it was to make the dataset an input. That is only
demonstrable by pointing the tool at a *different* dataset and watching it work, so this file does
that two ways:

  * against the generated fixture, always — a different CRS-agnostic scene, a different pose-CSV
    column layout, four-digit tile ids where Dražkov's are three;
  * against the real survey re-presented under another name, when it is mounted — same points, new
    directory layout, new filename convention, new tile-id width. Every product name must come out
    under the NEW convention.

The second form is what catches the failure the old code could not even express: writing a
correctly-computed product to a filename derived from somebody else's dataset, with no error.
"""
from __future__ import annotations

import pytest

from geovap.runtime import settings

TESTSITE = """
name = "testsite"
[paths]
data_root = "{root}"
workspace = "{ws}"
publish   = "{pub}"
[crs]
epsg = 5514
[poses]
adapter = "ladybug_export_csv"
file = "photos/trajectory.csv"
columns = {{ t = 0, file = 1, e = 2, n = 3, h = 4, roll = 11, pitch = 12, yaw = 13 }}
expect_columns = 17
[panos]
adapter = "equirect_dir"
dir = "photos"
width = 8000
height = 4000
[tiles]
adapter = "laz_dir"
dir = "cloud"
glob = "*.laz"
id_regex = 'SITE-B_tile(?P<id>\\d{{4}})\\.laz'
out_name = "SITE-B_{{id}}{{kind}}.laz"
[tiles.names]
cluster = "objects_{{id}}.laz"
[sensor]
pano_w = 8000
pano_h = 4000
zb_w = 2000
zb_h = 1000
r_min = 1.0
r_max = 40.0
point_spacing = 0.051
cell_size = 4.0
tol_abs = 0.15
tol_rel = 0.03
splat_k = 1.2
splat_min_px = 1
splat_max_px = 8
[tuning]
score_r0 = 8.0
incidence_max_deg = 80.0
top_k = 5
mad_cutoff = 2.5
"""


@pytest.fixture
def renamed_dataset(tmp_path):
    """The mounted dataset, re-presented: symlinks only, so it costs nothing and touches nothing."""
    real = settings.build(dataset="drazkov") if _drazkov_available() else None
    if real is None:
        pytest.skip("the reference dataset is not mounted")

    root = tmp_path / "site-b"
    (root / "photos").mkdir(parents=True)
    (root / "cloud").mkdir()

    src = real.poses.source_file()
    (root / "photos" / "trajectory.csv").symlink_to(src)
    for jpg in sorted(src.parent.glob("*.jpg")):
        (root / "photos" / jpg.name).symlink_to(jpg)
    for i, tile in enumerate(real.tiles.tiles(), start=1):
        (root / "cloud" / f"SITE-B_tile{i:04d}.laz").symlink_to(tile.path)

    toml = tmp_path / "testsite.toml"
    toml.write_text(TESTSITE.format(root=root, ws=tmp_path / "ws", pub=tmp_path / "pub"), encoding="utf-8")
    return settings.build(dataset=toml)


def _drazkov_available() -> bool:
    try:
        s = settings.build(dataset="drazkov")
        src = s.poses.source_file()
        return bool(src and src.exists() and s.tiles.tiles())
    except Exception:  # noqa: BLE001 - not configured on this machine
        return False


def test_the_same_survey_reads_through_a_different_descriptor(renamed_dataset):
    s = renamed_dataset
    reference = settings.build(dataset="drazkov")
    assert len(s.poses.load()) == len(reference.poses.load())
    assert len(s.tiles.tiles()) == len(reference.tiles.tiles())
    # ...and its pose table hashes identically: the adapter read the same numbers, not a copy of them
    assert s.poses.load().hash() == reference.poses.load().hash()


def test_tile_ids_are_four_characters_here_and_three_there(renamed_dataset):
    """The width difference is the point. `stem.split("_")[1][-3:]` and `name[-3:]` were correct
    only because Dražkov's ids happen to be three characters long."""
    assert {len(t.id.value) for t in renamed_dataset.tiles.tiles()} == {4}
    assert {len(t.id.value) for t in settings.build(dataset="drazkov").tiles.tiles()} == {3}


def test_every_product_name_follows_the_new_dataset(renamed_dataset):
    """The failure this prevents: a correctly-computed product written under another dataset's
    filename, silently."""
    ts = renamed_dataset.tiles
    names = [ts.out_name(t.id, kind) for t in ts.tiles() for kind in ("", "_colored", "_seg")]
    names += [ts.out_name(t.id, variant="cluster") for t in ts.tiles()]
    assert all(n.startswith(("SITE-B_", "objects_")) for n in names)
    assert not any("ID3432" in n for n in names), "a Dražkov filename leaked into another dataset"


def test_extent_is_known_without_a_tile_grid(renamed_dataset):
    """This descriptor configures no `grid`, which is optional. The extent still resolves, from the
    LAZ headers — found by running `doctor` against exactly this dataset, where every extent-based
    check failed for want of a file the dataset need not have."""
    tiles = renamed_dataset.tiles.tiles()
    assert all(t.ring is None for t in tiles)
    assert all(t.bbox is not None for t in tiles)


def test_doctor_passes_on_it(renamed_dataset):
    from geovap.app import doctor

    report = doctor.report(renamed_dataset)
    failed = [k for k, v in report["checks"].items() if v.get("ok") is False]
    assert not failed, f"{failed}\n{doctor.render(report)}"


def test_the_generated_fixture_is_also_not_drazkov(tmp_path, monkeypatch):
    """Always runs: the fixture needs no mounted data at all."""
    from geovap.io.datasets import synthetic

    root = tmp_path / "fx"
    synthetic.build(root)
    monkeypatch.setenv("GEOVAP_SYNTHETIC_ROOT", str(root))
    monkeypatch.setenv("GEOVAP_SYNTHETIC_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("GEOVAP_SYNTHETIC_PUBLISH", str(tmp_path / "pub"))
    s = settings.build(dataset="synthetic")
    assert s.name == "synthetic"
    assert not any("ID3432" in s.tiles.out_name(t.id) for t in s.tiles.tiles())
