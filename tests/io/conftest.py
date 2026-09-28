"""Synthetic dataset fixture for `geovap.io` tests: no real Dražkov data needed.

Builds a tiny stand-in for a dataset directory (a 3-row export.csv with a deliberately unusual
column layout, two empty .laz files whose names exercise the tile id regex, and a descriptor TOML
that references it through an env var), so the adapters' behaviour is tested against something
that is not Dražkov -- proving the column indices, tile regex and paths really come from the
descriptor rather than being hardcoded.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# 17 columns, matching Dražkov's export.csv shape, but with the pose fields at different indices
# (t=0, file=1, roll=2, pitch=3, yaw=4, e=5, n=6, h=7) to prove the adapter reads `[poses].columns`
# rather than assuming Dražkov's own (0,1,2,3,4,11,12,13) layout.
_HEADER = [f"col{i}" for i in range(17)]
_ROWS = [
    # t,    file,        roll, pitch, yaw,   e,       n,       h,     col8..col16 (unused filler)
    (100.0, "f000.jpg", 1.0, 2.0, 10.0, 500.0, 1000.0, 200.0),
    (99.0, "f001.jpg", 1.5, 2.5, 20.0, 501.0, 1001.0, 201.0),
    (101.0, "f002.jpg", 0.5, 1.5, 30.0, 502.0, 999.0, 199.0),
]


def _write_export_csv(path: Path, header: list[str]) -> None:
    lines = [",".join(header)]
    for t, filename, roll, pitch, yaw, e, n, h in _ROWS:
        row = [""] * len(header)
        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7] = (
            str(t), filename, str(roll), str(pitch), str(yaw), str(e), str(n), str(h),
        )
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


DESCRIPTOR_TEMPLATE = """
name = "testds"

[paths]
data_root = "${{TESTDS_DATA_ROOT}}"
workspace = "${{TESTDS_WORKSPACE}}"
publish   = "${{TESTDS_PUBLISH}}"

[crs]
epsg = 5514

[poses]
adapter = "ladybug_export_csv"
file = "panos/export.csv"
columns = {{ t = 0, file = 1, roll = 2, pitch = 3, yaw = 4, e = 5, n = 6, h = 7 }}
expect_columns = 17

[panos]
adapter = "equirect_dir"
dir = "panos"
width = 100
height = 50

[tiles]
adapter = "laz_dir"
dir = "tiles"
glob = "*.laz"
id_regex = 'tile_(?P<id>\\d+)\\.laz'
out_name = "OUT_{{id}}{{kind}}.laz"
{reference_block}

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
def dataset_dir(tmp_path: Path) -> Path:
    """A synthetic dataset directory: `panos/export.csv`, `panos/f000.jpg` (+ f001, f002), and
    `tiles/tile_007.laz` + `tiles/tile_012.laz` (empty files -- the tile adapter only parses
    names unless a grid geojson is given, which this fixture omits)."""
    root = tmp_path / "data"
    (root / "panos").mkdir(parents=True)
    (root / "tiles").mkdir(parents=True)
    _write_export_csv(root / "panos" / "export.csv", _HEADER)
    for _, filename, *_rest in _ROWS:
        (root / "panos" / filename).write_bytes(b"")
    (root / "tiles" / "tile_007.laz").write_bytes(b"")
    (root / "tiles" / "tile_012.laz").write_bytes(b"")
    return root


@pytest.fixture
def make_descriptor_file(tmp_path: Path):
    """Factory: writes a descriptor TOML (with an optional `[reference]` table) and returns its
    path, ready for `Descriptor.load(path, overrides=...)`."""

    def _make(*, with_reference: bool = False) -> Path:
        reference_block = ""
        if with_reference:
            reference_block = '\n[reference]\nadapter = "jvf_zps"\nfile = "ref.geojson"\n'
        text = DESCRIPTOR_TEMPLATE.format(reference_block=reference_block)
        path = tmp_path / "testds.toml"
        path.write_text(text, encoding="utf-8")
        return path

    return _make


@pytest.fixture
def env_paths(monkeypatch, dataset_dir: Path, tmp_path: Path) -> dict[str, str]:
    """Sets the three env vars the descriptor template references, pointing at the synthetic
    dataset built by `dataset_dir`."""
    values = {
        "TESTDS_DATA_ROOT": str(dataset_dir),
        "TESTDS_WORKSPACE": str(tmp_path / "workspace"),
        "TESTDS_PUBLISH": str(tmp_path / "publish"),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values
