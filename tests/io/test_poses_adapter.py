"""`ladybug_export_csv`: column indices come from the descriptor, and the header-shape check is
this adapter's own validation (not a hardcoded assertion)."""
from __future__ import annotations

import numpy as np
import pytest

from geovap.io.descriptor import Descriptor
from geovap.io.registry import build_pose_source

# mirrors conftest._ROWS: (t, file, roll, pitch, yaw, e, n, h)
_ROWS = [
    (100.0, "f000.jpg", 1.0, 2.0, 10.0, 500.0, 1000.0, 200.0),
    (99.0, "f001.jpg", 1.5, 2.5, 20.0, 501.0, 1001.0, 201.0),
    (101.0, "f002.jpg", 0.5, 1.5, 30.0, 502.0, 999.0, 199.0),
]


def test_load_reads_declared_columns(make_descriptor_file, env_paths):
    d = Descriptor.load(make_descriptor_file())
    poses = build_pose_source(d).load()

    assert len(poses) == len(_ROWS)
    # sorted by timestamp: row index 1 (t=99) comes first, then 0 (t=100), then 2 (t=101)
    assert list(poses.filename) == ["f001.jpg", "f000.jpg", "f002.jpg"]
    assert np.allclose(poses.t, sorted(t for t, *_ in _ROWS))
    assert poses.origin.shape == (3, 3)
    # row for f001.jpg: e=501, n=1001, h=201 (see conftest._ROWS)
    i = list(poses.filename).index("f001.jpg")
    assert np.allclose(poses.origin[i], [501.0, 1001.0, 201.0])
    assert np.allclose(poses.roll[i], 1.5)
    assert np.allclose(poses.yaw[i], 20.0)
    assert poses.source == "export"


def test_describe_reports_counts_and_span(make_descriptor_file, env_paths):
    d = Descriptor.load(make_descriptor_file())
    info = build_pose_source(d).describe()
    assert info["exists"] is True
    assert info["count"] == len(_ROWS)
    assert info["t_min"] == min(t for t, *_ in _ROWS)
    assert info["t_max"] == max(t for t, *_ in _ROWS)


def test_column_count_mismatch_raises_clear_message(make_descriptor_file, env_paths, dataset_dir):
    # rewrite export.csv with only 5 header columns -- expect_columns=17 in the descriptor should
    # reject it with a message naming both the expected and actual column counts.
    csv_path = dataset_dir / "panos" / "export.csv"
    csv_path.write_text("a,b,c,d,e\n1,2,3,4,5\n", encoding="utf-8")
    d = Descriptor.load(make_descriptor_file())
    source = build_pose_source(d)
    with pytest.raises(ValueError) as exc_info:
        source.load()
    message = str(exc_info.value)
    assert "17" in message
    assert "5" in message
