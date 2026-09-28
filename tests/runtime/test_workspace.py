"""The output layout, and the one piece of it that is a correctness guarantee rather than a
convention: how an export run and a corrected run are kept apart."""
from __future__ import annotations

from dataclasses import dataclass

from geovap.runtime import settings
from geovap.runtime.workspace import Workspace


@dataclass
class _FakePoses:
    """Just the two members `source_dir` needs -- typed structurally on purpose."""

    source: str
    _hash: str = "abcdef1234"

    def hash(self) -> str:
        return self._hash


def _ws(make_descriptor_file) -> Workspace:
    return settings.build(dataset=make_descriptor_file()).workspace


def test_export_outputs_keep_their_exact_paths(make_descriptor_file, env_paths):
    """THE regression anchor. Export-pose outputs must land on unsuffixed paths so a re-run can be
    diffed byte-for-byte against the baseline. If this test fails, every recorded measurement in the
    project's history becomes incomparable."""
    ws = _ws(make_descriptor_file)
    base = ws.out / "dataset"
    assert ws.source_dir(base, _FakePoses("export")) == base
    assert ws.frames_dir(_FakePoses("export")) == ws.frames


def test_a_corrected_table_never_writes_into_the_export_outputs(make_descriptor_file, env_paths):
    ws = _ws(make_descriptor_file)
    base = ws.out / "dataset"
    corrected = _FakePoses("poses_corrected", "a2d74f0580")
    assert ws.source_dir(base, corrected) == base.with_name("dataset_a2d74f")
    assert ws.frames_dir(corrected) == ws.frames / "a2d74f"


def test_source_dir_is_a_sibling_and_frames_dir_is_a_child(make_descriptor_file, env_paths):
    """These two look interchangeable and are not: the existing cache has `frames/<hash>/` but
    `out/dataset_<hash>/`. Unifying them would orphan a 30 GB product directory."""
    ws = _ws(make_descriptor_file)
    corrected = _FakePoses("poses_corrected")
    assert ws.source_dir(ws.out / "dataset", corrected).parent == ws.out
    assert ws.frames_dir(corrected).parent == ws.frames


def test_layout_is_rooted_in_the_descriptor_workspace(make_descriptor_file, env_paths):
    ws = _ws(make_descriptor_file)
    for p in (ws.store, ws.frames, ws.out, ws.poses, ws.pipeline, ws.consolidated_tiles, ws.derived):
        assert ws.root in p.parents
    assert ws.consolidated_tiles == ws.root / "out" / "consolidated" / "tiles"


def test_baseline_defaults_to_a_directory_beside_the_descriptor(make_descriptor_file, env_paths):
    """Baselines are measurements OF a dataset, so they travel with its descriptor rather than
    shipping inside the application (`pipeline.py:64` held one run's numbers as a Python dict)."""
    s = settings.build(dataset=make_descriptor_file())
    assert s.workspace.baseline == s.descriptor.file.resolve().parent / "testds" / "baseline"


def test_clean_frames_falls_back_to_the_baseline(make_descriptor_file, env_paths):
    """Before the screening stage has run there is no workspace copy; a stage that needs the clean
    set must still be runnable against the dataset's tracked one."""
    import json

    s = settings.build(dataset=make_descriptor_file())
    ws = s.workspace
    ws.baseline.mkdir(parents=True, exist_ok=True)
    (ws.baseline / "clean_frames.json").write_text(json.dumps({"clean": [3, 1, 2], "reject": []}))
    try:
        assert ws.clean_frames() == [1, 2, 3]
        ws.derived.mkdir(parents=True, exist_ok=True)
        ws.clean_frames_json.write_text(json.dumps({"clean": [9]}))
        assert ws.clean_frames() == [9]  # the workspace copy wins once it exists
    finally:
        (ws.baseline / "clean_frames.json").unlink()
