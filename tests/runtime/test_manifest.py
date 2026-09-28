"""Resume semantics: a stage is skipped only when it would produce the same thing again."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geovap.runtime import manifest, settings
from geovap.stages.base.spec import StageSpec


@dataclass
class _Stage:
    spec: StageSpec
    _inputs: dict
    _outputs: list

    def inputs(self, s):
        return self._inputs

    def outputs(self, s):
        return self._outputs


def test_inputs_hash_follows_content_not_mtime(tmp_path):
    f = tmp_path / "in.csv"
    f.write_text("a")
    h1 = manifest.inputs_hash({"f": f})
    f.write_text("a")  # rewritten, same bytes
    assert manifest.inputs_hash({"f": f}) == h1
    f.write_text("b")
    assert manifest.inputs_hash({"f": f}) != h1


def test_a_missing_input_hashes_differently_from_an_empty_one(tmp_path):
    """An input that appears later must make the stage stale; hashing a missing path as if it were
    an empty file would leave the stage green forever."""
    missing = manifest.inputs_hash({"f": tmp_path / "nope"})
    (tmp_path / "nope").write_text("")
    assert manifest.inputs_hash({"f": tmp_path / "nope"}) != missing


def test_large_files_are_hashed_by_stat_not_by_content(tmp_path, monkeypatch):
    """The point store is tens of GB; re-reading it on every `status` call is not an option."""
    monkeypatch.setattr(manifest, "HASH_CONTENT_MAX_BYTES", 4)
    big = tmp_path / "big.bin"
    big.write_bytes(b"0123456789")
    entry = manifest.hashable(big)
    assert entry[0] == "path" and entry[3] == 10  # (kind, path, mtime_ns, size)


def test_is_done_requires_marker_hash_and_outputs(make_descriptor_file, env_paths, tmp_path):
    s = settings.build(dataset=make_descriptor_file())
    ws = s.workspace
    src, out = tmp_path / "src", tmp_path / "out"
    src.write_text("v1")
    stage = _Stage(StageSpec(name="demo"), {"src": src}, [out])

    assert not manifest.is_done(stage, s)  # no marker
    manifest.write_marker(ws, "demo", manifest.make_marker(
        name="demo", rc=0, inputs={"src": src}, outputs=[out], metrics={}))
    assert not manifest.is_done(stage, s)  # marker green but the output is missing
    out.write_text("done")
    assert manifest.is_done(stage, s)
    src.write_text("v2")
    assert not manifest.is_done(stage, s)  # input changed -> stale


def test_a_corrupt_marker_means_not_done(make_descriptor_file, env_paths):
    s = settings.build(dataset=make_descriptor_file())
    s.workspace.pipeline.mkdir(parents=True, exist_ok=True)
    manifest.marker_path(s.workspace, "demo").write_text("{ not json")
    assert manifest.read_marker(s.workspace, "demo") is None


def test_run_manifest_round_trips(make_descriptor_file, env_paths):
    """Totals are measured into the manifest, never configured: `config.py:71` asserted against
    Dražkov's own 584 809 840 points, which no second dataset can satisfy."""
    s = settings.build(dataset=make_descriptor_file())
    m = manifest.RunManifest(dataset=s.name, descriptor=str(s.descriptor.file), total_points=123, n_tiles=2)
    m.save(s.workspace)
    back = manifest.RunManifest.load(s.workspace)
    assert (back.total_points, back.n_tiles, back.dataset) == (123, 2, "testds")
