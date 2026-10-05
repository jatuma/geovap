"""The driver: ordering, selection and resumption.

`mapping/cli/pipeline.py` kept the run order as a literal `_ORDER` list next to 24 inline stage
declarations, with an `assert` whose only job was to catch the two drifting apart. Here the order is
derived from what each stage declares, so the failure mode is gone -- but the derivation itself now
needs pinning, which is what this file does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from geovap.app import driver, profiles
from geovap.runtime import manifest, settings
from geovap.stages.base.spec import StageRegistry, StageSpec


@dataclass
class FakeStage:
    spec: StageSpec
    _available: bool = True
    _outputs: list = field(default_factory=list)
    _inputs: dict = field(default_factory=dict)
    ran: int = 0

    def available(self, s):
        return self._available

    def inputs(self, s):
        return self._inputs

    def outputs(self, s):
        return self._outputs

    def metrics(self, s):
        return {}

    def run(self, s, **opts):
        self.ran += 1


def _registry(*stages) -> StageRegistry:
    r = StageRegistry()
    for st in stages:
        r.stages[st.spec.name] = st  # bypass add(): these share one source file by construction
    return r


@pytest.fixture
def s(make_descriptor_file, env_paths, tmp_path):
    return settings.build(dataset=make_descriptor_file(), workspace=tmp_path / "ws")


def test_order_comes_from_what_stages_declare(s):
    r = _registry(
        FakeStage(StageSpec(name="c", after=("b",))),
        FakeStage(StageSpec(name="a")),
        FakeStage(StageSpec(name="b", after=("a",))),
    )
    assert r.order() == ["a", "b", "c"]


def test_an_unknown_predecessor_is_ignored_not_fatal(s):
    """An install without `[semantics]` has `merge` listing `label` as a predecessor while the
    semantics stages are absent. That must run, not crash."""
    r = _registry(FakeStage(StageSpec(name="merge", after=("label", "colorize"))),
                  FakeStage(StageSpec(name="colorize")))
    assert r.order() == ["colorize", "merge"]


def test_a_cycle_is_reported_with_the_stages_in_it(s):
    r = _registry(FakeStage(StageSpec(name="x", after=("y",))),
                  FakeStage(StageSpec(name="y", after=("x",))))
    with pytest.raises(ValueError, match="cycle.*x, y"):
        r.order()


def test_every_exclusion_says_which_kind_it_is(s, tmp_path):
    """Four different reasons a stage does not run, and they are not interchangeable: 'skipped' with
    no reason is useless when something downstream is missing."""
    done = FakeStage(StageSpec(name="done"), _outputs=[tmp_path / "out"])
    (tmp_path / "out").write_text("x")
    manifest.write_marker(s.workspace, "done", manifest.make_marker(
        name="done", rc=0, inputs={}, outputs=[tmp_path / "out"], metrics={}))

    r = _registry(
        done,
        FakeStage(StageSpec(name="gone"), _available=False),
        FakeStage(StageSpec(name="branch", optional=True)),
        FakeStage(StageSpec(name="asked")),
        FakeStage(StageSpec(name="unasked")),
    )
    plan = driver.select(r, s, only=("done", "gone", "branch", "asked"))
    # `branch` is optional but was named explicitly, so it runs -- naming a stage IS asking for it.
    assert plan.selected == ["asked", "branch"]
    assert plan.skipped["gone"] == "unavailable for this dataset"
    assert plan.skipped["done"] == "done"
    assert plan.skipped["unasked"] == "deselected"

    # without --only, the optional branch is left out, and says so rather than reporting "deselected"
    assert driver.select(r, s).skipped["branch"] == "optional"


def test_optional_stages_run_when_asked_for_by_name(s):
    """`--only cluster` must run clustering even though it is an optional branch; otherwise the
    stream that can be worked on in isolation cannot be invoked in isolation."""
    r = _registry(FakeStage(StageSpec(name="cluster", optional=True)))
    assert driver.select(r, s, only=("cluster",)).selected == ["cluster"]
    assert driver.select(r, s).selected == []
    assert driver.select(r, s, with_optional=True).selected == ["cluster"]


def test_force_reruns_a_done_stage(s, tmp_path):
    out = tmp_path / "o"
    out.write_text("x")
    st = FakeStage(StageSpec(name="s1"), _outputs=[out])
    manifest.write_marker(s.workspace, "s1", manifest.make_marker(
        name="s1", rc=0, inputs={}, outputs=[out], metrics={}))
    r = _registry(st)
    assert driver.select(r, s).selected == []
    assert driver.select(r, s, force=True).selected == ["s1"]


def test_pose_table_is_export_up_to_the_stage_that_builds_the_corrected_one(s):
    """The stages that BUILD the corrected pose table cannot read it -- it does not exist yet. The
    old driver held this as a hardcoded set of ten stage names; deriving it from the order means a
    new registration stage is handled without anyone remembering."""
    r = _registry(
        FakeStage(StageSpec(name="align")),
        FakeStage(StageSpec(name="assemble", after=("align",))),
        FakeStage(StageSpec(name="colorize", after=("assemble",))),
    )
    s2 = s.with_pose_table("corrected")
    assert driver.pose_table_for(r, s2, "align") == "export"
    assert driver.pose_table_for(r, s2, "assemble") == "export"
    assert driver.pose_table_for(r, s2, "colorize") == "corrected"


def test_a_failed_stage_is_not_done(s, tmp_path):
    out = tmp_path / "o"
    out.write_text("x")
    st = FakeStage(StageSpec(name="s1"), _outputs=[out])
    manifest.write_marker(s.workspace, "s1", manifest.make_marker(
        name="s1", rc=1, inputs={}, outputs=[out], metrics={}))
    assert not manifest.is_done(st, s)


def test_profiles_resolve_to_selections():
    for name in profiles.PROFILES:
        sel = profiles.as_selection(profiles.get(name))
        assert set(sel) <= {"skip", "with_optional", "end", "only", "start"}
    with pytest.raises(KeyError, match="unknown profile"):
        profiles.get("nope")


# --------------------------------------------------------------------------------- detached runs
def test_a_second_detached_run_is_refused(s, monkeypatch, capsys):
    """Two runs sharing one workspace interleave their markers and logs and overwrite each other's
    outputs mid-write. The guard is the only thing that stops it, since a full run is normally
    started detached and the terminal closed."""
    import os
    import subprocess

    s.workspace.mkdirs()
    driver.pid_file(s).write_text(str(os.getpid()))  # our own pid is certainly alive
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("spawned a second run"))
    assert driver.detach(s, ["run"]) != 0
    assert "already in progress" in capsys.readouterr().out


def test_a_stale_pid_file_does_not_block_a_run(s, monkeypatch):
    """A run that was killed leaves its pid file behind; that must not wedge the workspace."""
    import subprocess

    s.workspace.mkdirs()
    driver.pid_file(s).write_text("999999999")  # almost certainly not a live process
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: type("P", (), {"pid": 4242})())
    assert driver.detach(s, ["run"]) == 0
    assert driver.pid_file(s).read_text().strip() == "4242"


def test_force_detach_overrides_a_live_run(s, monkeypatch):
    import os
    import subprocess

    s.workspace.mkdirs()
    driver.pid_file(s).write_text(str(os.getpid()))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: type("P", (), {"pid": 777})())
    assert driver.detach(s, ["run"], force=True) == 0


def test_every_installed_stage_declares_a_cost():
    """`geovap run` prints an estimate before a multi-hour run; a stage with no estimate makes that
    number quietly wrong."""
    registry = driver.build_registry()
    missing = [n for n in registry.order() if registry[n].spec.est_min <= 0]
    assert not missing, f"stages with no time estimate: {missing}"
