"""Part B: `mapping.cli.pipeline` driver -- structure/marker/dry-run tests only, no data, no
subprocesses actually launched (Popen/run are monkeypatched to explode if the dry-run path ever
tries to call them)."""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from mapping.cli import pipeline as P

EXPECTED_ORDER = [
    "env-check",
    "baseline",
    "store-columns",
    "align",
    "traj-rot",
    "traj-validate",
    "refine",
    "register",
    "reg-conflict",
    "assemble",
    "products",
    "pose-report",
    "colorize",
    "colour-report",
    "quality",
    "promote",
    "segds",
    "seg-eval",
    "seg-project",
    "compare",
    "merge",
    "potree",
    "panos",
    "validate",
]


# --------------------------------------------------------------------------------- stage graph
def test_stage_order_matches_plan():
    assert P.stage_names() == EXPECTED_ORDER


def test_stage_names_unique():
    names = P.stage_names()
    assert len(names) == len(set(names))


def test_optional_stages_flagged():
    optional = {s.name for s in P.STAGES if s.optional}
    assert optional == {"traj-validate", "reg-conflict"}


def test_every_stage_has_commands_or_inprocess():
    ctx = P.Ctx()
    for stage in P.STAGES:
        if stage.inprocess is not None:
            continue
        cmds = stage.commands(ctx)
        assert isinstance(cmds, list)
        for cmd in cmds:
            assert isinstance(cmd, list) and all(isinstance(x, str) for x in cmd)


# --------------------------------------------------------------------------------- inputs hash
def test_inputs_hash_stable_and_sensitive_to_content(tmp_path):
    f = tmp_path / "a.json"
    f.write_text('{"x": 1}')
    stage = P.Stage(name="fake", commands=lambda ctx: [], inputs=lambda ctx: {"f": f})
    ctx = P.Ctx()
    h1 = P.compute_inputs_hash(stage, ctx)
    h2 = P.compute_inputs_hash(stage, ctx)
    assert h1 == h2

    f.write_text('{"x": 2}')
    h3 = P.compute_inputs_hash(stage, ctx)
    assert h3 != h1


def test_inputs_hash_handles_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    stage = P.Stage(name="fake", commands=lambda ctx: [], inputs=lambda ctx: {"f": missing})
    ctx = P.Ctx()
    h1 = P.compute_inputs_hash(stage, ctx)
    h2 = P.compute_inputs_hash(stage, ctx)
    assert h1 == h2  # deterministic even when the input doesn't exist yet


# --------------------------------------------------------------------------------- marker round trip
def test_marker_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)
    marker = {"rc": 0, "inputs_hash": "abc123", "outputs": ["/x"], "metrics": {"n": 3}, "cmds": []}
    P.write_marker("fake-stage", marker)
    back = P.read_marker("fake-stage")
    assert back == marker
    assert P.read_marker("nonexistent-stage") is None


# --------------------------------------------------------------------------------- is_done
def test_is_done_logic(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)
    inp = tmp_path / "in.txt"
    inp.write_text("hello")
    out = tmp_path / "out.txt"
    stage = P.Stage(name="s", commands=lambda ctx: [], inputs=lambda ctx: {"in": inp}, outputs=lambda ctx: [out])
    ctx = P.Ctx()

    # no marker yet -> not done
    assert not P.is_done(stage, ctx)

    ih = P.compute_inputs_hash(stage, ctx)
    P.write_marker("s", {"rc": 0, "inputs_hash": ih, "outputs": [str(out)], "metrics": {}, "cmds": []})
    # marker green but output missing -> not done
    assert not P.is_done(stage, ctx)

    out.write_text("done")
    assert P.is_done(stage, ctx)

    # inputs changed after the marker was written -> stale
    inp.write_text("changed")
    assert not P.is_done(stage, ctx)

    # restore input, but a failed marker (rc!=0) is never "done"
    inp.write_text("hello")
    P.write_marker("s", {"rc": 1, "inputs_hash": ih, "outputs": [str(out)], "metrics": {}, "cmds": []})
    assert not P.is_done(stage, ctx)


# --------------------------------------------------------------------------------- compare
def test_compare_renders_from_fixture_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)
    P.write_marker("baseline", {"rc": 0, "inputs_hash": "x", "outputs": [], "cmds": [], "metrics": {"classes": {"clean": 800, "unverified": 150, "usable": 300, "reject": 200}, "eomt_city_mIoU_core": 0.35}})
    P.write_marker("quality", {"rc": 0, "inputs_hash": "x", "outputs": [], "cmds": [], "metrics": {"clean": 830, "unverified": 163, "usable": 302, "reject": 208}})
    P.write_marker("register", {"rc": 0, "inputs_hash": "x", "outputs": [], "cmds": [], "metrics": {"rms_before_median": 0.102, "rms_after_median": 0.035}})
    ctx = P.Ctx()
    P._do_compare(ctx)
    md = (tmp_path / "comparison.md").read_text()
    js = json.loads((tmp_path / "comparison.json").read_text())
    assert "clean / unverified / usable / reject" in md
    assert js["rows"]
    assert any(r["metric"].startswith("pass registration: median RMS") for r in js["rows"])


def test_compare_handles_no_markers_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)
    ctx = P.Ctx()
    P._do_compare(ctx)  # must not raise even with nothing recorded yet
    assert (tmp_path / "comparison.md").exists()
    assert (tmp_path / "comparison.json").exists()


# --------------------------------------------------------------------------------- dry run
def test_dry_run_lists_every_stage_and_never_calls_subprocess(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)

    def _boom(*a, **k):
        raise AssertionError("subprocess must not be invoked in --dry-run")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    monkeypatch.setattr(subprocess, "run", _boom)

    ap = P.build_argparser()
    args = ap.parse_args(["run", "--dry-run", "--with-optional"])
    rc = P.cmd_run(args)
    assert rc == 0

    out = capsys.readouterr().out
    for name in P.stage_names():
        assert f"=== {name} " in out, f"stage {name!r} missing from dry-run output"
    # nothing should have been marked done by a dry run
    assert list(tmp_path.glob("*.json")) == []


def test_dry_run_without_optional_skips_optional_stages(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    ap = P.build_argparser()
    args = ap.parse_args(["run", "--dry-run"])
    P.cmd_run(args)
    out = capsys.readouterr().out
    assert "=== traj-validate " not in out
    assert "=== reg-conflict " not in out
    assert "=== align " in out


# --------------------------------------------------------------------------------- select_stages
def test_select_stages_from_to_only():
    ap = P.build_argparser()
    args = ap.parse_args(["run", "--from", "register", "--to", "products"])
    names = [s.name for s in P.select_stages(args)]
    assert names == ["register", "assemble", "products"]

    args2 = ap.parse_args(["run", "--only", "merge,potree"])
    names2 = [s.name for s in P.select_stages(args2)]
    assert names2 == ["merge", "potree"]


# --------------------------------------------------------------------------------- est_min
def test_every_stage_has_a_nonzero_estimate():
    for stage in P.STAGES:
        assert stage.est_min > 0, f"stage {stage.name!r} has no est_min set"


def test_dry_run_prints_nonzero_minutes_for_a_slow_stage(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    ap = P.build_argparser()
    args = ap.parse_args(["run", "--dry-run", "--only", "quality"])
    P.cmd_run(args)
    out = capsys.readouterr().out
    assert "=== quality (required, ~90 min) ===" in out


# --------------------------------------------------------------------------------- stage.cwd wiring
def test_run_one_uses_stage_cwd(tmp_path, monkeypatch):
    calls = []

    class _FakeProc:
        stdout = iter(["ok\n"])

        def wait(self):
            return 0

    def fake_popen(cmd, cwd=None, **kw):
        calls.append(cwd)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(P, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)
    monkeypatch.setattr(P, "PIPELINE_LOG", tmp_path / "pipeline.log")

    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    rc = P._run_one("fake", 0, ["true"], P.Ctx(), cwd=other_dir)
    assert rc == 0
    assert calls == [str(other_dir)]


def test_run_stage_passes_stage_cwd_to_run_one(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)
    seen = {}

    def fake_run_one(name, idx, cmd, ctx, cwd=P.REPO_ROOT):
        seen["cwd"] = cwd
        return 0

    monkeypatch.setattr(P, "_run_one", fake_run_one)
    custom = tmp_path / "custom_cwd"
    custom.mkdir()
    stage = P.Stage(name="fake-cwd-stage", commands=lambda ctx: [["true"]], cwd=custom)
    rc = P.run_stage(stage, P.Ctx(), force=True)
    assert rc == 0
    assert seen["cwd"] == custom


# --------------------------------------------------------------------------------- detach / pid guard
def test_detach_refuses_second_run_when_pid_alive(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)
    monkeypatch.setattr(P, "PID_FILE", tmp_path / "pipeline.pid")
    monkeypatch.setattr(P, "PIPELINE_LOG", tmp_path / "pipeline.log")
    P.PID_FILE.write_text(str(os.getpid()))  # our own pid is definitely alive

    def _boom(*a, **k):
        raise AssertionError("must not spawn a second detached run")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    ap = P.build_argparser()
    args = ap.parse_args(["run", "--detach"])
    rc = P._detach(args)
    assert rc != 0


def test_detach_force_detach_bypasses_pid_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)
    monkeypatch.setattr(P, "PID_FILE", tmp_path / "pipeline.pid")
    monkeypatch.setattr(P, "PIPELINE_LOG", tmp_path / "pipeline.log")
    P.PID_FILE.write_text(str(os.getpid()))

    class _FakeProc:
        pid = 999999

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
    ap = P.build_argparser()
    args = ap.parse_args(["run", "--detach", "--force-detach"])
    rc = P._detach(args)
    assert rc == 0
    assert P.PID_FILE.read_text().strip() == "999999"


def test_detach_starts_when_pid_file_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PIPELINE_DIR", tmp_path)
    monkeypatch.setattr(P, "PID_FILE", tmp_path / "pipeline.pid")
    monkeypatch.setattr(P, "PIPELINE_LOG", tmp_path / "pipeline.log")
    # a pid that (almost certainly) doesn't exist
    P.PID_FILE.write_text("999999999")

    class _FakeProc:
        pid = 123456

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
    ap = P.build_argparser()
    args = ap.parse_args(["run", "--detach"])
    rc = P._detach(args)
    assert rc == 0
    assert P.PID_FILE.read_text().strip() == "123456"
