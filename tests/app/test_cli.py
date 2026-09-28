"""The `geovap` command. Smoke-level, but it covers the paths that must work on a machine with no
dataset mounted at all -- which is the state a new team member starts in."""
from __future__ import annotations

import json

import pytest

from geovap.app import cli


def _run(capsys, argv) -> tuple[int, str]:
    rc = cli.main(argv)
    return rc, capsys.readouterr().out


def test_stages_lists_what_is_installed(capsys, make_descriptor_file, env_paths):
    rc, out = _run(capsys, ["stages", "--dataset", str(make_descriptor_file()), "--json"])
    rows = json.loads(out)
    assert rc == 0 and {r["stage"] for r in rows} >= {"ingest", "store", "products", "cluster"}
    assert all(set(r) >= {"after", "optional", "est_min", "summary", "available"} for r in rows)


def test_datasets_reports_where_each_descriptor_resolved_from(capsys, env_paths):
    rc, out = _run(capsys, ["datasets", "--json"])
    rows = {r["name"]: r for r in json.loads(out)}
    assert rc == 0 and "synthetic" in rows
    assert rows["synthetic"]["file"].endswith("synthetic.toml")


def test_run_dry_run_writes_nothing(capsys, make_descriptor_file, env_paths, tmp_path):
    ws = tmp_path / "untouched"
    rc, out = _run(capsys, ["run", "--dataset", str(make_descriptor_file()),
                            "--workspace", str(ws), "--dry-run"])
    assert rc == 0
    assert "roughly" in out
    assert not ws.exists(), "a dry run created the workspace"


def test_doctor_fails_loudly_on_a_dataset_it_cannot_read(capsys, make_descriptor_file, env_paths, tmp_path):
    """The point of doctor: a non-zero exit and a named problem, not a traceback and not silence."""
    rc, out = _run(capsys, ["doctor", "--dataset", str(make_descriptor_file()),
                            "--data-root", str(tmp_path / "nothing-here")])
    assert rc == 1
    assert "FAILED" in out


def test_list_profiles_needs_no_dataset(capsys):
    rc, out = _run(capsys, ["run", "--list-profiles"])
    assert rc == 0 and "objects" in out and "prepare" in out
