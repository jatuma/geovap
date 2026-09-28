"""What `settings` must guarantee: a dataset is chosen at call time, not at import time, and an
explicit argument always beats a stale environment variable."""
from __future__ import annotations

from pathlib import Path

import pytest

from geovap.io.descriptor import DescriptorError
from geovap.runtime import settings


def test_explicit_paths_beat_the_environment(make_descriptor_file, env_paths, tmp_path, monkeypatch):
    """The bug this prevents: a CLI flag silently ignored because an override was exported in the
    shell. Precedence is argument > `$GEOVAP_OVERRIDE_*` > the descriptor's own value."""
    stale, wanted = tmp_path / "stale", tmp_path / "wanted"
    monkeypatch.setenv(settings.ENV_OVERRIDE_DATA_ROOT, str(stale))
    monkeypatch.setenv(settings.ENV_OVERRIDE_WORKSPACE, str(stale))

    s = settings.configure(dataset=make_descriptor_file(), data_root=wanted)
    assert s.paths.data_root == wanted  # the flag wins
    assert s.paths.workspace == stale  # what the flag did not name still comes from the environment


def test_the_environment_can_redirect_a_dataset_without_editing_its_descriptor(
    make_descriptor_file, env_paths, tmp_path, monkeypatch
):
    """`$GEOVAP_OVERRIDE_*` beats the descriptor. That is how a `--data-root` flag reaches a stage
    subprocess, and how a developer points a dataset at a copy for one run."""
    monkeypatch.setenv(settings.ENV_OVERRIDE_WORKSPACE, str(tmp_path / "scratch"))
    s = settings.build(dataset=make_descriptor_file())
    assert s.paths.workspace == tmp_path / "scratch"


def test_a_descriptors_own_variable_is_not_a_global_override(
    make_descriptor_file, env_paths, dataset_dir, tmp_path, monkeypatch
):
    """The fixture dataset resolves `${GEOVAP_SYNTHETIC_ROOT}` and Dražkov resolves `${GEOVAP_DATA}`.
    If those variables doubled as blanket overrides, exporting one to work on Dražkov would silently
    point the fixture -- and every other dataset -- at Dražkov's directory too. The run would
    succeed, against the wrong data."""
    monkeypatch.setenv("GEOVAP_DATA", str(tmp_path / "some-other-dataset"))
    s = settings.build(dataset=make_descriptor_file())
    assert s.paths.data_root == dataset_dir  # its own ${TESTDS_DATA_ROOT}, untouched


def test_settings_do_not_touch_the_disk(make_descriptor_file, env_paths, tmp_path):
    """`geovap status --dataset other` has to work for a dataset whose drive is not mounted, so
    building `Settings` must not open a single file. Adapters are constructed on first use."""
    s = settings.build(dataset=make_descriptor_file(), data_root=tmp_path / "not-mounted")
    assert s.name == "testds"
    assert s.sensor.pano_w == 100  # descriptor values are available with no data present
    with pytest.raises(Exception):
        s.poses.load()  # only NOW does it try to read


def test_get_configures_from_the_environment_on_first_use(make_descriptor_file, env_paths, monkeypatch):
    """`python -m geovap.stages.colour.colorize` must work standalone, with no driver having called
    `configure()` first."""
    monkeypatch.setenv(settings.ENV_DATASET, str(make_descriptor_file()))
    assert not settings.is_configured()
    s = settings.get()
    assert settings.is_configured() and s.name == "testds"
    assert settings.get() is s  # cached, not rebuilt per call


def test_missing_environment_variable_names_itself(tmp_path, monkeypatch):
    """A descriptor referencing `${GEOVAP_DATA}` with nothing set must fail loudly at load, not
    resolve to a literal '${GEOVAP_DATA}' directory that then appears to be empty."""
    d = tmp_path / "x.toml"
    d.write_text(
        '[paths]\ndata_root = "${GEOVAP_NOT_SET_ANYWHERE}"\nworkspace = "/w"\npublish = "/p"\n'
        "[crs]\nepsg = 5514\n[sensor]\npano_w=1\npano_h=1\nzb_w=1\nzb_h=1\nr_min=1.0\nr_max=2.0\n"
        "point_spacing=0.1\ncell_size=1.0\ntol_abs=0.1\ntol_rel=0.1\nsplat_k=1.0\nsplat_min_px=1\n"
        "splat_max_px=2\n[tuning]\nscore_r0=1.0\nincidence_max_deg=1.0\ntop_k=1\nmad_cutoff=1.0\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEOVAP_NOT_SET_ANYWHERE", raising=False)
    # No `$GEOVAP_DATA` override in play, so the descriptor's own expansion is what has to fail.
    monkeypatch.delenv(settings.ENV_OVERRIDE_DATA_ROOT, raising=False)
    with pytest.raises(DescriptorError, match="GEOVAP_NOT_SET_ANYWHERE"):
        settings.build(dataset=d)


def test_env_round_trips_into_a_child_process(make_descriptor_file, env_paths, monkeypatch):
    """`runtime.procs` hands a stage subprocess `Settings.env()`; rebuilding from exactly that
    environment must give the same dataset, or a stage could silently process a different one."""
    parent = settings.build(dataset=make_descriptor_file(), poses="corrected")
    # monkeypatch, not os.environ.update: these are the same variables the session fixture sets for
    # the rest of the suite, and popping them here would break every later test that resolves a
    # descriptor.
    for key, value in parent.env().items():
        monkeypatch.setenv(key, value)
    child = settings.build()
    assert (child.name, child.paths, child.pose_table) == (parent.name, parent.paths, parent.pose_table)
