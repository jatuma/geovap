"""`Descriptor.load`: env-var expansion, its failure mode, and overrides precedence."""
from __future__ import annotations

from pathlib import Path

import pytest

from geovap.domain.model.crs import Crs
from geovap.domain.model.sensor import Sensor, Tuning
from geovap.io.descriptor import Descriptor, DescriptorError


def test_env_var_expansion(make_descriptor_file, env_paths, dataset_dir):
    file = make_descriptor_file()
    d = Descriptor.load(file)
    assert d.data_root == Path(env_paths["TESTDS_DATA_ROOT"])
    assert d.workspace == Path(env_paths["TESTDS_WORKSPACE"])
    assert d.publish == Path(env_paths["TESTDS_PUBLISH"])
    assert d.name == "testds"
    assert isinstance(d.crs, Crs) and d.crs.epsg == 5514
    assert isinstance(d.sensor, Sensor)
    assert isinstance(d.tuning, Tuning)


def test_unset_env_var_raises_naming_var_and_file(make_descriptor_file, monkeypatch, tmp_path):
    # deliberately do not set TESTDS_DATA_ROOT
    monkeypatch.delenv("TESTDS_DATA_ROOT", raising=False)
    monkeypatch.setenv("TESTDS_WORKSPACE", str(tmp_path / "w"))
    monkeypatch.setenv("TESTDS_PUBLISH", str(tmp_path / "p"))
    file = make_descriptor_file()
    with pytest.raises(DescriptorError) as exc_info:
        Descriptor.load(file)
    message = str(exc_info.value)
    assert "TESTDS_DATA_ROOT" in message
    assert str(file) in message


def test_overrides_win_over_env(make_descriptor_file, env_paths, tmp_path):
    file = make_descriptor_file()
    override_root = tmp_path / "overridden-root"
    d = Descriptor.load(file, overrides={"data_root": str(override_root)})
    assert d.data_root == override_root
    # the other two paths still come from the environment
    assert d.workspace == Path(env_paths["TESTDS_WORKSPACE"])


def test_overrides_win_even_when_env_unset(make_descriptor_file, monkeypatch, tmp_path):
    monkeypatch.delenv("TESTDS_DATA_ROOT", raising=False)
    monkeypatch.setenv("TESTDS_WORKSPACE", str(tmp_path / "w"))
    monkeypatch.setenv("TESTDS_PUBLISH", str(tmp_path / "p"))
    file = make_descriptor_file()
    override_root = tmp_path / "cli-given-root"
    d = Descriptor.load(file, overrides={"data_root": str(override_root)})
    assert d.data_root == override_root
