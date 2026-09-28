"""Shared fixtures.

`has_data` used to mean "does Dražkov's export.csv exist at its hardcoded path". It now means "does
the active descriptor's data_root resolve", so the suite skips cleanly on a checkout with nothing
mounted -- and, more importantly, so it can be pointed at a different dataset.
"""
import os
import tempfile

import pytest

_ENV_DEFAULTS = {
    # A descriptor is resolved for every test, so tests that only need naming or sensor values
    # (`tiles.out_name`, `sensor.pano_w`) do not depend on a mounted drive. Adapters are lazy, so
    # pointing these at a temporary directory costs nothing until something actually reads a file.
    "GEOVAP_DATA": None,
    "GEOVAP_WORKSPACE": None,
    "GEOVAP_PUBLISH": None,
}


# Ensure the three dataset roots are set for the whole session -- at IMPORT time, not inside a
# fixture. A few `mapping/seg` modules still resolve `geovap.runtime.settings.get()` at their own
# import time (a pre-existing anti-pattern, e.g. `mapping.seg.areas.segds_dir()`), and pytest imports
# every test module during collection, before any fixture -- even an autouse, session-scoped one --
# has run. Without this, importing such a module during collection raises `DescriptorError:
# ${GEOVAP_DATA} is not set` -- correct behaviour for a product, unhelpful for a unit test that only
# wanted a filename template. Values already exported by the developer (pointing at the real
# dataset) are left alone, so `-m slow` still runs against it.
_tmp_dataset_root = tempfile.mkdtemp(prefix="geovap-test-roots-")
_added_dataset_env = [k for k in _ENV_DEFAULTS if not os.environ.get(k)]
for _k in _added_dataset_env:
    os.environ[_k] = _tmp_dataset_root


@pytest.fixture(scope="session", autouse=True)
def _dataset_env():
    """The env defaulting itself happens at import time above; this fixture only provides the
    session-scoped teardown and a name for `has_data` to depend on for ordering."""
    yield
    for k in _added_dataset_env:
        os.environ.pop(k, None)


@pytest.fixture(scope="session")
def has_data(_dataset_env) -> bool:
    from geovap.runtime import settings

    src = settings.get().poses.source_file()
    return bool(src and src.exists())


@pytest.fixture(scope="session")
def poses(has_data):
    if not has_data:
        pytest.skip("dataset not available")
    from geovap.runtime import pose_tables

    return pose_tables.load()
