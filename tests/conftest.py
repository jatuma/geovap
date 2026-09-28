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


@pytest.fixture(scope="session", autouse=True)
def _dataset_env():
    """Ensure the three dataset roots are set for the whole session.

    Without this, importing any module that calls `settings.get()` raises `DescriptorError:
    ${GEOVAP_DATA} is not set` -- correct behaviour for a product, unhelpful for a unit test that
    only wanted a filename template. Values already exported by the developer (pointing at the real
    dataset) are left alone, so `-m slow` still runs against it.
    """
    tmp = tempfile.mkdtemp(prefix="geovap-test-roots-")
    added = [k for k in _ENV_DEFAULTS if not os.environ.get(k)]
    for k in added:
        os.environ[k] = tmp
    yield
    for k in added:
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
