"""Reuse the synthetic dataset fixtures from `tests/io`: the runtime layer must be testable with no
real data at all, which is the same property those fixtures were built to prove."""
from __future__ import annotations

import pytest

from tests.io.conftest import (  # noqa: F401  (re-exported as fixtures)
    dataset_dir,
    env_paths,
    make_descriptor_file,
)


@pytest.fixture(autouse=True)
def _no_leaked_settings():
    """Every test gets a clean process-wide `Settings`; a leaked one from a previous test would make
    failures depend on test order."""
    from geovap.runtime import settings

    settings.reset()
    yield
    settings.reset()
