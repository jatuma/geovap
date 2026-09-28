"""The package imports, and a dataset resolves.

This used to assert `mapping.config.PANO_W == 8000` -- a property of Dražkov's camera, asserted as
if it were a property of the software. It now asserts that the *active* dataset's descriptor
resolves and describes a panorama, whatever size that is.
"""
from __future__ import annotations


def test_a_dataset_resolves_and_describes_itself():
    from geovap.runtime import settings

    s = settings.get()
    assert s.name
    assert s.sensor.pano_w > 0 and s.sensor.pano_h > 0
    assert s.sensor.pano == (s.sensor.pano_w, s.sensor.pano_h)
    assert s.crs.epsg > 0


def test_every_layer_imports():
    """The four distributions share one namespace package; a broken `__init__` shows up here."""
    import geovap.domain.model.geometry  # noqa: F401
    import geovap.io.registry  # noqa: F401
    import geovap.runtime.settings  # noqa: F401
    import geovap.stages.base.spec  # noqa: F401

    from geovap.stages.base.discovery import discover

    assert discover().order(), "no stages are installed"
