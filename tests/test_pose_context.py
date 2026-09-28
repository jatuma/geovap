"""WIRE: pose-source-aware output paths and registration wiring (Workspace.source_dir,
geovap.stages.prepare.products.load_products, geovap.runtime.store.open_store).

Synthetic poses only -- no cache dependency (no export.csv, no store, no frame products on disk).
Proves the contract that matters for the whole WIRE change: the default ("export", GEOVAP_POSES
unset) path is byte-identical to what every consumer already used, and a corrected table's outputs
land somewhere else, keyed by its hash, with its registration wired through.

`Workspace.frames_dir` (the function `products.frames_dir` used to be, before the `prepare` stage
group migration) has its own thorough coverage in `tests/runtime/test_workspace.py`; this file only
re-derives it here to prove `products.load_products` threads it through correctly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geovap.runtime import settings
from geovap.runtime import store as store_mod
from geovap.stages.prepare import products
from geovap.domain.model.poses import Poses
from geovap.domain.model.rig import IDENTITY
from geovap.runtime.workspace import Workspace


def _synthetic_poses(source: str, registration: Path | None = None) -> Poses:
    """A minimal, self-contained Poses (one frame) -- no CSV/cache reads."""
    return Poses(
        filename=np.array(["f0000.jpg"], dtype=object),
        t=np.array([0.0]),
        origin=np.array([[0.0, 0.0, 0.0]]),
        roll=np.array([0.0]),
        pitch=np.array([0.0]),
        yaw=np.array([0.0]),
        pass_id=np.array([0], dtype=np.int32),
        speed=np.array([0.0]),
        source=source,
        registration=registration,
    )


EXPORT = _synthetic_poses("export")
CORRECTED = _synthetic_poses("corrected", registration=Path("/fake/cache/out/pass_reg/pass_transforms.json"))


def _fake_workspace(root: Path) -> Workspace:
    return Workspace(root=root, publish=Path("/fake/pub"), baseline=Path("/fake/baseline"), dataset="fake")


class _FakeSettings:
    """Just the `.workspace` attribute `load_products` needs -- typed structurally on purpose."""

    def __init__(self, root: Path):
        self.workspace = _fake_workspace(root)


# ------------------------------------------------------------------------------------- config.source_dir
def test_source_dir_export_is_identity():
    base = Path("/fake/cache/out/dataset")
    assert settings.get().workspace.source_dir(base, EXPORT) == base


def test_source_dir_corrected_hash_suffix():
    base = Path("/fake/cache/segds")
    out = settings.get().workspace.source_dir(base, CORRECTED)
    assert out != base
    assert out.parent == base.parent
    assert out.name == f"segds_{CORRECTED.hash()[:6]}"


def test_source_dir_corrected_hash_matches_frames_dir_scheme():
    """source_dir and Workspace.frames_dir must key on the same 6 hex chars, so a corrected run's
    segds tree and its frame products land under matching-looking siblings."""
    base = Path("/fake/cache/out/dataset")
    assert settings.get().workspace.source_dir(base, CORRECTED).name.endswith(CORRECTED.hash()[:6])
    ws = _fake_workspace(Path("/fake/cache"))
    assert ws.frames_dir(CORRECTED).name == CORRECTED.hash()[:6]


# ------------------------------------------------------------------------------------- store.open_store
class _RecordingCloudStore:
    """Stand-in for CloudStore that records its constructor args instead of touching disk."""

    last_kwargs: dict = {}

    def __init__(self, root, registration=None):
        _RecordingCloudStore.last_kwargs = {"root": root, "registration": registration}


@pytest.fixture(autouse=True)
def _patch_cloud_store(monkeypatch):
    monkeypatch.setattr(store_mod, "CloudStore", _RecordingCloudStore)
    yield


def test_open_store_none_poses_is_unregistered():
    store = store_mod.open_store(_FakeSettings(Path("/fake")), None)
    assert isinstance(store, _RecordingCloudStore)
    assert store.last_kwargs["registration"] is None
    assert store.last_kwargs["root"] == Path("/fake/store")


def test_open_store_export_poses_is_unregistered():
    store_mod.open_store(_FakeSettings(Path("/fake")), EXPORT)
    assert _RecordingCloudStore.last_kwargs["registration"] is None


def test_open_store_corrected_poses_passes_registration():
    store_mod.open_store(_FakeSettings(Path("/fake")), CORRECTED)
    assert _RecordingCloudStore.last_kwargs["registration"] == CORRECTED.registration
    assert _RecordingCloudStore.last_kwargs["root"] == Path("/fake/store")


# ------------------------------------------------------------------------------------- products.load_products
class _RecordingFrameProducts:
    last_kwargs: dict = {}

    @classmethod
    def load(cls, frame, rig=None, *, root=None, allow_stale=False, poses=None):
        cls.last_kwargs = {"frame": frame, "rig": rig, "root": root, "allow_stale": allow_stale, "poses": poses}
        return "sentinel"


def test_load_products_threads_root_and_poses(monkeypatch):
    monkeypatch.setattr(products, "FrameProducts", _RecordingFrameProducts)
    fake_root = Path("/fake/cache")
    res = products.load_products(7, CORRECTED, s=_FakeSettings(fake_root))
    assert res == "sentinel"
    kw = _RecordingFrameProducts.last_kwargs
    assert kw["frame"] == 7
    assert kw["rig"] is IDENTITY
    assert kw["root"] == _fake_workspace(fake_root).frames_dir(CORRECTED)
    assert kw["poses"] is CORRECTED


def test_load_products_export_matches_bare_frames_dir(monkeypatch):
    monkeypatch.setattr(products, "FrameProducts", _RecordingFrameProducts)
    fake_root = Path("/fake/cache")
    products.load_products(3, EXPORT, s=_FakeSettings(fake_root))
    assert _RecordingFrameProducts.last_kwargs["root"] == _fake_workspace(fake_root).frames
