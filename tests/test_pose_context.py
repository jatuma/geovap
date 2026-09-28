"""WIRE: pose-source-aware output paths and registration wiring (config.source_dir,
products.frames_dir/load_products, cloud_store.open_store).

Synthetic poses only -- no cache dependency (no export.csv, no store, no frame products on disk).
Proves the contract that matters for the whole WIRE change: the default ("export", GEOVAP_POSES
unset) path is byte-identical to what every consumer already used, and a corrected table's outputs
land somewhere else, keyed by its hash, with its registration wired through.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mapping import cloud_store, config, products
from mapping.poses import Poses
from mapping.rig import IDENTITY


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


# ------------------------------------------------------------------------------------- config.source_dir
def test_source_dir_export_is_identity():
    base = Path("/fake/cache/out/dataset")
    assert config.source_dir(base, EXPORT) == base


def test_source_dir_corrected_hash_suffix():
    base = Path("/fake/cache/segds")
    out = config.source_dir(base, CORRECTED)
    assert out != base
    assert out.parent == base.parent
    assert out.name == f"segds_{CORRECTED.hash()[:6]}"


def test_source_dir_corrected_hash_matches_frames_dir_scheme():
    """source_dir and products.frames_dir must key on the same 6 hex chars, so a corrected run's
    segds tree and its frame products land under matching-looking siblings."""
    base = Path("/fake/cache/out/dataset")
    assert config.source_dir(base, CORRECTED).name.endswith(CORRECTED.hash()[:6])
    assert products.frames_dir(CORRECTED, root=Path("/fake/cache/frames")).name == CORRECTED.hash()[:6]


# ------------------------------------------------------------------------------------- products.frames_dir
def test_frames_dir_export_unchanged():
    root = Path("/fake/cache/frames")
    assert products.frames_dir(EXPORT, root=root) == root


def test_frames_dir_corrected_is_hash_subdir():
    root = Path("/fake/cache/frames")
    out = products.frames_dir(CORRECTED, root=root)
    assert out == root / CORRECTED.hash()[:6]
    assert out != root


# ------------------------------------------------------------------------------------- cloud_store.open_store
class _RecordingCloudStore:
    """Stand-in for CloudStore that records its constructor args instead of touching disk."""

    last_kwargs: dict = {}

    def __init__(self, root, registration=None):
        _RecordingCloudStore.last_kwargs = {"root": root, "registration": registration}


@pytest.fixture(autouse=True)
def _patch_cloud_store(monkeypatch):
    monkeypatch.setattr(cloud_store, "CloudStore", _RecordingCloudStore)
    yield


def test_open_store_none_poses_is_unregistered():
    store = cloud_store.open_store(None, root=Path("/fake/store"))
    assert isinstance(store, _RecordingCloudStore)
    assert store.last_kwargs["registration"] is None
    assert store.last_kwargs["root"] == Path("/fake/store")


def test_open_store_export_poses_is_unregistered():
    cloud_store.open_store(EXPORT, root=Path("/fake/store"))
    assert _RecordingCloudStore.last_kwargs["registration"] is None


def test_open_store_corrected_poses_passes_registration():
    cloud_store.open_store(CORRECTED, root=Path("/fake/store"))
    assert _RecordingCloudStore.last_kwargs["registration"] == CORRECTED.registration
    assert _RecordingCloudStore.last_kwargs["root"] == Path("/fake/store")


# ------------------------------------------------------------------------------------- products.load_products
class _RecordingFrameProducts:
    last_kwargs: dict = {}

    @classmethod
    def load(cls, frame, rig=None, root=config.FRAMES_DIR, allow_stale=False, poses=None):
        cls.last_kwargs = {"frame": frame, "rig": rig, "root": root, "allow_stale": allow_stale, "poses": poses}
        return "sentinel"


def test_load_products_threads_root_and_poses(monkeypatch):
    monkeypatch.setattr(products, "FrameProducts", _RecordingFrameProducts)
    res = products.load_products(7, CORRECTED)
    assert res == "sentinel"
    kw = _RecordingFrameProducts.last_kwargs
    assert kw["frame"] == 7
    assert kw["rig"] is IDENTITY
    assert kw["root"] == products.frames_dir(CORRECTED)
    assert kw["poses"] is CORRECTED


def test_load_products_export_matches_bare_frames_dir(monkeypatch):
    monkeypatch.setattr(products, "FrameProducts", _RecordingFrameProducts)
    products.load_products(3, EXPORT)
    assert _RecordingFrameProducts.last_kwargs["root"] == config.FRAMES_DIR
