"""C1: `las_out.write_tile`'s `xyz=` override, `product_rgb` uint16 passthrough and
`verify(xyz_mode=...)`. Fully synthetic (a tiny hand-built LAZ + `cloud_store.build_tile`); no
dependency on the real store."""
from __future__ import annotations

import json

import laspy
import numpy as np
import pytest

from mapping import cloud_store, las_out
from mapping.cloud_store import SCALE, PassRegistration, TileData, TileInfo, build_tile
from mapping.poses import Poses


N = 40


def _write_source_laz(path, seed=0) -> None:
    rng = np.random.default_rng(seed)
    header = laspy.LasHeader(version="1.4", point_format=7)
    header.scales = [SCALE, SCALE, SCALE]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)
    # keep coordinates within a small, positive tile so int32 store coords are unambiguous
    las.X = rng.integers(600_000_000, 600_100_000, N, dtype=np.int32)
    las.Y = rng.integers(-1_050_000_000, -1_049_900_000, N, dtype=np.int32)
    las.Z = rng.integers(400_000, 401_000, N, dtype=np.int32)
    las.classification = rng.integers(0, 15, N).astype(np.uint8)
    las.intensity = rng.integers(0, 65535, N).astype(np.uint16)
    las.red = (rng.integers(0, 256, N).astype(np.uint16)) * 256
    las.green = (rng.integers(0, 256, N).astype(np.uint16)) * 256
    las.blue = (rng.integers(0, 256, N).astype(np.uint16)) * 256
    las.gps_time = np.sort(rng.uniform(0, 10, N))
    las.point_source_id = np.ones(N, np.uint16)
    las.write(str(path))


@pytest.fixture
def td(tmp_path, monkeypatch):
    # `TileInfo.dir` resolves against the module-global `cloud_store.STORE_DIR` (not a per-instance
    # root), so a synthetic tile must patch that global to its own tmp store to avoid reading the real
    # cache's tile "037".
    store_root = tmp_path / "store"
    monkeypatch.setattr(cloud_store, "STORE_DIR", store_root)
    src = tmp_path / "ID3432_000037_JTSK.laz"
    _write_source_laz(src)
    meta = build_tile(str(src), str(store_root / "tiles" / "037"))
    meta["row_offset"] = 0
    meta.pop("classes", None)
    meta["polygon"] = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    info = TileInfo(**meta)
    return TileData(info)


@pytest.fixture
def poses_and_registration():
    poses = Poses(
        filename=np.array(["a.jpg"], dtype=object), t=np.array([0.0]), origin=np.zeros((1, 3)),
        roll=np.zeros(1), pitch=np.zeros(1), yaw=np.zeros(1), pass_id=np.array([0], dtype=np.int32), speed=np.zeros(1),
    )
    shift = {"centre": [600_050.0, -1_049_950.0], "t": [0.20, -0.10, 0.05], "yaw_deg": 0.0}
    transforms = {"passes": {"0": shift}}
    reg = PassRegistration(transforms, poses=poses)
    return poses, reg, shift


# --------------------------------------------------------------------------------------- xyz override
def test_write_tile_default_xyz_is_bit_exact(td, tmp_path):
    n = len(td)
    rgb = np.zeros((n, 3), np.uint8)
    extras = {name: np.zeros(n, dt) for name, dt, _ in las_out.EXTRA_DIMS}
    out = las_out.write_tile(td, tmp_path / "out.laz", rgb, extras, {"poses_hash": "abc"})
    v = las_out.verify(out, td)
    assert v["xyz_exact"] is True
    assert v["provenance"] is True


def test_write_tile_xyz_override_writes_given_coordinates(td, tmp_path):
    n = len(td)
    rgb = np.zeros((n, 3), np.uint8)
    extras = {name: np.zeros(n, dt) for name, dt, _ in las_out.EXTRA_DIMS}
    xyz_override = (np.asarray(td.xyz) + np.array([1000, -2000, 500], np.int32)).astype(np.int32)  # +1 m, -2 m, +0.5 m
    out = las_out.write_tile(td, tmp_path / "out.laz", rgb, extras, {}, xyz=xyz_override)
    las = laspy.read(str(out))
    inv = np.asarray(td.orig_index)
    back = np.stack([np.asarray(las.X), np.asarray(las.Y), np.asarray(las.Z)], 1)
    store_order = back[inv]  # source order -> store order
    assert np.array_equal(store_order, xyz_override)
    # bit-exact verify must now fail (coordinates were deliberately shifted)
    assert las_out.verify(out, td)["xyz_exact"] is False


def test_product_rgb_uint16_passes_through_unscaled(td, tmp_path):
    n = len(td)
    rgb16 = np.zeros((n, 3), np.uint16)
    rgb16[:, 0] = np.arange(n) % 65535
    extras = {"cluster_id": np.full(n, -1, np.int32), "obj_class": np.zeros(n, np.uint8)}
    out = las_out.write_tile(td, tmp_path / "objects.laz", rgb16, extras, {}, extra_dims=las_out.OBJ_EXTRA_DIMS)
    las = laspy.read(str(out))
    inv = np.asarray(td.orig_index)
    red_store = np.asarray(las.red)[inv]  # source order -> store order
    assert np.array_equal(red_store, rgb16[:, 0])  # NOT *256


def test_product_rgb_uint8_is_scaled_to_16bit(td, tmp_path):
    n = len(td)
    rgb8 = np.zeros((n, 3), np.uint8)
    rgb8[:, 0] = np.arange(n) % 256
    extras = {name: np.zeros(n, dt) for name, dt, _ in las_out.EXTRA_DIMS}
    out = las_out.write_tile(td, tmp_path / "out.laz", rgb8, extras, {})
    las = laspy.read(str(out))
    inv = np.asarray(td.orig_index)
    red_store = np.asarray(las.red)[inv]  # source order -> store order
    assert np.array_equal(red_store, rgb8[:, 0].astype(np.uint16) * 256)


# --------------------------------------------------------------------------------------- verify registered
def test_verify_registered_reports_shift_mm(td, tmp_path, poses_and_registration):
    _, reg, shift = poses_and_registration
    td_reg = TileData(td.info, registration=reg)
    n = len(td_reg)
    rgb = np.zeros((n, 3), np.uint8)
    extras = {name: np.zeros(n, dt) for name, dt, _ in las_out.EXTRA_DIMS}
    xyz_registered = np.round(td_reg.xyz_m() / SCALE).astype(np.int32)
    out = las_out.write_tile(td_reg, tmp_path / "out.laz", rgb, extras, {}, xyz=xyz_registered)

    v_exact = las_out.verify(out, td_reg, xyz_mode="exact")
    assert v_exact["xyz_exact"] is False  # registered points differ from the raw source

    v_reg = las_out.verify(out, td_reg, xyz_mode="registered")
    assert v_reg["xyz_exact"] is True
    assert v_reg["n_moved"] == 0
    expected_shift_mm = 1000.0 * (shift["t"][0] ** 2 + shift["t"][1] ** 2 + shift["t"][2] ** 2) ** 0.5
    assert v_reg["shift_mm"]["max"] == pytest.approx(0.0, abs=1.0)  # xyz written == td_reg.xyz_m(), so 0 residual
    assert v_reg["shift_mm"]["p50"] >= 0.0

    # sanity: the registered coordinates really did move relative to the unregistered tile by ~|t|
    unreg = np.asarray(td.xyz).astype(np.int64)
    reg_int = xyz_registered.astype(np.int64)
    moved_mm = np.sqrt(((reg_int - unreg) ** 2).sum(1).astype(np.float64)) * (SCALE * 1000.0)
    assert moved_mm.max() == pytest.approx(expected_shift_mm, rel=0.05)


def test_verify_registered_detects_unwritten_registration(td, tmp_path, poses_and_registration):
    """Writing the *unregistered* (bit-exact) xyz but verifying in "registered" mode must flag the shift."""
    _, reg, shift = poses_and_registration
    td_reg = TileData(td.info, registration=reg)
    n = len(td_reg)
    rgb = np.zeros((n, 3), np.uint8)
    extras = {name: np.zeros(n, dt) for name, dt, _ in las_out.EXTRA_DIMS}
    out = las_out.write_tile(td_reg, tmp_path / "out.laz", rgb, extras, {})  # no xyz= -> bit-exact source copy

    v_reg = las_out.verify(out, td_reg, xyz_mode="registered", tol_mm=0.01)
    assert v_reg["xyz_exact"] is False
    assert v_reg["n_moved"] == n
    assert v_reg["shift_mm"]["max"] > 50.0  # the pass shift is ~0.2 m
