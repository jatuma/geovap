"""C2: `mapping.merge` on a synthetic pair of inputs (tw45 LAS + seg-label npys + a cluster LAZ) against
a tiny synthetic `CloudStore`. No dependency on the real cache."""
from __future__ import annotations

import json

import laspy
import numpy as np
import pytest

from mapping import cloud_store, las_out
from mapping.cloud_store import SCALE, CloudStore, PassRegistration, build_tile
from mapping.merge import MergeInputs, check_same_points, merge_tile, registered_xyz_int
from geovap.domain.model.poses import Poses

NAME = "037"
N = 30


def _write_source_laz(path, seed=0) -> None:
    rng = np.random.default_rng(seed)
    header = laspy.LasHeader(version="1.4", point_format=7)
    header.scales = [SCALE, SCALE, SCALE]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)
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
def poses():
    return Poses(
        filename=np.array(["a.jpg"], dtype=object), t=np.array([0.0]), origin=np.zeros((1, 3)),
        roll=np.zeros(1), pitch=np.zeros(1), yaw=np.zeros(1), pass_id=np.array([0], dtype=np.int32), speed=np.zeros(1),
    )


@pytest.fixture
def store_root(tmp_path, monkeypatch):
    # `TileInfo.dir` resolves against the module-global `cloud_store.STORE_DIR`, so a synthetic store
    # must patch that global to its own tmp root to avoid `TileData` reading the real cache's tile "037".
    root = tmp_path / "store"
    monkeypatch.setattr(cloud_store, "STORE_DIR", root)
    src = tmp_path / "ID3432_000037_JTSK.laz"
    _write_source_laz(src)
    meta = build_tile(str(src), str(root / "tiles" / NAME))
    meta["row_offset"] = 0
    meta["polygon"] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    (root / "tiles.json").write_text(json.dumps({"total": meta["n"], "cell_size": 4.0, "tiles": [meta]}))
    return root, src


@pytest.fixture
def registration(poses):
    transforms = {"passes": {"0": {"centre": [600_050.0, -1_049_950.0], "t": [0.20, -0.10, 0.05], "yaw_deg": 0.0}}}
    return PassRegistration(transforms, poses=poses)


def _write_tw45(store_root, out_dir, poses, seed=1) -> None:
    store = CloudStore(root=store_root)  # unregistered: tw45 is always a bit-exact-xyz copy
    td = store.tile(NAME)
    n = len(td)
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, (n, 3)).astype(np.uint8)
    extras = {name: np.zeros(n, dt) for name, dt, _ in las_out.EXTRA_DIMS}
    extras["n_views"] = rng.integers(0, 4, n).astype(np.uint8)
    extras["ref_r"], extras["ref_g"], extras["ref_b"] = (np.asarray(td.rgb)[:, i] for i in range(3))
    extras["dE00_med"] = rng.uniform(0, 5, n).astype(np.float32)
    prov = {"poses_source": poses.source, "poses_hash": poses.hash(), "git": None}
    out_dir.mkdir(parents=True, exist_ok=True)
    las_out.write_tile(td, out_dir / f"ID3432_000{NAME}_colored.laz", rgb, extras, prov)


def _write_seg_labels(out_dir, n, seed=2) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = rng.integers(0, 14, n).astype(np.uint8)
    label[::5] = 255
    np.save(out_dir / f"{NAME}.npy", label)
    np.save(out_dir / f"{NAME}_conf.npy", rng.integers(0, 256, n).astype(np.uint8))
    np.save(out_dir / f"{NAME}_nviews.npy", rng.integers(0, 5, n).astype(np.uint8))
    (out_dir / f"{NAME}_meta.json").write_text(json.dumps({"n": n}))


def _write_clusters(store_root, src_path, out_dir, seed=3) -> None:
    """Same point order as the vendor LAZ (source order), rgb = palette u16, cluster_id/hag/obj_class."""
    src = laspy.read(str(src_path))
    n = len(src.points)
    rng = np.random.default_rng(seed)
    header = laspy.LasHeader(version="1.4", point_format=7)
    header.scales = [SCALE, SCALE, SCALE]
    header.offsets = [0.0, 0.0, 0.0]
    header.add_extra_dim(laspy.ExtraBytesParams(name="cluster_id", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="hag", type=np.float32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="obj_class", type=np.uint8))
    las = laspy.LasData(header)
    las.X, las.Y, las.Z = src.X, src.Y, src.Z
    las.red = rng.integers(0, 65535, n).astype(np.uint16)
    las.green = rng.integers(0, 65535, n).astype(np.uint16)
    las.blue = rng.integers(0, 65535, n).astype(np.uint16)
    cluster_id = rng.integers(-1, 5, n).astype(np.int32)
    las.cluster_id = cluster_id
    las.hag = rng.uniform(0, 10, n).astype(np.float32)
    las.obj_class = rng.integers(0, 4, n).astype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    las.write(str(out_dir / f"objects_t000{NAME}.laz"))


def _make_inputs(tmp_path, store_root, src_path, poses, registration=None, allow_mixed_poses=False) -> MergeInputs:
    tw45_dir = tmp_path / "tw45" / "tiles"
    seg_dir = tmp_path / "seg_eomt" / "labels"
    clusters_dir = tmp_path / "clusters" / "src"
    n = len(CloudStore(root=store_root).tile(NAME))
    _write_tw45(store_root, tw45_dir, poses)
    _write_seg_labels(seg_dir, n)
    _write_clusters(store_root, src_path, clusters_dir)
    return MergeInputs(poses=poses, tw45_tiles=tw45_dir, seg_labels=seg_dir, clusters_src=clusters_dir,
                        out_dir=tmp_path / "out", rgb_fallback=True, allow_mixed_poses=allow_mixed_poses)


# --------------------------------------------------------------------------------------- check_same_points
def test_check_same_points_ok():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    check_same_points(a, a.copy())


def test_check_same_points_length_mismatch_raises():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2, 3]])
    with pytest.raises(ValueError):
        check_same_points(a, b)


def test_check_same_points_value_mismatch_raises():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = a.copy()
    b[0, 0] += 1
    with pytest.raises(ValueError, match="1/2 points"):
        check_same_points(a, b, name="thing")


# --------------------------------------------------------------------------------------- registered_xyz_int
def test_registered_xyz_int_matches_xyz_m(store_root, registration):
    root, _ = store_root
    store = CloudStore(root=root, registration=registration)
    td = store.tile(NAME)
    out = registered_xyz_int(td)
    assert out.dtype == np.int32
    assert np.array_equal(out, np.round(td.xyz_m() / SCALE).astype(np.int32))


# --------------------------------------------------------------------------------------- merge_tile
def test_merge_tile_writes_both_products(tmp_path, store_root, poses, registration):
    root, src_path = store_root
    store = CloudStore(root=root, registration=registration)
    inp = _make_inputs(tmp_path, root, src_path, poses)

    meta = merge_tile(NAME, store, inp)

    tiles_out = inp.out_dir / "tiles" / f"ID3432_000{NAME}.laz"
    objects_out = inp.out_dir / "objects" / f"ID3432_000{NAME}.laz"
    assert tiles_out.exists() and objects_out.exists()
    assert meta["verify"]["tile"]["xyz_exact"] is True
    assert meta["verify"]["objects"]["xyz_exact"] is True
    assert meta["n"] == N
    assert (inp.out_dir / "tiles" / f"{NAME}_meta.json").exists()

    las = laspy.read(str(tiles_out))
    # classification 255 for the rows the synthetic seg labels marked unlabelled
    label = np.load(inp.seg_labels / f"{NAME}.npy")
    td_unreg = CloudStore(root=root).tile(NAME)
    inv = np.asarray(td_unreg.orig_index)
    cls_store = np.asarray(las.classification)[inv]  # source order -> store order
    assert np.array_equal(cls_store, label)

    obj_las = laspy.read(str(objects_out))
    assert "cluster_id" in obj_las.point_format.dimension_names
    assert "obj_class" in obj_las.point_format.dimension_names
    assert "cluster_id" in las.point_format.dimension_names  # tiles product also carries cluster_id (CONS_EXTRA_DIMS)
    assert "hag" in las.point_format.dimension_names


def test_merge_tile_rgb_falls_back_where_n_views_zero(tmp_path, store_root, poses, registration):
    root, src_path = store_root
    store = CloudStore(root=root, registration=registration)
    inp = _make_inputs(tmp_path, root, src_path, poses)
    meta = merge_tile(NAME, store, inp)

    tiles_out = inp.out_dir / "tiles" / f"ID3432_000{NAME}.laz"
    las = laspy.read(str(tiles_out))
    td_unreg = CloudStore(root=root).tile(NAME)
    inv = np.asarray(td_unreg.orig_index)
    n_views_store = np.asarray(las.n_views)[inv]
    ref_store = np.stack([np.asarray(las.ref_r), np.asarray(las.ref_g), np.asarray(las.ref_b)], 1)[inv]
    rgb_store = (np.stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)], 1) >> 8)[inv]
    no_col = n_views_store == 0
    assert np.array_equal(rgb_store[no_col], ref_store[no_col])


def test_merge_tile_check_same_points_failure_aborts(tmp_path, store_root, poses, registration):
    root, src_path = store_root
    store = CloudStore(root=root, registration=registration)
    inp = _make_inputs(tmp_path, root, src_path, poses)
    # corrupt the cluster file's first point so it no longer matches the store's source points
    p = inp.clusters_src / f"objects_t000{NAME}.laz"
    las = laspy.read(str(p))
    las.X = np.asarray(las.X).copy()
    las.X[0] += 100_000
    las.write(str(p))

    with pytest.raises(ValueError, match="clusters"):
        merge_tile(NAME, store, inp)


def test_merge_tile_poses_mismatch_aborts_unless_allowed(tmp_path, store_root, poses, registration):
    root, src_path = store_root
    store = CloudStore(root=root, registration=registration)
    inp = _make_inputs(tmp_path, root, src_path, poses)
    # rewrite tw45 with a different poses_hash
    p = inp.tw45_tiles / f"ID3432_000{NAME}_colored.laz"
    las = laspy.read(str(p))
    vlrs = [v for v in las.header.vlrs if v.user_id != las_out.PROVENANCE_USER_ID]
    las.header.vlrs.clear()
    for v in vlrs:
        las.header.vlrs.append(v)
    las.header.vlrs.append(laspy.vlrs.VLR(user_id=las_out.PROVENANCE_USER_ID, record_id=las_out.PROVENANCE_RECORD_ID,
                                          description="x", record_data=json.dumps({"poses_source": "export", "poses_hash": "deadbeef"}).encode()))
    las.write(str(p))

    with pytest.raises(ValueError, match="poses_hash"):
        merge_tile(NAME, store, inp)

    inp.allow_mixed_poses = True
    meta = merge_tile(NAME, store, inp)
    assert any("poses_hash" in w for w in meta["warnings"])
