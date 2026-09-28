"""S0: pose-table infrastructure. Needs export.csv (fast: no store/frames required).

Also S_assemble (compose S3b/S4/S5b into `poses_corrected`, `mapping/cli/assemble_poses.py`) --
synthetic-data tests only, no cache dependency."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from geovap.domain.model import geometry
from geovap.stages.register import passes as pass_reg
from geovap.stages.register.passes import META_COLS, assemble, overlay_refined, transform_trajectory
from geovap.domain.model.poses import Poses
from geovap.runtime.pose_tables import load as load_poses, read as read_pose_table, write as write_pose_table
from geovap.stages.register.trajectory import CamSensorRig, Trajectory
from geovap.runtime import settings

PANO_W, PANO_H = settings.get().sensor.pano_w, settings.get().sensor.pano_h


@pytest.fixture(scope="module")
def export_poses():
    src = settings.get().poses.source_file()
    if src is None or not src.exists():
        pytest.skip("export.csv not found")
    return load_poses("export")


def test_load_poses_export_unchanged(export_poses):
    p = export_poses
    assert p.source == "export"
    assert p.traj is None
    assert p.registration is None
    assert len(p) == len(p.filename) == len(p.t) == len(p.pass_id)


def test_write_read_round_trip(export_poses, tmp_path: Path):
    p = export_poses
    meta = {
        "status": np.array(["kept"] * len(p), dtype=object),
        "src": np.array(["export"] * len(p), dtype=object),
        "dt_s": np.zeros(len(p)),
    }
    out = write_pose_table(p, meta, {"stage": "test"}, tmp_path / "poses_corrected.csv")
    p2 = read_pose_table(out)

    assert p2.source == "poses_corrected"
    assert list(p2.filename) == list(p.filename)
    np.testing.assert_array_equal(p2.t, p.t)  # t must round-trip exactly
    np.testing.assert_allclose(p2.origin, p.origin, atol=1e-9)
    np.testing.assert_allclose(p2.roll, p.roll, atol=1e-9)
    np.testing.assert_allclose(p2.pitch, p.pitch, atol=1e-9)
    np.testing.assert_allclose(p2.yaw, p.yaw, atol=1e-9)
    np.testing.assert_array_equal(p2.pass_id, p.pass_id)

    prov_path = out.with_suffix(".json")
    assert prov_path.exists()
    import json

    prov = json.loads(prov_path.read_text())
    assert prov["stage"] == "test"
    assert prov["poses_hash"] == p2.hash()
    assert "git" in prov and "pose_source_sha1" in prov


def test_read_pose_table_attaches_registration(export_poses, tmp_path: Path):
    """`read_pose_table` attaches Poses.registration from the sidecar's "registration" entry (the
    shape `assemble_poses.py` writes: {"path": ..., "sha1": ...}); a table without one stays None."""
    p = export_poses
    meta = {"status": np.array(["kept"] * len(p), dtype=object)}
    out = write_pose_table(p, meta, {}, tmp_path / "poses_corrected.csv")

    p_no_reg = read_pose_table(out)
    assert p_no_reg.registration is None

    prov = json.loads(out.with_suffix(".json").read_text())
    transforms_path = tmp_path / "pass_transforms.json"
    transforms_path.write_text("{}")
    prov["registration"] = {"path": str(transforms_path), "sha1": "deadbeef"}
    out.with_suffix(".json").write_text(json.dumps(prov))

    p_reg = read_pose_table(out)
    assert p_reg.registration == transforms_path


def test_load_poses_via_source(export_poses, tmp_path: Path):
    p = export_poses
    meta = {"status": np.array(["kept"] * len(p), dtype=object)}
    out = write_pose_table(p, meta, {}, tmp_path / "poses_corrected.csv")
    p2 = load_poses(str(out))
    assert p2.source == "poses_corrected"
    np.testing.assert_array_equal(p2.t, p.t)


def test_hash_stable(export_poses):
    p1 = load_poses("export")
    p2 = load_poses("export")
    assert p1.hash() == p2.hash() == export_poses.hash()


def test_pass_of_time_matches_stored(export_poses):
    p = export_poses
    computed = p.pass_of_time(p.t)
    np.testing.assert_array_equal(computed, p.pass_id)


def test_pass_of_time_clamps_ends(export_poses):
    p = export_poses
    before = p.pass_of_time(np.array([p.t[0] - 1000.0]))
    after = p.pass_of_time(np.array([p.t[-1] + 1000.0]))
    assert before[0] == p.pass_id[0]
    assert after[0] == p.pass_id[-1]


class _FakeTraj:
    """Covers a single pass entirely and returns a constant, distinctive pose there."""

    def __init__(self, pass_id: int):
        self.pass_id = pass_id
        self.origin_val = np.array([111.0, 222.0, 333.0])
        self.roll_val = 1.5
        self.pitch_val = -2.5
        self.yaw_val = 42.0

    def covers(self, pass_id, t_query):
        t_query = np.atleast_1d(t_query)
        return np.full(len(t_query), pass_id == self.pass_id, dtype=bool)

    def camera_pose(self, t_query, pass_id):
        t_query = np.atleast_1d(t_query)
        k = len(t_query)
        origin = np.tile(self.origin_val, (k, 1))
        roll = np.full(k, self.roll_val)
        pitch = np.full(k, self.pitch_val)
        yaw = np.full(k, self.yaw_val)
        return origin, roll, pitch, yaw


def test_interp_uses_traj_where_covered(export_poses):
    p = export_poses
    covered_pass = int(p.pass_id[0])
    other_pass = int(p.pass_id[p.pass_id != covered_pass][0])

    fake = _FakeTraj(covered_pass)
    p_traj = Poses(
        filename=p.filename,
        t=p.t,
        origin=p.origin,
        roll=p.roll,
        pitch=p.pitch,
        yaw=p.yaw,
        pass_id=p.pass_id,
        speed=p.speed,
        source=p.source,
        traj=fake,
    )

    sel_cov = np.flatnonzero(p.pass_id == covered_pass)[:3]
    sel_other = np.flatnonzero(p.pass_id == other_pass)[:3]

    tq = np.concatenate([p.t[sel_cov], p.t[sel_other]])
    ph = np.concatenate([p.pass_id[sel_cov], p.pass_id[sel_other]])

    origin, roll, pitch, yaw = p_traj.interp(tq, ph)
    n_cov = len(sel_cov)

    np.testing.assert_allclose(origin[:n_cov], np.tile(fake.origin_val, (n_cov, 1)))
    np.testing.assert_allclose(roll[:n_cov], fake.roll_val)
    np.testing.assert_allclose(pitch[:n_cov], fake.pitch_val)
    np.testing.assert_allclose(yaw[:n_cov], fake.yaw_val)

    # uncovered pass falls through to ordinary linear interpolation == plain Poses without traj
    origin_lin, roll_lin, pitch_lin, yaw_lin = p.interp(tq[n_cov:], ph[n_cov:])
    np.testing.assert_allclose(origin[n_cov:], origin_lin)
    np.testing.assert_allclose(roll[n_cov:], roll_lin)
    np.testing.assert_allclose(pitch[n_cov:], pitch_lin)
    np.testing.assert_allclose(yaw[n_cov:], yaw_lin)

    # pose_at(idx, 0.0) unaffected by traj
    o0, r0, pt0, y0 = p_traj.pose_at(sel_cov, 0.0)
    np.testing.assert_array_equal(o0, p.origin[sel_cov])


# =============================================================================== assemble (S_assemble)
def _straight_line_poses(n, t0=0.0, dt=1.0, pass_id=0):
    t = t0 + dt * np.arange(n)
    origin = np.stack([t, 2.0 * t, np.full(n, 20.0)], axis=1)
    return Poses(
        filename=np.array([f"f{i}.jpg" for i in range(n)], dtype=object),
        t=t, origin=origin, roll=np.zeros(n), pitch=np.zeros(n), yaw=np.zeros(n),
        pass_id=np.full(n, pass_id, dtype=np.int32), speed=np.ones(n), source="export",
    )


def _refined_row(base: Poses, i: int, status: str, **overrides) -> dict:
    row = {
        "filename": str(base.filename[i]), "status": status,
        "E": base.origin[i, 0], "N": base.origin[i, 1], "H": base.origin[i, 2],
        "roll": base.roll[i], "pitch": base.pitch[i], "yaw": base.yaw[i], "pass_id": int(base.pass_id[i]),
        "dt_s": 99.0, "dyaw": 99.0, "droll": 99.0, "dpitch": 99.0, "n_edge": 99.0, "rms_before": 99.0, "rms_after": 99.0,
    }
    row.update(overrides)
    return {k: str(v) for k, v in row.items()}


def test_overlay_refined_masks_and_stats():
    base = _straight_line_poses(5)
    rows = [_refined_row(base, i, "kept") for i in range(5)]
    rows[2] = _refined_row(base, 2, "refined", E=110.0, N=210.0, H=25.0, roll=1.0, pitch=2.0, yaw=3.0, pass_id=1, dt_s=0.02, dyaw=0.5, droll=0.1, dpitch=0.2, n_edge=500.0, rms_before=10.0, rms_after=5.0)

    overlaid, meta, stats = overlay_refined(base, rows)

    np.testing.assert_array_equal(meta["refined_mask"], [False, False, True, False, False])
    np.testing.assert_array_equal(meta["src_pass_changed"], [False, False, True, False, False])
    assert stats["n_overlaid"] == 1
    assert stats["n_pass_changed"] == 1
    exp_dpos = np.linalg.norm(np.array([110.0, 210.0, 25.0]) - base.origin[2])
    assert stats["dpos_median_m"] == pytest.approx(exp_dpos)
    assert stats["dyaw_median_deg"] == pytest.approx(3.0)  # base yaw was 0

    # overlaid row: new pose + reassigned pass
    np.testing.assert_allclose(overlaid.origin[2], [110.0, 210.0, 25.0])
    assert overlaid.roll[2] == 1.0 and overlaid.pitch[2] == 2.0 and overlaid.yaw[2] == 3.0
    assert overlaid.pass_id[2] == 1
    assert meta["n_edge"][2] == 500.0 and meta["rms_before"][2] == 10.0 and meta["rms_after"][2] == 5.0

    # untouched rows keep the base pose exactly, and the refinement columns are NaN (not the "kept" row's own 99.0 filler)
    for i in (0, 1, 3, 4):
        np.testing.assert_array_equal(overlaid.origin[i], base.origin[i])
        assert overlaid.pass_id[i] == base.pass_id[i]
        assert np.isnan(meta["n_edge"][i]) and np.isnan(meta["dt_s"][i])
    assert overlaid.traj is None


def test_overlay_refined_filename_mismatch_raises():
    base = _straight_line_poses(3)
    rows = [_refined_row(base, i, "kept") for i in range(3)]
    rows[1] = _refined_row(base, 1, "refined")
    rows[1]["filename"] = "wrong.jpg"
    with pytest.raises(ValueError, match="filename mismatch"):
        overlay_refined(base, rows)


def _synthetic_rot_only_traj(lin: Poses, pass_id: int, t_r: np.ndarray, rate_deg_s: float, R_cs, dt_s: float, segment) -> Trajectory:
    R_s = Rotation.from_euler("z", (rate_deg_s * t_r)[:, None], degrees=True).as_matrix()
    quat = Rotation.from_matrix(R_s).as_quat()
    rig = CamSensorRig(R_cs=R_cs, l_cs=np.zeros(3), dt_s=dt_s)
    return Trajectory(
        t=t_r, pass_id=np.full(len(t_r), pass_id, dtype=np.int32), S=np.zeros((len(t_r), 3)), quat=quat,
        segments={pass_id: [segment]}, rigs={pass_id: rig}, mode="rot_only",
        lin_t=lin.t, lin_origin=lin.origin, lin_roll=lin.roll, lin_pitch=lin.pitch, lin_yaw=lin.yaw, lin_pass_id=lin.pass_id,
    )


def _worst_px_invariance(poses_a: Poses, poses_b: Poses, tq: np.ndarray, pass_id: int, transforms: dict, rng) -> float:
    """max |du| (seam-wrapped), |dv| between world_to_pano(P, R_a, C_a) and
    world_to_pano(T_p(P), R_b, C_b) at query time(s) `tq`, `pass_id` -- the invariance
    `apply_pass_transforms`/`transform_trajectory` must preserve exactly."""
    tr = transforms["passes"][str(pass_id)]
    cx, cy = tr["centre"]
    a = np.radians(tr["yaw_deg"])
    Rp = pass_reg._rot_yaw(a)
    centre3 = np.array([cx, cy, 0.0])
    t_vec = np.array(tr["t"])

    worst = 0.0
    ph = np.full(len(tq), pass_id)
    o_a, r_a, p_a, y_a = poses_a.interp(tq, ph)
    o_b, r_b, p_b, y_b = poses_b.interp(tq, ph)
    R_a = geometry.vehicle_rotation(y_a, r_a, p_a)
    R_b = geometry.vehicle_rotation(y_b, r_b, p_b)
    for k in range(len(tq)):
        P = o_a[k] + rng.uniform(-25, 25, (500, 3)) * np.array([1, 1, 0.4])
        P_t = (P - centre3) @ Rp.T + centre3 + t_vec
        u1, v1, _, _ = geometry.world_to_pano(P, R_a[k], o_a[k], PANO_W, PANO_H, dtype=np.float64)
        u2, v2, _, _ = geometry.world_to_pano(P_t, R_b[k], o_b[k], PANO_W, PANO_H, dtype=np.float64)
        du = np.abs(u1 - u2)
        du = np.minimum(du, PANO_W - du)
        worst = max(worst, float(du.max()), float(np.abs(v1 - v2).max()))
    return worst


def test_transform_trajectory_invariant_with_apply_pass_transforms():
    """The core assemble step-3 contract: transforming the table (`pass_reg.apply_pass_transforms`)
    and the attached rot-only trajectory (`transform_trajectory`) by the *same* per-pass rigid
    transform must leave every rendered pixel exactly where it was -- at each frame's own time
    (`pose_at`-equivalent) AND at an interpolated time 0.3 s later, to 1e-6 px. This is the property
    that broke (up to several hundred px) when the pass rotation was folded into the rig's `R_cs`
    instead of the trajectory's per-sample orientation -- see `transform_trajectory`'s docstring."""
    lin = _straight_line_poses(9, dt=0.5, pass_id=0)  # t in [0, 4.0]
    R_cs = Rotation.from_euler("xyz", [2.0, -3.0, 5.0], degrees=True).as_matrix()
    t_r = np.linspace(-1.0, 5.0, 400)
    traj = _synthetic_rot_only_traj(lin, 0, t_r, rate_deg_s=37.0, R_cs=R_cs, dt_s=0.03, segment=(-1.0, 5.0))
    poses = Poses(filename=lin.filename, t=lin.t, origin=lin.origin, roll=lin.roll, pitch=lin.pitch, yaw=lin.yaw, pass_id=lin.pass_id, speed=lin.speed, source="test", traj=traj)

    cx, cy = float(lin.origin[:, 0].mean()), float(lin.origin[:, 1].mean())
    transforms = {"passes": {"0": {"yaw_deg": 4.3, "t": [0.6, -0.4, 0.12], "centre": [cx, cy]}}}

    registered = pass_reg.apply_pass_transforms(poses, transforms)
    traj_corrected = transform_trajectory(traj, transforms, registered)
    registered.traj = traj_corrected

    rng = np.random.default_rng(11)
    for k in (0, 3, 7):
        tq = np.array([poses.t[k], poses.t[k] + 0.3, poses.t[k] - 0.3])
        worst = _worst_px_invariance(poses, registered, tq, 0, transforms, rng)
        assert worst < 1e-6, (k, worst)

    # sanity: the transform actually moved things (a bug that no-ops the transform would pass trivially)
    assert not np.allclose(poses.origin, registered.origin)
    assert not np.allclose(traj.quat, traj_corrected.quat)


def test_assemble_end_to_end(tmp_path: Path, monkeypatch):
    """Writes a self-contained synthetic base/refined/transforms input set (no cache dependency),
    runs `assemble()`, and checks: per-frame status combination, S4 pass-reassignment carried
    through, provenance chaining, and that the corrected table round-trips through `load_poses`."""
    # keep write_pose_table's sha1 lookup a harmless no-op: `s.poses.source_file()` (the vendor
    # export path) is made to point at a file that does not exist, same intent as the old
    # `monkeypatch.setattr(config, "EXPORT_CSV", ...)` against the pre-refactor global; the pose
    # source is now the descriptor's adapter, and its file comes from `source_file()`.
    from geovap.runtime import settings

    monkeypatch.setattr(type(settings.get().poses), "source_file", lambda self: tmp_path / "no_such_export.csv")

    lin = _straight_line_poses(6, dt=1.0)
    lin.pass_id[3:] = 1  # frames 0-2 pass 0, frames 3-5 pass 1

    R_cs = np.eye(3)
    t_r = np.linspace(-1.0, 3.0, 200)
    traj = _synthetic_rot_only_traj(lin, 0, t_r, rate_deg_s=20.0, R_cs=R_cs, dt_s=0.0, segment=(-1.0, 3.0))  # only pass 0 covered

    base_path = tmp_path / "poses_traj_rot.csv"
    traj.save(tmp_path / "trajectory_test.npz")
    base_status = np.array(["traj_rot" if p == 0 else "kept" for p in lin.pass_id], dtype=object)
    write_pose_table(lin, {"status": base_status, "dt_s": np.zeros(6)}, {"stage": "test_S3b", "trajectory": "trajectory_test.npz"}, base_path)

    refined_path = tmp_path / "poses_refined_export.csv"
    fieldnames = ["filename", "status", "E", "N", "H", "roll", "pitch", "yaw", "pass_id", "dt_s", "dyaw", "droll", "dpitch", "n_edge", "rms_before", "rms_after"]
    rows = [_refined_row(lin, i, "kept") for i in range(6)]
    rows[1] = _refined_row(lin, 1, "refined", E=100.0, N=200.0, H=30.0, roll=1.0, pitch=2.0, yaw=3.0, pass_id=1, dt_s=0.01, dyaw=0.5, droll=0.1, dpitch=0.2, n_edge=500.0, rms_before=10.0, rms_after=5.0)
    with open(refined_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})

    transforms_path = tmp_path / "pass_transforms.json"
    transforms_path.write_text(json.dumps({"passes": {
        "0": {"yaw_deg": 4.0, "t": [0.5, -0.3, 0.05], "centre": [float(lin.origin[lin.pass_id == 0, 0].mean()), float(lin.origin[lin.pass_id == 0, 1].mean())]}},
        "1": {"yaw_deg": -2.0, "t": [-0.2, 0.4, 0.0], "centre": [float(lin.origin[lin.pass_id == 1, 0].mean()), float(lin.origin[lin.pass_id == 1, 1].mean())]},
        "summary": {"datum": "none"},
    }))

    out_path, report = assemble(
        base_path=base_path, refined_path=refined_path, transforms_path=transforms_path,
        out_path=tmp_path / "poses_corrected.csv", traj_out_path=tmp_path / "trajectory_corrected.npz", log=lambda *a: None,
    )

    result = read_pose_table(out_path)
    assert len(result) == 6
    assert result.traj is not None and result.traj.mode == "rot_only"
    assert result.pass_id[1] == 1  # S4 reassignment carried through
    assert result.registration == transforms_path  # sidecar "registration" round-trips

    status_rows = list(csv.DictReader(open(out_path)))
    assert [r["status"] for r in status_rows] == ["traj_rot+reg", "traj_rot+refined+reg", "traj_rot+reg", "kept+reg", "kept+reg", "kept+reg"]
    assert status_rows[1]["src_pass_changed"] == "True"
    assert float(status_rows[1]["n_edge"]) == 500.0
    assert status_rows[0]["n_edge"] == "nan"

    prov = json.loads(out_path.with_suffix(".json").read_text())
    assert prov["inputs"]["base"]["poses_hash"] and prov["inputs"]["refined"]["sha1"] and prov["inputs"]["pass_transforms"]["sha1"]
    assert prov["trajectory"] == "trajectory_corrected.npz"
    assert prov["registration"]["path"] == str(transforms_path)
    import hashlib

    assert prov["registration"]["sha1"] == hashlib.sha1(transforms_path.read_bytes()).hexdigest()
    assert prov["s4_overlay"]["n_overlaid"] == 1
    assert prov["status_counts"] == {"traj_rot+reg": 2, "traj_rot+refined+reg": 1, "kept+reg": 3}
    assert report["s4_overlay"]["n_overlaid"] == 1

    # pose_at(idx, 0) matches the table exactly, regardless of trajectory coverage
    o0, r0, p0, y0 = result.pose_at(np.arange(6), 0.0)
    np.testing.assert_array_equal(o0, result.origin)
    np.testing.assert_array_equal(y0, result.yaw)

    assert (tmp_path / "trajectory_corrected.npz").exists()
