"""S5: pass-to-pass registration and JVF datum. Mostly synthetic (no store/JVF file access); one test
(`test_cloud_store_registration_hook_...`) opens the real store if built, skipped otherwise. < 5 s."""
from __future__ import annotations

import numpy as np
import pytest

from geovap.domain.model import geometry
from geovap.stages.register import passes as pr
from geovap.domain.model.poses import Poses
from geovap.runtime.pose_tables import load as load_poses
from geovap.runtime import settings

PANO_W, PANO_H = settings.get().sensor.pano_w, settings.get().sensor.pano_h


# --------------------------------------------------------------------------------- register_pair
def _symmetric_patches(centre: np.ndarray, pass_id: int, seed: int = 0) -> pr.Patches:
    """Ground plane (grid, symmetric about `centre`) + a square building outline (4 finite straight
    walls, symmetric about `centre`) so the reference patch set's centroid is exactly `centre` --
    this makes the ICP's known-offset recovery exact in closed form (see test below). Finite straight
    walls (not a full ring/circle) are required: point-to-plane sensitivity to yaw at a point p is
    n.(z x (p-centre)), which is identically zero everywhere on a ring centered on the rotation axis
    (normal always radial) -- a classic ICP degeneracy -- but nonzero almost everywhere on a wall of
    a square centered the same way (normal constant, radius vector is not parallel to it off-centre)."""
    g = np.linspace(-40, 40, 21)
    gx, gy = np.meshgrid(g, g)
    ground_c = np.stack([gx.ravel() + centre[0], gy.ravel() + centre[1], np.zeros(gx.size)], axis=1)
    ground_n = np.tile([0.0, 0.0, 1.0], (len(ground_c), 1))

    s = np.linspace(-30, 30, 61)
    z = np.tile(np.linspace(0.2, 2.8, 7), (len(s) + 6) // 7 + 1)[: len(s)]
    walls_c, walls_n = [], []
    for sign, axis in ((+1, 0), (-1, 0), (+1, 1), (-1, 1)):
        pos = np.zeros((len(s), 3))
        pos[:, axis] = sign * 30.0
        pos[:, 1 - axis] = s
        pos[:, 2] = z
        pos[:, :2] += centre[:2]
        nrm = np.zeros((len(s), 3))
        nrm[:, axis] = sign
        walls_c.append(pos)
        walls_n.append(nrm)
    facade_c = np.concatenate(walls_c)
    facade_n = np.concatenate(walls_n)

    c = np.concatenate([ground_c, facade_c])
    n = np.concatenate([ground_n, facade_n])
    cls = np.concatenate([np.full(len(ground_c), pr.CLS_GROUND, np.uint8), np.full(len(facade_c), pr.CLS_FACADE, np.uint8)])
    w = np.ones(len(c))
    return pr.Patches(c, n, cls, w, pass_id)


def test_register_pair_recovers_known_offset():
    centre = np.array([642100.0, 1055700.0, 0.0])
    P_ref = _symmetric_patches(centre, pass_id=0)

    dE, dN, dH, dyaw = 0.42, -0.31, 0.08, 1.7
    x_true = np.array([dE, dN, dH, dyaw])
    q_c, q_n = pr._apply_4dof(P_ref.c, P_ref.n, centre, x_true)
    P_q = pr.Patches(q_c, q_n, P_ref.cls.copy(), P_ref.w.copy(), pass_id=1)

    res = pr.register_pair(P_q, P_ref)
    assert res["converged"], res

    # Both patch sets are symmetric about `centre`, so P_q's own centroid (what register_pair rotates
    # about internally) is exactly centre + (dE, dN, 0); closed form then gives recovered T == -x_true
    # exactly (see plan S5 step 6 "recover it within 1cm / 0.01deg").
    T = np.asarray(res["T"])
    np.testing.assert_allclose(T[:3], -x_true[:3], atol=0.01)
    assert abs(T[3] - (-dyaw)) < 0.01, res

    # applying the recovered transform should bring P_q back onto P_ref closely
    pr_c, _ = pr._apply_4dof(P_q.c, P_q.n, np.array(res["centre"] + [0.0]), T)
    # ground points (normal (0,0,1)) only constrain height; facade ring constrains xy exactly
    facade = P_q.cls == pr.CLS_FACADE
    d_xy = np.linalg.norm((pr_c[facade] - P_ref.c[facade])[:, :2], axis=1)
    assert d_xy.max() < 0.02, d_xy.max()


def test_register_pair_low_overlap_does_not_converge():
    centre = np.array([0.0, 0.0, 0.0])
    P_ref = _symmetric_patches(centre, pass_id=0)
    tiny = P_ref.select(np.arange(len(P_ref)) < 5)  # far too few points to register
    res = pr.register_pair(tiny, P_ref)
    assert not res["converged"]


# ----------------------------------------------------------------------------- curb/JVF corridor gate
def test_gate_curb_to_road_corridor_keeps_only_near_points():
    # a single straight JVF road-boundary sample line along the E axis at N=0
    road = pr.RoadBoundaryIndex([np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])])
    cc = np.array([[10.0, 0.5, 0.0], [10.0, 1.9, 0.0], [10.0, 5.0, 0.0], [50.0, -30.0, 0.0]])
    cn = np.tile([1.0, 0.0, 0.0], (4, 1))
    cw = np.ones(4)
    out_c, out_n, out_w = pr.gate_curb_to_road_corridor(cc, cn, cw, road, corridor_m=2.0)
    # only the two points within 2 m of the line (N=0.5, N=1.9) should survive -- the ploughed-field
    # style outliers at N=5 and N=-30 (see pass_reg module docstring "curb corridor") must not.
    assert len(out_c) == 2
    np.testing.assert_allclose(sorted(out_c[:, 1]), [0.5, 1.9])
    assert len(out_n) == 2 and len(out_w) == 2


def test_gate_curb_to_road_corridor_empty_input():
    road = pr.RoadBoundaryIndex([np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])])
    empty = np.empty((0, 3))
    out_c, out_n, out_w = pr.gate_curb_to_road_corridor(empty, empty, np.empty(0), road)
    assert len(out_c) == 0 and len(out_n) == 0 and len(out_w) == 0


# --------------------------------------------------------------------------- CloudStore registration
def test_cloud_store_registration_hook_transforms_only_target_pass():
    """`CloudStore(registration=...)` (S5 step 4): `xyz_m` applies the per-pass rigid transform to rows
    scanned during that pass (selected by `poses.pass_of_time(gps_time)`, see `PassRegistration.apply`
    docstring) and leaves rows from any other pass byte-identical. Real store test -- narrow
    `query_time` windows keep it well under 5 s even though the store is 18 GB; tile "011" and the two
    windows below were found (empirically, on this dataset) to each fall cleanly inside one pass."""
    from geovap.runtime import settings
    from geovap.runtime.store import CloudStore

    if not (settings.get().workspace.store / "tiles.json").exists():
        pytest.skip("store not built")

    poses = load_poses()
    store = CloudStore(settings.get().workspace.store)
    name = "011"
    win_target = (301660.0, 301662.0)  # lands entirely in pass 11
    win_other = (301900.0, 301902.0)  # lands entirely in pass 12
    rows_target = dict((t.name, r) for t, r in store.query_time(*win_target))[name]
    rows_other = dict((t.name, r) for t, r in store.query_time(*win_other))[name]
    assert len(rows_target) > 10 and len(rows_other) > 10

    td = store.tile(name)
    p_target = np.unique(poses.pass_of_time(np.asarray(td.gps_time[rows_target])))
    p_other = np.unique(poses.pass_of_time(np.asarray(td.gps_time[rows_other])))
    assert list(p_target) == [11] and list(p_other) == [12], (p_target, p_other)  # sanity on the fixed windows above

    dE, dN, dH, dyaw = 1.5, -0.7, 0.3, 4.0
    cx = float(poses.origin[poses.pass_id == 11, 0].mean())
    cy = float(poses.origin[poses.pass_id == 11, 1].mean())
    transforms = {"passes": {"11": {"yaw_deg": dyaw, "t": [dE, dN, dH], "centre": [cx, cy]}}}

    store_reg = CloudStore(settings.get().workspace.store, registration=transforms)
    xyz_target_raw = td.xyz_m(rows_target)
    xyz_target_reg = store_reg.tile(name).xyz_m(rows_target)
    xyz_other_raw = td.xyz_m(rows_other)
    xyz_other_reg = store_reg.tile(name).xyz_m(rows_other)

    # pass 12 has no entry in `transforms` -> untouched, byte-identical
    np.testing.assert_array_equal(xyz_other_raw, xyz_other_reg)

    # pass 11: reg must equal the 4-DoF transform applied by hand (independent of PassRegistration.apply)
    a = np.radians(dyaw)
    c, s = np.cos(a), np.sin(a)
    Rz = np.array([[c, -s], [s, c]])
    xy = xyz_target_raw[:, :2] - np.array([cx, cy])
    expect_xy = xy @ Rz.T + np.array([cx, cy]) + np.array([dE, dN])
    np.testing.assert_allclose(xyz_target_reg[:, :2], expect_xy, atol=1e-9)
    np.testing.assert_allclose(xyz_target_reg[:, 2], xyz_target_raw[:, 2] + dH, atol=1e-9)
    assert not np.allclose(xyz_target_raw, xyz_target_reg)  # actually moved


# ------------------------------------------------------------------------------------- invariance
def _synthetic_poses_single_pass(m: int, seed: int = 0) -> Poses:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(300_000, 300_200, m))
    origin = np.stack(
        [rng.uniform(-642_150, -642_050, m), rng.uniform(-1_055_150, -1_055_050, m), rng.uniform(220, 226, m)], 1
    )
    return Poses(
        filename=np.array([f"f{i}.jpg" for i in range(m)], dtype=object),
        t=t,
        origin=origin,
        roll=rng.uniform(-5, 5, m),
        pitch=rng.uniform(-4, 4, m),
        yaw=rng.uniform(-180, 180, m),
        pass_id=np.zeros(m, np.int32),
        speed=np.full(m, 8.0),
    )


def test_apply_pass_transforms_invariant_world_to_pano():
    rng = np.random.default_rng(7)
    poses = _synthetic_poses_single_pass(6)
    cx, cy = poses.origin[:, 0].mean(), poses.origin[:, 1].mean()
    transforms = {"passes": {"0": {"yaw_deg": 2.3, "t": [0.35, -0.22, 0.11], "centre": [float(cx), float(cy)]}}}
    poses2 = pr.apply_pass_transforms(poses, transforms)

    R, C = geometry.frame_rotations(poses)
    R2, C2 = geometry.frame_rotations(poses2)

    Rp = pr._rot_yaw(np.radians(2.3))
    t = np.array([0.35, -0.22, 0.11])
    centre3 = np.array([cx, cy, 0.0])

    worst = 0.0
    for k in range(len(poses)):
        P = poses.origin[k] + rng.uniform(-25, 25, (2000, 3)) * np.array([1, 1, 0.4])
        P_t = (P - centre3) @ Rp.T + centre3 + t

        u1, v1, r1, el1 = geometry.world_to_pano(P, R[k], C[k], PANO_W, PANO_H, dtype=np.float64)
        u2, v2, r2, el2 = geometry.world_to_pano(P_t, R2[k], C2[k], PANO_W, PANO_H, dtype=np.float64)
        du = np.abs(u1 - u2)
        du = np.minimum(du, PANO_W - du)  # seam wrap
        worst = max(worst, du.max(), np.abs(v1 - v2).max())
    assert worst < 1e-6, worst


# ---------------------------------------------------------------------------------- global solve
def test_solve_global_recovers_known_3_node_graph():
    truth = {0: [0.0, 0.0, 0.0, 0.0], 1: [0.10, -0.05, 0.02, 0.30], 2: [-0.05, 0.08, -0.01, -0.20]}
    pass_ids = [0, 1, 2]

    def pair_T(a, b):
        # perfect pairwise data: relative offset exactly matches the (small-angle) truth difference
        return [truth[b][i] - truth[a][i] for i in range(4)]

    pair_results = {
        (0, 1): {"T": pair_T(0, 1), "rms_after": 0.01, "n": 1000, "converged": True, "centre": [0.0, 0.0]},
        (1, 2): {"T": pair_T(1, 2), "rms_after": 0.01, "n": 1000, "converged": True, "centre": [0.0, 0.0]},
        (0, 2): {"T": pair_T(0, 2), "rms_after": 0.01, "n": 1000, "converged": True, "centre": [0.0, 0.0]},
    }
    jvf_results = {0: {"dE": truth[0][0], "dN": truth[0][1], "dH": truth[0][2], "rms": 0.01, "n": 1000}}

    gr = pr.solve_global(pass_ids, pair_results, jvf_results)
    # translations are anchored (node 0 has a JVF constraint, pairwise data is exact) -> recovered
    # near-exactly for every node.
    for p in pass_ids:
        np.testing.assert_allclose(gr.x[p][:3], truth[p][:3], atol=0.02)
    # yaw has no absolute anchor (JVF gives no yaw here): only relative yaws are constrained, and the
    # weak identity prior pulls the *mean* yaw across nodes toward 0 rather than any one node -- so
    # check the pairwise differences (fully determined) rather than absolute per-node yaw.
    assert abs((gr.x[1][3] - gr.x[0][3]) - (truth[1][3] - truth[0][3])) < 0.02
    assert abs((gr.x[2][3] - gr.x[0][3]) - (truth[2][3] - truth[0][3])) < 0.02
    assert gr.residuals["pairwise_m_median"] < 0.01
