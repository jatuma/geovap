"""S5 -- pass-to-pass registration and JVF datum (07_revize_geometrie_a_data.md, plan section S5).

Units: the 30 photo passes (`Poses.pass_id`); points are assigned to a pass by
`store.query_time(t0, t1)` over the pass' own time range (own-pass points only, no ±45 s window).
Where a pass is 1:1 with a `psid` (see `Geovap_cache/out/poses/pass_psid.json`) the `psid` column is
used as an extra, cheaper guard against points that fall in the same time bucket from a neighbouring
pass; passes with a split psid (12, 14, 20, 29) fall back to the time filter alone.

Pipeline: `extract_patches` (planar voxels + curb break-lines, cached per pass) -> `register_pair`
(pairwise 4-DoF point-to-plane ICP between passes whose frame origins come close) and `jvf_offset`
(own-pass curbs vs the JVF road-boundary polylines, the absolute datum) -> `solve_global` (a small
linear pose graph over 30 nodes) -> `pass_transforms.json`. Application: `CloudStore(registration=...)`
on the cloud side (see `geovap.runtime.store`), `apply_pass_transforms` here on the pose side.

Curb corridor (see `CURB_JVF_CORRIDOR_M`): `_curb_points` is a generic ground height-jump detector run
over each pass' whole (80 m-padded) bbox, which on this rural dataset picks up plough-furrow and field
edges as readily as real curbs -- an unfiltered run tagged 30k-80k "curb" points per pass, an order of
magnitude more than a few hundred metres of real kerb can produce (`out/pass_reg/qa/*.png` shows the
S5-run-2026-09-08 curb cloud for pass 0 as a diagonal fan across a ploughed field, not the road edge).
`extract_patches` therefore keeps a curb candidate only if it falls within `CURB_JVF_CORRIDOR_M` of a
*raw* JVF road-boundary polyline, before it is ever compared to JVF for the offset -- this is what
turns `jvf_offset` from an unconstrained nearest-line match (which converges to whatever agricultural
line happens to be close) into an actual road-curb-vs-road-boundary measurement.

**S5b conclusion (2026-09-08): the JVF absolute datum is unreliable here, production is pairwise-only.**
Two absolute-datum methods were tried (`jvf_offset`: cloud curbs vs JVF road boundary;
`jvf_offset_photometric`: per-pass photo-edge grid search vs the same lines) and both failed acceptance:
anchoring the pose graph on either made the near-field median (`cli.validate_pass_reg`, 200 frames)
WORSE (13.3 -> 15.6 px), with only 1/30 passes meeting the anchor criterion and the two methods
disagreeing on the passes both can measure -- this rural road has too few real kerbs, and both curb and
edge detectors saturate on plough-furrow/verge texture instead (`CURB_JVF_CORRIDOR_M`,
`MIN_EDGE_COMPONENT_PX` mitigate, do not eliminate, this). Pairwise ICP is solid (66/72 pairs converge,
rms 0.10 -> 0.035 m); `cli.register_passes solve` therefore defaults to `--datum none` (pairwise + weak
identity prior, no absolute anchor). Both JVF paths stay selectable (`--datum jvf-cloud|jvf-photo`) for
when a better absolute reference (SBET/POSPac, 07 §6) arrives.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from geovap.domain.model import geometry
from geovap.runtime.store import CloudStore
from .config import OUT_DIR, PANO_H, PANO_W, POSES_DIR, STORE_DIR, ZB_H, ZB_W
from geovap.domain.model.frames import FrameIndex
from geovap.domain.model.poses import Poses
from geovap.runtime.pose_tables import load as load_poses
from geovap.runtime.panos import pano_path
from geovap.stages.prepare.products import FrameProducts

PASS_REG_DIR = OUT_DIR / "pass_reg"

VOXEL = 0.5
MIN_PTS_VOXEL = 30
PLANAR_RATIO = 0.2  # keep voxel if lambda0 / lambda2 < this (lambda ascending: 0=smallest)
FACADE_NZ_MAX = 0.3  # |normal_z| below this on a planar voxel => facade candidate
GROUND_NZ_MIN = 0.7
GROUND_CLASS = 2  # LAZ classification code for ground

CURB_GRID = 0.25
CURB_DZ = 0.05  # m, minimum height jump between neighbour cells to call a curb

OVERLAP_RADIUS_M = 15.0  # pairwise overlap: frame origins within this distance
FRAME_MARGIN_M = 80.0  # bbox padding around a pass' frame origins

ROAD_BOUNDARY_CODE = "0100000304"  # "hranice dopravni stavby nebo plochy"
JVF_GATE_M = 1.0
CURB_JVF_CORRIDOR_M = 2.0  # keep a height-jump candidate as CLS_CURB only if within this of a raw
# (unshifted) JVF road-boundary polyline -- see module docstring "curb corridor". Without this gate
# `_curb_points` (a global per-pass height-jump grid, run over an 80 m-padded bbox) tags every plough
# furrow / driveway apron / field edge as a curb, which then matches whatever JVF line happens to be
# nearest and produces a per-pass offset dominated by agricultural noise, not the road curb (see
# `Geovap_cache/out/pass_reg/qa/p00_f0189.png`: pass 0's own "curb" points are a diagonal fan across a
# ploughed field, not the grass/asphalt edge the vehicle is driving beside). 2 m comfortably exceeds
# the 0.14 m JVF digitising tolerance and the largest offset the memory note/plan documents (0.6 m).
TRAJ_CORRIDOR_M = 12.0  # curb candidates are gathered only from ground points within this of the
# pass' OWN frame trajectory. Necessary in addition to `CURB_JVF_CORRIDOR_M`: the JVF road-boundary
# layer ("0100000304") also codes farm-track and field-access edges, so a ploughed-field height jump
# that happens to sit near a *different*, unrelated boundary polyline survives a JVF-only gate (this
# is exactly what the first (JVF-only) run of this fix did -- jvf_offsets.json was byte-identical
# before and after adding `CURB_JVF_CORRIDOR_M` alone). Gating on the vehicle's own path first removes
# any candidate that could not possibly be the curb of the road this pass drove, before the JVF gate
# ever runs; ~12 m covers a two-lane road plus verge on either side of the trajectory.

CLS_GROUND, CLS_FACADE, CLS_OTHER, CLS_CURB = 0, 1, 2, 3


# --------------------------------------------------------------------------------------- patches
@dataclass
class Patches:
    c: np.ndarray  # [N,3] centroid (m)
    n: np.ndarray  # [N,3] normal (unit; curb: (nx,ny,0))
    cls: np.ndarray  # [N] uint8, CLS_*
    w: np.ndarray  # [N] float64 weight
    pass_id: int

    def __len__(self) -> int:
        return len(self.c)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, c=self.c, n=self.n, cls=self.cls, w=self.w, pass_id=np.int64(self.pass_id))

    @staticmethod
    def load(path: Path) -> "Patches":
        d = np.load(path)
        return Patches(c=d["c"], n=d["n"], cls=d["cls"], w=d["w"], pass_id=int(d["pass_id"]))

    def select(self, mask: np.ndarray) -> "Patches":
        return Patches(self.c[mask], self.n[mask], self.cls[mask], self.w[mask], self.pass_id)


def _voxel_pca(xyz: np.ndarray, cls_raw: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Planar voxels of `xyz` (>= MIN_PTS_VOXEL pts, planarity below PLANAR_RATIO) -> (c, n, cls, w)."""
    if len(xyz) == 0:
        return (np.empty((0, 3)),) * 2 + (np.empty(0, np.uint8), np.empty(0))
    origin = xyz.min(axis=0)
    ijk = np.floor((xyz - origin) / voxel).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    counts = np.diff(np.r_[starts, len(key_s)])
    keep = counts >= MIN_PTS_VOXEL
    starts, counts = starts[keep], counts[keep]
    xyz_s = xyz[order]
    cls_s = cls_raw[order]
    out_c, out_n, out_cls, out_w = [], [], [], []
    for s, cnt in zip(starts.tolist(), counts.tolist()):
        pts = xyz_s[s : s + cnt]
        cent = pts.mean(axis=0)
        d = pts - cent
        cov = (d.T @ d) / cnt
        ev, evec = np.linalg.eigh(cov)  # ascending
        if ev[2] <= 1e-12 or ev[0] / ev[2] >= PLANAR_RATIO:
            continue
        normal = evec[:, 0]
        maj_cls = np.bincount(cls_s[s : s + cnt].astype(np.int64)).argmax()
        nz = abs(normal[2])
        if maj_cls == GROUND_CLASS and nz >= GROUND_NZ_MIN:
            c = CLS_GROUND
        elif nz <= FACADE_NZ_MAX:
            c = CLS_FACADE
        else:
            c = CLS_OTHER
        out_c.append(cent)
        out_n.append(normal)
        out_cls.append(c)
        out_w.append(float(cnt))
    if not out_c:
        return (np.empty((0, 3)),) * 2 + (np.empty(0, np.uint8), np.empty(0))
    return np.array(out_c), np.array(out_n), np.array(out_cls, dtype=np.uint8), np.array(out_w)


def _curb_points(xyz_ground: np.ndarray, grid: float = CURB_GRID, dz: float = CURB_DZ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Height-jump break-lines on ground points: 2D grid, cells whose height differs by >= dz from a
    4-neighbour cell become curb points at the cell centre (xy), height = cell mean, normal = 2D unit
    vector pointing from the lower cell to the higher one. Vectorised over a dense grid array (bounded
    by the pass' bbox / 0.25 m, a few 1e5 cells)."""
    if len(xyz_ground) < 4:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
    origin = xyz_ground[:, :2].min(axis=0)
    ij = np.floor((xyz_ground[:, :2] - origin) / grid).astype(np.int64)
    dims = ij.max(axis=0) + 1  # (nx, ny)
    if dims[0] * dims[1] > 20_000_000:  # pathological bbox: fall back to no curbs rather than OOM
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
    flat = ij[:, 0] * dims[1] + ij[:, 1]
    n_cells = int(dims[0] * dims[1])
    counts = np.bincount(flat, minlength=n_cells)
    sum_h = np.bincount(flat, weights=xyz_ground[:, 2], minlength=n_cells)
    sum_x = np.bincount(flat, weights=xyz_ground[:, 0], minlength=n_cells)
    sum_y = np.bincount(flat, weights=xyz_ground[:, 1], minlength=n_cells)
    occ = counts > 0
    h_grid = np.full(n_cells, np.nan)
    h_grid[occ] = sum_h[occ] / counts[occ]
    h_grid = h_grid.reshape(dims[0], dims[1])
    x_grid = np.full(n_cells, np.nan)
    x_grid[occ] = sum_x[occ] / counts[occ]
    x_grid = x_grid.reshape(dims[0], dims[1])
    y_grid = np.full(n_cells, np.nan)
    y_grid[occ] = sum_y[occ] / counts[occ]
    y_grid = y_grid.reshape(dims[0], dims[1])

    best_jump = np.zeros(dims, dtype=np.float64)
    best_di = np.zeros(dims, dtype=np.int8)
    best_dj = np.zeros(dims, dtype=np.int8)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb_h = np.full(dims, np.nan)
        src = h_grid
        # shifted view: nb_h[i,j] = h_grid[i+di, j+dj] where in bounds
        i0, i1 = max(0, -di), dims[0] - max(0, di)
        j0, j1 = max(0, -dj), dims[1] - max(0, dj)
        si0, si1 = max(0, di), dims[0] + min(0, di)
        sj0, sj1 = max(0, dj), dims[1] + min(0, dj)
        nb_h[i0:i1, j0:j1] = src[si0:si1, sj0:sj1]
        jump = nb_h - h_grid
        better = np.nan_to_num(np.abs(jump), nan=-1.0) > np.abs(best_jump)
        best_jump = np.where(better, np.nan_to_num(jump, nan=0.0), best_jump)
        best_di = np.where(better, di, best_di)
        best_dj = np.where(better, dj, best_dj)

    curb = occ.reshape(dims) & (np.abs(best_jump) >= dz)
    ii, jj = np.nonzero(curb)
    if len(ii) == 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
    jump = best_jump[ii, jj]
    d2 = np.stack([best_di[ii, jj].astype(np.float64), best_dj[ii, jj].astype(np.float64)], axis=1)
    d2 = d2 * np.sign(jump)[:, None] / (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-12)
    c = np.stack([x_grid[ii, jj], y_grid[ii, jj], h_grid[ii, jj]], axis=1)
    n3 = np.concatenate([d2, np.zeros((len(d2), 1))], axis=1)
    w = np.minimum(np.abs(jump) / dz, 4.0)
    return c, n3, w


_ROAD_INDEX_CACHE: "RoadBoundaryIndex | None" = None


def _road_corridor_index() -> "RoadBoundaryIndex":
    """Process-wide cached `RoadBoundaryIndex` of the raw (unshifted) JVF road boundary, used to gate
    curb candidates onto the actual road corridor (see `CURB_JVF_CORRIDOR_M`). Built lazily so a worker
    that never extracts curbs (e.g. re-running only the pairwise step) never pays for it."""
    global _ROAD_INDEX_CACHE
    if _ROAD_INDEX_CACHE is None:
        _ROAD_INDEX_CACHE = RoadBoundaryIndex(load_road_boundary())
    return _ROAD_INDEX_CACHE


def gate_curb_to_road_corridor(
    cc: np.ndarray, cn: np.ndarray, cw: np.ndarray, road_index: "RoadBoundaryIndex", corridor_m: float = CURB_JVF_CORRIDOR_M
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only curb candidates within `corridor_m` of `road_index` (pure/testable half of the
    `extract_patches` corridor gate; see `CURB_JVF_CORRIDOR_M`)."""
    if len(cc) == 0:
        return cc, cn, cw
    d_road, _ = road_index.query(cc)
    keep = d_road <= corridor_m
    return cc[keep], cn[keep], cw[keep]


def extract_patches(store: CloudStore, poses: Poses, pass_id: int, voxel: float = VOXEL, psid_filter: set[int] | None = None) -> Patches:
    sel = poses.pass_id == pass_id
    t0, t1 = float(poses.t[sel].min()), float(poses.t[sel].max())
    org = poses.origin[sel]
    bbox = (
        float(org[:, 0].min() - FRAME_MARGIN_M),
        float(org[:, 1].min() - FRAME_MARGIN_M),
        float(org[:, 0].max() + FRAME_MARGIN_M),
        float(org[:, 1].max() + FRAME_MARGIN_M),
    )
    xyz_chunks, cls_chunks, ground_chunks = [], [], []
    for tile, rows in store.query_time(t0, t1, bbox=bbox):
        td = store.tile(tile.name)
        if psid_filter is not None and td.psid is not None:
            m = np.isin(np.asarray(td.psid[rows]), list(psid_filter))
            rows = rows[m]
            if len(rows) == 0:
                continue
        xyz = td.xyz_m(rows)
        cls = np.asarray(td.classification[rows])
        xyz_chunks.append(xyz)
        cls_chunks.append(cls)
    if xyz_chunks:
        xyz = np.concatenate(xyz_chunks)
        cls = np.concatenate(cls_chunks)
    else:
        xyz, cls = np.empty((0, 3)), np.empty(0, dtype=np.uint8)

    c, n, vcls, w = _voxel_pca(xyz, cls, voxel)
    if len(xyz):
        gmask = cls == GROUND_CLASS
        ground = xyz[gmask]
        if len(ground):
            d_traj, _ = cKDTree(org[:, :2]).query(ground[:, :2], k=1)
            ground = ground[d_traj <= TRAJ_CORRIDOR_M]
        cc, cn, cw = _curb_points(ground)
        cc, cn, cw = gate_curb_to_road_corridor(cc, cn, cw, _road_corridor_index())
    else:
        cc, cn, cw = np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
    all_c = np.concatenate([c, cc]) if len(cc) else c
    all_n = np.concatenate([n, cn]) if len(cc) else n
    all_cls = np.concatenate([vcls, np.full(len(cc), CLS_CURB, dtype=np.uint8)]) if len(cc) else vcls
    all_w = np.concatenate([w, cw]) if len(cc) else w
    return Patches(all_c, all_n, all_cls, all_w, pass_id)


def _load_pass_psid() -> dict:
    p = POSES_DIR / "pass_psid.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())["passes"]


def _extract_one(args) -> tuple[int, dict]:
    pass_id, store_root, use_psid = args
    from geovap.runtime.pose_tables import load as _lp

    poses = _lp()
    store = CloudStore(store_root)
    psid_hist = _load_pass_psid().get(str(pass_id), {}).get("psid_hist", {})
    psid_filter = {int(k) for k in psid_hist} if (use_psid and len(psid_hist) == 1) else None
    pat = extract_patches(store, poses, pass_id, psid_filter=psid_filter)
    out = PASS_REG_DIR / f"patches_p{pass_id:02d}.npz"
    pat.save(out)
    store.release()
    counts = {int(c): int((pat.cls == c).sum()) for c in np.unique(pat.cls)} if len(pat) else {}
    used_psid = sorted(psid_filter) if psid_filter else None
    return pass_id, {"n": len(pat), "counts": counts, "used_psid": used_psid}


def run_extract_all(poses: Poses | None = None, workers: int = 8, store_root: Path = STORE_DIR) -> dict:
    """Extract and cache patches for all 30 passes. Returns per-pass counts report."""
    poses = poses or load_poses()
    PASS_REG_DIR.mkdir(parents=True, exist_ok=True)
    pass_ids = [int(p) for p in np.unique(poses.pass_id)]
    args = [(pid, store_root, True) for pid in pass_ids]
    report = {}
    with Pool(min(workers, 8)) as pool:
        for pid, info in pool.imap_unordered(_extract_one, args):
            report[pid] = info
            print(f"pass {pid:2d}: n={info['n']:7d} counts={info['counts']} psid={info['used_psid']}")
    (PASS_REG_DIR / "patches_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


# ------------------------------------------------------------------------------------ register_pair
def _rot_yaw(dyaw_rad: float) -> np.ndarray:
    c, s = np.cos(dyaw_rad), np.sin(dyaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _apply_4dof(pts: np.ndarray, normals: np.ndarray, centre: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x = (dE, dN, dH, dyaw_deg). Rotate about `centre` (xy) then translate."""
    Rz = _rot_yaw(np.radians(x[3]))
    d = pts - centre
    pr = d @ Rz.T + centre
    pr = pr + np.array([x[0], x[1], x[2]])
    nr = normals @ Rz.T
    return pr, nr


def register_pair(
    P_q: Patches,
    P_ref: Patches,
    T0: np.ndarray | None = None,
    dof: int = 4,
    max_iter: int = 25,
) -> dict:
    """Point-to-plane ICP, 4-DoF (dE, dN, dH, dyaw about the query centroid). Ground patches weighted
    to constrain dH; facade/curb patches weighted to constrain (dE, dN, dyaw) so ground alone cannot
    fix the horizontal. Returns dict(T=x, rms_before, rms_after, n, sv_min, converged)."""
    x = np.zeros(4) if T0 is None else np.asarray(T0, dtype=np.float64).copy()
    if len(P_q) < 20 or len(P_ref) < 20:
        return {"T": x.tolist(), "rms_before": None, "rms_after": None, "n": 0, "sv_min": 0.0, "converged": False}
    centre = P_q.c[:, :2].mean(axis=0)
    centre3 = np.array([centre[0], centre[1], 0.0])
    ref_tree = cKDTree(P_ref.c)
    ref_horiz = P_ref.cls != CLS_GROUND  # facade/curb: constrain E,N,yaw
    ground_gate_start, ground_gate_end = 1.0, 0.3

    rms_hist = []
    sv_min = 0.0
    x_prev = x.copy()
    for it in range(max_iter):
        gate = ground_gate_start + (ground_gate_end - ground_gate_start) * it / max(1, max_iter - 1)
        pr, nr = _apply_4dof(P_q.c, P_q.n, centre3, x)
        dist, idx = ref_tree.query(pr, k=1)
        cos_ang = np.abs(np.einsum("ij,ij->i", nr, P_ref.n[idx]))
        valid = (dist <= gate) & (cos_ang > np.cos(np.radians(20.0))) & (P_q.cls[:, None].ravel() == P_ref.cls[idx])
        if valid.sum() < 20:
            break
        q = pr[valid]
        nq = nr[valid]
        ridx = idx[valid]
        pref = P_ref.c[ridx]
        nref = P_ref.n[ridx]
        w = np.sqrt(P_q.w[valid] * P_ref.w[ridx])
        w = w / (w.mean() + 1e-9)
        is_horiz = ref_horiz[ridx]
        w_h = np.where(is_horiz, w * 3.0, w * 0.3)  # facade/curb dominate horizontal DoF
        res0 = np.einsum("ij,ij->i", q - pref, nref)
        rms_hist.append(float(np.sqrt(np.mean(res0**2))))

        def resid(dx):
            pr2, nr2 = _apply_4dof(P_q.c[valid], P_q.n[valid], centre3, x + dx)
            r = np.einsum("ij,ij->i", pr2 - pref, nref)
            return r * w_h

        sol = least_squares(resid, np.zeros(4), loss="soft_l1", f_scale=0.1, max_nfev=200)
        x = x + sol.x
        if np.linalg.norm(sol.x) < 1e-5 and np.allclose(x, x_prev, atol=1e-6):
            x_prev = x.copy()
            break
        x_prev = x.copy()
        # normal-matrix conditioning of the last linearised system (Gauss-Newton at solution)
        J = sol.jac if hasattr(sol, "jac") else None
        if J is not None:
            sv = np.linalg.svd(J, compute_uv=False)
            sv_min = float(sv[-1]) if len(sv) else 0.0

    # final residual
    pr, nr = _apply_4dof(P_q.c, P_q.n, centre3, x)
    dist, idx = ref_tree.query(pr, k=1)
    cos_ang = np.abs(np.einsum("ij,ij->i", nr, P_ref.n[idx]))
    valid = (dist <= ground_gate_end) & (cos_ang > np.cos(np.radians(20.0))) & (P_q.cls == P_ref.cls[idx])
    n = int(valid.sum())
    if n >= 20:
        res_final = np.einsum("ij,ij->i", pr[valid] - P_ref.c[idx[valid]], P_ref.n[idx[valid]])
        rms_after = float(np.sqrt(np.mean(res_final**2)))
    else:
        rms_after = None
    return {
        "T": x.tolist(),
        "centre": centre.tolist(),
        "rms_before": rms_hist[0] if rms_hist else None,
        "rms_after": rms_after,
        "n": n,
        "sv_min": sv_min,
        "converged": n >= 20,
    }


def overlap_pairs(poses: Poses) -> list[tuple[int, int]]:
    """Pairs of passes whose frame origins come within OVERLAP_RADIUS_M of each other."""
    pass_ids = sorted(int(p) for p in np.unique(poses.pass_id))
    trees = {}
    for p in pass_ids:
        org = poses.origin[poses.pass_id == p][:, :2]
        trees[p] = cKDTree(org)
    pairs = []
    for a, b in combinations(pass_ids, 2):
        d, _ = trees[a].query(poses.origin[poses.pass_id == b][:, :2], k=1)
        if d.min() <= OVERLAP_RADIUS_M:
            pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------------------- JVF datum
def load_road_boundary() -> list[np.ndarray]:
    """3D polylines (E,N,H) of the JVF road-boundary code, own object list (cached process-wide)."""
    from geovap.runtime import settings

    ref = settings.get().reference
    if ref is None:
        return []
    objs = ref.objects()
    return [o.coords for o in objs if o.code == ROAD_BOUNDARY_CODE and o.geom_type == "LineString" and len(o.coords) >= 2]


RESAMPLE_STEP_M = 0.1  # dense resampling of the JVF polylines for the nearest-sample KD-tree


def _resample_road(segs: list[np.ndarray], step: float = RESAMPLE_STEP_M) -> np.ndarray:
    """Densely resample all polylines to points step m apart (E, N, H) so a KD-tree nearest-neighbour
    query approximates point-to-segment distance (exact to within step/2, well under the 1 m gate)."""
    out = []
    for poly in segs:
        for a, b in zip(poly[:-1], poly[1:]):
            L = np.linalg.norm(b[:2] - a[:2])
            if L < 1e-9:
                out.append(a[None])
                continue
            n = max(1, int(np.ceil(L / step)))
            t = np.linspace(0.0, 1.0, n + 1)[:, None]
            out.append(a[None, :] + t * (b - a)[None, :])
    return np.concatenate(out) if out else np.empty((0, 3))


class RoadBoundaryIndex:
    """cKDTree over a dense resampling of the JVF road-boundary polylines, 2D nearest-neighbour with
    the sample's own height as an approximation of point-to-segment distance/z (see `_resample_road`)."""

    def __init__(self, segs: list[np.ndarray]):
        self.samples = _resample_road(segs)
        self.tree = cKDTree(self.samples[:, :2]) if len(self.samples) else None

    def query(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """pts[N,>=2] (E,N,...) -> (dist2d[N], z_nearest[N])."""
        if self.tree is None or len(pts) == 0:
            return np.full(len(pts), np.inf), np.full(len(pts), np.nan)
        d, j = self.tree.query(pts[:, :2], k=1)
        return d, self.samples[j, 2]


def _point_segment_dist2d(pts: np.ndarray, segs) -> tuple[np.ndarray, np.ndarray]:
    """pts[N,3] -> (dist[N], z_at_nearest[N]) vs the road boundary. `segs` is either a list of
    polylines (a `RoadBoundaryIndex` is built on the fly) or an already-built `RoadBoundaryIndex`."""
    index = segs if isinstance(segs, RoadBoundaryIndex) else RoadBoundaryIndex(segs)
    return index.query(pts)


JVF_GATES_M = (1.0, 0.5, 0.3)  # shrinking-gate rounds, same idea as register_pair


def jvf_offset(pass_id: int, patches: Patches | None = None, road_lines: list[np.ndarray] | None = None) -> dict:
    """Own-pass curb points vs JVF road-boundary polylines. Robust 2D shift (dE, dN) minimising
    point-to-segment distance, shrinking-gate rounds (`JVF_GATES_M`, 1.0 -> 0.3 m) so a handful of
    unrelated terrain breaks caught at the first, loose gate cannot bias the final fit; dH from
    ground z vs the JVF line z at the matched location, at the final (tightest) gate.

    Superseded as the absolute datum by `jvf_offset_photometric` (see its docstring): on this rural
    dataset a ploughed field right up against the road produces height-jump "curb" candidates over
    its whole surface, not just along the true edge, and `CURB_JVF_CORRIDOR_M`/`TRAJ_CORRIDOR_M`
    reduce but do not remove that -- both gates only ask "is this near *a* road/near the pass' own
    path", never "does the local geometry look like a kerb". `register_pair` still uses this
    function's CLS_CURB points for the pairwise horizontal DoF (facade/curb weighting), where being
    outnumbered by ground/facade voxels and cross-checked against another pass' own curb set is much
    less exploitable than an unconstrained one-sided match to a static reference layer."""
    if patches is None:
        patches = Patches.load(PASS_REG_DIR / f"patches_p{pass_id:02d}.npz")
    if road_lines is None:
        road_lines = load_road_boundary()
    curb = patches.select(patches.cls == CLS_CURB)
    if len(curb) < 10:
        return {"pass_id": pass_id, "n": 0, "dE": 0.0, "dN": 0.0, "dH": 0.0, "rms": None}
    pts = curb.c
    index = road_lines if isinstance(road_lines, RoadBoundaryIndex) else RoadBoundaryIndex(road_lines)
    x = np.zeros(2)
    for gate in JVF_GATES_M:
        p_shift = pts.copy()
        p_shift[:, :2] += x
        d, _ = index.query(p_shift)
        mask = d <= gate
        if mask.sum() < 10:
            break

        def resid(dx, m=mask):
            p2 = pts[m, :2] + x + dx
            d2, _ = index.query(p2)
            return d2

        sol = least_squares(resid, np.zeros(2), loss="soft_l1", f_scale=max(gate * 0.3, 0.03), max_nfev=100)
        x = x + sol.x
    p_final = pts.copy()
    p_final[:, :2] += x
    d_final, z_final = index.query(p_final)
    gate_final = JVF_GATES_M[-1]
    inl = d_final <= gate_final
    n = int(inl.sum())
    rms = float(np.sqrt(np.mean(d_final[inl] ** 2))) if n else None
    dH = float(np.median(z_final[inl] - p_final[inl, 2])) if n else 0.0
    return {"pass_id": pass_id, "n": n, "dE": float(x[0]), "dN": float(x[1]), "dH": dH, "rms": rms}


# ------------------------------------------------------------------------------ photometric JVF datum
# `jvf_offset` (cloud curbs vs JVF) is unreliable on this dataset (see its docstring); the alternative
# the plan/memory note proposes is a per-pass (E, N) offset chosen directly by photo evidence: shift
# the (static) JVF road-boundary samples and score them against each frame's own Canny edges -- the
# exact statistic `mapping.seg.nearfield` already uses to flag misaligned frames -- picking the shift
# that minimises the median band-to-edge distance. This never touches the point cloud, so it cannot be
# fooled by ground-height texture (the pass_reg problem this module opens with); it can still be
# fooled by ground-*image* texture (grass/gravel produce dense short Canny fragments, and "distance to
# the nearest edge" is small almost anywhere in clutter) -- `MIN_EDGE_COMPONENT_PX` below is the
# mitigation, and `PHOTO_TRAJ_CORRIDOR_M` keeps the candidate points on the road actually driven.
# ROWS/FLAG_PX match `seg.nearfield` so the two are directly comparable.
PHOTO_ROWS = (500, 850)  # ground rows, `mapping.seg.nearfield.ROWS`
PHOTO_FRAMES_PER_PASS = 6
PHOTO_TRAJ_CORRIDOR_M = 6.0  # tighter than TRAJ_CORRIDOR_M (12 m, used for the cloud curb detector):
# a wide corridor pulls in the opposite lane's curb, driveways and side-street boundaries, all
# candidate "edges" the grid search could just as easily lock onto; 6 m covers a typical two-lane
# road's own kerbs on both sides without much else.
PHOTO_COARSE_HALF_M = 0.8
PHOTO_COARSE_STEP_M = 0.1
PHOTO_FINE_HALF_M = 0.12
PHOTO_FINE_STEP_M = 0.02
PHOTO_MIN_PTS = 20  # minimum projected road-boundary samples in the ground rows for a frame to score
PHOTO_MAX_ROAD_PTS = 3000  # subsample the (0.1 m-dense) corridor samples above this: the score is a
# median over pixels, not a fit, so it does not need every sample -- a busy pass' corridor can hold
# 60k+ points at 0.1 m spacing, which made an early version of this grid search take tens of seconds
# per pass (458-627 `world_to_pano` calls x that many points); 3000 keeps each call sub-millisecond.


MIN_EDGE_COMPONENT_PX = 40  # drop connected Canny components smaller than this before the distance
# transform. Raw Canny on grass/gravel/ploughed ground produces dense short edge fragments everywhere
# (see module docstring "curb corridor" -- the same terrain that fools the cloud curb detector also
# fools a naive edge-distance score: a candidate shifted OFF the road and INTO textured verge can look
# like a *better* match than the true curb, since "distance to the nearest edge" is small almost
# everywhere in clutter). Kerbs, lane markings and building/fence lines are long, continuous
# components; isolated texture is not -- filtering by connected-component size keeps the former.


def _photo_edge_dist(poses: Poses, k: int, rows: tuple[int, int] = PHOTO_ROWS, min_component_px: int = MIN_EDGE_COMPONENT_PX) -> np.ndarray:
    """Canny-edge distance transform (px) of frame `k`'s photo at z-buffer (2000x1000) resolution --
    same computation as `mapping.seg.nearfield.band_edge_distance`, plus the component-size filter
    above; factored out so a per-frame result can be reused across every (dE, dN) grid point instead
    of being recomputed for each one."""
    photo = cv2.imread(pano_path(poses, k))
    photo = cv2.resize(photo, (ZB_W, ZB_H), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = (cv2.Canny(g, 40, 100) > 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component_px
    edges = keep[lbl]
    return ndimage.distance_transform_edt(~edges)


def jvf_offset_photometric(
    pass_id: int,
    poses: Poses,
    fi: FrameIndex,
    road_index: "RoadBoundaryIndex",
    frames: list[int] | None = None,
    n_frames: int = PHOTO_FRAMES_PER_PASS,
    rows: tuple[int, int] = PHOTO_ROWS,
) -> dict:
    """Per-pass (dE, dN) by 2D grid search: shift `road_index`'s dense road-boundary samples, project
    with the pass' own (unshifted, export) camera poses, and score the median distance to each frame's
    own Canny edge (`_photo_edge_dist`) in the ground rows. Coarse grid (`PHOTO_COARSE_*`, +-0.8 m /
    0.1 m) then a local refine (`PHOTO_FINE_*`, +-0.12 m / 0.02 m) around the coarse minimum -- two
    stages so the search stays a few hundred evaluations instead of the (1.6/0.05)^2 ~= 1000 a single
    0.05 m-step ±0.8 m grid would need per pass. `dH` comes from the (still corridor-gated, see
    `jvf_offset`) own-pass curb points, refit with a *tight* 0.3 m gate at the found (dE, dN): once the
    horizontal shift is right, only points genuinely on the boundary line survive that gate, which the
    photometric method itself cannot supply (a road edge in the photo is one line, not two, so it has
    no vertical-parallax signal for height).

    Sign convention (task S5 step 1 "be explicit which"): the grid search finds `x`, the shift applied
    to the *static* JVF samples (camera pose left at its raw/export value) that best matches the photo
    -- i.e. it measures how far the raw pose itself is displaced from truth, `x = C_raw - C_true`
    (worked out from `world_to_pano(P + x, R, C) == world_to_pano(P, R, C - x)`, exact for a pure
    translation with fixed `R`). The pass correction `apply_pass_transforms` needs is the opposite:
    `t = C_true - C_raw = -x`, applied to the CAMERA/CLOUD (JVF itself is never moved) -- so this
    function negates the grid-search result before using it for the height fit or returning it; every
    caller downstream (`solve_global`, `apply_pass_transforms`) receives `t`, matching `jvf_offset`'s
    convention (there `x` is already added directly to the pass' own curb points, so no such flip)."""
    idx_all = np.flatnonzero(poses.pass_id == pass_id)
    if len(idx_all) == 0 or road_index.tree is None:
        return {"pass_id": pass_id, "n_frames": 0, "dE": 0.0, "dN": 0.0, "dH": 0.0, "rms": None, "n": 0, "score_px": None, "method": "photometric"}
    if frames is None:
        frames = idx_all[np.linspace(0, len(idx_all) - 1, min(n_frames, len(idx_all))).round().astype(int)].tolist()

    org = poses.origin[idx_all]
    d_traj, _ = cKDTree(org[:, :2]).query(road_index.samples[:, :2], k=1)
    near = road_index.samples[d_traj <= PHOTO_TRAJ_CORRIDOR_M]
    if len(near) < PHOTO_MIN_PTS:
        return {"pass_id": pass_id, "n_frames": 0, "dE": 0.0, "dN": 0.0, "dH": 0.0, "rms": None, "n": 0, "score_px": None, "method": "photometric"}
    if len(near) > PHOTO_MAX_ROAD_PTS:
        near = near[np.random.default_rng(pass_id).choice(len(near), PHOTO_MAX_ROAD_PTS, replace=False)]

    from geovap.runtime import settings

    _frames_root = settings.get().workspace.frames_dir(poses)
    dist_maps, fps = {}, {}
    for k in frames:
        dist_maps[k] = _photo_edge_dist(poses, k, rows)
        try:
            fps[k] = FrameProducts.load(k, root=_frames_root)
        except Exception:
            fps[k] = None

    zb_scale = ZB_W / PANO_W

    def _project_valid(dE: float, dN: float, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shifted = near
        if dE or dN:
            shifted = near.copy()
            shifted[:, 0] += dE
            shifted[:, 1] += dN
        # FrameProducts.visible expects FULL-res (u, v) (it applies its own full-res -> z-buffer scale
        # internally, see FrameProducts.scale/visible) -- project at full res for that call, then scale
        # to z-buffer res ourselves for the row filter and the (2000x1000) dist_maps lookup.
        u, v, r, el = geometry.world_to_pano(shifted, fi.R[k], fi.C[k], PANO_W, PANO_H)
        uz, vz = u * zb_scale, v * zb_scale
        ok = (r > 0) & (uz >= 0) & (uz < ZB_W) & (vz >= rows[0]) & (vz < rows[1])
        fp = fps[k]
        if fp is not None and ok.any():
            ok = ok & fp.visible(r.astype(np.float32), u, v)
        return uz, vz, ok

    # baseline (zero-shift) support per frame: an offset that leaves a frame with much less of its own
    # baseline sample survives ONLY because the search pushed most of the corridor out of the visible
    # ground-row window / off the road surface (a depth-occlusion flip) -- the tiny, non-representative
    # remainder can land on the photo's Canny edges by pure chance and falsely look like a great fit.
    # An early version of this search had exactly that failure: every pass converged to (or past) the
    # +-0.92 m grid corner, not because that was the true offset, but because shrinking to a handful of
    # lucky points always drove the median down. Requiring >= 60% of each frame's own baseline count
    # keeps the comparison apples-to-apples across the whole grid.
    n0 = {}
    for k in frames:
        _, _, ok0 = _project_valid(0.0, 0.0, k)
        n0[k] = int(ok0.sum())

    def score(dE: float, dN: float) -> float | None:
        meds = []
        for k in frames:
            if n0[k] < PHOTO_MIN_PTS:
                continue
            uz, vz, ok = _project_valid(dE, dN, k)
            if ok.sum() < max(PHOTO_MIN_PTS, int(0.6 * n0[k])):
                continue
            uu = np.clip(uz[ok].astype(np.int64), 0, ZB_W - 1)
            vv = np.clip(vz[ok].astype(np.int64), 0, ZB_H - 1)
            meds.append(float(np.median(dist_maps[k][vv, uu])))
        if len(meds) < max(2, (len(frames) + 1) // 2):  # need most frames to agree, not a lucky one or two
            return None
        return float(np.median(meds))

    def grid_search(c_de: float, c_dn: float, half: float, step: float) -> tuple[float, float, float | None]:
        best_de, best_dn, best_s = c_de, c_dn, np.inf
        vals = np.arange(-half, half + 1e-9, step)
        for dE in c_de + vals:
            for dN in c_dn + vals:
                s = score(float(dE), float(dN))
                if s is not None and s < best_s:
                    best_de, best_dn, best_s = float(dE), float(dN), s
        return best_de, best_dn, (None if best_s == np.inf else best_s)

    dE0, dN0, s0 = grid_search(0.0, 0.0, PHOTO_COARSE_HALF_M, PHOTO_COARSE_STEP_M)
    # a coarse result sitting exactly on the +-0.8 m search edge means the search never turned around
    # -- the true optimum may lie further out, OR (the failure mode this module's fix history found
    # empirically) the objective is being gamed by shifting onto textured clutter outside the corridor.
    # Either way the result is not trustworthy; flag it rather than silently reporting a boundary value.
    boundary_hit = bool(np.isclose(abs(dE0), PHOTO_COARSE_HALF_M) or np.isclose(abs(dN0), PHOTO_COARSE_HALF_M))
    dE1, dN1, s1 = grid_search(dE0, dN0, PHOTO_FINE_HALF_M, PHOTO_FINE_STEP_M)
    if s1 is None:
        dE1, dN1, s1 = dE0, dN0, s0
    # x -> t: see the sign-convention paragraph in the docstring above.
    dE0, dN0 = -dE0, -dN0
    dE1, dN1 = -dE1, -dN1

    # tight-gate height fit at the found horizontal shift, from the (corridor-gated) curb points --
    # informational (`dH_n`/`dH_rms`) only; NOT what gates whether solve_global uses this constraint
    # (see PX_TO_M below), since curb-point support can be thin on short/sparse passes even when the
    # photometric horizontal fit itself is well determined by hundreds of frame-edge pixels.
    dH, n_h, rms_h = 0.0, 0, None
    try:
        patches = Patches.load(PASS_REG_DIR / f"patches_p{pass_id:02d}.npz")
        curb = patches.select(patches.cls == CLS_CURB)
        if len(curb) >= 5:
            p2 = curb.c.copy()
            p2[:, 0] += dE1
            p2[:, 1] += dN1
            d2, z2 = road_index.query(p2)
            inl = d2 <= 0.3
            n_h = int(inl.sum())
            if n_h >= 5:
                dH = float(np.median(z2[inl] - p2[inl, 2]))
                rms_h = float(np.sqrt(np.mean(d2[inl] ** 2)))
    except FileNotFoundError:
        pass

    # 1 px (ZB res) ~= 0.18 deg (`seg.nearfield`); at a typical near-field curb range of ~8 m that is
    # ~0.025 m of ground error -- used only as a relative-confidence proxy across passes for
    # `solve_global`'s weighting, not a claimed physical accuracy. A boundary hit gets a 10x rms
    # penalty (not exclusion -- the (dE, dN) value is still reported for QA) so `solve_global` leans on
    # this pass' pairwise constraints instead of trusting an unreliable datum fit for it.
    PX_TO_M = 0.02
    rms = max(s1, 3.0) * PX_TO_M if s1 is not None else None
    if rms is not None and boundary_hit:
        rms *= 10.0
    return {
        "pass_id": pass_id,
        "n_frames": len(frames),
        "dE": dE1,
        "dN": dN1,
        "dH": dH,
        "score_px": s1,
        "coarse": {"dE": dE0, "dN": dN0, "score_px": s0},
        "boundary_hit": boundary_hit,
        "n": 200 if s1 is not None else 0,  # nominal support, see PX_TO_M note
        "rms": rms,
        "dH_n": n_h,
        "dH_rms": rms_h,
        "n_road_pts": len(near),
        "method": "photometric",
    }


# ---------------------------------------------------------------------------------- global solve
@dataclass
class GlobalResult:
    node_ids: list[int]
    x: dict  # pass_id -> [dE, dN, dH, dyaw_deg]
    centre: dict  # pass_id -> [E, N]
    residuals: dict  # per-constraint residual after solve


def solve_global(
    pass_ids: list[int],
    pair_results: dict[tuple[int, int], dict],
    jvf_results: dict[int, dict],
    prior_sigma_m: float = 0.5,
    prior_sigma_deg: float = 0.5,
) -> GlobalResult:
    """Linear pose graph: 4-DoF state per pass node (dE, dN, dH, dyaw, all small -> linear).
    Pairwise constraint: x_b - x_a ~= T_pair(a,b) (weight 1/rms^2 * n). JVF constraint:
    x_p ~= (jvf.dE, jvf.dN, jvf.dH, 0) (weight n / rms^2). Weak identity prior (x_p ~= 0) with
    sigma (prior_sigma_m, prior_sigma_deg) stabilises components no constraint touches (e.g. dyaw
    on an isolated straight-road pass)."""
    idx = {p: i for i, p in enumerate(pass_ids)}
    n = len(pass_ids)
    dim = 4 * n
    rows, cols, vals, b = [], [], [], []
    row = 0

    def add_row(coeffs: dict[int, float], target: float, weight: float):
        nonlocal row
        for c, v in coeffs.items():
            rows.append(row)
            cols.append(c)
            vals.append(v * weight)
        b.append(target * weight)
        row += 1

    # weights are capped relative to the prior so no single well-observed constraint dominates the
    # conditioning of the combined (very different physical units) system; relative confidence between
    # constraints is preserved below the cap.
    w_cap = 50.0
    for (a, bb), res in pair_results.items():
        if not res.get("converged") or res.get("rms_after") is None:
            continue
        rms = max(res["rms_after"], 0.01)
        weight = min(np.sqrt(res["n"]) / rms, w_cap)
        Tab = res["T"]  # transform bringing b's patches onto a's frame: x_b_frame -> a
        for k in range(4):
            add_row({4 * idx[bb] + k: 1.0, 4 * idx[a] + k: -1.0}, Tab[k], weight)

    for p, jr in jvf_results.items():
        if p not in idx or jr.get("rms") is None or jr["n"] < 10:
            continue
        rms = max(jr["rms"], 0.02)
        weight = min(np.sqrt(jr["n"]) / rms, w_cap)
        add_row({4 * idx[p] + 0: 1.0}, jr["dE"], weight)
        add_row({4 * idx[p] + 1: 1.0}, jr["dN"], weight)
        add_row({4 * idx[p] + 2: 1.0}, jr["dH"], weight)

    w_prior_m = 1.0 / prior_sigma_m
    w_prior_deg = 1.0 / prior_sigma_deg
    for p in pass_ids:
        i = idx[p]
        add_row({4 * i + 0: 1.0}, 0.0, w_prior_m)
        add_row({4 * i + 1: 1.0}, 0.0, w_prior_m)
        add_row({4 * i + 2: 1.0}, 0.0, w_prior_m)
        add_row({4 * i + 3: 1.0}, 0.0, w_prior_deg)

    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    A = coo_matrix((vals, (rows, cols)), shape=(row, dim)).tocsr()
    b = np.array(b)
    sol = lsqr(A, b, atol=1e-10, btol=1e-10, iter_lim=2000)[0]
    x = {p: sol[4 * idx[p] : 4 * idx[p] + 4].tolist() for p in pass_ids}
    centre = {}
    for (a, bb), res in pair_results.items():
        if "centre" in res:
            centre.setdefault(bb, res["centre"])

    # unweighted, physical-unit residuals per constraint type (the weighted A@x-b above mixes wildly
    # different physical units/weights and is not itself interpretable)
    pair_res_m, pair_res_deg, jvf_res_m = [], [], []
    for (a, bb), res in pair_results.items():
        if not res.get("converged") or res.get("rms_after") is None:
            continue
        xa, xb, T = np.array(x[a]), np.array(x[bb]), np.array(res["T"])
        d = (xb - xa) - T
        pair_res_m.append(float(np.linalg.norm(d[:3])))
        pair_res_deg.append(float(abs(d[3])))
    for p, jr in jvf_results.items():
        if p not in idx or jr.get("rms") is None or jr["n"] < 10:
            continue
        xp = np.array(x[p])
        d = xp[:2] - np.array([jr["dE"], jr["dN"]])
        jvf_res_m.append(float(np.linalg.norm(d)))
    residuals = {
        "pairwise_m_median": float(np.median(pair_res_m)) if pair_res_m else None,
        "pairwise_m_max": float(np.max(pair_res_m)) if pair_res_m else None,
        "pairwise_deg_median": float(np.median(pair_res_deg)) if pair_res_deg else None,
        "jvf_m_median": float(np.median(jvf_res_m)) if jvf_res_m else None,
        "jvf_m_max": float(np.max(jvf_res_m)) if jvf_res_m else None,
        "n_pair_constraints": len(pair_res_m),
        "n_jvf_constraints": len(jvf_res_m),
    }
    return GlobalResult(node_ids=pass_ids, x=x, centre=centre, residuals=residuals)


# --------------------------------------------------------------------------------- application
def apply_pass_transforms(poses: Poses, transforms: dict) -> Poses:
    """Apply a `pass_transforms.json`-shaped dict (pass -> {t:[dE,dN,dH], yaw_deg, centre:[E,N]}) to a
    Poses table: C' = R_p (C - centre) + centre + t_p, R_v' = R_v R_p^T -> Euler via
    geometry.euler_from_vehicle_rotation. Trajectory (if any) is left untouched (S3 not applied here);
    callers that also carry a dense trajectory must transform it identically before attaching."""
    origin = poses.origin.copy()
    roll = poses.roll.copy()
    pitch = poses.pitch.copy()
    yaw = poses.yaw.copy()
    R_v_all = geometry.vehicle_rotation(poses.yaw, poses.roll, poses.pitch)
    for p_str, tr in transforms.get("passes", transforms).items():
        try:
            p = int(p_str)
        except ValueError:
            continue
        sel = poses.pass_id == p
        if not sel.any():
            continue
        dE, dN, dH = tr["t"]
        dyaw = tr["yaw_deg"]
        cx, cy = tr.get("centre", [poses.origin[sel, 0].mean(), poses.origin[sel, 1].mean()])
        Rp = _rot_yaw(np.radians(dyaw))
        C = origin[sel]
        Cc = C - np.array([cx, cy, 0.0])
        C2 = Cc @ Rp.T + np.array([cx, cy, 0.0]) + np.array([dE, dN, dH])
        origin[sel] = C2
        Rv = R_v_all[sel]
        Rv2 = Rv @ Rp.T
        y2, r2, p2 = geometry.euler_from_vehicle_rotation(Rv2)
        yaw[sel], roll[sel], pitch[sel] = y2, r2, p2
    return Poses(
        filename=poses.filename.copy(),
        t=poses.t.copy(),
        origin=origin,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        pass_id=poses.pass_id.copy(),
        speed=poses.speed.copy(),
        source=poses.source,
    )
