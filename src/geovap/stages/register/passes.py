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
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from geovap.domain.model import geometry
from geovap.runtime.store import CloudStore
from geovap.domain.model.frames import FrameIndex
from geovap.domain.model.poses import Poses
from geovap.runtime.pose_tables import load as load_poses
from geovap.runtime.panos import pano_path
from geovap.stages.prepare.products import FrameProducts
from geovap.stages.base.cli import stage_main
from geovap.stages.base.spec import StageSpec, registry

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings


def pass_reg_dir(s: "Settings | None" = None) -> Path:
    """`s.workspace.out / "pass_reg"` (was the module-level `PASS_REG_DIR = OUT_DIR / "pass_reg"`,
    which froze the dataset at import time -- see `tests/test_no_import_time_settings.py`). Every
    caller below resolves this late, inside a function body."""
    from geovap.runtime import settings

    return (s or settings.get()).workspace.out / "pass_reg"

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


def _load_pass_psid(s: "Settings | None" = None) -> dict:
    from geovap.runtime import settings

    p = (s or settings.get()).workspace.poses / "pass_psid.json"
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
    out = pass_reg_dir() / f"patches_p{pass_id:02d}.npz"
    pat.save(out)
    store.release()
    counts = {int(c): int((pat.cls == c).sum()) for c in np.unique(pat.cls)} if len(pat) else {}
    used_psid = sorted(psid_filter) if psid_filter else None
    return pass_id, {"n": len(pat), "counts": counts, "used_psid": used_psid}


def run_extract_all(poses: Poses | None = None, workers: int = 8, store_root: Path | None = None, s: "Settings | None" = None) -> dict:
    """Extract and cache patches for all 30 passes. Returns per-pass counts report."""
    from geovap.runtime import settings

    s = s or settings.get()
    store_root = store_root if store_root is not None else s.workspace.store
    poses = poses or load_poses(s=s)
    d = pass_reg_dir(s)
    d.mkdir(parents=True, exist_ok=True)
    pass_ids = [int(p) for p in np.unique(poses.pass_id)]
    args = [(pid, store_root, True) for pid in pass_ids]
    report = {}
    with Pool(min(workers, 8)) as pool:
        for pid, info in pool.imap_unordered(_extract_one, args):
            report[pid] = info
            print(f"pass {pid:2d}: n={info['n']:7d} counts={info['counts']} psid={info['used_psid']}")
    (d / "patches_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
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
        patches = Patches.load(pass_reg_dir() / f"patches_p{pass_id:02d}.npz")
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
# ROWS/FLAG_PX match `geovap.stages.semantics.pseudogt.nearfield` so the two are directly comparable
# -- duplicated, not imported: stage groups talk to each other through artifacts on disk, never
# through each other's Python (`.importlinter`'s stage-independence contract).
PHOTO_ROWS = (500, 850)  # ground rows, `nearfield.ROWS` convention
NEARFIELD_FLAG_PX = 20.0  # `nearfield.FLAG_PX` convention
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
    from geovap.runtime import settings

    _sensor = settings.get().sensor
    photo = cv2.imread(pano_path(poses, k))
    photo = cv2.resize(photo, (_sensor.zb_w, _sensor.zb_h), interpolation=cv2.INTER_AREA)
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

    _s = settings.get()
    _sensor = _s.sensor
    _frames_root = _s.workspace.frames_dir(poses)
    dist_maps, fps = {}, {}
    for k in frames:
        dist_maps[k] = _photo_edge_dist(poses, k, rows)
        try:
            fps[k] = FrameProducts.load(k, root=_frames_root)
        except Exception:
            fps[k] = None

    zb_scale = _sensor.zb_w / _sensor.pano_w

    def _project_valid(dE: float, dN: float, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shifted = near
        if dE or dN:
            shifted = near.copy()
            shifted[:, 0] += dE
            shifted[:, 1] += dN
        # FrameProducts.visible expects FULL-res (u, v) (it applies its own full-res -> z-buffer scale
        # internally, see FrameProducts.scale/visible) -- project at full res for that call, then scale
        # to z-buffer res ourselves for the row filter and the (2000x1000) dist_maps lookup.
        u, v, r, el = geometry.world_to_pano(shifted, fi.R[k], fi.C[k], _sensor.pano_w, _sensor.pano_h)
        uz, vz = u * zb_scale, v * zb_scale
        ok = (r > 0) & (uz >= 0) & (uz < _sensor.zb_w) & (vz >= rows[0]) & (vz < rows[1])
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
            uu = np.clip(uz[ok].astype(np.int64), 0, _sensor.zb_w - 1)
            vv = np.clip(vz[ok].astype(np.int64), 0, _sensor.zb_h - 1)
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
        patches = Patches.load(pass_reg_dir() / f"patches_p{pass_id:02d}.npz")
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


# ================================================================================================
# `mapping/cli/register_passes.py`, folded in below (its `cmd_*` functions become plain library
# functions here -- no more `pr.` prefix, this module IS "pr" now) as the `register` stage.

def cmd_extract(workers: int = 8, s: "Settings | None" = None):
    run_extract_all(workers=workers, s=s)


def cmd_pairs(workers: int = 8, s: "Settings | None" = None) -> dict:
    from geovap.runtime import settings

    s = s or settings.get()
    d = pass_reg_dir(s)
    poses = load_poses(s=s)
    pairs = overlap_pairs(poses)
    cache = {}

    def get(pid):
        if pid not in cache:
            cache[pid] = Patches.load(d / f"patches_p{pid:02d}.npz")
        return cache[pid]

    out = {}
    for a, b in pairs:
        Pa, Pb = get(a), get(b)
        res = register_pair(Pb, Pa)  # query=b onto ref=a
        out[f"{a}_{b}"] = res
        print(f"{a:2d}-{b:2d}: n={res['n']:5d} rms {res['rms_before']} -> {res['rms_after']}  T={np.round(res['T'], 3).tolist()} sv_min={res['sv_min']:.3g}")
    (d / "pairs.json").write_text(json.dumps(out, indent=2))
    return out


def cmd_jvf(s: "Settings | None" = None) -> dict:
    from geovap.runtime import settings

    s = s or settings.get()
    d = pass_reg_dir(s)
    poses = load_poses(s=s)
    pass_ids = sorted(int(p) for p in np.unique(poses.pass_id))
    road_lines = load_road_boundary()
    index = RoadBoundaryIndex(road_lines)
    print(f"road boundary: {len(road_lines)} polylines, {len(index.samples)} resampled points")
    out = {}
    for p in pass_ids:
        pat = Patches.load(d / f"patches_p{p:02d}.npz")
        res = jvf_offset(p, pat, index)
        out[p] = res
        anchor = res["rms"] is not None and max(abs(res["dE"]), abs(res["dN"])) < 0.1
        print(f"pass {p:2d}: n={res['n']:5d} dE={res['dE']:+.3f} dN={res['dN']:+.3f} dH={res['dH']:+.3f} rms={res['rms']} anchor={anchor}")
    (d / "jvf_offsets.json").write_text(json.dumps(out, indent=2))
    return out


def cmd_jvf_photo(n_frames: int = PHOTO_FRAMES_PER_PASS, s: "Settings | None" = None) -> dict:
    """Per-pass (dE, dN[, dH]) by photo-edge grid search (`jvf_offset_photometric`) -- this is the
    JVF datum `solve` actually consumes; see this module's docstring / `jvf_offset` docstring for
    why the cloud-curb version (`cmd_jvf`) is kept only as a secondary/informational signal."""
    from geovap.runtime import settings

    s = s or settings.get()
    d = pass_reg_dir(s)
    poses = load_poses(s=s)
    fi = FrameIndex(poses)
    pass_ids = sorted(int(p) for p in np.unique(poses.pass_id))
    road_lines = load_road_boundary()
    index = RoadBoundaryIndex(road_lines)
    print(f"road boundary: {len(road_lines)} polylines, {len(index.samples)} resampled points")
    out = {}
    for p in pass_ids:
        res = jvf_offset_photometric(p, poses, fi, index, n_frames=n_frames)
        out[p] = res
        anchor = res["score_px"] is not None and max(abs(res["dE"]), abs(res["dN"])) < 0.1
        print(f"pass {p:2d}: n_frames={res['n_frames']:2d} dE={res['dE']:+.3f} dN={res['dN']:+.3f} dH={res['dH']:+.3f} score_px={res['score_px']} dH_n={res['dH_n']} anchor={anchor}")
    (d / "jvf_offsets_photo.json").write_text(json.dumps(out, indent=2))
    return out


DATUM_CHOICES = ("none", "jvf-cloud", "jvf-photo")


def cmd_solve(datum: str = "none", s: "Settings | None" = None) -> dict:
    """`datum`: "none" (production default -- pairwise-only, weak identity prior, no absolute JVF
    constraint; see this module's docstring for why), "jvf-cloud" (jvf_offsets.json, informational
    -- see jvf_offset docstring for why it is unreliable), or "jvf-photo" (jvf_offsets_photo.json,
    the best absolute-datum attempt tried -- still not adopted, see jvf_offset_photometric docstring)."""
    from geovap.runtime import settings

    if datum not in DATUM_CHOICES:
        raise ValueError(f"--datum must be one of {DATUM_CHOICES}, got {datum!r}")
    s = s or settings.get()
    d = pass_reg_dir(s)
    poses = load_poses(s=s)
    pass_ids = sorted(int(p) for p in np.unique(poses.pass_id))
    pairs_raw = json.loads((d / "pairs.json").read_text())
    pair_results = {}
    for k, v in pairs_raw.items():
        a, b = (int(x) for x in k.split("_"))
        pair_results[(a, b)] = v

    jvf_results: dict[int, dict] = {}
    jvf_source = "none"
    if datum == "jvf-photo":
        jvf_source = "photometric"
        jvf_raw = json.loads((d / "jvf_offsets_photo.json").read_text())
        jvf_results = {int(k): v for k, v in jvf_raw.items()}
    elif datum == "jvf-cloud":
        jvf_source = "cloud_curb"
        jvf_raw = json.loads((d / "jvf_offsets.json").read_text())
        jvf_results = {int(k): v for k, v in jvf_raw.items()}

    gr = solve_global(pass_ids, pair_results, jvf_results)

    out = {"passes": {}, "summary": {**gr.residuals, "n_pairs": len(pair_results), "n_jvf_anchors": 0, "jvf_source": jvf_source, "datum": datum}, "poses_source": poses.source}
    n_anchor = 0
    t_all, yaw_all = [], []
    for p in pass_ids:
        x = gr.x[p]
        jr = jvf_results.get(p, {})
        is_anchor = jr.get("rms") is not None and max(abs(jr.get("dE", 0)), abs(jr.get("dN", 0))) < 0.1
        n_anchor += int(is_anchor)
        n_pairs_p = sum(1 for (a, b) in pair_results if a == p or b == p)
        cx, cy = gr.centre.get(p, [float(poses.origin[poses.pass_id == p, 0].mean()), float(poses.origin[poses.pass_id == p, 1].mean())])
        flag = bool(np.linalg.norm(x[:3]) > 0.3 or abs(x[3]) > 0.3)
        t_all.append(x[:3])
        yaw_all.append(x[3])
        out["passes"][str(p)] = {
            "yaw_deg": x[3],
            "t": [x[0], x[1], x[2]],
            "centre": [cx, cy],
            "rms": jr.get("rms"),
            "n_pairs": n_pairs_p,
            "anchor": is_anchor,
            "flag": flag,
            "jvf": jr,
        }
    out["summary"]["n_jvf_anchors"] = n_anchor
    # gauge check (datum=none): the weak identity prior alone fixes the global shift -- the mean
    # transform over all 30 nodes should sit near zero (no absolute datum is pulling it anywhere else).
    t_all = np.array(t_all)
    out["summary"]["mean_t"] = t_all.mean(axis=0).tolist()
    out["summary"]["mean_yaw_deg"] = float(np.mean(yaw_all))
    out["summary"]["n_flagged"] = sum(1 for v in out["passes"].values() if v["flag"])
    (d / "pass_transforms.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    return out


def cmd_poses(s: "Settings | None" = None):
    from geovap.runtime import settings
    from geovap.runtime.pose_tables import write as write_pose_table

    s = s or settings.get()
    d = pass_reg_dir(s)
    poses = load_poses(s=s)
    transforms = json.loads((d / "pass_transforms.json").read_text())
    poses2 = apply_pass_transforms(poses, transforms)
    meta = {
        "status": np.array(["registered"] * len(poses2), dtype=object),
        "src": np.array(["pass_reg"] * len(poses2), dtype=object),
        "dt_s": np.zeros(len(poses2)),
    }
    out = write_pose_table(
        poses2,
        meta,
        {
            "stage": "S5_pass_reg",
            "input_poses_hash": poses.hash(),
            "pass_transforms": str(d / "pass_transforms.json"),
            "datum": transforms.get("summary", {}).get("datum", "unknown"),
        },
        d / "poses_export_registered.csv",
    )
    print("wrote", out)
    return out


def run_register_all(datum: str = "none", workers: int = 8, s: "Settings | None" = None) -> None:
    """`register_passes.py all`: extract, pairs, jvf, jvf_photo, solve, poses -- unchanged order."""
    cmd_extract(workers=workers, s=s)
    cmd_pairs(workers=workers, s=s)
    cmd_jvf(s=s)
    cmd_jvf_photo(s=s)
    cmd_solve(datum=datum, s=s)
    cmd_poses(s=s)


# ================================================================================================
# `mapping/cli/validate_pass_reg.py`, folded in below as the `reg-conflict` stage (`run_conflict_metric`)
# plus its QA/near-field helpers, kept as library functions for ad hoc use (`main()` below).
#
# S5 validation / QA: sign-and-convention render check, the near-field acceptance metric
# before/after registration, and the cross-pass silhouette ("double surface") conflict metric of
# `geovap.stages.register.screen` before/after. Read-only user of `mapping.seg.nearfield` (the
# near-field metric definition) and `screen` (the pass-conflict silhouette residual
# `_silhouette_points`/`_residual`/`_photo_edges` and its `GEO_MIN_POINTS`/`CONFLICT_PX`
# thresholds -- reused, not redefined, so "conflict" means the same thing here as in
# `frame_quality.csv`); does not modify either.

def pass_reg_qa_dir(s: "Settings | None" = None) -> Path:
    return pass_reg_dir(s) / "qa"


def _road_objects(s: "Settings | None" = None):
    from geovap.runtime import settings

    ref = (s or settings.get()).reference
    if ref is None:
        return []
    objs = ref.objects()
    return [o for o in objs if o.code == ROAD_BOUNDARY_CODE and o.geom_type == "LineString" and len(o.coords) >= 2]


def _band_mask(objs, R, C, fp, scale: float) -> np.ndarray:
    from geovap.stages.prepare import vectors

    mask, _occ = vectors.render_objects(objs, R, C, fp, {ROAD_BOUNDARY_CODE: 1}, scale=scale)
    return mask


def _draw_band(photo: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = photo.copy()
    d = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    out[d > 0] = color
    return out


def _draw_curb_points(photo: np.ndarray, u: np.ndarray, v: np.ndarray, ok: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    out = photo.copy()
    for x, y in zip(u[ok], v[ok]):
        cv2.circle(out, (int(round(x)), int(round(y))), 2, color, -1)
    return out


def render_qa(pass_ids: list[int] = (0, 5), n_frames: int = 3, rows: tuple[int, int] | None = None, out_dir: Path | None = None, s: "Settings | None" = None) -> list[dict]:
    """For each pass, `n_frames` evenly spaced frames: three stacked crops (ground rows only) --
    (a) photo + JVF road boundary projected with the EXPORT pose, (b) photo + this pass' own curb
    points (also export pose -- curb points are raw store coordinates, never shifted), (c) photo +
    JVF projected with the pose AFTER the S5 transform (`apply_pass_transforms`); JVF itself is never
    moved -- see module docstring / `apply_pass_transforms`. If (c) sits closer to the photo
    edge than (a), the transform is corrective; if it sits further, the transform (or its sign) is
    wrong for that pass."""
    from geovap.runtime import settings

    s = s or settings.get()
    rows = rows or PHOTO_ROWS
    out_dir = out_dir or pass_reg_qa_dir(s)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = load_poses(s=s)
    fi = FrameIndex(poses)
    d = pass_reg_dir(s)
    transforms = json.loads((d / "pass_transforms.json").read_text())
    poses_corr = apply_pass_transforms(poses, transforms)
    fi_corr = FrameIndex(poses_corr)
    objs = _road_objects(s)
    scale = s.sensor.zb_w / 8000.0

    report = []
    for pid in pass_ids:
        idx_all = np.flatnonzero(poses.pass_id == pid)
        picks = idx_all[np.linspace(0, len(idx_all) - 1, n_frames).round().astype(int)]
        patches = Patches.load(d / f"patches_p{pid:02d}.npz")
        curb = patches.select(patches.cls == CLS_CURB)
        tr = transforms.get("passes", transforms).get(str(pid))
        for k in picks.tolist():
            photo = cv2.imread(pano_path(poses, k))
            photo = cv2.resize(photo, (s.sensor.zb_w, s.sensor.zb_h), interpolation=cv2.INTER_AREA)
            try:
                fp = FrameProducts.load(k, root=s.workspace.frames_dir(poses))
            except Exception:
                fp = None

            mask_before = _band_mask(objs, fi.R[k], fi.C[k], fp, scale)
            mask_after = _band_mask(objs, fi_corr.R[k], fi_corr.C[k], fp, scale)

            near = curb.c[np.linalg.norm(curb.c[:, :2] - fi.C[k, :2], axis=1) < 25.0]
            u, v, r, el = geometry.world_to_pano(near, fi.R[k], fi.C[k], w=s.sensor.zb_w, h=s.sensor.zb_h)
            ok = (r > 0) & (u >= 0) & (u < s.sensor.zb_w) & (v >= rows[0]) & (v < rows[1])

            row_a = _draw_band(photo, mask_before, (0, 255, 255))  # yellow = JVF, export pose
            row_b = _draw_curb_points(photo, u, v, ok, (0, 0, 255))  # red = own-pass curb pts
            row_c = _draw_band(photo, mask_after, (255, 0, 255))  # magenta = JVF, S5-corrected pose

            crop = np.concatenate([row_a[rows[0]:rows[1]], row_b[rows[0]:rows[1]], row_c[rows[0]:rows[1]]], axis=0)
            label = f"pass {pid} frame {k}  a=JVF/export(yellow) b=own-curb(red) c=JVF/S5-corrected(magenta) t={tr['t'] if tr else None}"
            cv2.putText(crop, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            path = out_dir / f"p{pid:02d}_f{k:04d}.png"
            cv2.imwrite(str(path), crop)
            report.append({"pass_id": pid, "frame": k, "path": str(path), "n_band_before": int((mask_before > 0).sum()), "n_band_after": int((mask_after > 0).sum()), "n_curb_drawn": int(ok.sum())})
    (out_dir / "qa_report.json").write_text(json.dumps(report, indent=1))
    return report


# ------------------------------------------------------------------------------- near-field metric
def _band_edge_distance_from_mask(photo_bgr: np.ndarray, mask: np.ndarray, rows) -> dict:
    """Same statistic as `mapping.seg.nearfield.band_edge_distance`, but taking an already-rendered
    band mask (so it can be recomputed for the S5-corrected pose) instead of reading `bands_erp/*.png`
    off disk (those were rendered once, with the export pose only)."""
    g = cv2.GaussianBlur(cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(g, 40, 100) > 0
    dist = ndimage.distance_transform_edt(~edges)
    sel = np.zeros(mask.shape, bool)
    sel[rows[0]:rows[1]] = mask[rows[0]:rows[1]] > 0
    n = int(sel.sum())
    if n < 200:
        return {"n": n, "median_px": None}
    inside = ndimage.distance_transform_edt(sel)
    centre = sel & (inside >= np.maximum(1, ndimage.maximum_filter(inside, 5) - 0.5))
    d = dist[centre]
    return {"n": int(centre.sum()), "median_px": round(float(np.median(d)), 1)}


def run_metric(frames: list[int] | None = None, n_subset: int = 200, out: Path | None = None, s: "Settings | None" = None) -> dict:
    """The acceptance test (task 2): band-to-photo-edge distance (`seg.nearfield`'s own statistic),
    computed per frame with (i) export pose vs the static JVF band [-> should equal `segds/nearfield.json`
    where MIN_PX/rows agree] and (ii) the S5-corrected pose vs the same static JVF band. Frames are a
    fixed, pass-stratified `n_subset` sample of the clean set unless `frames` is given explicitly."""
    from geovap.runtime import settings

    # ROWS/FLAG_PX match `geovap.stages.semantics.pseudogt.nearfield`'s convention, duplicated here
    # rather than imported: stage groups talk to each other through artifacts on disk, never
    # through each other's Python (`.importlinter`'s stage-independence contract).
    ROWS, FLAG_PX = PHOTO_ROWS, NEARFIELD_FLAG_PX
    s = s or settings.get()
    d = pass_reg_dir(s)
    out = out or (d / "nearfield_reg.json")
    poses = load_poses(s=s)

    if frames is None:
        clean = np.array(sorted(s.workspace.clean_frames("clean")))
        rng = np.random.default_rng(0)
        pass_of = poses.pass_id[clean]
        frames = []
        for pid in np.unique(pass_of):
            pf = clean[pass_of == pid]
            take = max(1, round(n_subset * len(pf) / len(clean)))
            frames.extend(rng.choice(pf, size=min(take, len(pf)), replace=False).tolist())
        frames = sorted(set(frames))

    transforms = json.loads((d / "pass_transforms.json").read_text())
    poses_corr = apply_pass_transforms(poses, transforms)
    fi = FrameIndex(poses)
    fi_corr = FrameIndex(poses_corr)
    objs = _road_objects(s)
    scale = s.sensor.zb_w / 8000.0

    per_frame = {}
    for k in frames:
        photo = cv2.imread(pano_path(poses, k))
        photo = cv2.resize(photo, (s.sensor.zb_w, s.sensor.zb_h), interpolation=cv2.INTER_AREA)
        try:
            fp = FrameProducts.load(k, root=s.workspace.frames_dir(poses))
        except Exception:
            fp = None
        mask_before = _band_mask(objs, fi.R[k], fi.C[k], fp, scale)
        mask_after = _band_mask(objs, fi_corr.R[k], fi_corr.C[k], fp, scale)
        r_before = _band_edge_distance_from_mask(photo, mask_before, ROWS)
        r_after = _band_edge_distance_from_mask(photo, mask_after, ROWS)
        per_frame[int(k)] = {"pass_id": int(poses.pass_id[k]), "before": r_before, "after": r_after}

    by_pass: dict[int, dict] = {}
    for k, r in per_frame.items():
        pid = r["pass_id"]
        by_pass.setdefault(pid, {"before": [], "after": [], "flag_before": 0, "flag_after": 0, "n": 0})
        agg = by_pass[pid]
        agg["n"] += 1
        mb, ma = r["before"]["median_px"], r["after"]["median_px"]
        if mb is not None:
            agg["before"].append(mb)
            agg["flag_before"] += mb > FLAG_PX
        if ma is not None:
            agg["after"].append(ma)
            agg["flag_after"] += ma > FLAG_PX
    summary = {}
    for pid, agg in sorted(by_pass.items()):
        summary[str(pid)] = {
            "n": agg["n"],
            "n_measured_before": len(agg["before"]),
            "n_measured_after": len(agg["after"]),
            "median_px_before": round(float(np.median(agg["before"])), 1) if agg["before"] else None,
            "median_px_after": round(float(np.median(agg["after"])), 1) if agg["after"] else None,
            "n_flagged_before": agg["flag_before"],
            "n_flagged_after": agg["flag_after"],
        }
    all_before = [v["median_px"] for v in [r["before"] for r in per_frame.values()] if v["median_px"] is not None]
    all_after = [v["median_px"] for v in [r["after"] for r in per_frame.values()] if v["median_px"] is not None]
    overall = {
        "n_frames": len(frames),
        "median_px_before": round(float(np.median(all_before)), 1) if all_before else None,
        "median_px_after": round(float(np.median(all_after)), 1) if all_after else None,
        "n_flagged_before": int(sum(1 for v in all_before if v > FLAG_PX)),
        "n_flagged_after": int(sum(1 for v in all_after if v > FLAG_PX)),
        "n_measured_before": len(all_before),
        "n_measured_after": len(all_after),
    }
    out.write_text(json.dumps({"rows": ROWS, "flag_px": FLAG_PX, "overall": overall, "by_pass": summary, "frames": per_frame}, indent=1))
    print(json.dumps(overall, indent=1))
    print(json.dumps(summary, indent=1))
    return {"overall": overall, "by_pass": summary}


# --------------------------------------------------------------------- cross-pass conflict metric
# Task 2: `screen.assess_frame`'s own "pass_conflict" residual (du2/dv2 -- the second silhouette
# residual, computed from OTHER-pass points gathered in the same +-45 s window as the frame's own
# geometry check, see `screen.py` module docstring) is the metric pairwise registration is actually
# meant to fix (double surfaces), unlike the near-field metric above (which never touches a second
# pass at all). Every frame with `n_edge_other > 0` in `frame_quality.csv` is recomputed here with
# the OTHER-pass points and the frame's own camera pose transformed by the S5 solve (points via
# `PassRegistration.apply` = `T[poses.pass_of_time(gps_time)]`, camera via `apply_pass_transforms`);
# "before" is recomputed the same way with the identity transform (T=0) rather than read back from the
# CSV, so before/after use byte-identical candidate gathering / edge images and differ only in the
# transform applied -- a fair paired comparison.
#
# Caveat found while building this (see this module's docstring and its report): the frame-level
# own/other split in `screen.assess_frame` is a fixed +-2 s pad around each PHOTO pass's own [t0,
# t1], not `poses.pass_of_time`'s gap-midpoint pass boundary. Near a pass' start/end (vehicle
# slowing/turning, camera not yet firing but the scanner still running) points screen.py calls
# "other" can still resolve to the SAME pass under `pass_of_time` -- for those frames pairwise
# registration moves the camera and those points by the identical transform, so du2/dv2 are invariant
# by construction (see the geometry invariance test) and correctly do not change. `frac_true_other`
# records, per frame, the fraction of its "other" candidate points that resolve to a genuinely
# different pass under `pass_of_time`; the headline before/after comparison is restricted to frames
# with `frac_true_other >= MIN_FRAC_TRUE_OTHER`, with the full (unfiltered) numbers reported alongside
# for transparency.
MIN_FRAC_TRUE_OTHER = 0.5

_CG: dict = {}


def _conflict_init(transforms_path: str) -> None:
    from geovap.runtime.store import CloudStore, PassRegistration
    from geovap.runtime.pose_tables import load as _lp
    from geovap.runtime import settings
    from geovap.stages.prepare.masks import VehicleMask

    poses = _lp()
    transforms = json.loads(Path(transforms_path).read_text())
    _CG["poses"] = poses
    _CG["fi"] = FrameIndex(poses)
    poses_corr = apply_pass_transforms(poses, transforms)
    _CG["fi_corr"] = FrameIndex(poses_corr)
    _CG["store"] = CloudStore(settings.get().workspace.store)
    _CG["reg"] = PassRegistration(transforms, poses=poses)
    _s = settings.get()
    _mask_path = _s.workspace.vehicle_mask
    _CG["vm"] = VehicleMask(_mask_path, *_s.sensor.pano) if _mask_path.exists() else None
    p = poses.pass_id
    _CG["pass_t0"] = np.array([poses.t[np.flatnonzero(p == q)].min() for q in range(int(p.max()) + 1)])
    _CG["pass_t1"] = np.array([poses.t[np.flatnonzero(p == q)].max() for q in range(int(p.max()) + 1)])


def _conflict_job(k: int) -> dict:
    from geovap.stages.register import screen as qmod
    from geovap.stages.prepare.products import TIME_WINDOW_S, gather_candidates

    poses, fi, fi_corr = _CG["poses"], _CG["fi"], _CG["fi_corr"]
    store, reg, vm = _CG["store"], _CG["reg"], _CG["vm"]
    pass_t0, pass_t1 = _CG["pass_t0"], _CG["pass_t1"]
    p = int(poses.pass_id[k])
    R, C = fi.R[k], fi.C[k]
    r_max = _settings_sensor_r_max()
    dominant, frac_true_other = None, 0.0
    try:
        xyz, _pid = gather_candidates(store, C, r_max)
        parts = store.query_disc(float(C[0]), float(C[1]), r_max)
        gps = np.concatenate([np.asarray(store.tile(t.name).gps_time[rows]) for t, rows in parts]) if parts else np.empty(0)
        win = np.abs(gps - poses.t[k]) <= TIME_WINDOW_S
        own = win & (gps >= pass_t0[p] - 2) & (gps <= pass_t1[p] + 2)
        other = win & ~own
        n_other = int(other.sum())
        dt_img, edge_idx, valid = qmod._photo_edges(poses, k, vm)

        e_before = qmod._silhouette_points(xyz[other], R, C) if n_other > 20000 else xyz[:0]
        n2b, du2b, dv2b, *_ = qmod._residual(e_before, R, C, dt_img, edge_idx, valid)

        if n_other:
            gps_other = gps[other]
            other_pass = poses.pass_of_time(gps_other)
            uniq, counts = np.unique(other_pass, return_counts=True)
            dominant = int(uniq[np.argmax(counts)])
            same = int(counts[uniq == p].sum()) if (uniq == p).any() else 0
            frac_true_other = float((counts.sum() - same) / counts.sum())
            xyz_after = reg.apply(xyz[other], gps_other)
        else:
            xyz_after = xyz[:0]

        Rc, Cc = fi_corr.R[k], fi_corr.C[k]
        e_after = qmod._silhouette_points(xyz_after, Rc, Cc) if n_other > 20000 else xyz_after[:0]
        n2a, du2a, dv2a, *_ = qmod._residual(e_after, Rc, Cc, dt_img, edge_idx, valid)
        store.release()
        err = None
    except Exception as e:  # keep the manifest complete, mirrors screen._job
        n_other = 0
        n2b = n2a = 0
        du2b = dv2b = du2a = dv2a = np.nan
        err = f"{type(e).__name__}: {e}"

    conflict_b = bool(n2b >= qmod.GEO_MIN_POINTS and np.isfinite(du2b) and (abs(du2b) > qmod.CONFLICT_PX or abs(dv2b) > qmod.CONFLICT_PX))
    conflict_a = bool(n2a >= qmod.GEO_MIN_POINTS and np.isfinite(du2a) and (abs(du2a) > qmod.CONFLICT_PX or abs(dv2a) > qmod.CONFLICT_PX))
    return {
        "frame": k,
        "own_pass": p,
        "other_pass_dominant": dominant,
        "frac_true_other": round(frac_true_other, 3),
        "n_other_pts": n_other,
        "n2_before": int(n2b),
        "du2_before": None if not np.isfinite(du2b) else round(float(du2b), 3),
        "dv2_before": None if not np.isfinite(dv2b) else round(float(dv2b), 3),
        "conflict_before": conflict_b,
        "n2_after": int(n2a),
        "du2_after": None if not np.isfinite(du2a) else round(float(du2a), 3),
        "dv2_after": None if not np.isfinite(dv2a) else round(float(dv2a), 3),
        "conflict_after": conflict_a,
        "error": err,
    }


def _settings_sensor_r_max() -> float:
    from geovap.runtime import settings

    return settings.get().sensor.r_max


def run_conflict_metric(
    frames: list[int] | None = None,
    workers: int = 6,
    out: Path | None = None,
    dataset_dir: Path | None = None,
    s: "Settings | None" = None,
) -> dict:
    """Task 2: cross-pass silhouette conflict (`screen`'s `pass_conflict` residual) before vs
    after the S5 pass-registration transform, over every frame `frame_quality.csv` records with
    `n_edge_other > 0`. See the module-docstring-adjacent comment above for the before/after definition
    and the `frac_true_other` caveat. Also folds in the pairwise-ICP cloud-only point-to-plane rms
    already recorded per overlap pair in `pairs.json` (task 2's third leg)."""
    import csv as _csv
    from multiprocessing import Pool
    from geovap.runtime import settings

    s = s or settings.get()
    d = pass_reg_dir(s)
    out = out or (d / "conflict_reg.json")
    dataset_dir = dataset_dir or s.workspace.derived

    if frames is None:
        rows = list(_csv.DictReader(open(dataset_dir / "frame_quality.csv")))
        frames = sorted(int(r["frame"]) for r in rows if int(r["n_edge_other"]) > 0)

    transforms_path = str(d / "pass_transforms.json")
    results: list[dict] = []
    with Pool(min(workers, 6), initializer=_conflict_init, initargs=(transforms_path,)) as pool:
        for r in pool.imap_unordered(_conflict_job, frames, chunksize=2):
            results.append(r)
    results.sort(key=lambda r: r["frame"])

    def _agg(rows: list[dict]) -> dict:
        du_b = [abs(r["du2_before"]) for r in rows if r["du2_before"] is not None]
        dv_b = [abs(r["dv2_before"]) for r in rows if r["dv2_before"] is not None]
        du_a = [abs(r["du2_after"]) for r in rows if r["du2_after"] is not None]
        dv_a = [abs(r["dv2_after"]) for r in rows if r["dv2_after"] is not None]
        return {
            "n_frames": len(rows),
            "n_measured_before": len(du_b),
            "n_measured_after": len(du_a),
            "median_abs_du2_before": round(float(np.median(du_b)), 2) if du_b else None,
            "median_abs_dv2_before": round(float(np.median(dv_b)), 2) if dv_b else None,
            "median_abs_du2_after": round(float(np.median(du_a)), 2) if du_a else None,
            "median_abs_dv2_after": round(float(np.median(dv_a)), 2) if dv_a else None,
            "n_conflict_before": sum(1 for r in rows if r["conflict_before"]),
            "n_conflict_after": sum(1 for r in rows if r["conflict_after"]),
        }

    true_other = [r for r in results if r["frac_true_other"] >= MIN_FRAC_TRUE_OTHER]
    same_pass_only = [r for r in results if r["frac_true_other"] < MIN_FRAC_TRUE_OTHER]
    overall_all = _agg(results)
    overall_true_other = _agg(true_other)
    overall_same_pass = _agg(same_pass_only)

    by_pair: dict[str, list[dict]] = {}
    for r in true_other:
        if r["other_pass_dominant"] is None:
            continue
        a, b = sorted((r["own_pass"], r["other_pass_dominant"]))
        by_pair.setdefault(f"{a}_{b}", []).append(r)
    pair_summary = {k: _agg(v) for k, v in sorted(by_pair.items())}

    cloud_icp = {}
    pairs_path = d / "pairs.json"
    if pairs_path.exists():
        raw = json.loads(pairs_path.read_text())
        for k, v in raw.items():
            cloud_icp[k] = {"n": v.get("n"), "rms_before": v.get("rms_before"), "rms_after": v.get("rms_after"), "converged": v.get("converged")}
        rb = [v["rms_before"] for v in cloud_icp.values() if v["rms_before"] is not None]
        ra = [v["rms_after"] for v in cloud_icp.values() if v["rms_after"] is not None and v["converged"]]
        cloud_icp_summary = {
            "n_pairs": len(cloud_icp),
            "n_converged": sum(1 for v in cloud_icp.values() if v["converged"]),
            "median_rms_before": round(float(np.median(rb)), 4) if rb else None,
            "median_rms_after": round(float(np.median(ra)), 4) if ra else None,
        }
    else:
        cloud_icp_summary = {}

    from geovap.stages.register import screen as _qmod

    report = {
        "min_frac_true_other": MIN_FRAC_TRUE_OTHER,
        "conflict_px": {"conflict_px": _qmod.CONFLICT_PX, "geo_min_points": _qmod.GEO_MIN_POINTS},
        "overall_all_n2gt0_frames": overall_all,
        "overall_true_cross_pass": overall_true_other,
        "overall_same_pass_only": overall_same_pass,
        "by_pass_pair": pair_summary,
        "cloud_icp_pairs_summary": cloud_icp_summary,
        "cloud_icp_pairs": cloud_icp,
        "frames": {r["frame"]: r for r in results},
    }
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"overall_true_cross_pass": overall_true_other, "overall_same_pass_only": overall_same_pass, "cloud_icp_pairs_summary": cloud_icp_summary}, indent=1))
    return report


# ================================================================================================
# `mapping/cli/assemble_poses.py`, folded in below as the `assemble` stage.
#
# S_assemble: compose the validated correction layers into the final `poses_corrected` pose table
# -- the one `geovap.runtime.pose_tables.load("corrected")` returns.
#
# Order (`07_revize_geometrie_a_data.md`, plan step "ASSEMBLE"):
#
# 1. Base = `out/poses/poses_traj_rot.csv` (S3b rot-only trajectory attached; frame-time values equal
#    export, orientation *between* frames comes from the dense scanner-plane trajectory).
# 2. Overlay S4 (`out/poses/poses_refined_export.csv`): rows with `status == "refined"` replace their
#    own (E, N, H, roll, pitch, yaw, pass_id) and carry their theta/rms columns; every other row is
#    left exactly as the base table has it.
# 3. Apply S5b (`out/pass_reg/pass_transforms.json`) with `apply_pass_transforms` to the overlaid
#    table, and transform the attached rot-only trajectory identically: `R_p^T` is right-composed
#    into every per-sample scanner quaternion (`R_s'(t) := R_s(t) @ R_p^T`, `R_cs` untouched -- see
#    `transform_trajectory` for why folding it into `R_cs` would be wrong), so `camera_pose`'s
#    orientation becomes `R_v' = R_v @ R_p^T` at every query time, and the trajectory's `lin_*`
#    origin (what `Trajectory._linear_origin` linearly interpolates *between* frames) is resynced to
#    this same transformed table -- not the frozen, pre-registration table `poses_traj_rot` was
#    built from. A rotation-about-a-fixed-centre-plus-translation is affine, so `camera_pose` stays
#    exactly consistent with the transformed table at every query time (see
#    `tests/test_poses_table.py::test_transform_trajectory_invariant_with_apply_pass_transforms`
#    and `::test_assemble_end_to_end`).
# 4. Write `poses_corrected.csv` (+ `.json` provenance chaining every input's `poses_hash`/file sha1,
#    per-frame `status`, and a counts summary) and `trajectory_corrected.npz/.json`.
#
# Honest limitation carried from S4/S3b (do not blur this in the output): a refined frame's pose is
# only used AT its own timestamp (`Poses.pose_at(idx, 0.0)`, and any query that happens to land exactly
# there). `Poses.interp` with the trajectory attached still gets its orientation *between* frames from
# the rot-only trajectory (S3b), never from S4's per-frame refinement -- the two are not merged; S4 and
# the trajectory are just two independent, non-conflicting sources for two different things (one frame's
# own-time pose vs. the interpolant between frames).

META_COLS = [
    "status", "src_pass_changed", "dt_s", "dyaw", "droll", "dpitch",
    "n_edge", "rms_before", "rms_after", "reg_dE", "reg_dN", "reg_dH", "reg_dyaw",
]


def _sha1_file(path: Path) -> str | None:
    import hashlib

    if not Path(path).exists():
        return None
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_rows(path: Path) -> list[dict]:
    import csv as _csv

    with open(path, newline="") as fh:
        return list(_csv.DictReader(fh))


def _sidecar_json(csv_path: Path) -> dict:
    p = Path(csv_path).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else {}


# ------------------------------------------------------------------------------ step 2: S4 overlay
def overlay_refined(base: Poses, refined_rows: list[dict]) -> tuple[Poses, dict[str, np.ndarray], dict]:
    """Overlay rows with `status == "refined"` in `refined_rows` (one row per frame, same order as
    `base`) onto `base`'s (E, N, H, roll, pitch, yaw, pass_id). Returns
    (overlaid Poses [traj=None], per-frame meta arrays for the refinement columns [NaN / False where
    not overlaid], stats dict: n_overlaid, n_pass_changed, |delta pos|/|delta yaw| distribution)."""
    n = len(base)
    if len(refined_rows) != n:
        raise ValueError(f"poses_refined_export has {len(refined_rows)} rows, base has {n}")

    origin = base.origin.copy()
    roll = base.roll.copy()
    pitch = base.pitch.copy()
    yaw = base.yaw.copy()
    pass_id = base.pass_id.copy()

    refined_mask = np.zeros(n, dtype=bool)
    src_pass_changed = np.zeros(n, dtype=bool)
    dt_s = np.full(n, np.nan)
    dyaw = np.full(n, np.nan)
    droll = np.full(n, np.nan)
    dpitch = np.full(n, np.nan)
    n_edge = np.full(n, np.nan)
    rms_before = np.full(n, np.nan)
    rms_after = np.full(n, np.nan)

    dpos_list, dyaw_abs_list = [], []
    n_pass_changed = 0
    for i, row in enumerate(refined_rows):
        if row["status"] != "refined":
            continue
        if row["filename"] != str(base.filename[i]):
            raise ValueError(f"row {i}: filename mismatch {row['filename']!r} != {base.filename[i]!r} -- tables out of sync")
        old_o, old_yaw = origin[i].copy(), yaw[i]
        new_o = np.array([float(row["E"]), float(row["N"]), float(row["H"])])
        new_pid = int(row["pass_id"])
        dpos_list.append(float(np.linalg.norm(new_o - old_o)))
        dyaw_abs_list.append(float(abs((float(row["yaw"]) - old_yaw + 180.0) % 360.0 - 180.0)))

        origin[i], roll[i], pitch[i], yaw[i] = new_o, float(row["roll"]), float(row["pitch"]), float(row["yaw"])
        if new_pid != int(pass_id[i]):
            src_pass_changed[i] = True
            n_pass_changed += 1
        pass_id[i] = new_pid
        refined_mask[i] = True

        def _f(key):
            v = row[key]
            return float(v) if v not in ("", "nan", None) else np.nan

        dt_s[i], dyaw[i], droll[i], dpitch[i] = _f("dt_s"), _f("dyaw"), _f("droll"), _f("dpitch")
        n_edge[i], rms_before[i], rms_after[i] = _f("n_edge"), _f("rms_before"), _f("rms_after")

    overlaid = Poses(
        filename=base.filename.copy(), t=base.t.copy(), origin=origin, roll=roll, pitch=pitch,
        yaw=yaw, pass_id=pass_id, speed=base.speed.copy(), source=base.source, traj=None,
    )
    stats = {
        "n_overlaid": int(refined_mask.sum()),
        "n_pass_changed": n_pass_changed,
        "dpos_median_m": float(np.median(dpos_list)) if dpos_list else None,
        "dpos_p95_m": float(np.percentile(dpos_list, 95)) if dpos_list else None,
        "dpos_max_m": float(np.max(dpos_list)) if dpos_list else None,
        "dyaw_median_deg": float(np.median(dyaw_abs_list)) if dyaw_abs_list else None,
        "dyaw_p95_deg": float(np.percentile(dyaw_abs_list, 95)) if dyaw_abs_list else None,
        "dyaw_max_deg": float(np.max(dyaw_abs_list)) if dyaw_abs_list else None,
    }
    meta = {
        "refined_mask": refined_mask, "src_pass_changed": src_pass_changed,
        "dt_s": dt_s, "dyaw": dyaw, "droll": droll, "dpitch": dpitch,
        "n_edge": n_edge, "rms_before": rms_before, "rms_after": rms_after,
    }
    return overlaid, meta, stats


# ------------------------------------------------------------------------- step 3: S5b + trajectory
def _pass_rotation_yaw(transforms: dict) -> dict[int, float]:
    passes = transforms.get("passes", transforms)
    out = {}
    for p_str, tr in passes.items():
        try:
            out[int(p_str)] = float(tr["yaw_deg"])
        except (KeyError, ValueError):
            continue
    return out


def transform_trajectory(traj, transforms: dict, transformed_table: Poses):
    """rot-only only. `camera_pose` computes `R_cam(t) = R_cs @ R_s(t)` (`R_cs` constant per pass,
    `R_s(t)` the interpolated per-sample scanner orientation) and we need
    `R_cam'(t) = R_cam(t) @ R_p^T` for *every* t in the pass (matching
    `apply_pass_transforms`'s `R_v' = R_v @ R_p^T` on the table, so the two stay exactly
    consistent -- see the invariance tests). `R_p^T` sits at the *right* end of that product, after
    the time-varying `R_s(t)`; folding it into `R_cs` instead (`R_cs' = R_cs @ R_p^T`) would insert
    it in the *middle* (`R_cs @ R_p^T @ R_s(t)`) which only agrees with the wanted
    `R_cs @ R_s(t) @ R_p^T` when `R_p` and `R_s(t)` commute -- false in general (checked: up to ~8 px
    at 25 m for this dataset's transforms, not the needed 1e-6). The fix is to fold `R_p^T` into the
    *sample* orientations instead: `R_s'(t) := R_s(t) @ R_p^T` (`R_cs` untouched). Quaternion
    composition is right-linear (`(a+b)*c = a*c + b*c` and right-multiplication by a unit quaternion
    preserves norm), so right-composing every per-sample quaternion with the *same* `R_p^T` commutes
    exactly with `camera_pose`'s linear-interpolate-then-renormalise step -- so this reproduces
    `R_cam(t) @ R_p^T` for every interpolated t, not only at the samples themselves.

    Also resyncs `lin_*` (the origin `Trajectory._linear_origin` linearly interpolates between
    frames) to `transformed_table` -- every frame, same order, so `covers`/`camera_pose` and the
    plain-linear fallback (frames/passes the trajectory does not cover) agree on the same table."""
    if traj.mode != "rot_only":
        raise NotImplementedError("transform_trajectory only implements the rot_only path (S3b); pos-mode trajectories are not in production use (see trajectory.py)")
    from scipy.spatial.transform import Rotation
    from geovap.stages.register.trajectory import CamSensorRig, Trajectory

    yaw_by_pass = _pass_rotation_yaw(transforms)
    quat2 = traj.quat.copy()
    for p in np.unique(traj.pass_id):
        Rp = _rot_yaw(np.radians(yaw_by_pass.get(int(p), 0.0)))
        m = traj.pass_id == p
        quat2[m] = (Rotation.from_quat(traj.quat[m]) * Rotation.from_matrix(Rp.T)).as_quat()
    new_rigs: dict[int, CamSensorRig] = {p: CamSensorRig(R_cs=rig.R_cs.copy(), l_cs=rig.l_cs.copy(), dt_s=rig.dt_s, stats=dict(rig.stats)) for p, rig in traj.rigs.items()}
    return Trajectory(
        t=traj.t.copy(), pass_id=traj.pass_id.copy(), S=traj.S.copy(), quat=quat2,
        segments={k: list(v) for k, v in traj.segments.items()}, rigs=new_rigs, sample_hz=traj.sample_hz,
        meta={**traj.meta, "pass_reg_applied": True},
        mode="rot_only",
        lin_t=transformed_table.t.copy(), lin_origin=transformed_table.origin.copy(),
        lin_roll=transformed_table.roll.copy(), lin_pitch=transformed_table.pitch.copy(),
        lin_yaw=transformed_table.yaw.copy(), lin_pass_id=transformed_table.pass_id.copy(),
    )


def _reg_deltas(pass_id: np.ndarray, transforms: dict) -> dict[str, np.ndarray]:
    """Broadcast each frame's *current* pass' (dE, dN, dH, dyaw_deg) transform onto per-frame arrays
    -- constant within a pass, informational (the table already carries the applied pose)."""
    passes = transforms.get("passes", transforms)
    n = len(pass_id)
    reg_dE, reg_dN, reg_dH, reg_dyaw = (np.full(n, np.nan) for _ in range(4))
    for p_str, tr in passes.items():
        try:
            p = int(p_str)
        except ValueError:
            continue
        sel = pass_id == p
        if not sel.any():
            continue
        dE, dN, dH = tr["t"]
        reg_dE[sel], reg_dN[sel], reg_dH[sel], reg_dyaw[sel] = dE, dN, dH, tr["yaw_deg"]
    return {"reg_dE": reg_dE, "reg_dN": reg_dN, "reg_dH": reg_dH, "reg_dyaw": reg_dyaw}


# ---------------------------------------------------------------------------------------- assemble
def assemble(
    base_path: Path,
    refined_path: Path,
    transforms_path: Path,
    out_path: Path,
    traj_out_path: Path,
    log=print,
) -> tuple[Path, dict]:
    from geovap.runtime.pose_tables import read as read_pose_table, write as write_pose_table

    base_path, refined_path, transforms_path = Path(base_path), Path(refined_path), Path(transforms_path)
    out_path, traj_out_path = Path(out_path), Path(traj_out_path)

    base = read_pose_table(base_path)
    if base.traj is None:
        raise RuntimeError(f"{base_path} has no attached trajectory (sidecar missing 'trajectory' or npz not found) -- assemble step 1 requires it")
    base_status = _read_csv_rows(base_path)
    if len(base_status) != len(base):
        raise RuntimeError(f"{base_path}: row count mismatch with its own reader")
    base_status_col = [r["status"] for r in base_status]

    refined_rows = _read_csv_rows(refined_path)
    overlaid, refine_meta, refine_stats = overlay_refined(base, refined_rows)
    log(f"S4 overlay: {refine_stats['n_overlaid']} / {len(overlaid)} rows refined "
        f"({refine_stats['n_pass_changed']} with a changed pass_id); "
        f"|dpos| median {refine_stats['dpos_median_m']:.3f} m p95 {refine_stats['dpos_p95_m']:.3f} m; "
        f"|dyaw| median {refine_stats['dyaw_median_deg']:.3f} deg p95 {refine_stats['dyaw_p95_deg']:.3f} deg")

    transforms = json.loads(transforms_path.read_text())
    registered = apply_pass_transforms(overlaid, transforms)
    traj_corrected = transform_trajectory(base.traj, transforms, registered)
    registered.traj = traj_corrected

    status = [f"{b}+refined+reg" if m else f"{b}+reg" for b, m in zip(base_status_col, refine_meta["refined_mask"])]
    reg_meta = _reg_deltas(registered.pass_id, transforms)

    per_frame_meta = {
        "status": np.array(status, dtype=object),
        "src_pass_changed": refine_meta["src_pass_changed"],
        "dt_s": refine_meta["dt_s"], "dyaw": refine_meta["dyaw"],
        "droll": refine_meta["droll"], "dpitch": refine_meta["dpitch"],
        "n_edge": refine_meta["n_edge"], "rms_before": refine_meta["rms_before"], "rms_after": refine_meta["rms_after"],
        **reg_meta,
    }
    assert list(per_frame_meta.keys()) == META_COLS

    from collections import Counter

    status_counts = dict(Counter(status))
    log(f"status counts: {status_counts}")

    traj_out_path.parent.mkdir(parents=True, exist_ok=True)
    traj_corrected.save(traj_out_path)
    log(f"wrote {traj_out_path}")

    base_prov = _sidecar_json(base_path)
    refined_prov = _sidecar_json(refined_path)
    provenance = {
        "stage": "assemble (S0-S5b composition)",
        "inputs": {
            "base": {"path": str(base_path), "stage": "S3b traj_rot", "poses_hash": base.hash(), "sha1": _sha1_file(base_path), "trajectory_npz_sha1": _sha1_file(base_path.parent / base_prov.get("trajectory", "trajectory_rot.npz"))},
            "refined": {"path": str(refined_path), "stage": "S4 pose_refine", "poses_hash": refined_prov.get("poses_hash"), "sha1": _sha1_file(refined_path)},
            "pass_transforms": {"path": str(transforms_path), "stage": "S5b pass_reg", "sha1": _sha1_file(transforms_path), "datum": transforms.get("summary", {}).get("datum")},
        },
        "order": ["poses_traj_rot (S3b)", "overlay S4 refined rows", "apply S5b pass_transforms to table and trajectory (per-sample orientation + lin_origin)"],
        "trajectory": traj_out_path.name,
        # the pass_transforms this table's points are registered against -- `Poses.registration`
        # (geovap.runtime.pose_tables.read) and `geovap.runtime.store.open_store` read this back so a
        # `CloudStore` opened for this pose table applies the same S5b transform the poses assume.
        "registration": {"path": str(transforms_path), "sha1": _sha1_file(transforms_path)},
        "s4_overlay": refine_stats,
        "status_counts": status_counts,
        "note": "S4's refined pose is used only at the frame's own timestamp; Poses.interp between "
                "frames still gets orientation from the S3b rot-only trajectory (never merged with S4).",
    }
    write_pose_table(registered, per_frame_meta, provenance, out_path)
    log(f"wrote {out_path} ({len(registered)} frames)")
    return out_path, {"s4_overlay": refine_stats, "status_counts": status_counts}


# ================================================================================================ stages
# Three stages live in this one module: `register` (pairwise pass-to-pass ICP + the S5b solve,
# `run_register_all`), `reg-conflict` (optional QA of that solve, `run_conflict_metric`), and
# `assemble` (compose S3b/S4/S5b into `poses_corrected`, `assemble()`). They share this module
# because `assemble` calls `apply_pass_transforms`/`transform_trajectory` directly and `reg-conflict`
# reads `register`'s own `pass_transforms.json` -- the same "several stages, one module" shape as
# `stages.prepare.store`'s `store`/`store-columns`.

class Register:
    spec = StageSpec(
        name="register", after=("refine",), est_min=13,
        summary="S5: pairwise pass-to-pass ICP + global pose-graph solve -> pass_transforms.json",
    )
    cli_args = ("--stage", "register")

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        return {"refined_csv": s.workspace.poses / "poses_refined_export.csv"}

    def outputs(self, s: "Settings") -> list[Path]:
        d = pass_reg_dir(s)
        return [d / "pass_transforms.json", d / "pairs.json"]

    def metrics(self, s: "Settings") -> dict:
        try:
            d = pass_reg_dir(s)
            pairs = json.loads((d / "pairs.json").read_text())
            rms_before = [v.get("rms_before") for v in pairs.values() if isinstance(v, dict) and v.get("rms_before") is not None]
            rms_after = [v.get("rms_after") for v in pairs.values() if isinstance(v, dict) and v.get("rms_after") is not None]
            converged = sum(1 for v in pairs.values() if isinstance(v, dict) and v.get("converged"))
            tr = json.loads((d / "pass_transforms.json").read_text())
            return {
                "n_pairs": len(pairs),
                "n_converged": converged,
                "rms_before_median": float(np.percentile(rms_before, 50)) if rms_before else None,
                "rms_after_median": float(np.percentile(rms_after, 50)) if rms_after else None,
                "summary": tr.get("summary", {}),
                "max_t_m": max((max(abs(float(v)) for v in p_.get("t", [0, 0, 0])) for p_ in tr.get("passes", {}).values()), default=None),
                "max_yaw_deg": max((abs(float(p_.get("yaw_deg", 0.0))) for p_ in tr.get("passes", {}).values()), default=None),
            }
        except Exception as e:  # noqa: BLE001 - metrics must always be reportable
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", *, workers: int = 8, datum: str = "none") -> None:
        run_register_all(datum=datum, workers=workers, s=s)


REGISTER = registry.add(Register())


class RegConflict:
    spec = StageSpec(
        name="reg-conflict", after=("register",), optional=True, est_min=30,
        summary="QA: cross-pass silhouette conflict before/after the S5 transform",
    )
    cli_args = ("--stage", "conflict")

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        return {"pass_transforms": pass_reg_dir(s) / "pass_transforms.json"}

    def outputs(self, s: "Settings") -> list[Path]:
        return []

    def metrics(self, s: "Settings") -> dict:
        return {}

    def run(self, s: "Settings", *, workers: int = 6) -> None:
        run_conflict_metric(workers=workers, s=s)


REG_CONFLICT = registry.add(RegConflict())


class Assemble:
    """`assemble` is special: the driver (`geovap.app.driver.POSE_TABLE_PRODUCER`) treats everything
    at or before it as running on EXPORT poses, so this stage's name must stay exactly `assemble`
    for an existing run's marker file to resume."""

    spec = StageSpec(
        name="assemble", after=("traj-rot", "refine", "register"), est_min=0.5,
        summary="compose S3b/S4/S5b into poses_corrected.csv",
    )
    cli_args = ("--stage", "assemble")

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        return {
            "traj_csv": s.workspace.poses / "poses_traj_rot.csv",
            "refined_csv": s.workspace.poses / "poses_refined_export.csv",
            "pass_transforms": pass_reg_dir(s) / "pass_transforms.json",
        }

    def outputs(self, s: "Settings") -> list[Path]:
        return [s.workspace.poses / "poses_corrected.csv"]

    def metrics(self, s: "Settings") -> dict:
        try:
            from geovap.runtime.pose_tables import load as load_poses

            return {"corrected_hash6": load_poses("corrected", s=s).hash()[:6]}
        except Exception as e:  # noqa: BLE001 - metrics must always be reportable
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", *, base: Path | None = None, refined: Path | None = None, transforms: Path | None = None, out: Path | None = None, traj_out: Path | None = None) -> None:
        base = base or (s.workspace.poses / "poses_traj_rot.csv")
        refined = refined or (s.workspace.poses / "poses_refined_export.csv")
        transforms = transforms or (pass_reg_dir(s) / "pass_transforms.json")
        out = out or (s.workspace.poses / "poses_corrected.csv")
        traj_out = traj_out or (s.workspace.poses / "trajectory_corrected.npz")
        assemble(base, refined, transforms, out, traj_out)


ASSEMBLE = registry.add(Assemble())


# ================================================================================================ cli
def _add_options_register(p) -> None:
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--datum", choices=DATUM_CHOICES, default="none")


def _to_opts_register(args) -> dict:
    return {"workers": args.workers, "datum": args.datum}


def _add_options_conflict(p) -> None:
    p.add_argument("--workers", type=int, default=6)


def _to_opts_conflict(args) -> dict:
    return {"workers": args.workers}


def _add_options_assemble(p) -> None:
    p.add_argument("--base", default=None)
    p.add_argument("--refined", default=None)
    p.add_argument("--transforms", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--traj-out", default=None)


def _to_opts_assemble(args) -> dict:
    return {
        "base": Path(args.base) if args.base else None,
        "refined": Path(args.refined) if args.refined else None,
        "transforms": Path(args.transforms) if args.transforms else None,
        "out": Path(args.out) if args.out else None,
        "traj_out": Path(args.traj_out) if args.traj_out else None,
    }


def main(argv=None) -> int:
    """One module, three stages (`register`, `reg-conflict`, `assemble`) -- `--stage` picks which,
    same convention as `stages.prepare.store`'s two-stage module. Additionally preserves
    `mapping/cli/register_passes.py`'s ad hoc sub-commands (`extract`/`pairs`/`jvf`/`jvf_photo`/
    `solve`/`poses`) and `mapping/cli/validate_pass_reg.py`'s (`qa`/`metric`/`conflict`) for dev use,
    routed via a first positional argument that is not `run`/`--status`."""
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    stage_choice = "register"
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--stage" and i + 1 < len(argv):
            stage_choice = argv[i + 1]
            i += 2
            continue
        if a.startswith("--stage="):
            stage_choice = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1

    dev_cmds = {"extract", "pairs", "jvf", "jvf_photo", "solve", "poses", "qa", "metric", "conflict-report"}
    if rest and rest[0] in dev_cmds:
        return _dev_main(rest)

    if stage_choice not in ("register", "conflict", "assemble"):
        raise SystemExit(f"--stage must be 'register', 'conflict' or 'assemble', got {stage_choice!r}")

    if stage_choice == "register":
        return stage_main(REGISTER, rest, add_options=_add_options_register, to_opts=_to_opts_register)
    if stage_choice == "conflict":
        return stage_main(REG_CONFLICT, rest, add_options=_add_options_conflict, to_opts=_to_opts_conflict)
    return stage_main(ASSEMBLE, rest, add_options=_add_options_assemble, to_opts=_to_opts_assemble)


def _dev_main(argv: list[str]) -> int:
    """`mapping/cli/register_passes.py` (`extract|pairs|jvf|jvf_photo|solve|poses|all`) and
    `mapping/cli/validate_pass_reg.py` (`qa|metric|conflict-report`) folded in unchanged, as dev
    tooling outside the eight resumable stages."""
    from geovap.stages.base.cli import add_dataset_flags, configure_from
    import argparse

    ap = argparse.ArgumentParser()
    add_dataset_flags(ap)
    ap.add_argument("cmd", choices=sorted({"all"} | {"extract", "pairs", "jvf", "jvf_photo", "solve", "poses", "qa", "metric", "conflict-report"}))
    ap.add_argument("--datum", choices=DATUM_CHOICES, default="none")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--passes", default="0,5")
    ap.add_argument("--n-frames", type=int, default=3)
    ap.add_argument("--n-subset", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    s = configure_from(a)

    if a.cmd == "all":
        run_register_all(datum=a.datum, workers=a.workers, s=s)
    elif a.cmd == "extract":
        cmd_extract(workers=a.workers, s=s)
    elif a.cmd == "pairs":
        cmd_pairs(workers=a.workers, s=s)
    elif a.cmd == "jvf":
        cmd_jvf(s=s)
    elif a.cmd == "jvf_photo":
        cmd_jvf_photo(s=s)
    elif a.cmd == "solve":
        cmd_solve(datum=a.datum, s=s)
    elif a.cmd == "poses":
        cmd_poses(s=s)
    elif a.cmd == "qa":
        pass_ids = [int(x) for x in a.passes.split(",")]
        render_qa(pass_ids, n_frames=a.n_frames, s=s)
    elif a.cmd == "metric":
        run_metric(n_subset=a.n_subset, out=Path(a.out) if a.out else None, s=s)
    elif a.cmd == "conflict-report":
        run_conflict_metric(workers=a.workers, out=Path(a.out) if a.out else None, s=s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
