"""Spherical z-buffer: scatter-min splatting of points into a reduced-resolution depth panorama.

Coordinates here are z-buffer pixels (ZB_W x ZB_H), i.e. full-res u,v scaled by ZB_W/PANO_W.
Depth is Euclidean range in metres (float32 while splatting, uint16 millimetres when stored).
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import ndimage

from .config import NO_POINT, POINT_SPACING, R_MAX, R_MIN, SPLAT_K, SPLAT_MAX_PX, SPLAT_MIN_PX, TOL_ABS, TOL_REL, ZB_H, ZB_W


@njit(cache=True)
def _splat_min(u, v, r, pid, depth, ids, zb_w, zb_h, rad_scale, rmin_px, rmax_px):
    """depth[H,W] float32 (inf = empty), ids[H,W] uint32. u,v in z-buffer px; r metres; pid uint32."""
    two_pi = 6.283185307179586
    for i in range(u.shape[0]):
        ri = r[i]
        rho = rad_scale / ri  # splat radius in px
        if rho < rmin_px:
            rho = rmin_px
        elif rho > rmax_px:
            rho = rmax_px
        cv = int(v[i])
        cu = int(u[i])
        if cv < 0:
            cv = 0
        elif cv >= zb_h:
            cv = zb_h - 1
        # elevation of the cell -> horizontal stretch near the poles
        el = (0.5 - (cv + 0.5) / zb_h) * 3.141592653589793  # rad, +pi/2 at v=0
        ce = abs(np.cos(el))
        if ce < 1e-6:
            ce = 1e-6
        rho_u = rho / ce
        if rho_u > zb_w / 2:
            rho_u = zb_w / 2
        irho = int(rho + 0.999)
        irho_u = int(rho_u + 0.999)
        inv_r2 = 1.0 / (rho * rho)
        inv_ru2 = 1.0 / (rho_u * rho_u)
        for dv in range(-irho, irho + 1):
            row = cv + dv
            if row < 0 or row >= zb_h:
                continue
            for du in range(-irho_u, irho_u + 1):
                if du * du * inv_ru2 + dv * dv * inv_r2 > 1.0:
                    continue
                col = (cu + du) % zb_w
                if ri < depth[row, col]:
                    depth[row, col] = ri
                    ids[row, col] = pid[i]


def splat(u_zb: np.ndarray, v_zb: np.ndarray, r: np.ndarray, pid: np.ndarray, zb_w: int = ZB_W, zb_h: int = ZB_H) -> tuple[np.ndarray, np.ndarray]:
    """Scatter-min of ranges. Returns (depth_m float32 [H,W] with inf for empty, point_id uint32 [H,W])."""
    depth = np.full((zb_h, zb_w), np.inf, dtype=np.float32)
    ids = np.full((zb_h, zb_w), NO_POINT, dtype=np.uint32)
    rad_scale = SPLAT_K * POINT_SPACING * zb_w / (2 * np.pi)  # px * m
    _splat_min(
        np.ascontiguousarray(u_zb, dtype=np.float32),
        np.ascontiguousarray(v_zb, dtype=np.float32),
        np.ascontiguousarray(r, dtype=np.float32),
        np.ascontiguousarray(pid, dtype=np.uint32),
        depth,
        ids,
        zb_w,
        zb_h,
        np.float32(rad_scale),
        np.float32(SPLAT_MIN_PX),
        np.float32(SPLAT_MAX_PX),
    )
    return depth, ids


def range_filter(r: np.ndarray, r_min: float = R_MIN, r_max: float = R_MAX) -> np.ndarray:
    return (r >= r_min) & (r <= r_max)


def depth_to_mm(depth_m: np.ndarray) -> np.ndarray:
    """inf -> 0; otherwise round(m*1000) as uint16 (65.535 m max, > R_MAX)."""
    out = np.zeros(depth_m.shape, dtype=np.uint16)
    ok = np.isfinite(depth_m)
    out[ok] = np.clip(np.rint(depth_m[ok] * 1000.0), 1, 65535).astype(np.uint16)
    return out


def depth_from_mm(depth_mm: np.ndarray) -> np.ndarray:
    d = depth_mm.astype(np.float32) * 0.001
    d[depth_mm == 0] = np.inf
    return d


def close_depth(depth_m: np.ndarray, size: int = 3) -> np.ndarray:
    """Fill small holes (empty/far cells) with the nearer surrounding depth: grey opening of range
    == closing of nearness. Wraps along u, clamps along v."""
    d = np.where(np.isfinite(depth_m), depth_m, np.float32(1e6)).astype(np.float32)
    ero = ndimage.minimum_filter(d, size=size, mode=("nearest", "wrap"))
    dil = ndimage.maximum_filter(ero, size=size, mode=("nearest", "wrap"))
    out = np.minimum(d, dil)  # never push a known surface farther
    out[out >= 1e6] = np.inf
    return out


SLOPE_TOL_K = 1.0  # tolerance grows with the local depth spread of the cell neighbourhood ...
SLOPE_TOL_CAP = 2.0  # ... up to this many metres (keeps real occlusion edges from leaking)
HEIGHT_TOL_M = 0.12  # a height error of this size on the ground becomes HEIGHT_TOL/sin|el| of range along the ray


def depth_spread(depth_closed_m: np.ndarray, size: int = 3) -> np.ndarray:
    """Local max-min of depth over a size x size neighbourhood (inf/empty ignored). Large on surfaces seen
    at grazing incidence (ground far away) and at occlusion edges."""
    d = np.where(np.isfinite(depth_closed_m), depth_closed_m, np.nan).astype(np.float32)
    big = np.where(np.isnan(d), -np.inf, d)
    small = np.where(np.isnan(d), np.inf, d)
    mx = ndimage.maximum_filter(big, size=size, mode=("nearest", "wrap"))
    mn = ndimage.minimum_filter(small, size=size, mode=("nearest", "wrap"))
    out = mx - mn
    out[~np.isfinite(out)] = 0.0
    return out


def visible(r: np.ndarray, u_zb: np.ndarray, v_zb: np.ndarray, depth_closed_m: np.ndarray, tol_abs: float = TOL_ABS, tol_rel: float = TOL_REL, spread_m: np.ndarray | None = None) -> np.ndarray:
    """r <= depth + tol at the point's z-buffer cell, tol = max(tol_abs, tol_rel*r, K*local depth spread).
    The slope term keeps ground seen at grazing angles (depth changes by ~0.5 m per cell at 20 m)
    from being 'occluded' by its own nearer neighbours. Empty cell -> not visible."""
    h, w = depth_closed_m.shape
    cv = np.clip(v_zb.astype(np.int64), 0, h - 1)
    cu = np.mod(u_zb.astype(np.int64), w)
    d = depth_closed_m[cv, cu]
    tol = np.maximum(tol_abs, tol_rel * r)
    if spread_m is not None:
        tol = np.maximum(tol, np.minimum(SLOPE_TOL_K * spread_m[cv, cu], SLOPE_TOL_CAP))
    # grazing incidence on (near-)horizontal surfaces: |el| from the cell row
    el = np.abs(np.radians(90.0 - (cv + 0.5) / h * 180.0))
    tol = np.maximum(tol, np.minimum(HEIGHT_TOL_M / np.maximum(np.sin(el), 0.05), SLOPE_TOL_CAP))
    return np.isfinite(d) & (r <= d + tol)
