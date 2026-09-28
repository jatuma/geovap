"""Photo edges, silhouette points and their residual -- the geometry of "does the cloud line up
with the picture".

These four functions used to be private helpers split between `mapping/quality.py` (frame screening)
and `mapping/calib/icp.py` (pose refinement), which made the two stages import each other: screening
reached into calibration for `_photo_edges`, while refinement reached into screening for the
residual conventions. They are pure array maths, so they belong here and neither stage needs the
other.

Nothing in this module loads an image or a pose table. `photo_edges` takes a luminance raster that
the caller has already decoded; `silhouette_points` and `residual` take world points and a frame
rotation. Thresholds arrive as arguments rather than as module constants.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
from scipy import ndimage

from ..model.geometry import world_to_pano
from ..model.sensor import Sensor
from . import depth as _depth


def photo_edges(
    lum: np.ndarray,
    *,
    pano_w: int,
    pano_h: int,
    vmask=None,
    edge_pct: float,
    sky_edge_pct: float,
    texture_win: int,
    texture_density: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sobel edges of a luminance panorama, thresholded by percentile and de-textured.

    `lum` is [H,W] float in [0,1] at whatever resolution the caller works in; `pano_w`/`pano_h` are
    the FULL-resolution panorama dimensions, used only to scale coordinates for `vmask`. Returns
    (edges, valid) as bool [H,W].

    Two thresholds: the sky half of the image gets the stricter of the two percentiles, so thin
    cloud texture does not drown the skyline. Cells whose local edge density exceeds
    `texture_density` are dropped -- dense foliage otherwise matches everything.
    """
    h, w = lum.shape
    g = cv2.GaussianBlur(lum, (0, 0), 1.0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    valid = lum > 0.01
    if vmask is not None:
        vv, uu = np.mgrid[0:h, 0:w]
        valid &= ~vmask(uu * (pano_w / w), vv * (pano_h / h))
    lower = valid.copy()
    lower[: h // 3] = False
    thr = np.percentile(mag[lower], edge_pct)
    sky = valid.copy()
    sky[h // 2 :] = False
    thr_sky = max(thr, np.percentile(mag[sky], sky_edge_pct))
    thr_map = np.where(np.arange(h)[:, None] < h // 2 - int(h * 5 / 180), thr_sky, thr)
    edges = (mag > thr_map) & valid
    density = cv2.blur(edges.astype(np.float32), (texture_win, texture_win))
    edges &= density <= texture_density
    return edges, valid


def edge_distance_transform(edges: np.ndarray, pad: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Seam-aware Euclidean distance to the nearest edge pixel, plus that pixel's index.

    The panorama wraps in u, so the image is padded with its own opposite margin before the
    transform and cropped back afterwards -- otherwise every edge within `pad` of the seam reports
    a spuriously large distance. Returns (dt float16 [H,W], idx int16 [2,H,W] as (row, col)).
    """
    w = edges.shape[1]
    e = np.concatenate([edges[:, -pad:], edges, edges[:, :pad]], axis=1)
    dt, (ir, ic) = ndimage.distance_transform_edt(~e, return_indices=True)
    ir = ir[:, pad:-pad]
    ic = (ic[:, pad:-pad] - pad) % w
    return dt[:, pad:-pad].astype(np.float16), np.stack([ir, ic]).astype(np.int16)


def silhouette_points(
    xyz: np.ndarray, R: np.ndarray, C: np.ndarray, sensor: Sensor, fine_w: int, fine_h: int
) -> np.ndarray:
    """Points of `xyz` that sit on a depth discontinuity as seen from (R, C). Returns an xyz subset.

    A point qualifies when its z-buffer cell is a depth edge -- a neighbour two cells away is much
    farther, or empty sky above the horizon -- and the point itself is the near surface of that
    cell rather than something hidden behind it. The splat cap is tightened to 3 px so fat splats
    do not smear the discontinuity away.
    """
    u, v, r, _el = world_to_pano(xyz, R, C, *sensor.pano)
    keep = _depth.range_filter(r, sensor.r_min, sensor.r_max)
    if keep.sum() < 1000:
        return xyz[:0]
    s = fine_w / sensor.pano_w
    x, y = u[keep] * s, v[keep] * s
    depth, _ = _depth.splat(
        x, y, r[keep], np.arange(keep.sum(), dtype=np.uint32), fine_w, fine_h,
        replace(sensor, splat_max_px=3),
    )
    d = _depth.close_depth(depth)
    fin = np.isfinite(d)
    el_rows = 90.0 - (np.arange(fine_h) + 0.5) / fine_h * 180.0
    edge = np.zeros_like(fin)
    for dy, dx in ((0, 2), (0, -2), (2, 0), (-2, 0)):
        dn = np.roll(d, (-dy, -dx), axis=(0, 1))
        fn = np.isfinite(dn)
        edge |= fin & fn & (dn - d > np.maximum(0.5, 0.15 * d))
        edge |= fin & ~fn & (el_rows > 0)[:, None]
    edge &= (el_rows >= -40)[:, None]
    cx = np.mod(np.floor(x).astype(np.int64), fine_w)
    cy = np.clip(np.floor(y).astype(np.int64), 0, fine_h - 1)
    near = r[keep] <= d[cy, cx] + np.maximum(0.15, 0.03 * r[keep])
    return xyz[keep][edge[cy, cx] & near]


def residual(
    xyz_edge: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    dt_img: np.ndarray,
    edge_idx: np.ndarray,
    valid: np.ndarray,
    sensor: Sensor,
    fine_w: int,
    fine_h: int,
    window: float = 20.0,
    n_max: int = 20000,
    inlier_px: float = 8.0,
) -> tuple[int, float, float, float, float, float]:
    """Median (du, dv) in full-resolution px between silhouette points and the nearest photo edge.

    Returns (n, median_du, median_dv, mad_du, mad_dv, inlier_fraction), all NaN when fewer than 50
    points land on a valid cell within `window`. `inlier_fraction` is the share of ALL edge points
    within `inlier_px` of a photo edge, not just of the matched ones, so a pose that matches a few
    points very well still scores low.
    """
    if len(xyz_edge) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan, np.nan
    if len(xyz_edge) > n_max:
        xyz_edge = xyz_edge[:: len(xyz_edge) // n_max]
    s = fine_w / sensor.pano_w
    u, v, _r, _el = world_to_pano(xyz_edge, R, C, *sensor.pano, dtype=np.float64)
    x, y = u * s, v * s
    xi = np.mod(np.floor(x).astype(np.int64), fine_w)
    yi = np.clip(np.floor(y).astype(np.int64), 0, fine_h - 1)
    d = dt_img[yi, xi].astype(np.float32)
    ok = valid[yi, xi] & (d <= window)
    n = int(ok.sum())
    if n < 50:
        return n, np.nan, np.nan, np.nan, np.nan, np.nan
    ex = edge_idx[1, yi[ok], xi[ok]].astype(np.float64) + 0.5
    ey = edge_idx[0, yi[ok], xi[ok]].astype(np.float64) + 0.5
    du = ((ex - x[ok] + fine_w / 2) % fine_w - fine_w / 2) / s
    dv = (ey - y[ok]) / s
    inl = float((d[ok] <= inlier_px).mean() * ok.mean())
    return (
        n,
        float(np.median(du)),
        float(np.median(dv)),
        float(np.median(np.abs(du - np.median(du)))),
        float(np.median(np.abs(dv - np.median(dv)))),
        inl,
    )
