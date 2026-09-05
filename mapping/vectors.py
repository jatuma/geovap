"""JVF DTM vectors (lines / points / polygons) -> panorama, with occlusion.

Fixes over experiments/e1 + e3: straight 3D segments are subdivided (they are curves in ERP),
seam crossings are split instead of skipped, the band width follows the 14 cm tolerance at each
sample's own range, and every sample is depth-tested against the frame's depth panorama.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry
from .config import PANO_H, PANO_W, R_MAX, R_MIN, TOL_ABS, TOL_REL
from .products import FrameProducts

TOLERANCE_M = 0.14  # legal xy accuracy budget (03_semanticka_segmentace.md SS1.7)
MAX_STEP_DEG = 0.25


@dataclass
class ProjectedPolyline:
    u: np.ndarray  # [K] full-res px, continuous (may run outside [0,W) within a piece; pieces never cross the seam)
    v: np.ndarray
    r: np.ndarray  # range [m]
    visible: np.ndarray  # [K] bool
    half_width_px: np.ndarray  # [K]


def subdivide_3d(coords: np.ndarray, C: np.ndarray, max_step_deg: float = MAX_STEP_DEG) -> np.ndarray:
    """Insert points along each 3D segment so consecutive samples subtend <= max_step_deg at the camera."""
    if len(coords) < 2:
        return coords
    out = [coords[:1]]
    tan_step = np.tan(np.radians(max_step_deg))
    for a, b in zip(coords[:-1], coords[1:]):
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-9:
            continue
        # closest approach of the segment to the camera bounds the angular rate (rad per metre = 1/d_min)
        t_min = np.clip(np.dot(C - a, ab) / (L * L), 0.0, 1.0)
        d_min = max(np.linalg.norm(a + ab * t_min - C), 0.5)
        n = max(1, int(np.ceil(L / (d_min * tan_step))))
        n = min(n, 20000)
        t = np.linspace(0, 1, n + 1)[1:]
        out.append(a[None] + ab[None] * t[:, None])
    return np.concatenate(out)


def split_at_seam(u: np.ndarray, *arrays) -> list[tuple]:
    """Split a projected polyline into pieces that do not wrap; the crossing is interpolated so both
    pieces reach the seam. Returns [(u, *arrays), ...]."""
    pieces = []
    start = 0
    n = len(u)
    if n == 0:
        return pieces
    u = u.copy()
    arrs = [a.copy() for a in arrays]
    cur_u, cur_a = [u[0]], [[a[0]] for a in arrs]
    for i in range(1, n):
        du = u[i] - u[i - 1]
        if abs(du) > PANO_W / 2:  # wrapped
            # unwrap u[i] into the frame of u[i-1]
            ui = u[i] + (PANO_W if du < 0 else -PANO_W)
            frac = ((PANO_W if du < 0 else 0.0) - u[i - 1]) / (ui - u[i - 1])
            edge_a = [a[i - 1] + (a[i] - a[i - 1]) * frac for a in arrs]
            edge_u = PANO_W if du < 0 else 0.0
            cur_u.append(edge_u)
            for c, e in zip(cur_a, edge_a):
                c.append(e)
            pieces.append((np.array(cur_u), *[np.array(c) for c in cur_a]))
            cur_u = [PANO_W - edge_u]
            cur_a = [[e] for e in edge_a]
        cur_u.append(u[i])
        for c, a in zip(cur_a, arrs):
            c.append(a[i])
    pieces.append((np.array(cur_u), *[np.array(c) for c in cur_a]))
    return pieces


def project_polyline(coords: np.ndarray, R: np.ndarray, C: np.ndarray, fp: FrameProducts | None, r_max: float = R_MAX) -> list[ProjectedPolyline]:
    """3D polyline (E,N,H) -> list of seam-free ERP pieces with visibility and band width per sample."""
    coords = np.asarray(coords, dtype=np.float64)
    if np.isnan(coords).any():
        return []
    pts = subdivide_3d(coords, C)
    u, v, r, el = geometry.world_to_pano(pts, R, C, dtype=np.float64)
    inr = (r >= R_MIN) & (r <= r_max)
    if fp is not None:
        # a vector lies ON the surface: allow the surface itself (tolerance) but reject what is behind nearer geometry
        vis = fp.visible(r.astype(np.float32), u, v) & inr
    else:
        vis = inr
    hw = TOLERANCE_M / np.maximum(r, 0.5) * PANO_W / (2 * np.pi)  # px
    pieces = []
    for pu, pv, pr, pvis, phw in split_at_seam(u, v, r, vis.astype(np.float64), hw):
        pieces.append(ProjectedPolyline(pu, pv, pr, pvis > 0.5, phw))
    return pieces


def draw_polyline(canvas: np.ndarray, piece: ProjectedPolyline, color, scale: float = 1.0, min_px: int = 1, occluded_color=None) -> None:
    """Draw visible sub-segments solid; occluded ones in `occluded_color` (None = skip). `scale` maps
    full-res px to canvas px."""
    u = piece.u * scale
    v = piece.v * scale
    for i in range(len(u) - 1):
        seg_vis = piece.visible[i] and piece.visible[i + 1]
        col = color if seg_vis else occluded_color
        if col is None:
            continue
        th = max(min_px, int(round((piece.half_width_px[i] + piece.half_width_px[i + 1]) * scale)))  # 2*hw average
        p0 = (int(round(u[i])), int(round(v[i])))
        p1 = (int(round(u[i + 1])), int(round(v[i + 1])))
        cv2.line(canvas, p0, p1, col, th, lineType=cv2.LINE_AA)


def draw_point(canvas: np.ndarray, u: float, v: float, r: float, color, scale: float = 1.0) -> None:
    rad = max(2, int(round(TOLERANCE_M / max(r, 0.5) * PANO_W / (2 * np.pi) * scale)))
    cv2.circle(canvas, (int(round(u * scale)), int(round(v * scale))), rad, color, -1, lineType=cv2.LINE_AA)


def render_objects(objects, R: np.ndarray, C: np.ndarray, fp: FrameProducts | None, class_ids: dict[str, int] | None = None, scale: float = 0.25, r_max: float = R_MAX, occluded_value: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise JVF objects into (mask [h,w] uint8 of class ids, occluded_mask [h,w] uint8).

    class_ids: jvfcode -> id (1..255). Objects with no id are skipped. `scale` sets output size (0.25 -> 2000x1000).
    """
    h, w = int(PANO_H * scale), int(PANO_W * scale)
    mask = np.zeros((h, w), np.uint8)
    occ = np.zeros((h, w), np.uint8)
    for obj in objects:
        cid = class_ids.get(obj.jvfcode, 0) if class_ids else 1
        if cid == 0:
            continue
        if obj.geom_type == "Point":
            u, v, r, el = geometry.world_to_pano(obj.coords, R, C, dtype=np.float64)
            if not (R_MIN <= r[0] <= r_max):
                continue
            vis = fp.visible(r.astype(np.float32), u, v)[0] if fp is not None else True
            draw_point(mask if vis else occ, float(u[0]), float(v[0]), float(r[0]), int(cid), scale)
            continue
        coords = obj.coords if obj.geom_type == "LineString" else np.concatenate([obj.coords, obj.coords[:1]])
        for piece in project_polyline(coords, R, C, fp, r_max):
            draw_polyline(mask, piece, int(cid), scale)
            draw_polyline(occ, ProjectedPolyline(piece.u, piece.v, piece.r, ~piece.visible, piece.half_width_px), int(cid), scale)
    return mask, occ
