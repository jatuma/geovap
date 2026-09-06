"""Horizon-levelled gnomonic views of a panorama (numpy + torch helpers).

A view is (yaw_deg, pitch_deg, fov_deg) relative to the *levelled* vehicle frame: yaw relative to the
vehicle heading, pitch relative to the true horizon (roll/pitch of the frame removed). View pixel (x right,
y down) -> ray in view axes (right, up, fwd) -> levelled body axes (x fwd, y left, z up) -> world -> camera
(R_k) -> ERP (u, v). The inverse `pano_to_view_xy` maps camera-axis rays to normalised view coords and is
shared by the fusion of view probabilities back to ERP.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .. import geometry
from ..config import PANO_H, PANO_W, ZB_H, ZB_W
from ..frame_select import FrameIndex
from ..poses import load_poses
from . import classes as C
from .areas import SEGDS_DIR
from .render_labels import LABELS_DIR, frames_arg

VIEWS_DIR = SEGDS_DIR / "views"
VIEW_SIZE = 1024
FOV = 90.0
# ring (8) + down ring (4, roads) + up ring (4, offset by 45 deg): the +35..+45 deg band at the ring seams is
# otherwise uncovered (gnomonic vertical half-angle at the view edge is only 35 deg for FOV 90).
VIEWS: list[tuple[float, float, float]] = (
    [(float(y), 0.0, FOV) for y in range(0, 360, 45)]
    + [(float(y), -45.0, FOV) for y in (0, 90, 180, 270)]
    + [(float(y), 45.0, FOV) for y in (45, 135, 225, 315)]
)
PAD_PX = PANO_W // 4 + 64


def view_name(yaw: float, pitch: float) -> str:
    return f"y{int(round(yaw)) % 360:03d}_p{int(round(pitch)):+03d}"


def view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """R_view [3,3]: levelled body axes (x fwd, y left, z up) -> view axes (right, up, fwd).

    Positive yaw turns the view left (CCW from above, same sense as the vehicle yaw); positive pitch looks up.
    """
    y, p = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    cy, sy, cp, sp = np.cos(y), np.sin(y), np.cos(p), np.sin(p)
    fwd = np.array([cy * cp, sy * cp, sp])  # view forward in body axes
    left = np.array([-sy, cy, 0.0])
    up = np.cross(fwd, left)  # = right-handed completion; (fwd, left, up) is body-like
    up /= np.linalg.norm(up)
    right = -left
    return np.stack([right, up, fwd])  # rows = view axes expressed in body axes


def level_rotation(poses, k: int) -> np.ndarray:
    """R_lev [3,3]: world -> levelled body of frame k (yaw only)."""
    return geometry.vehicle_rotation(poses.yaw[k], 0.0, 0.0)


def view_rays(size: int, fov_deg: float) -> np.ndarray:
    """[S,S,3] unit rays in view axes (right, up, fwd) for pixel centres."""
    f = 0.5 / np.tan(np.deg2rad(fov_deg / 2))
    lin = (np.arange(size) + 0.5) / size - 0.5
    gx, gy = np.meshgrid(lin, lin)
    d = np.stack([gx, -gy, np.full_like(gx, f)], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def view_to_pano_maps(R_cam: np.ndarray, R_lev: np.ndarray, view, size: int = VIEW_SIZE, w: int = PANO_W, h: int = PANO_H) -> tuple[np.ndarray, np.ndarray]:
    """(map_u, map_v) float32 [S,S]: ERP pixel coordinates (full-res, continuous) sampled by each view pixel."""
    yaw, pitch, fov = view
    Rv = view_rotation(yaw, pitch)
    d_view = view_rays(size, fov)
    d_body = d_view @ Rv  # rows: Rv^T d
    d_world = d_body @ R_lev  # R_lev^T d
    d_cam = d_world @ R_cam.T
    u, v, _, _ = geometry.cam_to_pano(d_cam, w, h)
    return u.astype(np.float32), v.astype(np.float32)


def pano_to_view_xy(d_cam: np.ndarray, R_cam: np.ndarray, R_lev: np.ndarray, view) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-axis rays [...,3] -> normalised view coords (x, y in [-0.5, 0.5] inside the FOV), in_front mask."""
    yaw, pitch, fov = view
    Rv = view_rotation(yaw, pitch)
    d_world = d_cam @ R_cam
    d_body = d_world @ R_lev.T
    d_view = d_body @ Rv.T
    f = 0.5 / np.tan(np.deg2rad(fov / 2))
    fwd = d_view[..., 2]
    ok = fwd > 1e-6
    fs = np.where(ok, fwd, 1.0)
    x = d_view[..., 0] / fs * f
    y = -d_view[..., 1] / fs * f
    return x, y, ok


def extract_image(pano_bgr: np.ndarray, map_u: np.ndarray, map_v: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Sample the ERP image with horizontal wrap (pad + shift)."""
    w = pano_bgr.shape[1]
    scale = w / PANO_W
    pad = int(PAD_PX * scale)
    padded = np.concatenate([pano_bgr[:, -pad:], pano_bgr, pano_bgr[:, :pad]], axis=1)
    mu = (map_u * scale + pad).astype(np.float32)
    mv = (map_v * scale).astype(np.float32)
    if interp == cv2.INTER_NEAREST:
        mu = np.floor(mu).astype(np.float32)  # cell-centre semantics: pixel i covers [i, i+1)
        mv = np.floor(mv).astype(np.float32)
    return cv2.remap(padded, mu, mv, interp, borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------------------------- export
_G: dict = {}


def _init():
    _G["poses"] = load_poses()
    _G["fi"] = FrameIndex(_G["poses"])


def export_frame(k: int, poses=None, fi=None, size: int = VIEW_SIZE, out_dir: Path = VIEWS_DIR, with_labels: bool = True) -> dict:
    poses = poses or _G["poses"]
    fi = fi or _G["fi"]
    photo = cv2.imread(poses.path(k))
    lab = cv2.imread(str(LABELS_DIR / f"f{k:04d}.png"), 0) if with_labels else None
    R_lev = level_rotation(poses, k)
    stats = {}
    for view in VIEWS:
        mu, mv = view_to_pano_maps(fi.R[k], R_lev, view, size)
        name = f"f{k:04d}_{view_name(view[0], view[1])}"
        img = extract_image(photo, mu, mv)
        cv2.imwrite(str(out_dir / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if lab is not None:
            lv = extract_image(lab, mu, mv, cv2.INTER_NEAREST)
            cv2.imwrite(str(out_dir / "labels" / f"{name}.png"), lv)
            stats[name] = np.bincount(lv.ravel(), minlength=256)[: C.N_CLASSES].tolist() + [int((lv == 255).sum())]
    return {"frame": k, "views": stats}


def _work(k):
    return export_frame(k)


def build(frames: str = "clean", workers: int = 8, limit: int | None = None, out_dir: Path = VIEWS_DIR) -> None:
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    ks = [k for k in frames_arg(frames) if (LABELS_DIR / f"f{k:04d}.png").exists()][:limit]
    res = []
    with Pool(workers, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, ks, chunksize=2)):
            res.append(r)
            if i % 50 == 0:
                print(f"[{i + 1}/{len(ks)}] frame {r['frame']}")
    res.sort(key=lambda r: r["frame"])
    (out_dir / "views_stats.json").write_text(json.dumps({"views": [list(v) for v in VIEWS], "size": VIEW_SIZE, "frames": res}))
