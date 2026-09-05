"""Forward and inverse mapping between world (S-JTSK E,N,H) and equirectangular pixels.

Forward chain (identity rig) is exactly `experiments/common/camera.py`, written as matrices:

    y = yaw, r = -roll, p = -pitch                      (radians; note the NEGATIVE signs)
    Ry = [[ cy, sy, 0], [-sy, cy, 0], [0, 0, 1]]
    Rr = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    Rp = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    R_v = Rp @ Rr @ Ry                                  world -> vehicle body (x fwd, y left, z up)
    x_cam = R_b @ R_v @ (P - C_cam)                     R_b = boresight (identity by default)
    az = atan2(y, x), el = atan2(z, hypot(x, y))
    u = (az mod 360) / 360 * W,  v = (90 - el) / 180 * H

Inverse: d_cam from (u, v); d_world = (R_b R_v)^T d_cam; P = C_cam + range * d_world.
"""
from __future__ import annotations

import numpy as np

from .config import PANO_H, PANO_W
from .poses import Poses
from .rig import IDENTITY, RigModel


# ---------------------------------------------------------------------------------- rotations
def _rx(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def _ry(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def _rz(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def vehicle_rotation(yaw_deg, roll_deg, pitch_deg) -> np.ndarray:
    """R_v[...,3,3]: world -> vehicle body, with the verified sign convention baked in."""
    y = np.deg2rad(np.asarray(yaw_deg, dtype=np.float64))
    r = np.deg2rad(-np.asarray(roll_deg, dtype=np.float64))
    p = np.deg2rad(-np.asarray(pitch_deg, dtype=np.float64))
    cy, sy = np.cos(y), np.sin(y)
    z, o = np.zeros_like(cy), np.ones_like(cy)
    Ry = np.stack([np.stack([cy, sy, z], -1), np.stack([-sy, cy, z], -1), np.stack([z, z, o], -1)], -2)
    return _ry(p) @ _rx(r) @ Ry  # Rp @ Rr @ Ry  (Rr == _rx(r), Rp == _ry(p))


def boresight_rotation(rig: RigModel) -> np.ndarray:
    """R_b[3,3]: body -> camera. Rz(kappa) @ Ry(phi) @ Rx(omega), angles in degrees."""
    w, f, k = (np.deg2rad(np.float64(a)) for a in rig.boresight_deg)
    return _rz(k) @ _ry(f) @ _rx(w)


def frame_rotations(poses: Poses, rig: RigModel = IDENTITY, idx=None) -> tuple[np.ndarray, np.ndarray]:
    """(R[K,3,3], C[K,3]) for frames `idx` (default: all) under the rig model.

    R maps world vectors to camera axes; C is the camera centre in world coordinates.
    """
    if idx is None:
        idx = np.arange(len(poses))
    idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
    origin, roll, pitch, yaw = poses.pose_at(idx, rig.dt_s)
    R_v = vehicle_rotation(yaw, roll, pitch)  # [K,3,3]
    lever = np.asarray(rig.lever_arm_m, dtype=np.float64)
    C = origin + np.einsum("kji,j->ki", R_v, lever) if np.any(lever) else origin  # R_v^T @ l
    R = boresight_rotation(rig) @ R_v if any(rig.boresight_deg) else R_v
    return R, C


# ---------------------------------------------------------------------------------- forward
def cam_to_pano(x_cam: np.ndarray, w: int = PANO_W, h: int = PANO_H):
    """x_cam[...,3] (float32 ok) -> (u, v, r, el_deg). u in [0,W), v in [0,H]."""
    x, y, z = x_cam[..., 0], x_cam[..., 1], x_cam[..., 2]
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arctan2(z, hxy))
    u = np.mod(az, 360.0) / 360.0 * w
    v = (90.0 - el) / 180.0 * h
    return u, v, r, el


def world_to_cam(P: np.ndarray, R: np.ndarray, C: np.ndarray, dtype=np.float32) -> np.ndarray:
    """P[N,3] float64 world, one frame (R[3,3], C[3]) -> x_cam[N,3] in `dtype`.

    Subtraction in float64 (coordinates have 7 significant digits), then cast. float32 costs
    ~1e-3 px at 40 m, which is far below any calibration signal; use float64 for exact tests.
    """
    d = (np.asarray(P, dtype=np.float64) - C).astype(dtype)
    return d @ R.astype(dtype).T


def world_to_pano(P: np.ndarray, R: np.ndarray, C: np.ndarray, w: int = PANO_W, h: int = PANO_H, dtype=np.float32):
    """P[N,3] world -> (u, v, r, el) for one frame. Outputs [N] in `dtype`."""
    return cam_to_pano(world_to_cam(P, R, C, dtype), w, h)


# ---------------------------------------------------------------------------------- inverse
def pano_rays(u, v, w: int = PANO_W, h: int = PANO_H) -> np.ndarray:
    """Unit direction in CAMERA axes for pixel coordinates (u, v) (continuous; add 0.5 for cell centres)."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    az = np.deg2rad(u / w * 360.0)
    el = np.deg2rad(90.0 - v / h * 180.0)
    ce = np.cos(el)
    return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], -1)


def pano_to_world_ray(u, v, R: np.ndarray, w: int = PANO_W, h: int = PANO_H) -> np.ndarray:
    """Unit direction in WORLD axes for pixel (u, v) of a frame with rotation R (world->camera)."""
    return pano_rays(u, v, w, h) @ R  # R^T applied to row vectors == d @ R


def pano_to_world(u, v, rng, R: np.ndarray, C: np.ndarray, w: int = PANO_W, h: int = PANO_H) -> np.ndarray:
    """World point at Euclidean range `rng` along the pixel ray. rng[N] or scalar."""
    d = pano_to_world_ray(u, v, R, w, h)
    return C + d * np.asarray(rng, dtype=np.float64)[..., None]
