"""Fuse per-view class probabilities back into ERP (torch, GPU), with the levelled view geometry of views.py.

For every ERP output pixel: camera ray -> view coordinates (x, y) through M = R_view R_lev R_cam^T; gather
the view's probability with `grid_sample`, weight by a cos^2 fall-off to the view edge (E2 recipe, 03 SS4.3),
sum, normalise. Argmax happens once, after fusion.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..config import PANO_H, PANO_W
from .views import view_rotation

_RAYS: dict = {}


def erp_rays(out_h: int, out_w: int, device) -> torch.Tensor:
    """[H,W,3] camera-axis unit rays for ERP cell centres (same convention as geometry.pano_rays)."""
    key = (out_h, out_w, str(device))
    if key not in _RAYS:
        u = (torch.arange(out_w, device=device, dtype=torch.float64) + 0.5) / out_w * 2 * torch.pi
        v = (torch.arange(out_h, device=device, dtype=torch.float64) + 0.5) / out_h
        el = torch.pi / 2 - v * torch.pi
        ce = torch.cos(el)[:, None]
        rays = torch.stack([ce * torch.cos(u)[None, :], ce * torch.sin(u)[None, :], torch.sin(el)[:, None].expand(out_h, out_w)], -1)
        _RAYS[key] = rays
    return _RAYS[key]


def view_grid(rays: torch.Tensor, R_cam: np.ndarray, R_lev: np.ndarray, view) -> tuple[torch.Tensor, torch.Tensor]:
    """(grid [1,H,W,2] for grid_sample, weight [1,1,H,W]) of one view."""
    yaw, pitch, fov = view
    M = view_rotation(yaw, pitch) @ R_lev @ R_cam.T
    Mt = torch.from_numpy(M).to(rays.device, rays.dtype)
    d = rays @ Mt.T  # [H,W,3] view axes (right, up, fwd)
    f = 0.5 / np.tan(np.deg2rad(fov / 2))
    fwd = d[..., 2]
    ok = fwd > 1e-6
    fs = torch.where(ok, fwd, torch.ones_like(fwd))
    x = d[..., 0] / fs * f
    y = -d[..., 1] / fs * f
    inside = ok & (x.abs() <= 0.5) & (y.abs() <= 0.5)
    grid = torch.stack([x * 2, y * 2], -1).unsqueeze(0).float()
    dist = torch.maximum(x.abs(), y.abs()) / 0.5
    w = torch.cos(dist.clamp(max=1.0) * (torch.pi / 2)) ** 2
    w = torch.where(inside, w, torch.zeros_like(w)).float()[None, None]
    return grid, w


def fuse(view_probs: list[torch.Tensor], views: list, R_cam: np.ndarray, R_lev: np.ndarray, out_h: int, out_w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """view_probs[i] [C,S,S] -> (probs [C,out_h,out_w] float32 normalised, coverage [out_h,out_w] bool)."""
    device = view_probs[0].device
    rays = erp_rays(out_h, out_w, device)
    C = view_probs[0].shape[0]
    acc = torch.zeros(C, out_h, out_w, device=device)
    wsum = torch.zeros(1, out_h, out_w, device=device)
    for p, view in zip(view_probs, views):
        grid, w = view_grid(rays, R_cam, R_lev, view)
        s = F.grid_sample(p.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        acc += s * w[0]
        wsum += w[0]
    coverage = wsum[0] > 1e-3
    return acc / wsum.clamp(min=1e-3), coverage


def to_common(probs: torch.Tensor, native_to_common: torch.Tensor, n_common: int) -> torch.Tensor:
    out = torch.zeros(n_common, *probs.shape[1:], device=probs.device, dtype=probs.dtype)
    out.index_add_(0, native_to_common, probs)
    return out
