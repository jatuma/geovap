"""Client A: render point-cloud attributes into a panorama (cloud -> pano).

Everything is a gather over the frame's point_id panorama: attr[point_id]. Labels are copied
(nearest cell), never interpolated. Hole filling copies the nearest valid cell within `fill_px`.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from .cloud_store import CloudStore
from .config import NO_POINT, PANO_H, PANO_W, RENDERS_DIR
from .products import FrameProducts

LAYERS = ("depth", "classification", "intensity", "rgb", "point_id")

# ASPRS-ish palette for the classes present (1 unclassified, 2 ground) + room for 64+ semantic classes
CLASS_PALETTE = np.zeros((256, 3), np.uint8)
CLASS_PALETTE[0] = (40, 40, 40)
CLASS_PALETTE[1] = (200, 200, 200)
CLASS_PALETTE[2] = (140, 90, 40)
_rng = np.random.default_rng(7)
CLASS_PALETTE[3:] = _rng.integers(40, 230, (253, 3))


def fill_holes(valid: np.ndarray, fill_px: int) -> tuple[np.ndarray, np.ndarray]:
    """For each invalid cell within fill_px of a valid one, index of the nearest valid cell.

    Returns (src_rows, src_cols) [H,W] with the cell's own index where no fill applies.
    Wrap along u is handled by padding.
    """
    if fill_px <= 0:
        h, w = valid.shape
        rr, cc = np.indices((h, w))
        return rr, cc
    h, w = valid.shape
    pad = fill_px
    vpad = np.concatenate([valid[:, -pad:], valid, valid[:, :pad]], axis=1)
    dist, (ir, ic) = ndimage.distance_transform_edt(~vpad, return_indices=True)
    ir = ir[:, pad : pad + w]
    ic = (ic[:, pad : pad + w] - pad) % w
    dist = dist[:, pad : pad + w]
    rr, cc = np.indices((h, w))
    use = dist <= fill_px
    return np.where(use, ir, rr), np.where(use, ic, cc)


def gather(store: CloudStore, fp: FrameProducts, layer: str, fill_px: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """(values [H,W(,3)], valid [H,W]) of `layer` for the frame. depth comes from the product itself."""
    pid = fp.point_id
    valid = pid != NO_POINT
    rr, cc = fill_holes(valid, fill_px)
    pid_f = pid[rr, cc]
    valid_f = valid[rr, cc]
    if layer == "depth":
        d = fp.depth_m[rr, cc]
        return np.where(valid_f, d, np.inf).astype(np.float32), valid_f
    if layer == "point_id":
        return np.where(valid_f, pid_f, NO_POINT).astype(np.uint32), valid_f
    ti, local = store.locate(pid_f[valid_f])
    if layer == "rgb":
        out = np.zeros((*pid.shape, 3), np.uint8)
        vals = np.empty((valid_f.sum(), 3), np.uint8)
    elif layer == "intensity":
        out = np.zeros(pid.shape, np.uint16)
        vals = np.empty(valid_f.sum(), np.uint16)
    elif layer == "classification":
        out = np.zeros(pid.shape, np.uint8)
        vals = np.empty(valid_f.sum(), np.uint8)
    else:
        raise ValueError(layer)
    for t_idx in np.unique(ti):
        m = ti == t_idx
        td = store.tile(store.tiles[t_idx].name)
        col = getattr(td, layer)
        vals[m] = col[np.sort(local[m])][np.argsort(np.argsort(local[m]))]  # sorted access on memmap, restore order
    out[valid_f] = vals
    return out, valid_f


def to_png(layer: str, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """8-bit BGR visualisation of a layer."""
    if layer == "depth":
        d = np.where(valid, values, np.nan)
        x = np.clip(np.log1p(np.nan_to_num(d, nan=0.0)) / np.log1p(40.0), 0, 1)
        img = cv2.applyColorMap((255 * (1 - x)).astype(np.uint8), cv2.COLORMAP_TURBO)
    elif layer == "intensity":
        x = np.log1p(values.astype(np.float32)) / np.log1p(65535.0)
        img = cv2.cvtColor((255 * x).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif layer == "classification":
        img = CLASS_PALETTE[values][..., ::-1].copy()
    elif layer == "rgb":
        img = values[..., ::-1].copy()
    elif layer == "point_id":
        img = cv2.applyColorMap((values % 251).astype(np.uint8), cv2.COLORMAP_JET)
    else:
        raise ValueError(layer)
    img[~valid] = 0
    return img


def render_frame(store: CloudStore, fp: FrameProducts, layers=LAYERS, fill_px: int = 3, out_dir: Path | None = None, save_npz: bool = True) -> dict:
    """Render layers; optionally save PNG + NPZ under out_dir/f{frame}_{layer}.png|.npz. Returns {layer: (values, valid)}."""
    res = {}
    for layer in layers:
        vals, valid = gather(store, fp, layer, fill_px)
        res[layer] = (vals, valid)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer, (vals, valid) in res.items():
            cv2.imwrite(str(out_dir / f"f{fp.frame:04d}_{layer}.png"), to_png(layer, vals, valid))
        if save_npz:
            np.savez_compressed(out_dir / f"f{fp.frame:04d}.npz", valid=res[layers[0]][1], **{k: v[0] for k, v in res.items()})
    return res


def overlay_on_photo(photo_bgr: np.ndarray, layer_bgr: np.ndarray, valid: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a z-buffer-resolution layer over the (downscaled) photo."""
    h, w = layer_bgr.shape[:2]
    small = cv2.resize(photo_bgr, (w, h), interpolation=cv2.INTER_AREA)
    out = small.copy()
    out[valid] = (alpha * layer_bgr[valid] + (1 - alpha) * small[valid]).astype(np.uint8)
    return out
