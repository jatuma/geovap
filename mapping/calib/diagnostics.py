"""Where is the residual? Per-frame, per-sector (du, dv) between rendered intensity and photo by phase
correlation of gradient magnitudes, plus the fits the docs ask for (02 §7.4 / §11 phase 2):
  dv(az) = a + b sin(az) + c cos(az)  -> roll / pitch boresight;   du vs speed -> dt;   |shift| vs 1/r -> lever arm.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

from .. import geometry
from ..config import DEG_PER_PX, OUT_DIR, PANO_W
from ..poses import Poses
from ..rig import RigModel
from .objective import OBJ_H, OBJ_W, CalibFrame, render_intensity

N_AZ_SECTORS = 8
EL_BANDS = ((-60.0, -20.0), (-20.0, 20.0))  # low (road/verge), high (facades/vegetation)
PX_DEG = 360.0 / OBJ_W  # deg per objective pixel


@dataclass
class SectorShift:
    frame: int
    az_centre_deg: float
    el_band: int
    du_px: float  # positive = photo content lies at larger u than the render
    dv_px: float
    error: float
    n_valid: int
    median_range_m: float
    speed_mps: float
    yaw_deg: float


def _sector_shift(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> tuple[float, float, float]:
    """Phase correlation of gradient magnitudes inside mask m (zeros outside), upsampled 10x."""
    if m.mean() < 0.2:
        return np.nan, np.nan, np.nan
    aa = np.where(m, a, 0.0)
    bb = np.where(m, b, 0.0)
    aa = (aa - aa[m].mean()) * m
    bb = (bb - bb[m].mean()) * m
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1])).astype(np.float32)
    shift, err, _ = phase_cross_correlation(bb * win, aa * win, upsample_factor=10, normalization=None)
    return float(shift[1]), float(shift[0]), float(err)  # (du, dv, err): shift to apply to `a` to match `b`


def frame_shifts(cf: CalibFrame, poses: Poses, rig: RigModel) -> list[SectorShift]:
    R, C = geometry.frame_rotations(poses, rig, np.array([cf.frame]))
    img, valid = render_intensity(cf, R[0], C[0])
    u, v, r, el = geometry.world_to_pano(cf.xyz, R[0], C[0])
    gx, gy = cv2.Sobel(img, cv2.CV_32F, 1, 0, 3) / 8, cv2.Sobel(img, cv2.CV_32F, 0, 1, 3) / 8
    ga = np.hypot(gx, gy)
    gb = np.hypot(cf.photo_gx, cf.photo_gy)
    valid_all = valid & cf.valid_photo
    out = []
    sec_w = OBJ_W // N_AZ_SECTORS
    for s in range(N_AZ_SECTORS):
        c0 = s * sec_w
        az_c = (c0 + sec_w / 2) * PX_DEG
        for bi, (e0, e1) in enumerate(EL_BANDS):
            r0 = int((90 - e1) / 180 * OBJ_H)
            r1 = int((90 - e0) / 180 * OBJ_H)
            m = valid_all[r0:r1, c0 : c0 + sec_w]
            du, dv, err = _sector_shift(ga[r0:r1, c0 : c0 + sec_w], gb[r0:r1, c0 : c0 + sec_w], m)
            in_sec = (u >= c0 / OBJ_W * PANO_W) & (u < (c0 + sec_w) / OBJ_W * PANO_W) & (el >= e0) & (el < e1)
            med_r = float(np.median(r[in_sec])) if in_sec.any() else np.nan
            out.append(SectorShift(cf.frame, az_c, bi, du * (OBJ_W and PANO_W / OBJ_W), dv * (PANO_W / OBJ_W), err, int(m.sum()), med_r, float(poses.speed[cf.frame]), float(poses.yaw[cf.frame])))
    return out


def analyse(shifts: list[SectorShift]) -> dict:
    """Fit the diagnostic models. Shifts in full-res px; 1 px = DEG_PER_PX deg."""
    S = [s for s in shifts if np.isfinite(s.du_px) and s.n_valid > 2000]
    if len(S) < 10:
        return {"n": len(S)}
    az = np.radians([s.az_centre_deg for s in S])
    du = np.array([s.du_px for s in S])
    dv = np.array([s.dv_px for s in S])
    inv_r = np.array([1.0 / max(s.median_range_m, 0.5) for s in S])
    speed = np.array([s.speed_mps for s in S])
    # dv = a + b sin(az) + c cos(az): a ~ pitch-like constant? (no: constant dv is a pitch/omega mix; see docs)
    A = np.column_stack([np.ones_like(az), np.sin(az), np.cos(az)])
    coef_dv, *_ = np.linalg.lstsq(A, dv, rcond=None)
    res_dv = dv - A @ coef_dv
    # du = a + b * speed + c * (1/r)
    B = np.column_stack([np.ones_like(az), speed, inv_r])
    coef_du, *_ = np.linalg.lstsq(B, du, rcond=None)
    res_du = du - B @ coef_du
    return {
        "n": len(S),
        "du_median_px": float(np.median(du)),
        "dv_median_px": float(np.median(dv)),
        "du_mad_px": float(np.median(np.abs(du - np.median(du)))),
        "dv_mad_px": float(np.median(np.abs(dv - np.median(dv)))),
        "dv_fit": {"const_px": float(coef_dv[0]), "sin_az_px": float(coef_dv[1]), "cos_az_px": float(coef_dv[2]), "resid_mad_px": float(np.median(np.abs(res_dv)))},
        "dv_fit_deg": {"const": float(coef_dv[0] * DEG_PER_PX), "omega_like(sin)": float(coef_dv[1] * DEG_PER_PX), "phi_like(cos)": float(coef_dv[2] * DEG_PER_PX)},
        "du_fit": {"const_px": float(coef_du[0]), "per_mps_px": float(coef_du[1]), "per_inv_r_px": float(coef_du[2]), "resid_mad_px": float(np.median(np.abs(res_du)))},
        "du_const_deg(kappa_like)": float(coef_du[0] * DEG_PER_PX),
        # du/speed slope in px per (m/s): at range r a time offset dt shifts by dt*speed/r rad -> use median r
        "dt_estimate_s": float(coef_du[1] * np.radians(DEG_PER_PX) * np.median(1 / inv_r)),
    }


def save(shifts: list[SectorShift], summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary, "shifts": [asdict(s) for s in shifts]}, indent=1))


def plot(shifts: list[SectorShift], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = [s for s in shifts if np.isfinite(s.du_px) and s.n_valid > 2000]
    az = np.array([s.az_centre_deg for s in S])
    du = np.array([s.du_px for s in S])
    dv = np.array([s.dv_px for s in S])
    band = np.array([s.el_band for s in S])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for b, col in ((0, "tab:blue"), (1, "tab:orange")):
        m = band == b
        axs[0].scatter(az[m] + (b * 3), dv[m], s=6, c=col, label=f"el band {EL_BANDS[b]}")
        axs[1].scatter(az[m] + (b * 3), du[m], s=6, c=col)
    axs[0].set_xlabel("azimuth in image [deg]")
    axs[0].set_ylabel("dv [px] (render→photo)")
    axs[0].legend()
    axs[1].set_xlabel("azimuth in image [deg]")
    axs[1].set_ylabel("du [px]")
    axs[2].scatter([1 / max(s.median_range_m, 0.5) for s in S], np.hypot(du, dv), s=6)
    axs[2].set_xlabel("1 / median range [1/m]")
    axs[2].set_ylabel("|shift| [px]")
    for a in axs:
        a.grid(alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "shifts.png", dpi=120)
    plt.close(fig)
