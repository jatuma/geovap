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

from geovap.domain.model import geometry
from ..config import DEG_PER_PX, OUT_DIR, PANO_H, PANO_W
from ..poses import Poses
from geovap.domain.model.rig import RigModel
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
    u, v, r, el = geometry.world_to_pano(cf.xyz, R[0], C[0], PANO_W, PANO_H)
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


# --------------------------------------------------------------------------------------------
# S2 residual maps: edge-ICP (rig identity) residual structure by (azimuth, elevation) and by
# (azimuth, 1/range), on the same 120-frame calibration selection as `calib.fit.run`.
#
# du, dv sign convention (full-resolution px, same as `icp.EdgeICP.structure`): photo edge pixel
# minus projected cloud-silhouette pixel — `du = edge_col - proj_col`, `dv = edge_row - proj_row`
# (icp.py: `du = ec - x[ok]`, `dv = er - y[ok]`, `ec`/`er` = nearest photo-edge pixel). A positive
# du means the photo's edge lies at a larger azimuth (more to the right) than the projected cloud
# point. `az` is the camera-frame azimuth of the projected cloud point, 0..360 deg (0 = the pano
# seam = yaw direction), read straight from `geometry.world_to_pano`'s `u / PANO_W * 360`.

AZ_BIN_DEG = 5.0
EL_BIN_DEG = 10.0
N_INV_R_BINS = 6
INV_R_LO, INV_R_HI = 1.0 / 40.0, 1.0 / 1.0  # spans R_MIN..R_MAX
R_REPORT_M = 4.4  # near-field range the fitted amplitude is reported at (07 doc: up to 11 px there)
MIN_BIN_N = 20  # bins with fewer associations report a NaN median


def _fourier_design(az_deg: np.ndarray, inv_r: np.ndarray, order: int) -> np.ndarray:
    """[N, 1+2*order] design for `du ~ (1/r) * (a0 + sum_k a_k cos(k az) + b_k sin(k az))`."""
    az = np.radians(np.asarray(az_deg, dtype=np.float64))
    cols = [np.ones_like(az)]
    for k in range(1, order + 1):
        cols.append(np.cos(k * az))
        cols.append(np.sin(k * az))
    return np.asarray(inv_r, dtype=np.float64)[:, None] * np.stack(cols, axis=1)


def _fourier_fit(du: np.ndarray, az_deg: np.ndarray, inv_r: np.ndarray, order: int) -> dict:
    """Least-squares fit of `du ~ (1/r) f(az)`, f a Fourier series to `order`; R^2 and the
    peak-to-peak amplitude of f(az)/2 evaluated at r = R_REPORT_M, in px."""
    X = _fourier_design(az_deg, inv_r, order)
    coef, *_ = np.linalg.lstsq(X, du, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((du - pred) ** 2))
    ss_tot = float(np.sum((du - du.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    az_grid = np.linspace(0.0, 360.0, 721)
    f_grid = _fourier_design(az_grid, np.ones_like(az_grid), order) @ coef  # f(az) at r=1 m
    amp = float((f_grid.max() - f_grid.min()) / 2.0 / R_REPORT_M)
    return {"order": order, "r2": round(r2, 4), "coef": coef.round(4).tolist(), "amplitude_at_4.4m_px": round(amp, 3)}


def residual_maps(n_frames: int = 120, seed: int = 0, n_points: int = 30_000, window: float = 20.0, out_dir: Path = OUT_DIR / "diag", log=print) -> dict:
    """Edge-ICP residual structure at rig identity, on the calibration frame selection.

    Bins median du/dv (full-res px) and counts by (az 5deg, el 10deg) and by (az 5deg, 1/r in
    N_INV_R_BINS bins spanning 1/40..1/1 m^-1); saves `residual_maps.npz` (grids) +
    `residual_maps.json` (summary) + `residual_maps.png` (heatmaps) to `out_dir`. Fits
    `du ~ (1/r) f(az)` with f a Fourier series (order 1 vs order 5/6) to test for Ladybug
    stitching-seam parallax: a periodic du(az) with ~5-6 sign flips whose amplitude grows with
    1/r. Returns the summary dict (also the `verdict`: "present" / "absent" / "inconclusive").
    """
    from ..cloud_store import CloudStore
    from geovap.domain.model.frames import FrameIndex
    from ..poses import load_poses
    from geovap.domain.model.rig import IDENTITY
    from ..vehicle_mask import MASK_PATH, VehicleMask
    from . import icp as I
    from .fit import select_frames

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = CloudStore()
    poses = load_poses()
    fi = FrameIndex(poses, IDENTITY)
    vmask = VehicleMask() if MASK_PATH.exists() else None
    frames_idx = select_frames(poses, n_frames, seed)
    frames = [I.prepare_frame(int(f), store, fi, vmask, n_points) for f in frames_idx]
    frames = [f for f in frames if len(f.xyz) >= 200]
    log(f"prepared {len(frames)} calibration frames, {sum(len(f.xyz) for f in frames)} edge points")

    icp = I.EdgeICP(frames, poses)
    theta0 = np.zeros(len(icp.free))
    _, meta = icp.residuals(theta0, window, return_meta=True)
    az = np.concatenate([m["az"] for m in meta])
    el = np.concatenate([m["el"] for m in meta])
    r = np.concatenate([m["r"] for m in meta])
    du = np.concatenate([m["du"] for m in meta]) / icp.s
    dv = np.concatenate([m["dv"] for m in meta]) / icp.s
    n = len(du)
    inv_r = 1.0 / np.clip(r, 0.5, None)
    log(f"n associations = {n} (window {window} CHAM px)")

    from scipy.stats import binned_statistic_2d

    az_edges = np.arange(0.0, 360.0 + AZ_BIN_DEG, AZ_BIN_DEG)
    el_edges = np.arange(-90.0, 90.0 + EL_BIN_DEG, EL_BIN_DEG)
    invr_edges = np.linspace(INV_R_LO, INV_R_HI, N_INV_R_BINS + 1)

    def _bin(y, edges, values):
        med = binned_statistic_2d(az, y, values, statistic="median", bins=[az_edges, edges]).statistic
        cnt = binned_statistic_2d(az, y, values, statistic="count", bins=[az_edges, edges]).statistic.astype(np.int64)
        med = np.where(cnt >= MIN_BIN_N, med, np.nan)
        return med, cnt

    du_azel, cnt_azel = _bin(el, el_edges, du)
    dv_azel, _ = _bin(el, el_edges, dv)
    du_azr, cnt_azr = _bin(inv_r, invr_edges, du)
    dv_azr, _ = _bin(inv_r, invr_edges, dv)

    fit1 = _fourier_fit(du, az, inv_r, 1)
    fit5 = _fourier_fit(du, az, inv_r, 5)
    fit6 = _fourier_fit(du, az, inv_r, 6)

    # Coverage across the 6 1/r bins: edge-ICP associations are depth-edge / sky-silhouette points,
    # which on this rural scene are almost all distant rooflines / horizon, not near-field facades
    # or curbs — so most or all of the 6 bins spanning 1/40..1/1 (r = 40..1 m) can end up empty. The
    # "amplitude at 4.4 m" is then an extrapolation beyond the data, not a measurement, and the
    # near-field sign-flip count is meaningless if that bin itself has no data.
    bin_n = cnt_azr.sum(0)  # associations per 1/r bin, summed over azimuth
    populated_bins = int((bin_n >= MIN_BIN_N * 5).sum())  # bins with enough data to be usable at all
    near_populated = bool(bin_n[-1] >= MIN_BIN_N * 5)
    r_covered_min = float(1.0 / inv_r[np.argmax(inv_r)]) if n else float("nan")  # closest r actually seen

    # sign-flip count of median du(az) in the nearest (largest 1/r) range bin with data
    near_col = du_azr[:, -1]
    sign = np.sign(near_col[np.isfinite(near_col)])
    sign = sign[sign != 0]
    n_flips = int(np.sum(np.diff(sign) != 0)) if len(sign) > 1 else 0

    r2_gain = fit6["r2"] - fit1["r2"]
    if n < 20_000:
        verdict = "inconclusive (too few associations)"
    elif populated_bins <= 1:
        verdict = f"inconclusive (no near-field associations: all {n} within r >= {r_covered_min:.1f} m, only the outermost 1/r bin has data — cannot test amplitude growth with 1/r)"
    elif not near_populated:
        verdict = f"inconclusive (near-field 1/r bin empty; associations span r >= {r_covered_min:.1f} m only)"
    elif n_flips >= 4 and r2_gain > 0.02:
        verdict = "present"
    elif n_flips <= 2 and r2_gain < 0.01:
        verdict = "absent"
    else:
        verdict = "inconclusive"

    summary = {
        "n_frames": len(frames), "n_assoc": int(n), "window_cham_px": window,
        "az_bin_deg": AZ_BIN_DEG, "el_bin_deg": EL_BIN_DEG, "n_inv_r_bins": N_INV_R_BINS, "inv_r_range": [INV_R_LO, INV_R_HI],
        "du_median_px": float(np.median(du)), "dv_median_px": float(np.median(dv)),
        "fourier_order1": fit1, "fourier_order5": fit5, "fourier_order6": fit6,
        "near_field_sign_flips": n_flips, "r2_gain_order1_to_6": round(r2_gain, 4),
        "inv_r_bin_counts": bin_n.astype(int).tolist(), "inv_r_populated_bins": populated_bins, "r_covered_min_m": round(r_covered_min, 2),
        "verdict": verdict,
    }
    np.savez_compressed(
        out_dir / "residual_maps.npz",
        az_edges=az_edges, el_edges=el_edges, invr_edges=invr_edges,
        du_azel=du_azel, dv_azel=dv_azel, cnt_azel=cnt_azel,
        du_azr=du_azr, dv_azr=dv_azr, cnt_azr=cnt_azr,
        frames=np.array([f.frame for f in frames]),
    )
    (out_dir / "residual_maps.json").write_text(json.dumps(summary, indent=1, default=float))
    _plot_residual_maps(az_edges, el_edges, invr_edges, du_azel, dv_azel, du_azr, dv_azr, out_dir)
    log("residual_maps summary: " + json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, default=float))
    log(f"  fourier R^2: order1={fit1['r2']:.3f} order5={fit5['r2']:.3f} order6={fit6['r2']:.3f}  "
        f"amplitude@4.4m: order1={fit1['amplitude_at_4.4m_px']:.2f}px order6={fit6['amplitude_at_4.4m_px']:.2f}px  "
        f"near-field sign flips={n_flips}  verdict={verdict}")
    return summary


def _plot_residual_maps(az_edges, el_edges, invr_edges, du_azel, dv_azel, du_azr, dv_azr, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    az_c = (az_edges[:-1] + az_edges[1:]) / 2
    el_c = (el_edges[:-1] + el_edges[1:]) / 2
    invr_c = (invr_edges[:-1] + invr_edges[1:]) / 2
    finite = np.concatenate([g[np.isfinite(g)] for g in (du_azel, dv_azel, du_azr, dv_azr)])
    vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 5.0

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    for ax, grid, title, ylab, yc in (
        (axs[0, 0], du_azel, "du(az, el)", "elevation [deg]", el_c),
        (axs[0, 1], dv_azel, "dv(az, el)", "elevation [deg]", el_c),
        (axs[1, 0], du_azr, "du(az, 1/r)", "1/r [1/m]", invr_c),
        (axs[1, 1], dv_azr, "dv(az, 1/r)", "1/r [1/m]", invr_c),
    ):
        im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[az_edges[0], az_edges[-1], yc[0], yc[-1]])
        ax.set_xlabel("azimuth [deg]")
        ax.set_ylabel(ylab)
        ax.set_title(f"{title}  (photo edge - projected cloud, px)")
        fig.colorbar(im, ax=ax, label="px")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "residual_maps.png", dpi=120)
    plt.close(fig)
