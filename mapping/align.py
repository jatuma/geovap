"""Image -> pose re-registration: which trajectory pose does a photo actually belong to?

Some frames' export.csv poses are several metres / seconds off (clusters in passes 11, 12, 20-22; also
turning frames). For a frame we render the cloud's stored RGB at low resolution from candidate poses:
  * the recorded pass at t_k + dt, dt in [-DT_MAX, +DT_MAX]           (time offset along the pass)
  * frames of OTHER passes whose camera centre is within XPASS_M     (pose taken from the wrong pass)
and score each candidate by the masked, zero-mean normalised cross-correlation of the rendered grey
image against the photo grey, maximised over a cyclic column shift (a yaw offset). The best candidate
is refined (dt step 0.05 s, yaw 0.1 deg).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import geometry, zbuffer
from .cloud_store import CloudStore
from .config import PANO_H, PANO_W, R_MAX, R_MIN
from .poses import Poses
from .products import gather_candidates
from .sample import load_pano_rgb

W, H = 500, 250
DT_MAX = 8.0
DT_STEP = 0.25
XPASS_M = 15.0
N_POINTS = 400_000
GATHER_R = 90.0  # enough to cover +-8 s at 10 m/s around the recorded position
EL_MIN, EL_MAX = -45.0, 8.0  # rows compared (sky and vehicle excluded)
MIN_DE_GAIN = 1.5  # accept another pose only if the median colour ΔE improves by at least this much


@dataclass
class Alignment:
    frame: int
    score0: float  # median colour ΔE76 at the recorded pose (lower is better)
    score: float  # median colour ΔE76 at the chosen pose
    dt_s: float  # time offset along the recorded pass (0 if another pass was chosen)
    yaw_offset_deg: float  # additional yaw applied to the trajectory yaw
    src_pass: int  # pass whose trajectory the pose comes from
    src_time: float  # trajectory time of the chosen pose
    origin: tuple[float, float, float]
    roll: float
    pitch: float
    yaw: float  # final yaw (trajectory yaw + offset)
    suspicious: bool  # score0 much worse than score, or pose moved a lot


class Aligner:
    def __init__(self, store: CloudStore, poses: Poses, vmask=None):
        self.store = store
        self.poses = poses
        self.vmask = vmask
        rows = np.arange(H)
        el = 90.0 - (rows + 0.5) / H * 180.0
        self.row_ok = (el >= EL_MIN) & (el <= EL_MAX)
        self.mask_static = np.repeat(self.row_ok[:, None], W, 1)
        if vmask is not None:
            vv, uu = np.mgrid[0:H, 0:W]
            self.mask_static &= ~vmask(uu * (PANO_W / W), vv * (PANO_H / H))

    # ------------------------------------------------------------------ data per frame
    @staticmethod
    def _edges(g: np.ndarray) -> np.ndarray:
        g = cv2.GaussianBlur(g, (0, 0), 1.2)
        e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        return np.sqrt(e)  # compress dynamic range so a few strong edges do not dominate

    def photo_gray(self, k: int) -> np.ndarray:
        """Edge-magnitude image of the photo (the comparison feature)."""
        g = cv2.cvtColor(load_pano_rgb(self.poses.path(k)), cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
        return self._edges(g)

    def gather(self, k: int, seed: int = 0):
        C = self.poses.origin[k]
        parts = self.store.query_disc(float(C[0]), float(C[1]), GATHER_R)
        xyz = np.concatenate([self.store.tile(t.name).xyz_m(rows) for t, rows in parts])
        rgb = np.concatenate([np.asarray(self.store.tile(t.name).rgb[rows]) for t, rows in parts])
        gps = np.concatenate([np.asarray(self.store.tile(t.name).gps_time[rows]) for t, rows in parts])
        rng = np.random.default_rng(seed + k)
        if len(xyz) > N_POINTS:
            sel = rng.choice(len(xyz), N_POINTS, replace=False)
            xyz, rgb, gps = xyz[sel], rgb[sel], gps[sel]
        gray = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).astype(np.float32)
        return xyz, gray, gps, rgb

    # ------------------------------------------------------------------ rendering + scoring
    def render(self, xyz, gray, R, C) -> tuple[np.ndarray, np.ndarray]:
        u, v, r, el = geometry.world_to_pano(xyz, R, C)
        keep = zbuffer.range_filter(r, R_MIN, R_MAX)
        s = W / PANO_W
        depth, ids = zbuffer.splat(u[keep] * s, v[keep] * s, r[keep], np.arange(keep.sum(), dtype=np.uint32), W, H)
        valid = np.isfinite(depth)
        img = np.zeros((H, W), np.float32)
        img[valid] = gray[keep][ids[valid]]
        # fill small holes so cell boundaries do not become edges, then edge magnitude; empty = 0
        filled = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        e = self._edges(filled)
        e[~cv2.dilate(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = 0.0
        return e, valid

    def score(self, img, valid, photo) -> tuple[float, int]:
        """max over cyclic column shifts of zero-mean NCC of edge images, over the static mask restricted to
        pixels the render covers; multiplied by sqrt(coverage) so a pose that sees little cloud scores low."""
        cov = cv2.dilate(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        m = self.mask_static & cov
        frac = m.sum() / max(self.mask_static.sum(), 1)
        if frac < 0.05:
            return -1.0, 0
        a = np.where(m, img, 0.0)
        a = np.where(m, a - a[m].mean(), 0.0)
        na = np.sqrt((a**2).sum()) + 1e-6
        b = np.where(m, photo, 0.0)
        b = np.where(m, b - b[m].mean(), 0.0)
        # cyclic column shift via FFT cross-correlation, row-summed
        Fa = np.fft.rfft(a, axis=1)
        Fb = np.fft.rfft(b, axis=1)
        cc = np.fft.irfft(Fa * np.conj(Fb), n=W, axis=1).sum(0)  # cc[s] = sum a[x] b[x - s]
        nb = np.sqrt((b**2).sum()) + 1e-6
        ncc = cc / (na * nb) * np.sqrt(frac)
        s = int(np.argmax(ncc))
        shift = s if s <= W // 2 else s - W
        return float(ncc[s]), -shift  # photo rolled by +shift (returned) matches the render

    # ------------------------------------------------------------------ candidates
    def candidates(self, k: int):
        """[(label, pass, time)] to try."""
        p = self.poses
        out = [("dt", int(p.pass_id[k]), float(p.t[k] + dt)) for dt in np.arange(-DT_MAX, DT_MAX + 1e-9, DT_STEP)]
        C = p.origin[k, :2]
        d = np.hypot(p.origin[:, 0] - C[0], p.origin[:, 1] - C[1])
        near = np.flatnonzero((d <= XPASS_M) & (p.pass_id != p.pass_id[k]))
        for j in near:
            for dt in np.arange(-2.0, 2.01, 0.5):
                out.append(("xpass", int(p.pass_id[j]), float(p.t[j] + dt)))
        return out

    def pose_at(self, pass_id: int, t: float):
        o, r, pi, y = self.poses.interp(np.array([t]), np.array([pass_id]))
        return o[0], float(r[0]), float(pi[0]), float(y[0])

    # ------------------------------------------------------------------ stage 2: colour agreement
    def colour_de(self, k_photo, xyz, rgb, gps, pass_id, t, yaw_off, n=40_000) -> float:
        """Median CIE76 between the photo (nearest pixel) and the points' stored RGB, for a pose at
        (pass_id, t) turned by yaw_off. Lower is better: ~5-7 for a correct pose, 10+ for a wrong one."""
        from . import metrics

        o, roll, pitch, yaw = self.pose_at(pass_id, t)
        R = geometry.vehicle_rotation(np.array([yaw + yaw_off]), np.array([roll]), np.array([pitch]))[0]
        tw = np.abs(gps - t) <= 45.0
        u, v, r, el = geometry.world_to_pano(xyz[tw], R, o)
        ok = (r > 3.5) & (r <= R_MAX) & (el > -45) & (el < 20)
        if self.vmask is not None:
            ok &= ~self.vmask(u, v)
        idx = np.flatnonzero(ok)
        if len(idx) < 2000:
            return 99.0
        if len(idx) > n:
            idx = idx[:: max(1, len(idx) // n)]
        s = self.ps.nearest(u[idx], v[idx])
        d76, _ = metrics.delta_e(s, rgb[tw][idx])
        return float(np.median(d76))

    def align(self, k: int) -> Alignment:
        from .sample import PanoSampler

        photo = self.photo_gray(k)
        xyz, gray, gps, rgb = self.gather(k)
        self.ps = PanoSampler(load_pano_rgb(self.poses.path(k)), footprint=False, gradient=False)
        p = self.poses
        pass0, t0 = int(p.pass_id[k]), float(p.t[k])

        def evaluate(pass_id, t):
            o, roll, pitch, yaw = self.pose_at(pass_id, t)
            R = geometry.vehicle_rotation(np.array([yaw]), np.array([roll]), np.array([pitch]))[0]
            tw = np.abs(gps - t) <= 45.0
            if tw.sum() < 1000:
                return -1.0, 0
            img, valid = self.render(xyz[tw], gray[tw], R, o)
            return self.score(img, valid, photo)

        # stage 1: edge-NCC over all candidates (position + cyclic yaw)
        cands = [(pass0, t0)] + [(pid, t) for _, pid, t in self.candidates(k)]
        scored = []
        for pid, t in cands:
            sc, sh = evaluate(pid, t)
            scored.append((sc, sh, pid, t))
        scored.sort(key=lambda c: -c[0])
        self.last_candidates = scored
        # stage 2: colour ΔE decides among the recorded pose and the best NCC proposals
        de0 = self.colour_de(k, xyz, rgb, gps, pass0, t0, 0.0)
        best = (de0, pass0, t0, 0.0)
        tried = {(pass0, round(t0, 2))}
        for sc, sh, pid, t in scored[:8]:
            if (pid, round(t, 2)) in tried:
                continue
            tried.add((pid, round(t, 2)))
            yaw_ncc = sh * 360.0 / W
            for yo in (yaw_ncc, yaw_ncc - 3, yaw_ncc + 3, 0.0):
                de = self.colour_de(k, xyz, rgb, gps, pid, t, yo)
                if de < best[0]:
                    best = (de, pid, t, yo)
        de, pid, t, yo = best
        if de0 - de < MIN_DE_GAIN:  # keep the recorded pose unless clearly better
            de, pid, t, yo = de0, pass0, t0, 0.0
        else:
            # local refinement: time then yaw
            for step in (0.25, 0.1):
                for t2 in (t - step, t + step):
                    d2 = self.colour_de(k, xyz, rgb, gps, pid, t2, yo)
                    if d2 < de:
                        de, t = d2, t2
            for step in (2.0, 0.7):
                for yo2 in (yo - step, yo + step):
                    d2 = self.colour_de(k, xyz, rgb, gps, pid, t, yo2)
                    if d2 < de:
                        de, yo = d2, yo2
        o, roll, pitch, yaw = self.pose_at(pid, t)
        yaw_final = (yaw + yo + 180.0) % 360.0 - 180.0
        moved = float(np.linalg.norm(o - p.origin[k]))
        return Alignment(
            frame=k, score0=round(de0, 3), score=round(de, 3), dt_s=round(t - t0, 3) if pid == pass0 else 0.0,
            yaw_offset_deg=round(yo, 2), src_pass=pid, src_time=t, origin=tuple(np.round(o, 3).tolist()), roll=roll, pitch=pitch, yaw=yaw_final,
            suspicious=bool(moved > 1.0 or abs(yo) > 2.0 or de0 > 9.0),
        )


def save(alignments: list[Alignment], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(a) for a in alignments], indent=1))
