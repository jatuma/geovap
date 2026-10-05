"""`align`: image -> pose re-registration -- which trajectory pose does a photo actually belong to?

Some frames' export.csv poses are several metres / seconds off (clusters in passes 11, 12, 20-22; also
turning frames). For a frame we render the cloud's stored RGB at low resolution from candidate poses:
  * the recorded pass at t_k + dt, dt in [-DT_MAX, +DT_MAX]           (time offset along the pass)
  * frames of OTHER passes whose camera centre is within XPASS_M     (pose taken from the wrong pass)
and score each candidate by the masked, zero-mean normalised cross-correlation of the rendered grey
image against the photo grey, maximised over a cyclic column shift (a yaw offset). The best candidate
is refined (dt step 0.05 s, yaw 0.1 deg).

Ported from `mapping/align.py` (library) + `mapping/cli/align_frames.py` (CLI, now this module's
`main()`), turned into the `align` stage of `geovap.stages.register`.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from geovap.domain.model import geometry
from geovap.domain.math import depth as zbuffer
from geovap.runtime.store import CloudStore
from geovap.domain.model.poses import Poses
from geovap.runtime.panos import pano_path
from geovap.stages.prepare.products import gather_candidates
from geovap.io.images import load_pano_rgb
from geovap.stages.base.cli import stage_main
from geovap.stages.base.spec import StageSpec, registry

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings

W, H = 500, 250
DT_MAX = 8.0
DT_STEP = 0.25
XPASS_M = 15.0
N_POINTS = 400_000
GATHER_R = 90.0  # enough to cover +-8 s at 10 m/s around the recorded position
EL_MIN, EL_MAX = -45.0, 8.0  # rows compared (sky and vehicle excluded)
MIN_DE_GAIN = 1.5  # accept another pose only if the median colour ΔE improves by at least this much

YAW_RATE_THRESH_DEG_S = 8.0
TIME_BUDGET_S = 2.5 * 3600.0  # wall-clock budget for the whole run, at --workers workers
N_WORKERS_DEFAULT = 8
N_STRAIGHT_DEFAULT = 100
N_STRAIGHT_CAPPED = 40


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
            from geovap.runtime import settings

            _sensor = settings.get().sensor
            vv, uu = np.mgrid[0:H, 0:W]
            self.mask_static &= ~vmask(uu * (_sensor.pano_w / W), vv * (_sensor.pano_h / H))

    # ------------------------------------------------------------------ data per frame
    @staticmethod
    def _edges(g: np.ndarray) -> np.ndarray:
        g = cv2.GaussianBlur(g, (0, 0), 1.2)
        e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        return np.sqrt(e)  # compress dynamic range so a few strong edges do not dominate

    def photo_gray(self, k: int) -> np.ndarray:
        """Edge-magnitude image of the photo (the comparison feature)."""
        g = cv2.cvtColor(load_pano_rgb(pano_path(self.poses, k)), cv2.COLOR_RGB2GRAY)
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
        from geovap.runtime import settings

        _sensor = settings.get().sensor
        u, v, r, el = geometry.world_to_pano(xyz, R, C, _sensor.pano_w, _sensor.pano_h)
        keep = zbuffer.range_filter(r, _sensor.r_min, _sensor.r_max)
        s = W / _sensor.pano_w
        depth, ids = zbuffer.splat(u[keep] * s, v[keep] * s, r[keep], np.arange(keep.sum(), dtype=np.uint32), W, H, _sensor)
        valid = np.isfinite(depth)
        img = np.zeros((H, W), np.float32)
        img[valid] = gray[keep][ids[valid]]
        # fill small holes so cell boundaries do not become edges, then edge magnitude; empty = 0
        filled = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        e = self._edges(filled)
        e[~cv2.dilate(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)] = 0.0
        return e, valid

    @staticmethod
    def shift_to_yaw_deg(sh: int, w: int = W) -> float:
        """Cyclic column shift `sh` from `score` (defined so that ``photo == np.roll(render, sh)``,
        e.g. photo[x] == render[x - sh]) -> the yaw offset in degrees to ADD to the yaw used to
        render, so that re-rendering at (yaw + offset) matches the photo.

        Derivation under the CURRENT pano column convention in `geometry.py`
        (``u = ((180 - az) mod 360) / 360 * W``, az = atan2(y, x) in camera axes x-fwd/y-left, so
        columns run clockwise with the seam at the rear): for a fixed world point, increasing the
        render yaw by `delta` decreases az by `delta` (one-to-one, since yaw only rotates the world
        into the vehicle frame about z before roll/pitch), hence increases u by
        ``-(-delta) * W / 360 = +delta * W / 360``. So rendering at yaw + delta reproduces the
        yaw-render shifted by ``np.roll(render, delta * W / 360)``. Since `score` returns `sh` such
        that ``photo == np.roll(render, sh)``, matching the photo requires
        ``delta = sh * 360 / W`` -- i.e. the offset has the SAME sign as `sh` under this convention
        (this flips relative to the older ``u = (az mod 360) / 360 * W`` convention, under which the
        correct formula would have been ``-sh * 360 / W``).
        """
        return sh * 360.0 / w

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
        from geovap.domain.math import colour_metrics as metrics
        from geovap.runtime import settings

        _sensor = settings.get().sensor
        o, roll, pitch, yaw = self.pose_at(pass_id, t)
        R = geometry.vehicle_rotation(np.array([yaw + yaw_off]), np.array([roll]), np.array([pitch]))[0]
        tw = np.abs(gps - t) <= 45.0
        u, v, r, el = geometry.world_to_pano(xyz[tw], R, o, _sensor.pano_w, _sensor.pano_h)
        ok = (r > 3.5) & (r <= _sensor.r_max) & (el > -45) & (el < 20)
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
        from geovap.domain.math.sampling import PanoSampler

        photo = self.photo_gray(k)
        xyz, gray, gps, rgb = self.gather(k)
        self.ps = PanoSampler(load_pano_rgb(pano_path(self.poses, k)), footprint=False, gradient=False)
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
            yaw_ncc = self.shift_to_yaw_deg(sh, W)
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


# ================================================================================================
# `mapping/cli/align_frames.py`, folded in below as this module's stage + `main()`.
#
# Re-registers frames against the point cloud (`Aligner`) for the turning frames (|yaw_rate| >
# YAW_RATE_THRESH_DEG_S, `geovap.stages.register.screen.yaw_rates`) plus a set of straight, clean
# frames evenly spaced from the workspace's `clean_frames.json` "clean" list; writes
# `out/poses/align.json` (`save`'s format: a list of Alignment dicts).
#
# Runs a single-frame probe first to estimate the total wall-clock at `--workers` workers; if that
# estimate exceeds ~2.5 h the straight-frame count is capped to 40 and the cap is logged (skip with
# --no-cap).

_G: dict = {}


def _init() -> None:
    from geovap.runtime.store import open_store
    from geovap.runtime.pose_tables import load as load_poses
    from geovap.runtime import settings
    from geovap.stages.prepare.masks import VehicleMask

    _G["poses"] = load_poses()
    _G["store"] = open_store(poses=_G["poses"])
    _s = settings.get()
    _mask_path = _s.workspace.vehicle_mask
    vmask = VehicleMask(_mask_path, *_s.sensor.pano) if _mask_path.exists() else None
    _G["aligner"] = Aligner(_G["store"], _G["poses"], vmask)


def _job(k: int) -> Alignment:
    r = _G["aligner"].align(int(k))
    _G["store"].release()  # keep this worker's RSS small (memmap pages), like screen.py / colorize.py
    return r


def straight_frames(n: int, s: "Settings | None" = None) -> np.ndarray:
    """n frames evenly spaced (by index into the sorted list) from the workspace's `clean_frames.json`
    "clean" class."""
    from geovap.runtime import settings as _settings

    s = s or _settings.get()
    clean = np.array(sorted(s.workspace.clean_frames("clean")))
    if n >= len(clean):
        return clean
    idx = np.unique(np.linspace(0, len(clean) - 1, n).round().astype(int))
    return clean[idx]


def build_frame_list(poses: Poses, n_straight: int) -> tuple[np.ndarray, np.ndarray]:
    """(turning, straight) frame index arrays, disjoint."""
    from geovap.stages.register.screen import yaw_rates

    yr = np.nan_to_num(yaw_rates(poses), nan=0.0)
    turning = np.flatnonzero(np.abs(yr) > YAW_RATE_THRESH_DEG_S)
    straight = np.setdiff1d(straight_frames(n_straight), turning)
    return turning, straight


def _pctile_summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "median": round(float(med), 3), "iqr": round(float(q3 - q1), 3), "min": round(float(x.min()), 3), "max": round(float(x.max()), 3)}


def summarise(results: list[Alignment], poses: Poses, kinds: dict[int, str]) -> dict:
    out: dict = {"n_total": len(results)}
    for kind in ("turning", "straight"):
        dt = [r.dt_s for r in results if kinds.get(r.frame) == kind]
        out[f"dt_s_{kind}"] = _pctile_summary(np.array(dt))
    out["yaw_offset_deg"] = _pctile_summary(np.array([r.yaw_offset_deg for r in results]))
    out["n_suspicious"] = int(sum(r.suspicious for r in results))
    out["n_src_pass_mismatch"] = int(sum(1 for r in results if r.src_pass != int(poses.pass_id[r.frame])))
    out["n_turning"] = sum(1 for r in results if kinds.get(r.frame) == "turning")
    out["n_straight"] = sum(1 for r in results if kinds.get(r.frame) == "straight")
    return out


def run_align(s: "Settings", *, workers: int = N_WORKERS_DEFAULT, n_straight: int = N_STRAIGHT_DEFAULT, out: Path | None = None, no_cap: bool = False) -> Path:
    from multiprocessing import Pool
    from geovap.runtime.pose_tables import load as load_poses

    out = out or (s.workspace.poses / "align.json")
    poses = load_poses(s=s)
    turning, straight = build_frame_list(poses, n_straight)
    print(f"turning frames (|yaw_rate| > {YAW_RATE_THRESH_DEG_S} deg/s): {len(turning)}; straight (clean, evenly spaced): {len(straight)}")

    if not no_cap and len(turning) + len(straight) > 0:
        _init()
        probe_frame = int(turning[0]) if len(turning) else int(straight[0])
        t0 = time.time()
        _job(probe_frame)
        per_frame_s = time.time() - t0
        n_total = len(turning) + len(straight)
        est_h = per_frame_s * n_total / workers / 3600.0
        print(f"probe: frame {probe_frame} took {per_frame_s:.1f} s single-process -> est. {est_h:.2f} h total with {workers} workers")
        if per_frame_s * n_total / workers > TIME_BUDGET_S and n_straight > N_STRAIGHT_CAPPED:
            straight = np.setdiff1d(straight_frames(N_STRAIGHT_CAPPED, s), turning)
            n_total = len(turning) + len(straight)
            print(f"CAPPED: estimated runtime {est_h:.2f} h exceeds the {TIME_BUDGET_S/3600:.1f} h budget -> "
                  f"reduced to turning frames + {N_STRAIGHT_CAPPED} straight frames ({n_total} total, "
                  f"est. {per_frame_s * n_total / workers / 3600:.2f} h)")

    frames = np.sort(np.concatenate([turning, straight])).astype(int)
    kinds = {int(k): "turning" for k in turning} | {int(k): "straight" for k in straight}
    print(f"aligning {len(frames)} frames with {workers} workers")

    t0 = time.time()
    with Pool(workers, initializer=_init) as pool:
        results = list(pool.imap_unordered(_job, [int(k) for k in frames], chunksize=1))
    results.sort(key=lambda r: r.frame)
    print(f"aligned {len(results)} frames in {(time.time()-t0)/60:.1f} min")

    out = Path(out)
    save(results, out)
    print(f"wrote {out}")

    summary = summarise(results, poses, kinds)
    (out.parent / "align_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("summary: " + json.dumps(summary, indent=1, default=float))
    return out


# ================================================================================================ stage
class Align:
    spec = StageSpec(
        name="align", after=("store",), est_min=8,
        summary="re-register frames against the cloud (turning + straight clean frames)",
    )

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        return {"export_csv": s.poses.source_file()}

    def outputs(self, s: "Settings") -> list[Path]:
        return [s.workspace.poses / "align.json"]

    def metrics(self, s: "Settings") -> dict:
        try:
            d = json.loads((s.workspace.poses / "align.json").read_text())
            yaw = [r.get("yaw_offset_deg") for r in d if isinstance(r, dict) and r.get("yaw_offset_deg") is not None]
            return {"n_total": len(d), "median_yaw_offset_deg": float(np.median(yaw)) if yaw else None}
        except Exception as e:  # noqa: BLE001 - metrics must always be reportable
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", *, workers: int = N_WORKERS_DEFAULT, straight: int = N_STRAIGHT_DEFAULT, out: Path | None = None, no_cap: bool = False) -> None:
        run_align(s, workers=workers, n_straight=straight, out=out, no_cap=no_cap)


STAGE = registry.add(Align())


def _add_options(p) -> None:
    p.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
    p.add_argument("--straight", type=int, default=N_STRAIGHT_DEFAULT)
    p.add_argument("--out", default=None)
    p.add_argument("--no-cap", action="store_true", help="skip the runtime probe / budget cap")


def _to_opts(args) -> dict:
    return {"workers": args.workers, "straight": args.straight, "out": Path(args.out) if args.out else None, "no_cap": args.no_cap}


def main(argv=None) -> int:
    return stage_main(STAGE, argv, add_options=_add_options, to_opts=_to_opts)


if __name__ == "__main__":
    raise SystemExit(main())
