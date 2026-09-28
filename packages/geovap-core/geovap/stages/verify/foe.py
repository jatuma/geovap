"""Focus-of-expansion check: model-free second opinion on the camera model's azimuth convention.

`sphere.py` catches a Potree/panos.py convention that has drifted from `geovap.domain.model.geometry`,
but both an offline render and a point-cloud pinhole render still ultimately trust *some* piece of
this repo's geometry (poses, `frame_rotations`). This check adds one more, cheaper, independent
signal that does not even need the point cloud: the FOCUS OF EXPANSION (FOE) of frame-to-frame
photo motion.

On a (nearly) straight stretch, everything the vehicle drives towards appears to radiate outward
from a single column of the panorama as it moves closer - the FOE - and that column is, by
definition, the panorama column the camera model assigns to the travel direction
`C[k+1] - C[k]`. We measure the photo FOE purely from pixels (template-matching a horizontal-flow
profile along the horizon band between frame k and k+1 - no camera model, no `cam_to_pano`
involved), then compare it to the column `geovap.domain.model.geometry.cam_to_pano` predicts for the same
travel direction via `R[k]`. If the model's azimuth convention is reflected (as the pre-2026-09-16
`experiments/common/camera.py` formula was), the two disagree by ~half the panorama width; a
correct model gives near-zero circular difference.

Frames where the vehicle moved < 1 m or turned > 2 deg between k and k+1 (pose table), or where the flow
divergence score is below 0.3 (no usable expansion pattern), are skipped
(FOE is meaningless when the "forward" motion is dominated by yaw/turning). Frames whose photo,
pose, or flow-profile computation is unavailable are marked "skipped" rather than failing the run.

    uv run python -m geovap.stages.verify.foe --frames 100,250,498,591,900 \
        --poses corrected [--out <workspace>/out/consolidated/validation/foe_check.json] \
        [--width 4000 --height 2000]

Verdict: pass if the median circular |u_photo/W - u_model/W| over all non-skipped frames is <= 0.1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from geovap.stages.base.cli import add_dataset_flags, configure_from, describe
from geovap.stages.base.spec import StageSpec, registry

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings

MIN_MOVE_M = 1.0
MAX_TURN_DEG = 2.0  # straight segments only: a turn moves the FOE off the travel column
PASS_THRESHOLD = 0.1  # fraction of W, circular
MIN_DIVERGENCE_SCORE = 0.3  # flow-sign agreement below this = no usable expansion pattern (repetitive/blank horizon)

DEFAULT_FRAMES = (100, 150, 250, 300, 400, 498, 550, 650, 700, 750, 800, 850, 950, 1000, 1100, 1200)


def flow_profile(a: np.ndarray, b: np.ndarray, w: int, h: int, y0: int | None = None, y1: int | None = None,
                  bin_px: int = 100, search_px: int = 300) -> np.ndarray:
    """[[col_centre, du_px, ncc], ...] horizontal shift of each `bin_px`-wide column bin of the
    horizon band (`y0:y1`, default the middle third) of `a`, found by template-matching against
    `b` within +-`search_px` columns (wrapping across the seam). Pure pixels, no camera model."""
    import cv2

    if y0 is None:
        y0 = h // 3
    if y1 is None:
        y1 = h - h // 3
    prof = []
    for c in range(0, w, bin_px):
        t = a[y0:y1, c:c + bin_px]
        if t.shape[1] < bin_px:
            continue
        lo = c - search_px
        cols = np.arange(lo, c + bin_px + search_px) % w
        s = b[y0:y1][:, cols]
        r = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
        j = int(np.argmax(r))
        prof.append((c + bin_px / 2, j - search_px, float(r.max())))
    return np.asarray(prof)


def photo_foe_u(prof: np.ndarray, w: int, good_ncc: float = 0.5, near_deg_frac: float = 0.5, min_good: int = 10) -> tuple[float | None, float | None, int]:
    """Column (px) where the flow field diverges (FOE): the candidate column that best splits the
    profile into "moving left of it -> negative du" / "right of it -> positive du" (mean of
    sign(offset)*sign(du) over bins within `near_deg_frac` of the width, maximised over candidates).
    Returns (u_px, score, n_good_bins); (None, None, n_good) if too few usable bins."""
    if len(prof) == 0:
        return None, None, 0
    good = prof[:, 2] > good_ncc
    if good.sum() < min_good:
        return None, None, int(good.sum())
    u = prof[good, 0]
    du = prof[good, 1]
    near = w * near_deg_frac
    best = (None, -9.0)
    for c in np.arange(0, w, w / 160):
        rel = (u - c + w / 2) % w - w / 2
        m = np.abs(rel) < near
        if m.sum() < 6:
            continue
        score = float((np.sign(rel[m]) * np.sign(du[m])).mean())
        if score > best[1]:
            best = (c, score)
    return best[0], best[1], int(good.sum())


def model_travel_u(R: np.ndarray, mv: np.ndarray, w: int, h: int) -> float:
    """Column `geovap.domain.model.geometry.cam_to_pano` assigns to the travel direction `mv` (world), for a
    frame with rotation `R` (world -> camera, from `frame_rotations`)."""
    from geovap.domain.model import geometry

    d_cam = (mv / np.linalg.norm(mv)) @ R.T
    u, _, _, _ = geometry.cam_to_pano(d_cam[None, :], w, h)
    return float(u[0])


def circ_diff_frac(a_frac: float, b_frac: float) -> float:
    """|a - b| on the circle [0, 1), shortest way round."""
    d = (a_frac - b_frac + 0.5) % 1.0 - 0.5
    return abs(d)


def check_frame(poses, k: int, w: int, h: int) -> dict:
    from geovap.domain.model import geometry
    from geovap.domain.model.rig import IDENTITY
    from geovap.runtime.panos import pano_path

    row: dict = {"frame": k, "frame_next": k + 1}
    try:
        R, C = geometry.frame_rotations(poses, IDENTITY, [k, k + 1])
    except Exception as e:  # noqa: BLE001
        row.update(status="skipped", reason=f"poses: {e}")
        return row
    mv = C[1] - C[0]
    dist = float(np.linalg.norm(mv))
    yaw, _, _ = geometry.euler_from_vehicle_rotation(R)
    dyaw = float((yaw[1] - yaw[0] + 180.0) % 360.0 - 180.0)
    row.update(move_m=dist, dyaw_deg=dyaw)
    if dist < MIN_MOVE_M:
        row.update(status="skipped", reason=f"moved {dist:.2f} m < {MIN_MOVE_M} m")
        return row
    if abs(dyaw) > MAX_TURN_DEG:
        row.update(status="skipped", reason=f"turned {dyaw:+.1f} deg > {MAX_TURN_DEG} deg")
        return row

    try:
        import cv2

        a = cv2.imread(pano_path(poses, k), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(pano_path(poses, k + 1), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            row.update(status="skipped", reason="photo missing/unreadable")
            return row
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    except Exception as e:  # noqa: BLE001
        row.update(status="skipped", reason=f"photo load failed: {e}")
        return row

    prof = flow_profile(a, b, w, h)
    u_photo, score, n_good = photo_foe_u(prof, w)
    row["n_good_bins"] = n_good
    if u_photo is None:
        row.update(status="skipped", reason=f"only {n_good} usable flow bins")
        return row

    if score is None or score < MIN_DIVERGENCE_SCORE:
        row.update(status="skipped", reason=f"weak flow divergence (score {score:+.2f} < {MIN_DIVERGENCE_SCORE})", divergence_score=score)
        return row

    u_model = model_travel_u(R[0], mv, w, h)
    u_photo_frac, u_model_frac = u_photo / w, u_model / w
    diff = circ_diff_frac(u_photo_frac, u_model_frac)
    row.update(status="ok", divergence_score=score, u_photo_frac=u_photo_frac, u_model_frac=u_model_frac,
               abs_diff_frac=diff)
    return row


def run_foe_check(frames: list[int], poses_source: str, out_path: Path, width: int = 4000, height: int = 2000) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from geovap.runtime.pose_tables import load as load_poses

        poses = load_poses(poses_source)
    except Exception as e:  # noqa: BLE001
        result = {"poses": poses_source, "rows": [{"frame": k, "status": "skipped", "reason": f"load_poses failed: {e}"} for k in frames],
                  "verdict": {"pass": False, "note": f"could not load pose table {poses_source!r}: {e}"}}
        out_path.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return result

    rows = []
    for k in frames:
        try:
            row = check_frame(poses, k, width, height)
        except Exception as e:  # noqa: BLE001
            row = {"frame": k, "status": "skipped", "reason": f"unexpected error: {e}"}
        rows.append(row)
        if row["status"] == "ok":
            print(f"f{k:04d}->{k+1:04d}: move {row['move_m']:.1f} m, dyaw {row['dyaw_deg']:+.1f} deg | "
                  f"photo FOE u/W={row['u_photo_frac']:.3f} (score {row['divergence_score']:+.2f}) | "
                  f"model u/W={row['u_model_frac']:.3f} | diff {row['abs_diff_frac']:.3f}")
        else:
            print(f"f{k:04d}->{k+1:04d}: SKIPPED ({row.get('reason')})")

    ok_rows = [r for r in rows if r["status"] == "ok"]
    diffs = [r["abs_diff_frac"] for r in ok_rows]
    median_diff = float(np.median(diffs)) if diffs else None
    verdict = {
        "n_frames": len(frames), "n_ok": len(ok_rows), "n_skipped": len(frames) - len(ok_rows),
        "median_abs_diff_frac": median_diff,
        "pass": median_diff is not None and median_diff <= PASS_THRESHOLD,
        "note": f"pass iff median circular |u_photo/W - u_model/W| over non-skipped frames <= {PASS_THRESHOLD} "
                "(a reflected/half-turn azimuth convention would put this near 0.5, or near 0/1 at the seam).",
    }
    result = {"poses": poses_source, "width": width, "height": height, "rows": rows, "verdict": verdict}
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(verdict, indent=1))
    return result


# ================================================================================================ stage
class Foe:
    spec = StageSpec(
        name="foe", after=(), est_min=5,
        summary="focus-of-expansion: model-free second opinion on the camera model's azimuth convention",
    )
    cli_args: tuple[str, ...] = ()

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        return {}

    def outputs(self, s: "Settings") -> list[Path]:
        return [s.workspace.consolidated / "validation" / "foe_check.json"]

    def metrics(self, s: "Settings") -> dict[str, Any]:
        try:
            p = s.workspace.consolidated / "validation" / "foe_check.json"
            return json.loads(p.read_text()).get("verdict", {})
        except Exception as e:  # noqa: BLE001 - metrics must always be reportable
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", *, frames: list[int] | None = None, poses_source: str | None = None,
            out: str | None = None, width: int = 4000, height: int = 2000) -> None:
        fr = list(frames) if frames else list(DEFAULT_FRAMES)
        src = poses_source if poses_source is not None else s.pose_table
        out_path = Path(out) if out else s.workspace.consolidated / "validation" / "foe_check.json"
        run_foe_check(fr, src, out_path, width=width, height=height)


STAGE = registry.add(Foe())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_flags(ap)
    ap.add_argument("--status", action="store_true", help="report whether this stage is done, then exit")
    ap.add_argument("--frames", default=",".join(str(f) for f in DEFAULT_FRAMES), help="comma list of frame ids k (uses k and k+1)")
    ap.add_argument("--out", default=None, help="default: <workspace>/out/consolidated/validation/foe_check.json")
    ap.add_argument("--width", type=int, default=4000, help="downsample width for the flow/model comparison")
    ap.add_argument("--height", type=int, default=2000, help="downsample height for the flow/model comparison")
    args = ap.parse_args(argv)
    s = configure_from(args)

    if args.status:
        print(describe(STAGE, s))
        return 0

    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    poses_source = s.pose_table
    out_path = Path(args.out) if args.out else s.workspace.consolidated / "validation" / "foe_check.json"
    result = run_foe_check(frames, poses_source, out_path, width=args.width, height=args.height)
    return 0 if result["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
