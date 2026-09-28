"""Sphere-convention check: does Potree show the panorama the way our camera model says it should?

For every photo-only screenshot produced by screenshots.py (camera at the sphere centre, known
Potree view yaw/pitch, vertical FOV, viewport), render the SAME view offline from the source
panorama through `mapping.geometry` (the camera model), then correlate the two greyscale images
(NCC after downsampling). A correct `panos.py` convention (AZ_OFFSET, course/pitch/roll
re-parametrisation, texture.repeat.x) gives NCC close to 1; a half-turn or a mirrored texture
gives ~0 or negative.

WHY THE NCC-ALONE CHECK WAS CIRCULAR (2026-09-16): `render_offline` maps the panorama onto the
Potree view using `mapping.geometry.cam_to_pano` - the SAME camera model that `export_panos.py`
used to build the sphere's UV coordinates in the first place. If that model has, say, its azimuth
convention reflected (front/back swapped, as the pre-2026-09-16 `experiments/common/camera.py`
formula was), then BOTH the exported sphere texture and this offline render are wrong in exactly
the same way, and they still agree pixel-for-pixel: NCC = 1.000 for a convention that is simply
wrong. NCC-vs-offline can only ever catch a *drift* between panos.py and geometry.py, never a
shared, wrong convention baked into both.

THE PINHOLE REFERENCE (added 2026-09-16): for the same Potree view (yaw/pitch/fov), also render a
PINHOLE projection of the REGISTERED POINT CLOUD from the camera centre C. This uses only
model-independent 3D geometry - C (world position, from `frame_rotations`), the Potree view axes
(fwd/side/up, the same `potree_rays` used for the offline render), and a plain perspective
projection (x/z, y/z against tan(fov/2)). It never calls `cam_to_pano` or `pano_rays`, so it cannot
share a camera-model bug with either the sphere texture or the offline render: it is a genuinely
independent silhouette of "what should be visible from here, looking this way" straight from the
point cloud's own (E, N, H) coordinates. Comparing that silhouette to the PHOTO screenshot (via
`edge_metric.silhouette_agreement`) is the model-independent half of the check: if the camera
model's azimuth/seam convention is wrong, the photo texture is rotated/mirrored on the sphere and
its edges will line up worse with the pinhole silhouette than the correctly-mapped photo does -
even though the NCC-vs-offline number stays 1.000. Points are painted flat grey (200) with no
depth/colour coding: only the point-density silhouette (roofline, tree crowns, kerbs against the
photo) matters here, not per-point colour.

Potree View (potree.js `class View`): dir = Rz(yaw) Rx(pitch) (0,1,0); side = Rz(yaw) (1,0,0);
up = side x dir; camera.fov is the VERTICAL field of view (three.js PerspectiveCamera).

    uv run python pointcloud-tools/validate/sphere_check.py --shots-dir <screenshots out dir> \
        --variants cloud_panos,panos_corr0,panos_exp180,panos_exp0 --poses corrected [--fov 60] \
        [--no-pinhole]
Variants whose name contains "exp" are rendered with export poses, the others with --poses.
`panos_corr180`, `panos_exp0`, `panos_exp180` are the fixed A/B CONTROL variants; every other
variant is a PRIMARY candidate and must beat all three controls' pinhole-silhouette score on
every shared (frame, yaw).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SHOT_RE = re.compile(r"f(\d{4})_y(\d{3})_photo\.png$")
CONTROL_VARIANTS = {"panos_corr180", "panos_exp0", "panos_exp180"}


def potree_axes(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fwd, side, up) unit WORLD axes of a Potree/three.js perspective view. Pure 3D geometry -
    no camera model, no panorama involved."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # dir = Rz(yaw) Rx(pitch) (0,1,0);  Rx(pitch)(0,1,0) = (0, cos p, sin p)
    fwd = np.array([-sy * cp, cy * cp, sp])
    side = np.array([cy, sy, 0.0])
    up = np.cross(side, fwd)
    return fwd, side, up


def potree_rays(w: int, h: int, yaw: float, pitch: float, fov_deg: float) -> np.ndarray:
    """Unit WORLD rays [h,w,3] for every pixel of a Potree/three.js perspective view."""
    fwd, side, up = potree_axes(yaw, pitch)
    t = np.tan(np.deg2rad(fov_deg) / 2)
    aspect = w / h
    xs = (2 * (np.arange(w) + 0.5) / w - 1) * t * aspect
    ys = (1 - 2 * (np.arange(h) + 0.5) / h) * t
    rays = fwd[None, None] + xs[None, :, None] * side[None, None] + ys[:, None, None] * up[None, None]
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def render_offline(pano_bgr: np.ndarray, R: np.ndarray, w: int, h: int, yaw: float, pitch: float, fov_deg: float) -> np.ndarray:
    import cv2

    from mapping import geometry

    rays_w = potree_rays(w, h, yaw, pitch, fov_deg).reshape(-1, 3)
    d_cam = rays_w @ R.T  # world -> camera axes
    ph, pw = pano_bgr.shape[:2]
    u, v, _, _ = geometry.cam_to_pano(d_cam, pw, ph)
    mx = np.mod(u, pw).astype(np.float32).reshape(h, w)
    my = np.clip(v, 0, ph - 1).astype(np.float32).reshape(h, w)
    return cv2.remap(pano_bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def render_pinhole(P: np.ndarray, C: np.ndarray, w: int, h: int, yaw: float, pitch: float, fov_deg: float, splat_px: int = 3) -> np.ndarray:
    """Grey silhouette-only pinhole render [h,w] uint8 of the registered point cloud `P` [N,3] from
    camera centre `C`, for the given Potree view. Uses ONLY `C` and the Potree view axes
    (`potree_axes`) - no `mapping.geometry.cam_to_pano`/`pano_rays`, so it cannot inherit a camera-
    model convention bug. Every visible point is painted the same grey (200); only occupancy
    (the silhouette) is meaningful, not colour or depth."""
    import cv2

    fwd, side, up = potree_axes(yaw, pitch)
    Q = np.asarray(P, dtype=np.float64) - np.asarray(C, dtype=np.float64)
    z = Q @ fwd
    m = z > 0.5  # in front of the camera, not right on top of it
    Q, z = Q[m], z[m]
    img = np.zeros((h, w), dtype=np.uint8)
    if len(z) == 0:
        return img
    t = np.tan(np.deg2rad(fov_deg) / 2)
    aspect = w / h
    x = (Q @ side) / z / (t * aspect)
    y = (Q @ up) / z / t
    m2 = (np.abs(x) < 1) & (np.abs(y) < 1)
    if not np.any(m2):
        return img
    px = np.clip(((x[m2] + 1) / 2 * w).astype(np.int64), 0, w - 1)
    py = np.clip(((1 - y[m2]) / 2 * h).astype(np.int64), 0, h - 1)
    img[py, px] = 200
    k = max(1, splat_px)
    return cv2.dilate(img, np.ones((k, k), np.uint8))


def query_cloud_points(store, cx: float, cy: float, radius: float) -> np.ndarray:
    pts = [store.xyz_registered(t, rows) for t, rows in store.query_disc(cx, cy, radius)]
    return np.concatenate(pts) if pts else np.empty((0, 3), dtype=np.float64)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def compare(shot_png: Path, pano_path: Path, R: np.ndarray, yaw: float, pitch: float, fov_deg: float, small_w: int = 400) -> dict:
    import cv2

    shot = cv2.imread(str(shot_png), cv2.IMREAD_COLOR)
    if shot is None:
        raise FileNotFoundError(shot_png)
    h, w = shot.shape[:2]
    pano = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
    if pano is None:
        raise FileNotFoundError(pano_path)
    off = render_offline(pano, R, w, h, yaw, pitch, fov_deg)
    small = (small_w, int(round(small_w * h / w)))
    g1 = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(shot, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(off, small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    # also score the two wrong alternatives so the margin is explicit
    g_flip = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(render_offline(pano, R, w, h, yaw + np.pi, pitch, fov_deg), small, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    return {"ncc": ncc(g1, g2), "ncc_half_turn": ncc(g1, g_flip), "ncc_mirror": ncc(g1, g2[:, ::-1]), "offline": off, "w": w, "h": h}


def pinhole_scores(pinhole_img: np.ndarray, shot_png: Path, offline_bgr: np.ndarray) -> dict:
    """silhouette_agreement of the model-independent pinhole render against (a) the photo shot and
    (b) the offline camera-model render, so the two references can be told apart in the output."""
    from edge_metric import silhouette_agreement

    vs_photo = silhouette_agreement(pinhole_img, shot_png)
    vs_offline = silhouette_agreement(pinhole_img, offline_bgr)
    return {
        "pinhole_sil_median_px": vs_photo["median_px"],
        "pinhole_sil_inlier6": vs_photo["inlier6"],
        "pinhole_sil_n": vs_photo["n_sil"],
        "pinhole_sil_vs_offline_median_px": vs_offline["median_px"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots-dir", required=True, help="screenshots.py --out-dir (holds <variant>/fNNNN_yYYY_photo.png)")
    ap.add_argument("--variants", required=True, help="comma list of subdir names under --shots-dir")
    ap.add_argument("--poses", default="corrected", help="pose table for variants not containing 'exp'")
    ap.add_argument("--fov", type=float, default=60.0, help="vertical FOV the page used (index.html default 60)")
    ap.add_argument("--pitch", type=float, default=0.0, help="view pitch (rad) the driver set for every shot")
    ap.add_argument("--panos-dir", default=None, help="dir with f%%04d.jpg to sample (default: original panoramas via poses.path)")
    ap.add_argument("--save-offline", action="store_true", help="write <shot>_offline.png next to each screenshot")
    ap.add_argument("--no-pinhole", action="store_true", help="skip the point-cloud pinhole reference (needs the store; NCC-only check)")
    ap.add_argument("--pinhole-radius", type=float, default=45.0, help="disc radius (m) queried from the store around the camera centre")
    args = ap.parse_args(argv)

    from mapping import geometry
    from mapping.poses import load_poses
    from mapping.rig import IDENTITY

    do_pinhole = not args.no_pinhole
    store_cache: dict[str, object] = {}
    pinhole_cache: dict[tuple, np.ndarray] = {}  # (src, frame, yaw_deg) -> pinhole render (poses-only, model-free)

    def get_store(src: str):
        if src not in store_cache:
            from mapping.cloud_store import open_store

            store_cache[src] = open_store(poses_cache[src])
        return store_cache[src]

    def get_pinhole(src: str, k: int, C: np.ndarray, yaw: float, w: int, h: int) -> np.ndarray:
        key = (src, k, round(np.rad2deg(yaw)))
        if key not in pinhole_cache:
            store = get_store(src)
            P = query_cloud_points(store, C[0], C[1], args.pinhole_radius)
            pinhole_cache[key] = render_pinhole(P, C, w, h, yaw, args.pitch, args.fov)
        return pinhole_cache[key]

    shots_dir = Path(args.shots_dir)
    rows = []
    poses_cache = {}
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        src = "export" if "exp" in variant else args.poses
        if src not in poses_cache:
            poses_cache[src] = load_poses(src)
        poses = poses_cache[src]
        vdir = shots_dir / variant
        for png in sorted(vdir.glob("f*_photo.png")):
            m = SHOT_RE.search(png.name)
            if not m:
                continue
            k, yaw_deg = int(m.group(1)), float(m.group(2))
            R, C = geometry.frame_rotations(poses, IDENTITY, [k])
            R, C = R[0], C[0]
            pano_path = Path(args.panos_dir) / f"f{k:04d}.jpg" if args.panos_dir else Path(poses.path(k))
            yaw_rad = np.deg2rad(yaw_deg)
            res = compare(png, pano_path, R, yaw_rad, args.pitch, args.fov)
            if args.save_offline:
                import cv2

                cv2.imwrite(str(png.with_name(png.stem + "_offline.png")), res["offline"])
            row = {"variant": variant, "poses": src, "frame": k, "yaw_deg": yaw_deg, "ncc": res["ncc"],
                   "ncc_half_turn": res["ncc_half_turn"], "ncc_mirror": res["ncc_mirror"]}
            pin_msg = ""
            if do_pinhole:
                try:
                    pin_img = get_pinhole(src, k, C, yaw_rad, res["w"], res["h"])
                    row.update(pinhole_scores(pin_img, png, res["offline"]))
                    pin_msg = f", pinhole sil {row['pinhole_sil_median_px']:.1f}px" if row["pinhole_sil_median_px"] == row["pinhole_sil_median_px"] else ""
                except Exception as e:  # noqa: BLE001 - store may be unavailable; don't kill the whole run
                    print(f"  (pinhole skipped for {variant} f{k:04d}: {e})", file=sys.stderr)
                    row.update({"pinhole_sil_median_px": None, "pinhole_sil_inlier6": None, "pinhole_sil_n": None, "pinhole_sil_vs_offline_median_px": None})
            rows.append(row)
            print(f"{variant} f{k:04d} y{int(yaw_deg):03d}: ncc {res['ncc']:.3f} (half-turn {res['ncc_half_turn']:.3f}, mirror {res['ncc_mirror']:.3f}){pin_msg}")

    out_csv = shots_dir / "sphere_check.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["variant"])
        w.writeheader()
        w.writerows(rows)

    def _median(vals):
        vals = [v for v in vals if v is not None and v == v]
        return float(np.median(vals)) if vals else None

    summary = {}
    for variant in sorted({r["variant"] for r in rows}):
        vs = [r for r in rows if r["variant"] == variant]
        summary[variant] = {"n": len(vs), "ncc_median": float(np.median([r["ncc"] for r in vs])),
                            "ncc_half_turn_median": float(np.median([r["ncc_half_turn"] for r in vs])),
                            "ncc_mirror_median": float(np.median([r["ncc_mirror"] for r in vs])),
                            "pinhole_sil_median_px": _median([r.get("pinhole_sil_median_px") for r in vs]),
                            "pinhole_sil_inlier6_median": _median([r.get("pinhole_sil_inlier6") for r in vs])}
    # variants whose name carries "180" are the A/B CONTROLS (sphere turned by half a revolution): they are
    # expected to FAIL the model match and to prefer the half-turn alternative. Everything else is a
    # candidate convention and must match the camera model.
    # Convention controls are the *180 sets (half-turned on purpose). panos_exp0 is NOT a convention
    # control: it is the same convention on export poses (a pose A/B set), so it must pass the model
    # match like the primary and is reported, not gated, in the pinhole comparison.
    is_control = {v: ("180" in v) for v in summary}
    ok_primary = {v: s["ncc_median"] > 0.6 and s["ncc_median"] > s["ncc_half_turn_median"] + 0.15 for v, s in summary.items() if not is_control[v]}
    ok_control = {v: s["ncc_half_turn_median"] > s["ncc_median"] for v, s in summary.items() if is_control[v]}

    # model-independent check: the pinhole (point cloud, no camera model) silhouette of the primary
    # variant must sit closer to ITS photo than the half-turned controls' silhouettes sit to THEIR
    # photos - the part the NCC-vs-offline number cannot see (see module docstring). Judged on the
    # per-variant medians plus a per-(frame, yaw) win rate, because single symmetric scenes (a road
    # between two hedges) can score the same either way.
    by_key = {(r["variant"], r["frame"], r["yaw_deg"]): r for r in rows}

    def _pin(r):
        x = r.get("pinhole_sil_median_px")
        return x if x is not None and x == x else None

    pinhole_pass: dict[str, dict] = {}
    for v in summary:
        if is_control[v]:
            continue
        prows = [r for r in rows if r["variant"] == v and _pin(r) is not None]
        wins = n_checked = 0
        failures = []
        for r in prows:
            for c in [c for c in summary if is_control[c]]:
                cr = by_key.get((c, r["frame"], r["yaw_deg"]))
                if cr is None or _pin(cr) is None:
                    continue
                n_checked += 1
                if _pin(r) < _pin(cr):
                    wins += 1
                else:
                    failures.append({"frame": r["frame"], "yaw_deg": r["yaw_deg"], "control": c, "primary_px": _pin(r), "control_px": _pin(cr)})
        med_ok = all(summary[v]["pinhole_sil_median_px"] is not None and summary[c]["pinhole_sil_median_px"] is not None
                     and summary[v]["pinhole_sil_median_px"] < summary[c]["pinhole_sil_median_px"] for c in summary if is_control[c])
        win_rate = wins / n_checked if n_checked else 0.0
        pinhole_pass[v] = {"pass": n_checked > 0 and med_ok and win_rate >= 0.67, "median_beats_all_controls": med_ok,
                           "win_rate_vs_controls": round(win_rate, 3), "n_checked": n_checked, "n_failures": len(failures), "failures": failures[:20]}

    # export-vs-corrected (same convention): informational only
    pinhole_pose_ab = {v: summary[v]["pinhole_sil_median_px"] for v in summary if not is_control[v]}

    ncc_strict = {v: (s["ncc_median"] is not None and s["ncc_median"] >= 0.95) for v, s in summary.items() if not is_control[v]}
    model_independent_pass = {v: bool(ncc_strict.get(v)) and bool(pinhole_pass.get(v, {}).get("pass")) for v in summary if not is_control[v]}

    verdict = {"primary_pass": ok_primary, "control_behaves_as_expected": ok_control,
               "pinhole_pass": pinhole_pass, "pinhole_median_px_same_convention": pinhole_pose_ab,
               "model_independent_pass": model_independent_pass,
               "pass": all(ok_primary.values()) and all(ok_control.values()) and bool(ok_primary)
               and (not do_pinhole or (all(model_independent_pass.values()) and bool(model_independent_pass))),
               "note": "primary (AZ 0) sets must match the camera model (NCC > 0.6 and clearly above the half-turn alternative); "
                       "the *180 control sets must NOT (their half-turn alternative wins). NCC-vs-offline alone is CIRCULAR "
                       "(both use mapping.geometry.cam_to_pano) - model_independent_pass additionally requires NCC(offline, "
                       "photo) >= 0.95 AND the pinhole (point-cloud, model-free) silhouette to be closer to the photo than "
                       "every control variant's pinhole silhouette on the same frame/yaw. That combination is the "
                       "AZ_OFFSET_DEG = 0 verdict."}
    (shots_dir / "sphere_check.json").write_text(json.dumps({"variants": summary, "verdict": verdict}, indent=1))
    print(json.dumps({"variants": summary, "verdict": verdict}, indent=1))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
