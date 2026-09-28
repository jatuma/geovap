"""Panorama spheres in the Potree viewer (`Potree.Images360`): export + angle convention.

Builds the asset directory the viewer loads next to an octree:

    <cloud>/panos/coordinates.txt   TAB-separated, header + one line per frame
    <cloud>/panos/f%04d.jpg         downscaled equirectangular image
    <cloud>/panos/panos.json        provenance (pose source + hash, rig, frames, image size)

`coordinates.txt` is Potree's own format (`Images360Loader.load`): columns
`file time longitude latitude altitude course pitch roll`, angles in degrees. With the default
identity transform Potree reads longitude/latitude as scene x/y, so we write S-JTSK (E, N, H)
straight through -- the same coordinates PotreeConverter stamps into `metadata.json`.

Course/pitch/roll are NOT our roll/pitch/yaw: Potree turns them into the sphere's orientation as

    R_mesh = Rz(90 - course) @ Ry(-pitch) @ Rx(roll + 90)        (three.js Euler order "ZYX")

so the three numbers are a re-parametrisation of the whole rotation, not the vehicle angles.
`potree_angles` inverts that formula for the rotation the sphere actually needs (`sphere_rotation`).

Sphere convention (three.js r124 `SphereGeometry` + equirectangular texture, derived in
`tests/test_panos.py`): Potree's `Images360` loader sets `texture.repeat.x = -1`, and that is
CORRECT for our panoramas -- a normal photo (lettering readable) seen from inside a three.js
sphere is mirrored, and -1 undoes it; both viewer pages keep it and only expose `?flip=1` as an
A/B override. With repeat.x = -1 the texel of pixel u sits at texture s = 1 - u/w, which makes
the mesh-local direction of image pixel (u, v)

    L = (cos(az) cos(el), sin(el), -sin(az) cos(el)) = POTREE_M @ d_cam

with (az, el) the same angles as `geometry.pano_rays`. Hence R_mesh = (POTREE_M @ R)^T for a
frame with rotation R (world -> camera).

AZ_OFFSET_DEG (default 0) is spun into that rotation about the camera's own vertical axis; it is
0 by derivation. The convention was fixed on 2026-09-16 (the camera model had been reflected
about the lateral axis) -- an older analysis had claimed 0 vs. 180 was settled by NCC correlation
against Potree screenshots via `pointcloud-tools/validate/sphere_check.py`, but that check rendered
through the same wrong camera model on both sides and so could not have detected the error; it is
not evidence either way. What IS evidence: the user visually confirmed, on the export produced by
the OLD camera model with `az_offset_deg = 180` and repeat.x = -1, that the spheres line up
correctly. That fixture (`tests/fixtures/panos_corr180_coordinates.txt`, built from
`tests/fixtures/poses_corrected_34bca9.csv`) is reproduced line-for-line by this derivation with
az 0 -- see `test_new_export_reproduces_user_confirmed_panos_corr180` -- which is what ties the new
math back to a confirmed-correct display. `--az-offset 180` still exists for A/B sets only.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from geovap.domain.model import geometry
from .poses import Poses, load_poses
from geovap.domain.model.rig import IDENTITY, RigModel

# camera axes (x fwd, y left, z up) -> sphere-local axes, for Potree's texture.repeat.x = -1
# (three.js: y up; POTREE_M_old @ Rz(180deg), fixed 2026-09-16 -- see the module docstring)
POTREE_M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
# 0 by derivation; reproduces the user-confirmed display, see the module docstring
AZ_OFFSET_DEG = 0.0

HEADER = "file\ttime\tlongitude\tlatitude\taltitude\tcourse\tpitch\troll"


def _rz_cam(deg: float) -> np.ndarray:
    a = np.deg2rad(float(deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sphere_rotation(R: np.ndarray, az_offset_deg: float = AZ_OFFSET_DEG) -> np.ndarray:
    """R[...,3,3] world->camera  ->  R_mesh[...,3,3] sphere-local -> world."""
    return np.swapaxes(POTREE_M @ _rz_cam(az_offset_deg) @ np.asarray(R, dtype=np.float64), -1, -2)


def potree_angles(R: np.ndarray, az_offset_deg: float = AZ_OFFSET_DEG) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R[...,3,3] world->camera -> (course, pitch, roll) in degrees for `coordinates.txt`.

    Decomposes R_mesh = Rz(c) Ry(b) Rx(a) and undoes Potree's offsets. No gimbal lock in practice:
    the middle angle is b = -pitch (a few degrees), the singularity sits at |b| = 90 deg.
    """
    M = sphere_rotation(R, az_offset_deg)
    b = np.arcsin(np.clip(-M[..., 2, 0], -1.0, 1.0))
    c = np.arctan2(M[..., 1, 0], M[..., 0, 0])
    a = np.arctan2(M[..., 2, 1], M[..., 2, 2])
    course = 90.0 - np.degrees(c)
    pitch = -np.degrees(b)
    roll = np.degrees(a) - 90.0
    return (course + 180.0) % 360.0 - 180.0, pitch, roll


def coordinates_text(names: list[str], t: np.ndarray, C: np.ndarray, R: np.ndarray, az_offset_deg: float = AZ_OFFSET_DEG) -> str:
    course, pitch, roll = potree_angles(R, az_offset_deg)
    lines = [HEADER]
    for i, name in enumerate(names):
        lines.append(
            f"{name}\t{t[i]:.6f}\t{C[i, 0]:.4f}\t{C[i, 1]:.4f}\t{C[i, 2]:.4f}"
            f"\t{course[i]:.6f}\t{pitch[i]:.6f}\t{roll[i]:.6f}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------- frames
def select_frames(poses: Poses, frames: list[int] | None, bbox: tuple[float, float, float, float] | None, stride: int) -> list[int]:
    idx = np.arange(len(poses)) if frames is None else np.asarray(sorted(set(frames)), dtype=np.int64)
    if bbox is not None:
        e, n = poses.origin[idx, 0], poses.origin[idx, 1]
        keep = (e >= bbox[0]) & (e <= bbox[2]) & (n >= bbox[1]) & (n <= bbox[3])
        idx = idx[keep]
    return idx[:: max(1, stride)].tolist()


def cloud_bbox(cloud_dir: Path, margin: float = 30.0) -> tuple[float, float, float, float]:
    """(e_min, n_min, e_max, n_max) of a PotreeConverter octree, grown by `margin` metres."""
    bb = json.loads((Path(cloud_dir) / "metadata.json").read_text())["boundingBox"]
    return (bb["min"][0] - margin, bb["min"][1] - margin, bb["max"][0] + margin, bb["max"][1] + margin)


# ---------------------------------------------------------------------------------- images
_G: dict = {}


def _init(src_paths, out_dir, width, quality):
    _G.update(src=src_paths, out=Path(out_dir), width=width, quality=quality)


def _resize_one(job: tuple[int, str]) -> tuple[str, bool]:
    k, name = job
    dst = _G["out"] / name
    if dst.exists():
        return name, False
    img = cv2.imread(_G["src"][k], cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(_G["src"][k])
    w = _G["width"]
    if w and img.shape[1] > w:
        img = cv2.resize(img, (w, w // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, _G["quality"]])
    return name, True


def source_path(poses: Poses, k: int, image_dir: Path | None) -> str:
    """Photo used as the sphere texture: the pose table's own file, or f%04d.jpg in `image_dir`
    (e.g. a segds `qa/` directory -- the label overlay is the best alignment target there is)."""
    return poses.path(k) if image_dir is None else str(Path(image_dir) / f"f{k:04d}.jpg")


def export(
    out_dir: str | Path,
    frames: list[int] | None = None,
    poses_source: str | None = None,
    rig: RigModel = IDENTITY,
    width: int = 4096,
    quality: int = 85,
    image_dir: str | Path | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    stride: int = 1,
    az_offset_deg: float = AZ_OFFSET_DEG,
    workers: int = 6,
    log=print,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    poses = load_poses(poses_source)
    idx = select_frames(poses, frames, bbox, stride)
    if not idx:
        raise SystemExit("no frames selected (check --frames / --bbox / --cloud)")
    R, C = geometry.frame_rotations(poses, rig, np.asarray(idx))
    names = [f"f{k:04d}.jpg" for k in idx]

    src = {k: source_path(poses, k, Path(image_dir) if image_dir else None) for k in idx}
    written = 0
    with Pool(workers, initializer=_init, initargs=(src, out, width, quality)) as pool:
        for i, (_name, did) in enumerate(pool.imap_unordered(_resize_one, list(zip(idx, names)), chunksize=4)):
            written += did
            if i % 100 == 0:
                log(f"[{i + 1}/{len(idx)}] images")

    (out / "coordinates.txt").write_text(coordinates_text(names, poses.t[idx], C, R, az_offset_deg))
    manifest = {
        "n_frames": len(idx),
        "frames": idx,
        "poses_source": poses.source,
        "poses_hash": poses.hash(),
        "registration": str(poses.registration) if poses.registration else None,
        "rig": asdict(rig),
        "az_offset_deg": az_offset_deg,
        "image_dir": str(image_dir) if image_dir else None,
        "width": width,
        "quality": quality,
        "images_written": written,
        "crs": "EPSG:5514 (S-JTSK / Krovak East North), written as longitude=E, latitude=N",
    }
    (out / "panos.json").write_text(json.dumps(manifest, indent=1))
    log(f"{len(idx)} frames -> {out} ({written} images written, {len(idx) - written} already there)")
    return manifest
