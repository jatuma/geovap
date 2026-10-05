"""Run zero-shot models over frames with the levelled multi-view recipe; save fused predictions per frame.

bench/<tag>/f%04d_native.png  argmax of the model's own classes (uint8, 255 = uncovered)
bench/<tag>/f%04d_common.png  argmax after mapping probabilities to the common taxonomy
bench/<tag>/f%04d_conf.png    max common probability * 255
bench/<tag>/timing.csv        frame, seconds, peak GPU MiB
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from geovap.domain.model.frames import FrameIndex
from geovap.runtime.pose_tables import load as load_poses
from geovap.runtime.panos import pano_path
from geovap.domain.scheme import taxonomy as T
from geovap.stages.semantics.pseudogt.areas import segds_root
from geovap.stages.semantics.pseudogt.views import VIEWS, VIEW_SIZE, extract_image, level_rotation, view_to_pano_maps
from .fusion import fuse, to_common
from .models import SPECS, SegModel


# Pose-independent: a 2D prediction depends only on the photo (and the rig's fixed leveling
# rotation), never on the panorama's position/pose-correction state, so bench predictions always
# live under the export root (segds_root()), not the pose-source-aware segds_dir()
# (`Workspace.source_dir`) -- unlike areas/point_labels/labels_erp/... which are derived from the
# pose-dependent point cloud.
def bench_dir(s=None) -> Path:
    return segds_root(s) / "bench"


def dataset_seg_dir(s=None) -> Path:
    """Per-run dataset metadata directory (`id2label_<tag>.json`, `bench_frames.json`, ...) --
    same directory `stages.semantics.pseudogt.dataset.dataset_dir` writes, kept here too since
    `bench`/`evaluate` are their own module and must not import a dataset from `__file__`, which
    is what the old flat module's `DATASET_SEG_DIR = Path(__file__).resolve().parents[2] /
    "dataset" / "seg"` constant did."""
    from geovap.runtime import settings

    return (s or settings.get()).workspace.derived / "seg"


def missing_predictions(tag: str, frames: list[int], out_root: Path | None = None) -> list[int]:
    """Frames in `frames` that don't yet have a `f%04d_common.png` prediction for `tag`."""
    out_root = bench_dir() if out_root is None else out_root
    out = out_root / tag
    return [k for k in frames if not (out / f"f{k:04d}_common.png").exists()]


def frame_views(photo_bgr: np.ndarray, R_cam: np.ndarray, R_lev: np.ndarray, pano_w: int, pano_h: int, size: int = VIEW_SIZE) -> list[np.ndarray]:
    out = []
    for view in VIEWS:
        mu, mv = view_to_pano_maps(R_cam, R_lev, view, pano_w, pano_h, size)
        out.append(extract_image(photo_bgr, mu, mv, pano_w)[..., ::-1].copy())  # RGB
    return out


@torch.no_grad()
def segment_frame(model: SegModel, photo_bgr: np.ndarray, R_cam: np.ndarray, R_lev: np.ndarray, pano_w: int, pano_h: int, out_h: int, out_w: int):
    views = frame_views(photo_bgr, R_cam, R_lev, pano_w, pano_h)
    probs = [model.probs(v) for v in views]
    fused, cov = fuse(probs, VIEWS, R_cam, R_lev, out_h, out_w)
    common = to_common(fused, model.to_common, len(T.COMMON))
    native = fused.argmax(0).to(torch.uint8)
    cmn = common.argmax(0).to(torch.uint8)
    conf = (common.max(0).values * 255).clamp(0, 255).to(torch.uint8)
    native[~cov] = 255
    cmn[~cov] = 255
    return native.cpu().numpy(), cmn.cpu().numpy(), conf.cpu().numpy(), cov.cpu().numpy()


def run(tags: list[str], frames: list[int], out_root: Path | None = None, amp: bool = True, skip_existing: bool = True, poses_source: str | None = None) -> None:
    from geovap.runtime import settings

    s = settings.get()
    out_root = bench_dir(s) if out_root is None else out_root
    pano_w, pano_h = s.sensor.pano
    zb_w, zb_h = s.sensor.zb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poses = load_poses(poses_source)
    fi = FrameIndex(poses)
    dataset_dir = dataset_seg_dir(s)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        spec = SPECS[tag]
        out = out_root / tag
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        model = SegModel(spec, device, amp=amp)
        print(f"[{tag}] loaded {spec.checkpoint} ({model.num_labels} labels) in {time.time() - t0:.0f} s")
        (dataset_dir / f"id2label_{tag}.json").write_text(json.dumps({"checkpoint": spec.checkpoint, "taxonomy": spec.taxonomy, "id2label": model.id2label}, indent=0))
        timing_path = out / "timing.csv"
        done = set()
        if timing_path.exists() and skip_existing:
            done = {int(r["frame"]) for r in csv.DictReader(timing_path.open())}
        with timing_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if not done:
                w.writerow(["frame", "seconds", "peak_mib"])
            for i, k in enumerate(frames):
                if k in done:
                    continue
                photo = cv2.imread(pano_path(poses, k))
                torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
                t1 = time.time()
                native, cmn, conf, cov = segment_frame(model, photo, fi.R[k], level_rotation(poses, k), pano_w, pano_h, zb_h, zb_w)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t1
                peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0
                cv2.imwrite(str(out / f"f{k:04d}_native.png"), native)
                cv2.imwrite(str(out / f"f{k:04d}_common.png"), cmn)
                cv2.imwrite(str(out / f"f{k:04d}_conf.png"), conf)
                w.writerow([k, f"{dt:.3f}", f"{peak:.0f}"])
                fh.flush()
                if i % 10 == 0:
                    print(f"[{tag}] {i + 1}/{len(frames)} frame {k}: {dt:.2f} s, peak {peak:.0f} MiB, coverage {cov.mean():.3f}")
        del model
        torch.cuda.empty_cache()
