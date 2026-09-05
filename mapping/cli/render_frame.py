"""Render cloud attributes and JVF vectors into a panorama (cloud -> pano).

uv run python -m mapping.cli.render_frame 367 [--layers rgb depth classification intensity point_id]
                                              [--jvf] [--overlay] [--out DIR]
Writes DIR/f0367_<layer>.png (+ .npz with raw arrays), DIR/f0367_jvf.png (class-id mask, occluded
parts in a second channel) and DIR/f0367_overlay.jpg (photo with layers / vectors blended).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .. import compat, render, vectors
from ..cloud_store import CloudStore
from ..config import RENDERS_DIR
from ..frame_select import FrameIndex
from ..poses import load_poses
from ..products import FrameProducts
from ..rig import IDENTITY, RigModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=int, nargs="+")
    ap.add_argument("--layers", nargs="*", default=["rgb", "depth", "classification"])
    ap.add_argument("--jvf", action="store_true", help="also rasterise JVF vectors with occlusion")
    ap.add_argument("--overlay", action="store_true", help="write photo overlays")
    ap.add_argument("--rig", default=None)
    ap.add_argument("--out", default=str(RENDERS_DIR))
    a = ap.parse_args()

    store = CloudStore()
    poses = load_poses()
    rig = RigModel.from_json(a.rig) if a.rig else IDENTITY
    fi = FrameIndex(poses, rig)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    objects = None
    class_ids = None
    if a.jvf:
        compat.ensure_experiments_on_path()
        from common import io_data

        objects = io_data.load_jvf_objects()
        codes = sorted({o.jvfcode for o in objects})
        class_ids = {c: i + 1 for i, c in enumerate(codes)}
        (out / "jvf_class_ids.txt").write_text("\n".join(f"{i}\t{c}" for c, i in class_ids.items()))

    for k in a.frames:
        fp = FrameProducts.load(k, rig)
        res = render.render_frame(store, fp, layers=a.layers, out_dir=out)
        photo = None
        if a.overlay:
            photo = cv2.imread(poses.path(k))
            small = cv2.resize(photo, (fp.depth_mm.shape[1], fp.depth_mm.shape[0]), interpolation=cv2.INTER_AREA)
            for layer, (vals, valid) in res.items():
                cv2.imwrite(str(out / f"f{k:04d}_{layer}_overlay.jpg"), render.overlay_on_photo(photo, render.to_png(layer, vals, valid), valid, 0.55), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if objects is not None:
            mask, occ = vectors.render_objects(objects, fi.R[k], fi.C[k], fp, class_ids, scale=0.25)
            cv2.imwrite(str(out / f"f{k:04d}_jvf.png"), np.stack([mask, occ, np.zeros_like(mask)], -1))
            if a.overlay:
                small = cv2.resize(photo if photo is not None else cv2.imread(poses.path(k)), (2000, 1000), interpolation=cv2.INTER_AREA)
                vis = small.copy()
                vis[mask > 0] = (0.3 * vis[mask > 0] + np.array([0, 0, 178])).astype(np.uint8)
                vis[(occ > 0) & (mask == 0)] = (0.5 * vis[(occ > 0) & (mask == 0)] + np.array([127, 0, 0])).astype(np.uint8)
                cv2.imwrite(str(out / f"f{k:04d}_jvf_overlay.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"frame {k}: {', '.join(a.layers)}{' + jvf' if objects is not None else ''} -> {out}")


if __name__ == "__main__":
    main()
