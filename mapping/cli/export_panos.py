"""Panorama spheres for the Potree viewer (Potree.Images360).

    uv run python -m mapping.cli.export_panos --cloud pointcloud-tools/output/eomt_city_seg
    uv run python -m mapping.cli.export_panos --cloud .../eomt_city_seg --poses corrected \
        --image-dir Geovap_cache/segds_e8f3e1/qa --out .../eomt_city_seg/panos_qa --width 2048

Default output is `<cloud>/panos`, which `view.html` picks up automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..panos import AZ_OFFSET_DEG, cloud_bbox, export
from geovap.domain.model.rig import IDENTITY, RigModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", default=None, help="octree dir: default --out and a bbox frame filter")
    ap.add_argument("--out", default=None, help="output dir (default <cloud>/panos)")
    ap.add_argument("--frames", default=None, help="'clean', or a comma-separated list (default: all)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth selected frame")
    ap.add_argument("--margin", type=float, default=30.0, help="bbox margin around --cloud, metres")
    ap.add_argument("--no-bbox", action="store_true", help="do not filter frames by the cloud bbox")
    ap.add_argument("--poses", default=None, help="pose source: export (default) / corrected / csv path")
    ap.add_argument("--rig", default=None, help="rig json (default identity)")
    ap.add_argument("--image-dir", default=None, help="use f%%04d.jpg from here instead of the raw photos")
    ap.add_argument("--width", type=int, default=4096, help="downscale width, 0 = keep source")
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--az-offset", type=float, default=AZ_OFFSET_DEG, help="sphere yaw offset in deg (0 = derived + verified convention, see panos.py; 180 only for A/B sets)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    if a.cloud is None and a.out is None:
        ap.error("--cloud or --out is required")
    out = Path(a.out) if a.out else Path(a.cloud) / "panos"
    bbox = None if (a.cloud is None or a.no_bbox) else cloud_bbox(Path(a.cloud), a.margin)

    frames = None
    if a.frames == "clean":
        from ..seg.render_labels import clean_frames

        frames = clean_frames()
    elif a.frames:
        frames = [int(x) for x in a.frames.split(",")]

    export(
        out,
        frames=frames,
        poses_source=a.poses,
        rig=RigModel.from_json(a.rig) if a.rig else IDENTITY,
        width=a.width,
        quality=a.quality,
        image_dir=a.image_dir,
        bbox=bbox,
        stride=a.stride,
        az_offset_deg=a.az_offset,
        workers=a.workers,
    )


if __name__ == "__main__":
    main()
