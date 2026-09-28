"""Build per-frame products (depth + point-id panoramas) for a pose table.

uv run python -m mapping.cli.build_frames [--workers 16] [--poses corrected] [--force] [--frames 0,1,2]

`--poses`: pose table to build products for; default None -> env GEOVAP_POSES / "export". Output
lands under `mapping.products.frames_dir(poses)` (byte-identical to the historical default frames
dir for "export"; a hash-suffixed sibling for any corrected table).

By default frames whose product file already exists are skipped (resumable); `--force` rebuilds
everything requested instead. `--frames` restricts to an explicit comma-separated list of frame
indices; omitted, all frames in the pose table are built.
"""
from __future__ import annotations

import argparse

from ..products import build_all_frames
from ..rig import IDENTITY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--poses", default=None, help="pose table: 'export' | 'corrected' | None -> env GEOVAP_POSES / export")
    ap.add_argument("--force", action="store_true", help="rebuild even if the product file already exists")
    ap.add_argument("--frames", default=None, help="comma-separated frame indices (default: all frames in the pose table)")
    a = ap.parse_args()
    frames = [int(f) for f in a.frames.split(",")] if a.frames else None
    build_all_frames(frames=frames, rig=IDENTITY, workers=a.workers, poses_source=a.poses, skip_existing=not a.force)


if __name__ == "__main__":
    main()
