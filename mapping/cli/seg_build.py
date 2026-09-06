"""Build the segmentation dataset step by step.

uv run python -m mapping.cli.seg_build areas [--all-lines]
uv run python -m mapping.cli.seg_build rasters
uv run python -m mapping.cli.seg_build points [--workers 4]
uv run python -m mapping.cli.seg_build labels [--frames clean] [--workers 6]
uv run python -m mapping.cli.seg_build views [--workers 8]
uv run python -m mapping.cli.seg_build dataset
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("areas")
    a.add_argument("--hard-only", action="store_true", help="polygonize only hard boundary codes (default: all lines)")
    sub.add_parser("rasters")
    p = sub.add_parser("points")
    p.add_argument("--workers", type=int, default=4)
    l = sub.add_parser("labels")
    l.add_argument("--frames", default="clean", help="clean | all | comma list")
    l.add_argument("--workers", type=int, default=6)
    l.add_argument("--limit", type=int, default=None)
    v = sub.add_parser("views")
    v.add_argument("--workers", type=int, default=8)
    v.add_argument("--limit", type=int, default=None)
    sub.add_parser("dataset")
    args = ap.parse_args()

    if args.cmd == "areas":
        from ..seg import areas

        faces, rep = areas.build(all_lines=not args.hard_only)
        print(json.dumps({k: v for k, v in rep.items() if k not in ("conflicts", "unresolved_largest")}, indent=1, ensure_ascii=False))
        print(f"conflicts: {len(rep['conflicts'])}, unresolved listed: {len(rep['unresolved_largest'])} -> {areas.AREAS_DIR}")
    elif args.cmd == "rasters":
        from ..seg import rasters

        rasters.build()
    elif args.cmd == "points":
        from ..seg import point_labels

        point_labels.build(workers=args.workers)
    elif args.cmd == "labels":
        from ..seg import render_labels

        render_labels.build(frames=args.frames, workers=args.workers, limit=args.limit)
    elif args.cmd == "views":
        from ..seg import views

        views.build(workers=args.workers, limit=args.limit)
    elif args.cmd == "dataset":
        from ..seg import dataset

        dataset.build()


if __name__ == "__main__":
    main()
