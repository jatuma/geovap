"""Build the segmentation dataset step by step.

uv run python -m mapping.cli.seg_build [--poses export|corrected|<csv>] areas [--all-lines]
uv run python -m mapping.cli.seg_build [--poses ...] rasters
uv run python -m mapping.cli.seg_build [--poses ...] points [--workers 4]
uv run python -m mapping.cli.seg_build [--poses ...] labels [--frames clean] [--workers 6]
uv run python -m mapping.cli.seg_build [--poses ...] views [--workers 8]
uv run python -m mapping.cli.seg_build [--poses ...] dataset

`--poses` (before the subcommand) selects the pose table: default None -> env GEOVAP_POSES /
"export" (unchanged behaviour); a value here also sets GEOVAP_POSES for the rest of the process
(before any `mapping.seg.*` import), so it is equivalent to exporting the env var yourself.
`rasters`/`points` read the cloud through it (registered when corrected); `labels` and everything
downstream additionally uses it for per-frame products. Ignored by `areas` (JVF-only, no cloud or
poses involved). Outputs land under a hash-suffixed sibling of the default segds root for any
non-export table (see `mapping.seg.areas.SEGDS_DIR`), so a corrected run never overwrites the
export one.
"""
from __future__ import annotations

import argparse
import json
import os


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
    # every subcommand: pose table override (default: env GEOVAP_POSES / "export", unchanged
    # behaviour). "rasters"/"points" pull points through the REGISTERED cloud when set; "labels" and
    # anything downstream of per-frame products (point_id panoramas) picks up both the registered
    # cloud and the corrected camera geometry. Segds outputs land under a hash-suffixed sibling of
    # the default segds root (see mapping.seg.areas.SEGDS_DIR) so a corrected run never overwrites
    # the export one.
    ap.add_argument("--poses", default=None, help='pose table: "export" (default), "corrected", or a CSV path')
    args = ap.parse_args()

    # `mapping.seg.areas.SEGDS_DIR` (and every output-dir constant derived from it in the seg
    # submodules below) is resolved once at import time from `config.POSES_SOURCE` (env
    # `GEOVAP_POSES`), not from `--poses`. Set the env var here, before any `mapping.seg.*` import
    # below, so a `--poses corrected` run (used instead of exporting GEOVAP_POSES) writes into the
    # matching hash-suffixed segds tree rather than silently mixing corrected-pose outputs into the
    # export one. Must happen before the first `from ..seg import ...` in this function.
    if args.poses is not None:
        os.environ["GEOVAP_POSES"] = args.poses

    if args.cmd == "areas":
        from ..seg import areas

        faces, rep = areas.build(all_lines=not args.hard_only)
        print(json.dumps({k: v for k, v in rep.items() if k not in ("conflicts", "unresolved_largest")}, indent=1, ensure_ascii=False))
        print(f"conflicts: {len(rep['conflicts'])}, unresolved listed: {len(rep['unresolved_largest'])} -> {areas.AREAS_DIR}")
    elif args.cmd == "rasters":
        from ..seg import rasters

        rasters.build(poses_source=args.poses)
    elif args.cmd == "points":
        from ..seg import point_labels

        point_labels.build(workers=args.workers, poses_source=args.poses)
    elif args.cmd == "labels":
        from ..seg import render_labels

        render_labels.build(frames=args.frames, workers=args.workers, limit=args.limit, poses_source=args.poses)
    elif args.cmd == "views":
        from ..seg import views

        views.build(workers=args.workers, limit=args.limit, poses_source=args.poses)
    elif args.cmd == "dataset":
        from ..seg import dataset

        dataset.build(poses_source=args.poses)


if __name__ == "__main__":
    main()
