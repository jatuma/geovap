"""Project 2D segmentation predictions into the point cloud and report.

uv run python -m mapping.cli.seg_project run [--tiles 037 ...] [--workers 5] [--rgb ref|tw45] [--subsample N] [--frame-limit N]
uv run python -m mapping.cli.seg_project eval
uv run python -m mapping.cli.seg_project render
uv run python -m mapping.cli.seg_project potree-classes
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tiles", nargs="*", default=None)
    r.add_argument("--workers", type=int, default=5)
    r.add_argument("--tag", default="eomt_city", help="bench tag under segds/bench")
    r.add_argument("--rgb", default="ref", choices=["ref", "tw45"])
    r.add_argument("--rmax", type=float, default=40.0)
    r.add_argument("--min-conf", type=int, default=0)
    r.add_argument("--edge-px", type=float, default=3.0)
    r.add_argument("--no-occlusion", action="store_true")
    r.add_argument("--no-las", action="store_true")
    r.add_argument("--subsample", type=int, default=None)
    r.add_argument("--frame-limit", type=int, default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--las-dir", default=None)
    sub.add_parser("eval")
    rn = sub.add_parser("render")
    rn.add_argument("--frames", default=None, help="comma list of frames for the ERP round-trip panels")
    rn.add_argument("--tiles", nargs="*", default=None, help="tiles for BEV / oblique renders")
    pc = sub.add_parser("potree-classes")
    pc.add_argument("--out", default=None, help="directory of the Potree octree (default pointcloud-tools/output/eomt_city_seg)")
    a = ap.parse_args()

    if a.cmd == "run":
        from ..seg.project import Options, run

        opt = Options(bench_tag=a.tag, r_max=a.rmax, occlusion=not a.no_occlusion, min_conf=a.min_conf, edge_px=a.edge_px, rgb=a.rgb,
                      write_las=not a.no_las, subsample=a.subsample, frame_limit=a.frame_limit)
        if a.out:
            opt.out_dir = Path(a.out)
        if a.las_dir:
            opt.las_dir = Path(a.las_dir)
        run(a.tiles, opt, workers=a.workers)
    elif a.cmd == "eval":
        from ..seg import project_report

        project_report.evaluate()
    elif a.cmd == "render":
        from ..seg import project_report

        frames = [int(x) for x in a.frames.split(",")] if a.frames else None
        project_report.render_all(frames=frames, tiles=a.tiles)
    elif a.cmd == "potree-classes":
        from ..seg import project_report

        project_report.write_potree_classes(Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()
