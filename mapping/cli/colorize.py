"""uv run python -m mapping.cli.colorize [--tiles 037 001] [--workers 8] [--tag run] [--no-occlusion] ..."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..colorize import Options, run
from ..config import OUT_DIR
from ..rig import IDENTITY, RigModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--rig", default=None, help="rig json (default identity)")
    ap.add_argument("--rmax", type=float, default=40.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--sampling", default="footprint", choices=["nearest", "bilinear", "footprint"])
    ap.add_argument("--no-las", action="store_true")
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    rig = RigModel.from_json(a.rig) if a.rig else IDENTITY
    opt = Options(rig=rig, r_max=a.rmax, top_k=a.k, occlusion=not a.no_occlusion, sampling=a.sampling, write_las=not a.no_las, subsample=a.subsample, out_dir=Path(a.out), tag=a.tag)
    run(a.tiles, opt, workers=a.workers)


if __name__ == "__main__":
    main()
