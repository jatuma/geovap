"""uv run python -m mapping.cli.calibrate [--frames 120] [--points 30000] [--no-cv] [--tag calib]"""
from __future__ import annotations

import argparse

from ..calib.fit import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--points", type=int, default=30_000)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--tag", default="calib")
    ap.add_argument("--free", nargs="*", default=["omega", "phi", "kappa", "dt", "lx", "ly", "lz"])
    a = ap.parse_args()
    rig = run(n_frames=a.frames, n_points=a.points, free=tuple(a.free), do_cv=not a.no_cv, tag=a.tag)
    print("fitted rig:", rig)


if __name__ == "__main__":
    main()
