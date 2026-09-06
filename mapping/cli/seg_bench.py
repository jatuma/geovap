"""Zero-shot benchmark: run models, evaluate against the JVF-derived labels, write the report tables.

uv run python -m mapping.cli.seg_bench run --models all|tag,tag --frames bench|clean|1,2,3 [--no-amp]
uv run python -m mapping.cli.seg_bench evaluate [--models ...] [--frames bench]
uv run python -m mapping.cli.seg_bench report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _frames(spec: str) -> list[int]:
    from ..seg.bench import DATASET_SEG_DIR
    from ..seg.render_labels import frames_arg

    if spec == "bench":
        return json.loads((DATASET_SEG_DIR / "bench_frames.json").read_text())["frames"]
    return frames_arg(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", default="all")
    r.add_argument("--frames", default="bench")
    r.add_argument("--no-amp", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--models", default="all")
    e.add_argument("--frames", default="bench")
    sub.add_parser("report")
    a = ap.parse_args()
    from ..seg.models import SPECS

    if a.cmd == "run":
        from ..seg import bench

        tags = list(SPECS) if a.models == "all" else a.models.split(",")
        bench.run(tags, _frames(a.frames), amp=not a.no_amp)
    elif a.cmd == "evaluate":
        from ..seg import evaluate

        tags = list(SPECS) if a.models == "all" else a.models.split(",")
        evaluate.run(tags, _frames(a.frames))
    elif a.cmd == "report":
        from ..seg import evaluate

        evaluate.report()


if __name__ == "__main__":
    main()
