"""Zero-shot benchmark: run models, evaluate against the JVF-derived labels, write the report tables.

uv run python -m mapping.cli.seg_bench run --models all|tag,tag --frames bench|clean|1,2,3|missing:<spec>|<path.json> [--no-amp]
uv run python -m mapping.cli.seg_bench evaluate [--models ...] [--frames bench|clean|1,2,3|<path.json>]
uv run python -m mapping.cli.seg_bench report

`--frames missing:<spec>` (e.g. `missing:clean`, `missing:bench`, `missing:1,2,3`) resolves `<spec>` the
usual way and then, per model, drops any frame that already has a prediction (`bench.missing_predictions`)
-- so `run` only recomputes what's missing. `--frames` also accepts a path to a JSON file containing a
list of frame indices (e.g. a saved baseline bench list), so `evaluate` can be pointed at a fixed frame set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _base_frames(spec: str) -> list[int]:
    from ..seg.bench import DATASET_SEG_DIR
    from ..seg.render_labels import frames_arg

    if spec == "bench":
        return json.loads((DATASET_SEG_DIR / "bench_frames.json").read_text())["frames"]
    p = Path(spec)
    if p.suffix == ".json" and p.exists():
        data = json.loads(p.read_text())
        return data["frames"] if isinstance(data, dict) else list(data)
    return frames_arg(spec)


def _frames_for_model(spec: str, tag: str) -> list[int]:
    """Resolve `spec` to a frame list for model `tag`; `missing:<spec>` filters to frames lacking
    a prediction for `tag` (see `mapping.seg.bench.missing_predictions`)."""
    if spec.startswith("missing:"):
        from ..seg.bench import missing_predictions

        base = _base_frames(spec[len("missing:") :])
        return missing_predictions(tag, base)
    return _base_frames(spec)


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
    e.add_argument("--out", default=None, help="output dir for results.json + CSVs (default dataset/seg/bench); use another dir to keep a second evaluation separate")
    sub.add_parser("report")
    a = ap.parse_args()
    from ..seg.models import SPECS

    if a.cmd == "run":
        from ..seg import bench

        tags = list(SPECS) if a.models == "all" else a.models.split(",")
        for tag in tags:
            frames = _frames_for_model(a.frames, tag)
            if not frames:
                print(f"[{tag}] nothing to run ({a.frames}: 0 frames)")
                continue
            bench.run([tag], frames, amp=not a.no_amp)
    elif a.cmd == "evaluate":
        from ..seg import evaluate

        tags = list(SPECS) if a.models == "all" else a.models.split(",")
        evaluate.run(tags, _base_frames(a.frames), out_dir=Path(a.out) if a.out else evaluate.OUT_DIR)
    elif a.cmd == "report":
        from ..seg import evaluate

        evaluate.report()


if __name__ == "__main__":
    main()
