"""S7: final validation of `poses_corrected` (`load_poses("corrected")`) against `export` and the
intermediate tables.

    uv run python -m mapping.cli.pose_report run [--a export] [--b corrected] [--workers 8]
        [--out-dir Geovap_cache/out/poses]

Runs, in order: `pose_report.compare_pose_sources` (silhouette residual, own pose per source, on
turning / straight / refined-or-big-registration frame groups), `pose_report.colour_de_comparison`
(colour dE on turning frames), `pose_report.conflict_summary` (cites the existing S5b
`conflict_reg.json`), `pose_report.interpolation_benefit` (dense trajectory vs. neighbour-linear
interpolation on turning frames, + 3 example plots), and `pose_report.invariance_check` (real-data
pass-transform invariance). Writes `<out-dir>/report_final.md` + `.json`.

and `pose_report.slow_regression_037` (tile-037 pilot colour recipe, export vs corrected, run in-process --
pass `--no-slow` to skip it, e.g. while store/frames aren't built). Writes `<out-dir>/report_final.md` +
`.json`, including the Summary and slow-regression sections generated from this run's own numbers.

Individual steps can also be run standalone (`compare`, `colour`, `interp`, `invariance`, `slow`) for
faster iteration while developing.
"""
from __future__ import annotations

import argparse

from .. import pose_report as pr
from ..config import POSES_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "compare", "colour", "interp", "invariance", "slow"])
    ap.add_argument("--a", default="export")
    ap.add_argument("--b", default="corrected")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=str(POSES_DIR))
    ap.add_argument("--no-slow", action="store_true", help="skip slow_regression_037 in `run` (store/frames not built)")
    a = ap.parse_args()

    if a.cmd == "run":
        pr.run_final_report(a.a, a.b, out_dir=a.out_dir, workers=a.workers, run_slow=not a.no_slow)
    elif a.cmd == "compare":
        pr.compare_pose_sources(a.a, a.b, out_dir=a.out_dir, workers=a.workers)
    elif a.cmd == "colour":
        pr.colour_de_comparison(a.a, a.b, out_dir=a.out_dir, workers=a.workers)
    elif a.cmd == "interp":
        pr.interpolation_benefit(a.b, out_dir=a.out_dir)
    elif a.cmd == "invariance":
        print(pr.invariance_check(corrected_source=a.b))
    elif a.cmd == "slow":
        print(pr.slow_regression_037())


if __name__ == "__main__":
    main()
