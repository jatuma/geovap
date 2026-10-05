"""The command line every stage shares.

The restructuring requires each stage to be runnable on its own -- `python -m
geovap.stages.colour.colorize run --tag tw45` must work in a checkout where the `geovap` command is not
even installed, because that is what lets a team own a stage without owning the driver. That only
stays true if the dataset flags are identical everywhere and cost a stage one line to adopt.

The flags are deliberately the same five on every stage, and they are resolved into `Settings`
BEFORE the stage's implementation module is imported. That ordering is the whole point: the old code
froze `mapping/config.py`'s paths at import time, so by the time argparse had run, forty modules had
already captured Dražkov and no flag could change them.
"""
from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings
    from geovap.stages.base.spec import Stage


def add_dataset_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The five flags that select a dataset. Each falls back to its `$GEOVAP_*` variable, which is
    also how `runtime.procs` passes the parent's choice to a stage subprocess."""
    g = parser.add_argument_group("dataset")
    g.add_argument("--dataset", metavar="NAME|PATH",
                   help="descriptor name (looked up on $GEOVAP_DATASETS, ./datasets, then the "
                        "built-ins) or a path to one; default $GEOVAP_DATASET")
    g.add_argument("--data-root", metavar="DIR", help="override [paths].data_root")
    g.add_argument("--workspace", metavar="DIR", help="override [paths].workspace")
    g.add_argument("--publish", metavar="DIR", help="override [paths].publish")
    g.add_argument("--poses", metavar="NAME|CSV",
                   help='pose table: "export" (the regression anchor) or a corrected table')
    return parser


def configure_from(args: argparse.Namespace) -> "Settings":
    """Install the process-wide `Settings` from parsed flags. Call this before importing anything
    that reads settings at module scope."""
    from geovap.runtime import settings

    return settings.configure(
        dataset=args.dataset,
        data_root=args.data_root,
        workspace=args.workspace,
        publish=args.publish,
        poses=args.poses,
    )


def stage_parser(stage: "Stage", *, description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m {type(stage).__module__}",
        description=description or stage.spec.summary,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_dataset_flags(parser)
    parser.add_argument("--status", action="store_true",
                        help="report whether this stage is done and what it would read, then exit")
    return parser


def stage_main(
    stage: "Stage",
    argv: Sequence[str] | None = None,
    *,
    add_options: Callable[[argparse.ArgumentParser], None] | None = None,
    to_opts: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
) -> int:
    """A complete `main()` for one stage.

        def main(argv=None) -> int:
            return stage_main(STAGE, argv, add_options=_options, to_opts=_opts)

    `--status` prints the stage's own view of itself -- available, done, inputs, outputs, metrics --
    which is what `geovap status` aggregates and what a developer needs when a stage is skipped and
    they want to know why.
    """
    parser = stage_parser(stage)
    if add_options is not None:
        add_options(parser)
    args = parser.parse_args(argv)
    s = configure_from(args)

    if not stage.available(s):
        if stage.spec.optional:
            print(f"{stage.spec.name}: unavailable for dataset {s.name!r}; skipping (stage is optional)")
            return 0
        parser.error(f"{stage.spec.name}: unavailable for dataset {s.name!r}")

    if args.status:
        print(describe(stage, s))
        return 0

    stage.run(s, **(to_opts(args) if to_opts else {}))
    return 0


def describe(stage: "Stage", s: "Settings") -> str:
    """Human-readable status of one stage against one dataset. Never raises: `metrics()` is defined
    to return `{"error": ...}` rather than throw, so this works on a checkout with no data."""
    from geovap.runtime import manifest

    lines = [
        f"{stage.spec.name}  [{s.name}]  {stage.spec.summary}",
        f"  available : {stage.available(s)}",
        f"  done      : {manifest.is_done(stage, s)}",
    ]
    for label, items in (("inputs", stage.inputs(s).items()),
                         ("outputs", ((None, p) for p in stage.outputs(s)))):
        lines.append(f"  {label:<10}:")
        for key, path in items:
            from pathlib import Path

            mark = "ok     " if Path(path).exists() else "MISSING"
            lines.append(f"    {mark} {key + ' = ' if key else ''}{path}")
    lines.append(f"  metrics   : {stage.metrics(s)}")
    return "\n".join(lines)
