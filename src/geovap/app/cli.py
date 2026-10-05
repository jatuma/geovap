"""`geovap` -- the one command.

    geovap doctor  [--dataset D]              can this dataset be processed? writes nothing
    geovap stages                             what is installed, and in what order it would run
    geovap datasets                           which descriptors resolve, and from where
    geovap run     [--from/--to/--only/--skip] [--profile P] [--force] [--dry-run]
    geovap status                             what has been done, when, and with what result
    geovap compare [--baseline DIR]           this run's metrics against the dataset's baseline

Every subcommand takes the same five dataset flags as every stage, and a stage remains runnable on
its own -- `python -m geovap.stages.colour.colorize --dataset drazkov` -- so a team can own a stage
without the `geovap` command at all. This command is a convenience over that, never a
requirement.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from geovap.stages.base.cli import add_dataset_flags, configure_from


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geovap", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name, help_text):
        p = sub.add_parser(name, help=help_text)
        add_dataset_flags(p)
        p.add_argument("--json", action="store_true", help="machine-readable output")
        return p

    add("doctor", "resolve the dataset and report everything wrong, before writing anything")
    add("stages", "list the installed stages in dependency order")
    add("datasets", "list the descriptors that resolve, and where from")
    add("status", "what has been done, when, and with what result")

    run = add("run", "run the pipeline")
    run.add_argument("--profile", help="named selection of stages (see `geovap run --list-profiles`)")
    run.add_argument("--list-profiles", action="store_true")
    run.add_argument("--from", dest="start", metavar="STAGE", help="start at this stage")
    run.add_argument("--to", dest="end", metavar="STAGE", help="stop after this stage")
    run.add_argument("--only", metavar="A,B", help="run exactly these stages")
    run.add_argument("--skip", metavar="A,B", default="", help="leave these out")
    run.add_argument("--with-optional", action="store_true", help="include the optional branches")
    run.add_argument("--force", action="store_true", help="re-run stages whose marker is green")
    run.add_argument("--dry-run", action="store_true", help="print the plan and the commands, run nothing")
    run.add_argument("--detach", action="store_true",
                     help="run in the background and return; a full run takes most of a day")
    run.add_argument("--force-detach", action="store_true",
                     help="detach even though another run holds the pid file")

    cmp_ = add("compare", "this run's metrics against the dataset's tracked baseline")
    cmp_.add_argument("--baseline", metavar="DIR", help="override the dataset's baseline directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "run" and args.list_profiles:
        from geovap.app import profiles

        for p in profiles.PROFILES.values():
            print(f"  {p.name:<9} {p.summary}")
        return 0

    if args.cmd == "datasets":
        return _cmd_datasets(args)

    s = configure_from(args)

    if args.cmd == "doctor":
        return _cmd_doctor(s, args)
    if args.cmd == "stages":
        return _cmd_stages(s, args)
    if args.cmd == "status":
        return _cmd_status(s, args)
    if args.cmd == "run":
        return _cmd_run(s, args)
    if args.cmd == "compare":
        return _cmd_compare(s, args)
    raise AssertionError(args.cmd)


# --------------------------------------------------------------------------------------- commands
def _cmd_doctor(s, args) -> int:
    from geovap.app import doctor

    rep = doctor.report(s)
    print(json.dumps(rep, indent=1, default=str) if args.json else doctor.render(rep))
    return 0 if doctor.ok(rep) else 1


def _cmd_stages(s, args) -> int:
    from geovap.app import driver

    registry = driver.build_registry()
    rows = [
        {"stage": n, "after": list(registry[n].spec.after), "optional": registry[n].spec.optional,
         "est_min": registry[n].spec.est_min, "summary": registry[n].spec.summary,
         "available": registry[n].available(s)}
        for n in registry.order()
    ]
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    for r in rows:
        flags = "".join([" (optional)" if r["optional"] else "",
                         "" if r["available"] else "  [unavailable for this dataset]"])
        after = f"  after {', '.join(r['after'])}" if r["after"] else ""
        print(f"  {r['stage']:<14} {r['est_min']:>4.0f}m  {r['summary']}{after}{flags}")
    return 0


def _cmd_status(s, args) -> int:
    from geovap.app import driver

    rows = driver.status(s)
    if args.json:
        print(json.dumps(rows, indent=1, default=str))
        return 0
    print(f"dataset {s.name}  poses {s.pose_table}  workspace {s.paths.workspace}")
    for r in rows:
        mark = "done" if r["done"] else ("FAIL" if r["rc"] not in (None, 0) else "    ")
        when = (r["finished"] or "")[:19]
        secs = f"{r['seconds']:>6.0f}s" if r.get("seconds") else "       "
        note = "" if r["available"] else "  (unavailable)"
        print(f"  [{mark}] {r['stage']:<14} {when:<19} {secs}{note}")
    return 0


def _cmd_run(s, args) -> int:
    from geovap.app import driver, profiles

    selection: dict = {}
    if args.profile:
        selection.update(profiles.as_selection(profiles.get(args.profile)))
    if args.start:
        selection["start"] = args.start
    if args.end:
        selection["end"] = args.end
    if args.only:
        selection["only"] = tuple(args.only.split(","))
    if args.skip:
        selection["skip"] = tuple(selection.get("skip", ())) + tuple(args.skip.split(","))
    if args.with_optional:
        selection["with_optional"] = True
    selection["force"] = args.force
    if args.detach:
        argv = [a for a in sys.argv[1:] if a not in ("--detach", "--force-detach")]
        return driver.detach(s, argv, force=args.force_detach)
    return driver.run(s, dry_run=args.dry_run, **selection)


def _cmd_compare(s, args) -> int:
    """Measured metrics against the dataset's tracked baseline.

    The baseline is a directory beside the dataset's descriptor, not a dict in the source. A
    previous run's numbers hardcoded in Python (`mapping/cli/pipeline.py:64-79`) are meaningless for
    any other dataset, and there was no way to ship them with one.
    """
    from pathlib import Path

    from geovap.app import driver

    baseline = Path(args.baseline) if args.baseline else s.workspace.baseline
    rows = driver.status(s)
    measured = {r["stage"]: r["metrics"] for r in rows if r["metrics"]}
    ref_path = baseline / "metrics.json"
    reference = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}

    if args.json:
        print(json.dumps({"baseline": str(baseline), "measured": measured, "reference": reference},
                         indent=1, default=str))
        return 0

    print(f"baseline: {baseline}" + ("" if ref_path.exists() else "  (no metrics.json yet)"))
    for stage in sorted(set(measured) | set(reference)):
        m, r = measured.get(stage, {}), reference.get(stage, {})
        for key in sorted(set(m) | set(r)):
            mv, rv = m.get(key), r.get(key)
            if mv == rv:
                continue
            print(f"  {stage}.{key}: {rv!r} -> {mv!r}")
    return 0


def _cmd_datasets(args) -> int:
    from geovap.io.descriptor import Descriptor, DescriptorError, search_path

    found = []
    for directory in search_path():
        if not directory.is_dir():
            continue
        for toml in sorted(directory.glob("*.toml")):
            row = {"name": toml.stem, "file": str(toml)}
            try:
                d = Descriptor.load(toml)
                row.update(data_root=str(d.data_root), resolves=True,
                           data_root_exists=d.data_root.exists())
            except DescriptorError as e:
                row.update(resolves=False, error=str(e))
            found.append(row)
    if args.json:
        print(json.dumps(found, indent=1))
        return 0
    print("search path: " + ", ".join(str(p) for p in search_path()))
    for row in found:
        if not row["resolves"]:
            print(f"  {row['name']:<12} {row['file']}\n      unresolved: {row['error']}")
        else:
            mark = "ok " if row["data_root_exists"] else "MISSING"
            print(f"  {row['name']:<12} {row['file']}\n      [{mark}] data_root = {row['data_root']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
