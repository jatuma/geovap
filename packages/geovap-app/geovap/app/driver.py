"""Running a pipeline: ordering the installed stages, skipping the ones already done, launching
each in its own process, and recording what happened.

This replaces `mapping/cli/pipeline.py`, whose 1320 lines mixed four separate jobs: the declaration
of all 24 stages (as inline `_stage(...)` calls carrying lambdas), a hand-written `_ORDER` list that
had to be kept in sync with them (there was an `assert` in that file whose only purpose was to catch
the two drifting apart), the subprocess runner, and every stage's metrics reader. Only the first of
those belonged to the stages, and it now lives with them.

What remains here is genuinely the driver's: pick stages, order them, decide what is stale, run,
mark. It knows nothing about geometry, colour or segmentation, and adding a stage requires no edit
to this file.

Stages run as SUBPROCESSES. That is deliberate and predates the restructuring: a stage's memmaps and
CUDA context are released when its process exits, which is what lets a 24-stage run survive on one
machine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from geovap.runtime import manifest, procs
from geovap.stages.base.discovery import discover
from geovap.stages.base.spec import Stage, StageRegistry, command_for

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings

#: Stages that BUILD the corrected pose table must themselves read the export one: the corrected
#: table does not exist while they run. Named by the stage that produces it, so the rule is
#: derivable rather than a hand-kept list -- everything at or before `assemble` in the order runs on
#: export poses.
POSE_TABLE_PRODUCER = "assemble"


@dataclass
class Plan:
    """What a run intends to do, before it does any of it."""

    order: list[str]
    selected: list[str]
    skipped: dict[str, str] = field(default_factory=dict)  # stage -> why

    @property
    def est_min(self) -> float:
        return 0.0


def build_registry() -> StageRegistry:
    return discover()


def select(
    registry: StageRegistry,
    s: "Settings",
    *,
    only: Iterable[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    skip: Iterable[str] = (),
    with_optional: bool = False,
    force: bool = False,
) -> Plan:
    """Which stages this run will execute, and why each of the others will not.

    A stage is left out for one of four reasons, and they are NOT interchangeable -- a run that
    reports "skipped" without saying which one is useless when something is missing downstream:
      unavailable  the dataset cannot support it at all (no [reference] adapter configured)
      optional     it is a branch this run did not ask for
      done         its marker is green, its inputs hash the same, its outputs exist
      deselected   --only / --from / --to / --skip excluded it
    """
    order = registry.order()
    skip = set(skip)
    chosen = list(order)

    if only:
        wanted = set(only)
        unknown = wanted - set(order)
        if unknown:
            raise KeyError(f"unknown stage(s): {', '.join(sorted(unknown))}; installed: {', '.join(order)}")
        chosen = [n for n in order if n in wanted]
    else:
        if start is not None:
            chosen = chosen[order.index(start):]
        if end is not None:
            chosen = chosen[: chosen.index(end) + 1]

    plan = Plan(order=order, selected=[])
    for name in order:
        stage = registry[name]
        if name not in chosen:
            plan.skipped[name] = "deselected"
        elif name in skip:
            plan.skipped[name] = "skipped by request"
        elif not stage.available(s):
            plan.skipped[name] = "unavailable for this dataset"
        elif stage.spec.optional and not with_optional and not only:
            plan.skipped[name] = "optional"
        elif not force and manifest.is_done(stage, s):
            plan.skipped[name] = "done"
        else:
            plan.selected.append(name)
    return plan


def pose_table_for(registry: StageRegistry, s: "Settings", name: str) -> str:
    """Which pose table a stage must run against.

    Everything up to and including the stage that PRODUCES the corrected table runs on export poses,
    because the corrected one does not exist yet. `mapping/cli/pipeline.py` expressed this as a
    hardcoded set of ten stage names; deriving it from the order means a new registration stage is
    handled without anyone remembering to add it.
    """
    order = registry.order()
    if POSE_TABLE_PRODUCER not in order:
        return s.pose_table
    return "export" if order.index(name) <= order.index(POSE_TABLE_PRODUCER) else s.pose_table


def run_stage(registry: StageRegistry, s: "Settings", name: str, *, extra_args: tuple[str, ...] = ()) -> int:
    """Launch one stage, wait for it, and write its marker. Returns its exit code.

    The marker is written whether the stage succeeded or not -- a failed stage's marker records the
    non-zero rc, which is what stops `is_done` from treating it as complete on the next run.
    """
    stage = registry[name]
    ws = s.workspace
    ws.mkdirs()
    started = manifest.now()
    cmd = [*command_for(stage), *extra_args]

    log_line = f"[{started}] [{name}] $ {' '.join(cmd)}"
    print(log_line)
    _append(ws.pipeline / "pipeline.log", log_line + "\n")

    t0 = time.time()
    rc = procs.run(
        cmd, s,
        log_path=ws.logs / f"{name}.log",
        tee=(ws.pipeline / "pipeline.log",),
        cwd=_repo_root(),
        pose_table=pose_table_for(registry, s, name),
    )
    elapsed = time.time() - t0

    marker = manifest.make_marker(
        name=name, rc=rc,
        inputs=stage.inputs(s), outputs=stage.outputs(s),
        metrics=_safe_metrics(stage, s),
        poses_hash=_safe_poses_hash(s), started=started, repo=_repo_root(),
    )
    marker["seconds"] = round(elapsed, 1)
    manifest.write_marker(ws, name, marker)
    print(f"[{name}] rc={rc} in {elapsed:.0f}s")
    return rc


def run(
    s: "Settings",
    *,
    registry: StageRegistry | None = None,
    dry_run: bool = False,
    **selection,
) -> int:
    """Run a selection of stages in order. Stops at the first failure: a later stage reading a
    half-written artifact produces a wrong result rather than an error, which is worse."""
    registry = registry or build_registry()
    plan = select(registry, s, **selection)

    print(f"dataset {s.name}  poses {s.pose_table}  workspace {s.paths.workspace}")
    for name in plan.order:
        why = plan.skipped.get(name)
        mark = "RUN " if why is None else "    "
        note = "" if why is None else f"  ({why})"
        print(f"  {mark}{name}{note}")
    total = sum(registry[n].spec.est_min for n in plan.selected)
    print(f"  -> {len(plan.selected)} stage(s), roughly {total:.0f} min")

    if dry_run:
        for name in plan.selected:
            print(f"    $ {' '.join(command_for(registry[name]))}")
        return 0

    for name in plan.selected:
        rc = run_stage(registry, s, name)
        if rc != 0:
            print(f"stopped: {name} failed with rc={rc}")
            return rc
    return 0


def status(s: "Settings", registry: StageRegistry | None = None) -> list[dict]:
    """One row per installed stage: what it is, whether it can run, whether it is done."""
    registry = registry or build_registry()
    rows = []
    for name in registry.order():
        stage = registry[name]
        marker = manifest.read_marker(s.workspace, name) or {}
        rows.append({
            "stage": name,
            "available": stage.available(s),
            "optional": stage.spec.optional,
            "done": manifest.is_done(stage, s),
            "rc": marker.get("rc"),
            "finished": marker.get("finished"),
            "seconds": marker.get("seconds"),
            "metrics": marker.get("metrics", {}),
            "summary": stage.spec.summary,
        })
    return rows


# ------------------------------------------------------------------------------------- internals
def _repo_root() -> Path | None:
    """Where to run a stage from. `uv run` needs the workspace root; fall back to the cwd when the
    package is installed outside a checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "packages").is_dir():
            return parent
    return None


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _safe_metrics(stage: Stage, s: "Settings") -> dict:
    try:
        return stage.metrics(s)
    except Exception as e:  # noqa: BLE001 - a marker must always be writable
        return {"error": f"{type(e).__name__}: {e}"}


def _safe_poses_hash(s: "Settings") -> str | None:
    try:
        from geovap.runtime import pose_tables

        return pose_tables.load(s.pose_table, s=s).hash()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None
