"""The shape every pipeline step shares.

`mapping/cli/pipeline.py` declared all 24 stages inline: each one a `_stage(...)` call carrying
lambdas for its commands, inputs, outputs and metrics, ordered by a literal `_ORDER` list that had
to be kept in sync by hand (there is an `assert` in that file whose only job is to catch the two
drifting apart). That put every stage's knowledge in the driver, so adding a step meant editing the
orchestrator, and a team could not own a stage without touching a shared file.

Here a stage declares itself. The driver only orders them and runs them.

A stage group *exports* `StageSpec`s -- it is not one-to-one with a module. `stages.register`
exports `align`, `traj-rot`, `refine`, `assemble` and the rest, so the existing marker files,
resume semantics and `--from/--to/--only` selectors keep working unchanged.

Stages still execute as SUBPROCESSES. That is deliberate and predates this refactor: a stage's
memmaps and CUDA context are released when its process exits, which is what makes a 24-stage run
survive on one machine. So `Stage` supplies the *declaration* and `geovap.app.driver` supplies the
*execution*; `run()` is what the stage's own CLI calls once the driver has spawned it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid an import edge from stages up into runtime
    from geovap.runtime.settings import Settings


@dataclass(frozen=True)
class StageSpec:
    """Identity and ordering of one resumable step."""

    name: str
    #: Stage names that must have completed first. The driver topologically sorts on this, so no
    #: module holds the global order any more.
    after: tuple[str, ...] = ()
    #: An optional stage that reports `available() is False` is SKIPPED, not failed. This is how a
    #: dataset with no reference vectors still completes a run, and how clustering stays a branch
    #: that can be left out entirely.
    optional: bool = False
    #: Rough cost, used only to print an estimate before a long run.
    est_min: float = 1.0
    #: Human summary for `geovap status`.
    summary: str = ""


@runtime_checkable
class Stage(Protocol):
    """What a stage module must expose.

    Every method takes the resolved `Settings` rather than reading globals, so the same stage object
    can be interrogated for one dataset while another is being processed -- which is what `geovap
    status --dataset X` does without touching X's data.
    """

    spec: StageSpec

    def available(self, s: Settings) -> bool:
        """False when this stage cannot run against this dataset at all -- e.g. no `[reference]`
        adapter is configured, so there is nothing to derive pseudo-ground-truth from. Distinct
        from "not done yet"."""
        ...

    def inputs(self, s: Settings) -> dict[str, Path]:
        """Named paths whose content identity decides whether the stage is stale. The driver hashes
        these; unchanged hashes plus existing outputs mean the stage is skipped on a re-run."""
        ...

    def outputs(self, s: Settings) -> list[Path]:
        """Paths that must exist for the stage to count as done."""
        ...

    def metrics(self, s: Settings) -> dict[str, Any]:
        """Numbers to record in the stage marker. Must not raise: a stage run out of order, or
        against a checkout with no data, returns `{"error": ...}` so a marker is always writable."""
        ...

    def run(self, s: Settings, **opts: Any) -> None:
        """Do the work. Called in the stage's own subprocess by its CLI."""
        ...


@dataclass
class StageRegistry:
    """Every StageSpec the installed distributions contribute, resolved into a run order.

    Stage groups register themselves on import; which groups exist depends on what is installed, so
    a core-only install simply has no `semantics` stages rather than a broken pipeline.
    """

    stages: dict[str, Stage] = field(default_factory=dict)

    def add(self, stage: Stage) -> Stage:
        if stage.spec.name in self.stages:
            raise ValueError(f"duplicate stage name {stage.spec.name!r}")
        self.stages[stage.spec.name] = stage
        return stage

    def __contains__(self, name: str) -> bool:
        return name in self.stages

    def __getitem__(self, name: str) -> Stage:
        try:
            return self.stages[name]
        except KeyError:
            raise KeyError(
                f"unknown stage {name!r}; installed stages: {', '.join(sorted(self.stages))}"
            ) from None

    def order(self) -> list[str]:
        """Stage names in dependency order (Kahn, ties broken by name so a run is reproducible).

        An `after` naming a stage that is not installed is ignored rather than fatal: that is the
        normal state of a core-only install, where `merge` lists `label` as a predecessor but the
        semantics distribution is absent.
        """
        known = set(self.stages)
        pending = {n: {a for a in st.spec.after if a in known} for n, st in self.stages.items()}
        out: list[str] = []
        while pending:
            ready = sorted(n for n, deps in pending.items() if not deps)
            if not ready:
                raise ValueError(f"cycle among stages: {', '.join(sorted(pending))}")
            for n in ready:
                out.append(n)
                del pending[n]
            for deps in pending.values():
                deps.difference_update(ready)
        return out


#: The process-wide registry. Stage groups call `registry.add(...)` at import time.
registry = StageRegistry()
