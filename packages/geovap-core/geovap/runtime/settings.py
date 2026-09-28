"""The resolved dataset: descriptor values plus the adapters built from them, available to any
module through `settings.get()`.

This replaces `mapping/config.py`, whose constants were frozen at *import* time. That is the reason
the old code could not choose a dataset from the command line: by the time `argparse` had run,
forty modules had already executed `from .config import LAZ_DIR, PANO_W, ...` and captured Dražkov.
Here nothing is captured at import; a module holds a reference to this module and calls `get()`
inside the function that needs a value:

    from geovap.runtime import settings

    def colorize(frame: int) -> None:
        s = settings.get()
        w, h = s.sensor.pano
        ...

`configure()` is called once per process -- by a stage's `cli.py` after parsing its shared flags, or
by `geovap.runtime.procs` in a subprocess from the environment its parent exported. `get()` without
a prior `configure()` falls back to the environment, which is what makes `python -m
geovap.stages.colour.colorize` work standalone.

Adapters are built lazily and cached: constructing `Settings` must not touch the disk, so `geovap
status --dataset other` can report on a dataset whose data is not mounted.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from geovap.domain.model.crs import Crs
from geovap.domain.model.sensor import Sensor, Tuning
from geovap.io.descriptor import Descriptor

if TYPE_CHECKING:
    from geovap.io.protocols import PanoSource, PoseSource, ReferenceVectors, TileSource
    from geovap.runtime.workspace import Workspace

#: Environment variables read when `configure()` is not given an explicit value. They are also what
#: `runtime.procs` exports to a stage subprocess, so a child resolves to exactly its parent's
#: dataset without inheriting a half-resolved object.
ENV_DATASET = "GEOVAP_DATASET"
ENV_DATA_ROOT = "GEOVAP_DATA"
ENV_WORKSPACE = "GEOVAP_WORKSPACE"
ENV_PUBLISH = "GEOVAP_PUBLISH"
ENV_POSES = "GEOVAP_POSES"

#: Used when `$GEOVAP_DATASET` is unset. A descriptor of this name must resolve -- either
#: `./datasets/<name>.toml` or one shipped in `geovap/io/datasets/`.
DEFAULT_DATASET = "drazkov"

#: Pose table to read when none is named: the export table, which is the regression anchor (its
#: outputs must stay byte-identical across this refactor -- see `Workspace.source_dir`).
DEFAULT_POSE_TABLE = "export"


@dataclass(frozen=True)
class Paths:
    """The three roots a dataset is pinned to. Everything else is derived by `runtime.workspace`."""

    data_root: Path  # read-only input: panoramas, LAZ tiles, reference vectors
    workspace: Path  # our own intermediate artifacts (store, frames, products, markers)
    publish: Path  # what is handed to a viewer or a customer (octrees, panorama exports)


@dataclass(frozen=True)
class Settings:
    """One resolved dataset. Immutable; the adapter cache is the only mutable part and is keyed by
    kind, so two `Settings` for two datasets never share one."""

    name: str
    descriptor: Descriptor
    paths: Paths
    crs: Crs
    sensor: Sensor
    tuning: Tuning
    #: "export", or the name/path of a corrected pose table (was `config.POSES_SOURCE`). Selects
    #: which pose table `stages` read AND which output subdirectory they write to, so an export run
    #: and a corrected run never mix. See `Workspace.source_dir`.
    pose_table: str = DEFAULT_POSE_TABLE
    _adapters: dict = field(default_factory=dict, compare=False, repr=False)

    # -- adapters ------------------------------------------------------------------------------
    # Built on first use, not in __init__: `Settings` must be constructible for a dataset whose
    # data_root is not mounted (that is exactly the case `geovap doctor` has to report on).

    @property
    def poses(self) -> "PoseSource":
        return self._adapter("poses")

    @property
    def tiles(self) -> "TileSource":
        return self._adapter("tiles")

    @property
    def panos(self) -> "PanoSource":
        return self._adapter("panos")

    @property
    def reference(self) -> "ReferenceVectors | None":
        """`None` when the descriptor has no `[reference]` table -- reference vectors are optional,
        and a stage that needs them reports `available() is False` rather than failing the run."""
        return self._adapter("reference")

    def _adapter(self, kind: str):
        if kind not in self._adapters:
            from geovap.io import registry

            build = {
                "poses": registry.build_pose_source,
                "tiles": registry.build_tile_source,
                "panos": registry.build_pano_source,
                "reference": registry.build_reference,
            }[kind]
            self._adapters[kind] = build(self.descriptor)
        return self._adapters[kind]

    # -- derived -------------------------------------------------------------------------------

    @property
    def workspace(self) -> "Workspace":
        """Output layout for this dataset. Cached alongside the adapters."""
        if "workspace" not in self._adapters:
            from geovap.runtime.workspace import Workspace

            self._adapters["workspace"] = Workspace.of(self)
        return self._adapters["workspace"]

    def with_pose_table(self, pose_table: str) -> "Settings":
        """A copy reading a different pose table. Used by stages that compare export against a
        corrected run; the adapter cache is deliberately NOT shared, since `poses` differs."""
        return Settings(
            name=self.name,
            descriptor=self.descriptor,
            paths=self.paths,
            crs=self.crs,
            sensor=self.sensor,
            tuning=self.tuning,
            pose_table=pose_table,
        )

    def env(self) -> dict[str, str]:
        """The environment a subprocess needs to resolve to this exact dataset (see `runtime.procs`)."""
        return {
            ENV_DATASET: str(self.descriptor.file),
            ENV_DATA_ROOT: str(self.paths.data_root),
            ENV_WORKSPACE: str(self.paths.workspace),
            ENV_PUBLISH: str(self.paths.publish),
            ENV_POSES: self.pose_table,
        }


def build(
    *,
    dataset: str | Path | None = None,
    data_root: str | Path | None = None,
    workspace: str | Path | None = None,
    publish: str | Path | None = None,
    poses: str | None = None,
) -> Settings:
    """Resolve a descriptor into `Settings` WITHOUT installing it as the process-wide one.

    Precedence for each path is: explicit argument > `${ENV}` inside the descriptor > literal value
    in the descriptor. The environment is consulted only for values not given here, so a CLI flag
    always wins over a stale exported variable.
    """
    name = dataset if dataset is not None else os.environ.get(ENV_DATASET, DEFAULT_DATASET)
    overrides = {
        "data_root": data_root if data_root is not None else os.environ.get(ENV_DATA_ROOT),
        "workspace": workspace if workspace is not None else os.environ.get(ENV_WORKSPACE),
        "publish": publish if publish is not None else os.environ.get(ENV_PUBLISH),
    }
    descriptor = Descriptor.load(name, overrides={k: v for k, v in overrides.items() if v is not None})
    pose_table = poses if poses is not None else os.environ.get(ENV_POSES, DEFAULT_POSE_TABLE)
    return Settings(
        name=descriptor.name,
        descriptor=descriptor,
        paths=Paths(
            data_root=descriptor.data_root,
            workspace=descriptor.workspace,
            publish=descriptor.publish,
        ),
        crs=descriptor.crs,
        sensor=descriptor.sensor,
        tuning=descriptor.tuning,
        pose_table=pose_table,
    )


# ------------------------------------------------------------------------------- process-wide one
_current: Settings | None = None


def configure(**kwargs) -> Settings:
    """Resolve and install the process-wide `Settings`. Arguments are `build()`'s.

    Calling this twice is allowed (a test fixture, a driver reconfiguring between datasets) and
    simply replaces the current one; every module reads through `get()`, so nothing holds a stale
    copy -- which is the whole point of late binding.
    """
    global _current
    _current = build(**kwargs)
    return _current


def get() -> Settings:
    """The process-wide `Settings`, configuring from the environment on first use."""
    global _current
    if _current is None:
        _current = build()
    return _current


def is_configured() -> bool:
    return _current is not None


def reset() -> None:
    """Forget the process-wide settings. For tests; production code configures once."""
    global _current
    _current = None
