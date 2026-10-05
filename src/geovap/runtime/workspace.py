"""Where our own artifacts live: one object owning every output path.

`mapping/config.py` spread this over a dozen module-level constants (`STORE_DIR`, `FRAMES_DIR`,
`OUT_DIR`, `POSES_DIR`, `PIPELINE_DIR`, `CONSOLIDATED_DIR`, ...) derived from a `CACHE_ROOT` that
was itself guessed by probing two hardcoded directories. Collecting them here means a second dataset
needs no new constants, and the one piece of real logic in that file -- `source_dir`, which keeps an
export run and a corrected run from overwriting each other -- has a home next to what it partitions.

Layout under `<workspace>`, unchanged from the existing cache so a built store stays usable:

    store/                      columnar point store (runtime.store)
    frames/                     per-frame products; corrected pose tables get a hashed subdir
    gray/  renders/             luminance and debug renders
    segds/                      segmentation training dataset
    dataset/                    derived per-run summaries (clean_frames.json, frame_quality.csv, seg/)
    out/poses/                  pose tables and their sidecars
    out/pipeline/               stage markers, logs, run manifest
    out/consolidated/tiles/     the delivered product: one LAZ per tile

NOTE (deviation from the restructuring plan, deliberate): the plan sketched
`<workspace>/<dataset>/<run>/...`. The `<dataset>` level is NOT inserted. A descriptor already names
its own `[paths].workspace`, so two datasets are separated by pointing them at two workspaces --
explicit, greppable, and it keeps the existing 18 GB Dražkov store where it is rather than requiring
it to be moved under a new level. The `<run>` level is kept, as `source_dir` below.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings


class _PoseTable(Protocol):
    """What `source_dir` needs of a pose table: which source it came from and its content hash.
    Typed structurally so `runtime` does not depend on the module that loads poses."""

    source: str

    def hash(self) -> str: ...


@dataclass(frozen=True)
class Workspace:
    root: Path  # settings.paths.workspace
    publish: Path  # settings.paths.publish
    baseline: Path  # git-tracked reference results for this dataset (read-only)
    dataset: str

    @classmethod
    def of(cls, s: "Settings") -> "Workspace":
        return cls(
            root=Path(s.paths.workspace),
            publish=Path(s.paths.publish),
            baseline=baseline_dir(s),
            dataset=s.name,
        )

    # -- input-shaped caches -------------------------------------------------------------------
    @property
    def store(self) -> Path:
        return self.root / "store"

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def gray(self) -> Path:
        return self.root / "gray"

    @property
    def renders(self) -> Path:
        return self.root / "renders"

    @property
    def segds(self) -> Path:
        return self.root / "segds"

    @property
    def vehicle_mask(self) -> Path:
        return self.root / "vehicle_mask.npz"

    # -- outputs -------------------------------------------------------------------------------
    @property
    def out(self) -> Path:
        return self.root / "out"

    @property
    def poses(self) -> Path:
        return self.out / "poses"

    @property
    def pipeline(self) -> Path:
        return self.out / "pipeline"

    @property
    def logs(self) -> Path:
        return self.pipeline / "logs"

    @property
    def consolidated(self) -> Path:
        return self.out / "consolidated"

    @property
    def consolidated_tiles(self) -> Path:
        """The delivered product. One LAZ per tile carrying every dimension -- there is deliberately
        no `objects/` or `vendor/` sibling any more; those duplicated 585 M XYZ triples twice over
        only to hand a viewer a different RGB."""
        return self.consolidated / "tiles"

    @property
    def derived(self) -> Path:
        """Per-run summaries that used to be written into the git-tracked `dataset/` directory:
        `clean_frames.json`, `frame_quality.csv`, `seg/`. They are outputs, so they belong here;
        the tracked copies are the *baseline* to compare against, not the live ones."""
        return self.root / "dataset"

    @property
    def clean_frames_json(self) -> Path:
        return self.derived / "clean_frames.json"

    @property
    def quality_csv(self) -> Path:
        return self.derived / "frame_quality.csv"

    # -- per-pose-source partitioning ----------------------------------------------------------
    def source_dir(self, base: Path, poses: _PoseTable) -> Path:
        """Per-pose-source output root: `base` itself for the default "export" pose table, a
        hash-suffixed *sibling* directory for any corrected one.

        This is the regression anchor of the whole refactor. Export-pose outputs must keep landing on
        byte-identical paths, so that a re-run can be diffed against the baseline; a corrected table
        must never write into them. Ported unchanged from `mapping/config.py:source_dir`.
        """
        base = Path(base)
        if poses.source == "export":
            return base
        return base.with_name(base.name + "_" + poses.hash()[:6])

    def frames_dir(self, poses: _PoseTable) -> Path:
        """Products directory for a pose table. Note this is a *sub*directory, not a sibling as
        `source_dir` gives -- ported unchanged from `mapping/products.py:frames_dir`, where the
        difference is load-bearing for the existing cache layout."""
        if poses.source == "export":
            return self.frames
        return self.frames / poses.hash()[:6]

    # -- reading ------------------------------------------------------------------------------
    def clean_frames(self, kind: str = "clean") -> list[int]:
        """Frame indices of the named quality class ("clean" | "unverified" | "usable" | "reject"),
        from the workspace's `clean_frames.json`, falling back to the dataset's tracked baseline when
        the screening stage has not run yet. Replaces `mapping/seg/render_labels.py:clean_frames`."""
        for path in (self.clean_frames_json, self.baseline / "clean_frames.json"):
            if path.is_file():
                return sorted(json.loads(path.read_text(encoding="utf-8"))[kind])
        raise FileNotFoundError(
            f"no clean_frames.json in {self.clean_frames_json.parent} or {self.baseline}; "
            "run the frame-screening stage first"
        )

    def mkdirs(self) -> None:
        """Create the output directories. Inputs-shaped caches are created by whatever builds them."""
        for d in (self.out, self.poses, self.pipeline, self.logs, self.consolidated_tiles, self.derived):
            d.mkdir(parents=True, exist_ok=True)


def baseline_dir(s: "Settings") -> Path:
    """Git-tracked reference results for a dataset: `[paths].baseline` if the descriptor sets one,
    else `<descriptor dir>/<name>/baseline`.

    These are measurements of one dataset (Dražkov's frame classes, its bench mIoU), so they travel
    with the descriptor rather than shipping inside the application -- `mapping/cli/pipeline.py:64`
    held a previous run's numbers as a Python dict, which a product cannot do.
    """
    configured = getattr(s.descriptor, "baseline", None)
    if configured is not None:
        return Path(configured)
    return Path(s.descriptor.file).resolve().parent / s.descriptor.name / "baseline"
