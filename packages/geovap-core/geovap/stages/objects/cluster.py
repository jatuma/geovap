"""The `cluster` stage: run the clustering algorithms over a dataset's tiles.

This is the only module in the group that knows what a dataset is. `cluster_laz.py` and
`merge_tiles.py` take a LAZ path in and write a LAZ path out, importing nothing from the project --
that is deliberate, and it is what makes this whole stream detachable: it can be worked on by
someone with no Geovap context, it can run before or after anything else, and `optional=True` means
a run that skips it still produces merged tiles (the merge stage already tolerates missing cluster
input).

What this module replaces is `pointcloud-tools/clusters/batch.py`, which derived tile names by
slicing filenames (`basename(f).split("_")[1]`) and had the dataset's LAZ directory as a literal
default. Both now come from the descriptor: the input tiles from `TileSource.tiles()`, the output
names from `TileSource.out_name(..., variant="cluster")`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from geovap.stages.base.cli import stage_main
from geovap.stages.base.spec import StageSpec, registry

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings

#: The parameters the Dražkov run used. They are tuning, not dataset identity, so they stay here as
#: defaults rather than in the descriptor; `--arg` overrides any of them.
DEFAULT_ARGS = ("--eps", "0.3", "--hmin", "0.4", "--voxel", "0.1", "--hsplit", "2.5",
                "--watershed", "--classes")


def out_dir(s: "Settings") -> Path:
    return s.workspace.out / "clusters"


def _one(job) -> tuple[str, str]:
    """Cluster one tile in its own process: `cluster_laz` writes the rgb product, then the palette
    pass writes the objects product. Kept as a subprocess call, as before, so one tile's memory is
    returned to the OS before the next starts."""
    import laspy

    tile_id, src, rgb_path, obj_path, args = job
    if Path(obj_path).exists():
        return tile_id, "skip"
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "geovap.stages.objects.cluster_laz", str(src), str(rgb_path), *args],
        capture_output=True, text=True,
    )
    if r.returncode:
        return tile_id, "FAIL " + r.stderr[-300:]

    las = laspy.read(str(rgb_path))
    c = np.asarray(las.cluster_id)
    # A per-tile palette seeded from the tile id, so re-running one tile reproduces its colours.
    rng = np.random.default_rng(abs(hash(tile_id)) % (2**32))
    pal = rng.integers(40, 255, (max(int(c.max()), 0) + 1, 3)).astype(np.uint16) * 257
    col = np.zeros((len(c), 3), np.uint16)
    m = c >= 0
    col[m] = pal[c[m]]
    col[c == -1] = 140 * 257  # ground
    col[c == -2] = 60 * 257  # noise / below hmin
    las.red, las.green, las.blue = col[:, 0], col[:, 1], col[:, 2]
    las.write(str(obj_path))

    n_clusters = [ln for ln in r.stdout.splitlines() if ln.startswith("clusters")]
    return tile_id, f"{len(c):,} pts, {n_clusters[0] if n_clusters else ''}  {time.time() - t0:.0f}s"


class Cluster:
    spec = StageSpec(
        name="cluster",
        after=(),  # reads raw tiles, nothing else -- this stream can run from day one
        optional=True,
        est_min=120,
        summary="segment each tile into ground + objects, write a cluster_id dimension",
    )

    def available(self, s: "Settings") -> bool:
        return bool(s.tiles.tiles())

    def inputs(self, s: "Settings") -> dict[str, Path]:
        return {f"tile_{t.id.value}": t.path for t in s.tiles.tiles()}

    def outputs(self, s: "Settings") -> list[Path]:
        d = out_dir(s)
        return [d / s.tiles.out_name(t.id, variant="cluster") for t in s.tiles.tiles()]

    def metrics(self, s: "Settings") -> dict:
        try:
            d = out_dir(s)
            done = [p for p in self.outputs(s) if p.exists()]
            labels = d / "global_labels.npy"
            out = {"n_tiles": len(self.outputs(s)), "n_done": len(done)}
            if labels.exists():
                lab = np.load(labels)
                out["n_global_objects"] = int(len(np.unique(lab)))
            return out
        except Exception as e:  # noqa: BLE001 - a marker must always be writable
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", *, workers: int = 6, args: tuple[str, ...] = DEFAULT_ARGS,
            merge: bool = True, force: bool = False) -> None:
        d = out_dir(s)
        d.mkdir(parents=True, exist_ok=True)
        jobs = []
        for tile in s.tiles.tiles():
            rgb = d / s.tiles.out_name(tile.id, variant="cluster_rgb")
            obj = d / s.tiles.out_name(tile.id, variant="cluster")
            if force:
                for p in (rgb, obj):
                    p.unlink(missing_ok=True)
            jobs.append((tile.id.value, tile.path, rgb, obj, list(args)))
        # Largest first: the tail of a parallel run is one huge tile, so start it early.
        jobs.sort(key=lambda j: -Path(j[1]).stat().st_size)
        from concurrent.futures import ProcessPoolExecutor

        from geovap.runtime.procs import pool_context

        with ProcessPoolExecutor(workers, mp_context=pool_context()) as ex:
            for tile_id, msg in ex.map(_one, jobs):
                print(tile_id, msg, flush=True)

        if merge:
            # `merge_tiles` reads its directory from $CLUSTERS_DATA: that is how the dataset reaches
            # a module that must not import the project.
            env = dict(os.environ, CLUSTERS_DATA=str(d))
            rc = subprocess.run(
                [sys.executable, "-m", "geovap.stages.objects.merge_tiles"], env=env
            ).returncode
            if rc:
                raise RuntimeError(f"merge_tiles failed with rc={rc}")


STAGE = registry.add(Cluster())


def _add_options(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-merge", action="store_true", help="skip the cross-tile object merge")
    ap.add_argument("--force", action="store_true", help="recluster tiles that are already done")
    ap.add_argument("--arg", action="append", default=None, metavar="FLAG",
                    help=f"replace the clustering parameters (default: {' '.join(DEFAULT_ARGS)})")


def _to_opts(a: argparse.Namespace) -> dict:
    return {
        "workers": a.workers,
        "merge": not a.no_merge,
        "force": a.force,
        "args": tuple(a.arg) if a.arg else DEFAULT_ARGS,
    }


def main(argv=None) -> int:
    return stage_main(STAGE, argv, add_options=_add_options, to_opts=_to_opts)


if __name__ == "__main__":
    raise SystemExit(main())
