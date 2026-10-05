"""`ingest`: make a dataset legible before anything touches it.

This is the first thing any run does (`after=()`). It writes NOTHING but bookkeeping -- the run
manifest (`geovap.runtime.manifest.RunManifest`) and its own stage marker -- and it must never
modify the dataset it is reading. Everything it reports is a MEASUREMENT of the dataset that was
just resolved, never a property of the software.

That distinction is the whole point of this stage. `mapping/config.py:71` had

    EXPECTED_TOTAL_POINTS = 584_809_840

Dražkov's own point count, hardcoded and asserted against as though any dataset that used this
codebase had to have exactly that many points. It didn't scale to a second dataset and it couldn't
be checked without opening every LAZ file's `.points` payload. `ingest` replaces it: it sums each
tile's `laspy.open(...).header.point_count` -- reading only the LAZ header, not the compressed point
data -- and writes the result into `RunManifest.total_points`, where it belongs to the run, not to
the codebase. `n_tiles`, `n_frames` and `poses_hash` are the same idea applied to the other numbers
`config.py` and `mapping/cli/pipeline.py`'s `REFERENCE` dict used to freeze as literals.

`ingest` also absorbs two legacy scripts wholesale:

  - `experiments/e0_data_sanity.py` ("E0 -- sanity check dat pred cimkoli dalsim"), which printed,
    in Czech, for Dražkov only: export.csv vs panoramas on disk (both directions), pose roll/pitch
    and origin extents, JVF reference-class coverage, and one tile's header + classification.
  - `mapping/cli/pipeline.py`'s `env-check` stage, which printed, for one machine only: whether
    `$GEOVAP_CACHE` exists, whether `store/tiles.json` exists, a hardcoded `--fix-symlink` dance for
    a `/mnt` mount and a `Geovap_cache` symlink, and whether CUDA is importable.

Here both become one English, per-dataset report, driven entirely by the resolved `Settings` and its
adapters' own `describe()` -- no dataset-specific paths, no machine-specific symlink, no `/mnt`
hardcode.

Every check below is run through `_check()`, which never lets one unreadable input take down the
rest of the report: the whole reason to run `ingest` is to see everything that is wrong in one pass,
not to stop at the first `FileNotFoundError`. Only two things are treated as hard failures rather
than a failed check -- the pose table failing to load at all, and the tile source enumerating zero
tiles (or every tile's LAZ header being unreadable) -- because at that point there is nothing to
measure and `ingest` has nothing honest to write into the manifest. Those raise, naming exactly what
is missing, rather than making `available()` lie and say the dataset can't be ingested.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geovap.stages.base.cli import stage_main
from geovap.stages.base.spec import StageSpec, registry

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings


# --------------------------------------------------------------------------------------- helpers
def _check(fn) -> dict:
    """Run one check function; never raises. A check that raises becomes `{"ok": False, "error":
    ...}` instead of aborting the rest of the report -- this is what makes the negative case (one
    missing panorama AND one truncated CSV row) show up as two failed checks, not a traceback."""
    try:
        result = fn()
        result.setdefault("ok", True)
        return result
    except Exception as e:  # noqa: BLE001 - a check must always be reportable
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _safe_describe(adapter) -> dict:
    try:
        return adapter.describe()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _existing_ancestor(p: Path) -> Path:
    p = Path(p)
    while not p.exists():
        parent = p.parent
        if parent == p:
            break
        p = parent
    return p


def _check_frames_vs_panoramas(s: "Settings", poses) -> dict:
    """Both directions: a frame named in the pose table but missing on disk, and a panorama on disk
    that no pose row names. `e0_data_sanity.py` did this with a hardcoded `glob.glob(PANO_DIR +
    "/*.jpg")`; here the directory (if the pano adapter has one -- `describe()`'s `dir` key) is
    listed generically, so a non-jpg or non-flat-directory pano adapter still gets the first
    direction checked (via `panos.path()`), even if the second direction cannot be computed."""
    named = {str(f) for f in poses.filename}
    desc = _safe_describe(s.panos)
    result: dict[str, Any] = {"n_named_in_poses": len(named)}

    pano_dir = desc.get("dir") if isinstance(desc, dict) else None
    if not pano_dir or not Path(pano_dir).is_dir():
        missing = sorted(f for f in named if not s.panos.path(f).exists())
        result.update(
            missing_on_disk=missing[:20], n_missing_on_disk=len(missing),
            on_disk_not_named=None, n_on_disk_not_named=None,
            note="pano adapter has no listable directory; only the 'named but missing' direction was checked",
        )
        result["ok"] = not missing
        return result

    # Restrict the directory listing to the extensions actually named in the pose table: the pano
    # directory can (and, for the synthetic fixture, does) also hold the pose CSV itself, and that
    # is not a "panorama on disk the pose table doesn't know about" -- it is the pose table.
    named_exts = {Path(f).suffix.lower() for f in named}
    on_disk = {p.name for p in Path(pano_dir).iterdir() if p.is_file() and p.suffix.lower() in named_exts}
    missing_on_disk = sorted(named - on_disk)
    extra_on_disk = sorted(on_disk - named)
    result.update(
        missing_on_disk=missing_on_disk[:20], n_missing_on_disk=len(missing_on_disk),
        on_disk_not_named=extra_on_disk[:20], n_on_disk_not_named=len(extra_on_disk),
    )
    result["ok"] = not missing_on_disk and not extra_on_disk
    return result


def _check_pose_time_span(poses) -> dict:
    return {
        "t_min": float(poses.t.min()),
        "t_max": float(poses.t.max()),
        "span_s": float(poses.t.max() - poses.t.min()),
        "n_passes": int(poses.pass_id.max()) + 1 if len(poses) else 0,
    }


def _check_tiles_extent(tile_refs) -> dict:
    boxes = [t.bbox for t in tile_refs if t.bbox is not None]
    if not boxes:
        return {"ok": False, "error": "no tile has a bbox/ring to derive an extent from"}
    min_e = min(b[0] for b in boxes)
    min_n = min(b[1] for b in boxes)
    max_e = max(b[2] for b in boxes)
    max_n = max(b[3] for b in boxes)
    return {"n_tiles": len(tile_refs), "extent_e": [min_e, max_e], "extent_n": [min_n, max_n]}


def _check_reference_coverage(s: "Settings") -> dict:
    """Fraction of reference-vector objects whose taxonomy code is known to the adapter's own class
    table (`e0_data_sanity.py`'s "pokryto class_map.py: known/(known+unknown)"). `None` (not a
    failed check) when the dataset has no `[reference]` adapter at all -- that is optional, not
    broken."""
    ref = s.reference
    if ref is None:
        return {"configured": False}
    objects = ref.objects()
    classes = ref.classes()
    known = sum(1 for o in objects if o.code in classes)
    total = len(objects)
    return {
        "configured": True,
        "n_objects": total,
        "n_known_class": known,
        "coverage": (known / total) if total else None,
    }


def _check_disk_free(s: "Settings") -> dict:
    root = Path(s.workspace.root)
    du = shutil.disk_usage(_existing_ancestor(root))
    return {"path": str(root), "free_gb": round(du.free / 1e9, 1), "total_gb": round(du.total / 1e9, 1)}


def _check_cuda() -> dict:
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch installed is a normal, non-failing outcome
        return {"importable": False, "available": False}
    return {"importable": True, "available": bool(torch.cuda.is_available())}


def _print_report(report: dict) -> None:
    print(f"ingest: dataset {report['dataset']!r} ({report['descriptor']})")
    print(f"  n_frames     : {report['n_frames']}")
    print(f"  n_tiles      : {report['n_tiles']}")
    print(f"  total_points : {report['total_points']}  (measured: sum of LAZ header point_count)")
    print(f"  poses_hash   : {report['poses_hash']}")
    for name, c in report["checks"].items():
        status = "ok" if c.get("ok", True) else "FAILED"
        detail = {k: v for k, v in c.items() if k != "ok"}
        print(f"  [{status:>6}] {name}: {detail}")
    if report["failed_checks"]:
        print(f"  {len(report['failed_checks'])} check(s) FAILED: {report['failed_checks']}")
    else:
        print("  all checks passed")


# ------------------------------------------------------------------------------------------ stage
def checks_for(s: "Settings", poses, tile_refs, tile_errors: dict[str, str] | None = None) -> dict[str, dict]:
    """Every soft check, each individually fault-tolerant: one unreadable input yields one failed
    check, never a traceback, because the whole point is to list everything wrong in a single pass.

    Public because `geovap doctor` runs exactly these. A doctor that disagreed with the stage --
    passing where the stage then fails -- would be worse than no doctor at all.
    """
    tile_errors = tile_errors or {}
    checks: dict[str, dict] = {
        "frames_vs_panoramas": _check(lambda: _check_frames_vs_panoramas(s, poses)),
        "pose_time_span": _check(lambda: _check_pose_time_span(poses)),
        "tiles_extent": _check(lambda: _check_tiles_extent(tile_refs)),
        "crs": _check(lambda: {"epsg": s.crs.epsg, "proj4": s.crs.proj4, "axes": s.crs.axes}),
        "reference_coverage": _check(lambda: _check_reference_coverage(s)),
        "disk_free": _check(lambda: _check_disk_free(s)),
        "cuda": _check(_check_cuda),
    }
    checks["tile_headers"] = (
        {"ok": False, "unreadable": tile_errors, "n_unreadable": len(tile_errors), "n_tiles": len(tile_refs)}
        if tile_errors else {"ok": True, "n_tiles": len(tile_refs)}
    )
    return checks


def checks(s: "Settings") -> dict[str, dict]:
    """`checks_for` with the inputs resolved, each failure contained. Writes nothing -- this is what
    `geovap doctor` calls."""
    try:
        poses = s.poses.load()
    except Exception as e:  # noqa: BLE001
        poses = None
        loaded = {"ok": False, "error": f"cannot load poses: {type(e).__name__}: {e}"}
    else:
        loaded = {"ok": True, "n_frames": len(poses)}
    try:
        tile_refs = s.tiles.tiles()
    except Exception as e:  # noqa: BLE001
        tile_refs = []
        tiles_ok = {"ok": False, "error": f"cannot list tiles: {type(e).__name__}: {e}"}
    else:
        tiles_ok = {"ok": bool(tile_refs), "n_tiles": len(tile_refs)}
        if not tile_refs:
            tiles_ok["error"] = "dataset has no tiles -- nothing to build a store from"

    out = {"poses_load": loaded, "tiles_list": tiles_ok}
    if poses is not None and tile_refs:
        out.update(checks_for(s, poses, tile_refs))
    return out


class Ingest:
    spec = StageSpec(
        name="ingest", after=(), est_min=2,
        summary="dataset legibility report: sources, measured totals, consistency checks (writes only the manifest)",
    )

    def available(self, s: "Settings") -> bool:
        # Always True: a dataset that cannot be ingested FAILS `run()` with a clear message naming
        # the missing input, rather than quietly reporting itself unavailable and letting a driver
        # skip past a broken dataset as if it were merely an optional stage.
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        d: dict[str, Path] = {"descriptor": Path(s.descriptor.file)}
        src = s.poses.source_file()
        if src is not None:
            d["poses_source"] = src
        return d

    def outputs(self, s: "Settings") -> list[Path]:
        from geovap.runtime.manifest import RunManifest

        return [RunManifest.path(s.workspace)]

    def metrics(self, s: "Settings") -> dict:
        try:
            from geovap.runtime.manifest import RunManifest

            m = RunManifest.load(s.workspace)
            if m is None:
                return {"error": "no manifest yet; run `ingest`"}
            return {
                "total_points": m.total_points, "n_tiles": m.n_tiles,
                "n_frames": m.n_frames, "poses_hash": m.poses_hash,
            }
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", *, as_json: bool = False) -> None:
        from geovap.runtime import manifest

        started = manifest.now()

        # -- hard requirements: without these, there is nothing to measure -----------------------
        try:
            poses = s.poses.load()
        except Exception as e:
            raise RuntimeError(
                f"ingest: cannot load poses for dataset {s.name!r} ({s.poses.describe()}): "
                f"{type(e).__name__}: {e}"
            ) from e
        try:
            tile_refs = s.tiles.tiles()
        except Exception as e:
            raise RuntimeError(f"ingest: cannot list tiles for dataset {s.name!r}: {type(e).__name__}: {e}") from e
        if not tile_refs:
            raise RuntimeError(f"ingest: dataset {s.name!r} has no tiles -- nothing to build a store from")

        n_frames = len(poses)
        poses_hash = poses.hash()
        n_tiles = len(tile_refs)

        # measured total: LAZ HEADER point_count only, never the point payload (see module docstring)
        import laspy

        total_points = 0
        tile_errors: dict[str, str] = {}
        for ref in tile_refs:
            try:
                with laspy.open(ref.path) as reader:
                    total_points += int(reader.header.point_count)
            except Exception as e:  # noqa: BLE001
                tile_errors[ref.id.value] = f"{type(e).__name__}: {e}"
        if tile_errors and len(tile_errors) == len(tile_refs):
            raise RuntimeError(f"ingest: none of {n_tiles} tile LAZ headers could be read: {tile_errors}")

        # -- sources: describe() of every adapter, exactly what a run consumed -------------------
        sources = {
            "poses": _safe_describe(s.poses),
            "tiles": _safe_describe(s.tiles),
            "panos": _safe_describe(s.panos),
            "reference": _safe_describe(s.reference) if s.reference is not None else None,
        }

        # -- soft checks: each individually fault-tolerant ----------------------------------------
        checks = checks_for(s, poses, tile_refs, tile_errors)

        failed_checks = sorted(k for k, v in checks.items() if not v.get("ok", True))

        report = {
            "dataset": s.name,
            "descriptor": str(s.descriptor.file),
            "n_frames": n_frames,
            "n_tiles": n_tiles,
            "total_points": total_points,
            "poses_hash": poses_hash,
            "sources": sources,
            "checks": checks,
            "failed_checks": failed_checks,
        }

        if as_json:
            print(json.dumps(report, indent=2, default=str))
        else:
            _print_report(report)

        # -- write: the manifest, then this stage's own marker. Nothing else. --------------------
        m = manifest.RunManifest(
            dataset=s.name,
            descriptor=str(s.descriptor.file),
            git_rev=manifest.git_rev(),
            total_points=total_points,
            n_tiles=n_tiles,
            n_frames=n_frames,
            poses_hash=poses_hash,
            sources=sources,
            extra={"checks": checks, "failed_checks": failed_checks},
        )
        m.save(s.workspace)

        rc = 0 if not failed_checks else 1
        marker = manifest.make_marker(
            name=self.spec.name, rc=rc, inputs=self.inputs(s), outputs=self.outputs(s),
            metrics=self.metrics(s), poses_hash=poses_hash, started=started,
        )
        manifest.write_marker(s.workspace, self.spec.name, marker)

        if failed_checks:
            raise RuntimeError(
                f"ingest: {len(failed_checks)} check(s) failed for dataset {s.name!r}: {failed_checks} "
                "(see the report above for what and where)"
            )


STAGE = registry.add(Ingest())


def _add_options(p) -> None:
    p.add_argument("--json", action="store_true", help="dump the report as JSON instead of the human-readable form")


def _to_opts(args) -> dict:
    return {"as_json": args.json}


def main(argv=None) -> int:
    return stage_main(STAGE, argv, add_options=_add_options, to_opts=_to_opts)


if __name__ == "__main__":
    raise SystemExit(main())
