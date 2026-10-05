"""`merge`: the consolidated 3D product -- ONE LAZ per tile, every dimension.

Merges, per tile, the store's own reference columns with three per-tile products (colour-stage
fused colour, semantics-stage projected labels, objects-stage clusters) that all share the store's
row count and per-row point identity (see `geovap.io.las_writer` and
`geovap.runtime.store.TileData.orig_index`) into ONE LAS 1.4 PF7 file per tile:

    <workspace>/out/consolidated/tiles/<tile out_name>.laz
        rgb            = fused colour (reference RGB where n_views == 0)
        classification = common15 label (255 = unlabelled)
        extras         = geovap.io.las_writer.CONSOLIDATED_DIMS (22 dimensions)
        xyz            = registered (S5)

THIS IS A COLLAPSE, NOT A NEW FEATURE. The previous version of this module (`mapping/merge.py`,
before this rewrite) wrote THREE parallel LAZ sets over the same ~585 M points -- `tiles/` (fused
colour), `objects/` (cluster palette in RGB), `vendor/` (reference RGB) -- 75 GB total, two thirds
of which was the identical XYZ written three times just to hand a viewer a different colour.
`objects/`'s `cluster_id`/`obj_class` were already columns of `tiles/`; `vendor/`'s RGB was already
`ref_r/g/b`. That is why `objects/` and `vendor/` are GONE here, `write_vendor_tile` and `run_vendor`
no longer exist, and the octree stage (`stages.deliver.octree`) builds one cloud instead of three.
The viewer shades by attribute over the single cloud instead of loading a second one.

Every input is checked against the store's own points (`check_same_points`) before use, and against
the current pose table's hash where it carries one (the colour product's provenance VLR); a
mismatch aborts unless `--allow-mixed-poses` (the seg-labels/clusters products predate S5b and may
only exist under an older, or the export, pose table).

A missing PRODUCT (colour tile, seg labels, cluster tile) does not fail the merge -- clustering in
particular is an optional stage (`stages.objects.cluster`, `optional=True`) and a dataset with no
clustering must still produce a complete consolidated tile. The affected dimensions get their
documented neutral default and a warning lands in the tile's `_meta.json` and `summary.json`.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from geovap.domain.scheme import taxonomy as T
from geovap.io.las_writer import CONSOLIDATED_DIMS, PROVENANCE_USER_ID, write_tile
from geovap.runtime.las_check import verify_tile
from geovap.runtime.pose_tables import load as load_poses
from geovap.runtime.store import SCALE, CloudStore, TileData, open_store
from geovap.stages.base.cli import stage_main
from geovap.stages.base.spec import StageSpec, registry

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings


def _out_name(s: "Settings", name: str, kind: str = "", *, variant: str | None = None) -> str:
    """Product filename for a tile, from the dataset descriptor. Replaces the literal `ID3432_000`
    prefix, which was Dražkov's and was duplicated across ten modules -- on another dataset it wrote
    every product under the wrong name, with no error anywhere."""
    from geovap.domain.model.tiles import TileId

    return s.tiles.out_name(TileId(name), kind, variant=variant)


# --------------------------------------------------------------------------------------- locating inputs
def _non_empty(d: Path) -> bool:
    return d.is_dir() and any(d.iterdir())


@dataclass
class MergeInputs:
    """Resolved inputs for one `run`/`merge_tile` call. `poses` is the loaded `Poses` table the
    consolidated product is registered against (see `geovap.runtime.store.open_store`); `poses_hash`
    is a convenience for provenance.

    Every path field defaults to `None` and is resolved in `locate_inputs`/`run`, never at
    definition time -- `tests/test_no_import_time_settings.py` forbids a dataset value being
    resolved by a default argument.
    """

    poses: object
    #: The selector passed to `pose_tables.load` ("export" | "corrected" | a CSV path), kept
    #: separately from `poses.source` (which may be a bare file stem, e.g. "poses_corrected") so a
    #: worker process can reload the SAME table rather than mis-resolving a stem as a relative path.
    poses_source: str | None = None
    tw45_tiles: Path | None = None
    seg_labels: Path | None = None
    clusters_src: Path | None = None
    out_dir: Path | None = None
    rgb_fallback: bool = True
    allow_mixed_poses: bool = False

    @property
    def poses_hash(self) -> str:
        return self.poses.hash()


def locate_inputs(s: "Settings", poses_source: str | None = None, out_dir: Path | str | None = None,
                   tw45_tiles: Path | str | None = None, colour_tag: str = "colour",
                   seg_labels: Path | str | None = None, clusters_src: Path | str | None = None,
                   rgb_fallback: bool = True, allow_mixed_poses: bool = False) -> MergeInputs:
    """Resolves the default input directories against `s` (never against a module-level constant --
    `s` must be an explicit, already-resolved `Settings`, so this stays late-bound). Missing
    tw45/seg inputs are tolerated (the tile falls back to reference RGB / all-unlabelled); pass
    explicit paths to override any of them.

    `colour_tag` mirrors `geovap.stages.colour.colorize.Options.tag` (default `"colour"`, the
    `coloured_tiles` artifact's frozen location is `s.workspace.out / tag / "tiles"`) -- pass the
    `--tag` a colorize run was made with if it differs from the default (Dražkov's pipeline used
    `"tw45"` before the colour stage was ported)."""
    poses = load_poses(poses_source, s=s)
    tw45 = Path(tw45_tiles) if tw45_tiles is not None else s.workspace.out / colour_tag / "tiles"
    if tw45_tiles is None and not _non_empty(tw45):
        tw45 = None
    seg = Path(seg_labels) if seg_labels is not None else s.workspace.out / "seg_eomt" / "labels"
    if seg_labels is None and not _non_empty(seg):
        seg = None
    clusters = Path(clusters_src) if clusters_src is not None else s.workspace.out / "clusters"
    out = Path(out_dir) if out_dir is not None else s.workspace.consolidated
    return MergeInputs(poses=poses, poses_source=poses_source, tw45_tiles=tw45, seg_labels=seg, clusters_src=clusters,
                        out_dir=out, rgb_fallback=rgb_fallback, allow_mixed_poses=allow_mixed_poses)


# --------------------------------------------------------------------------------------- point identity
def check_same_points(a: np.ndarray, b: np.ndarray, name: str = "input") -> None:
    """Both int arrays [n,3] (or broadcastable), same STORE row order. Raises `ValueError` (not a bool
    return -- a silently mismatched merge would write garbage) if the lengths or coordinates differ."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"{name}: point count/shape mismatch: {a.shape} vs store's {b.shape}")
    if not np.array_equal(a, b):
        n_diff = int((a != b).any(axis=-1).sum()) if a.ndim > 1 else int((a != b).sum())
        raise ValueError(f"{name}: {n_diff}/{len(a)} points do not match the store (different point order or source)")


def registered_xyz_int(td: TileData) -> np.ndarray:
    """int32 [n,3] LAS integers of `td.xyz_m()` (honours `td.registration`), STORE row order. Registered
    mm coordinates fit int32 at the store's scale (0.001 m, see `geovap.runtime.store.SCALE`)."""
    xyz = np.round(td.xyz_m() / SCALE)
    out = xyz.astype(np.int32)
    assert np.array_equal(out, xyz), "registered coordinates overflow int32 at 1 mm scale"
    return out


def _reorder_to_store(arr_src: np.ndarray, orig_index: np.ndarray) -> np.ndarray:
    """A source-order array (e.g. straight off a LAS file) -> STORE row order."""
    return np.asarray(arr_src)[np.asarray(orig_index)]


def _read_provenance(las) -> dict | None:
    for v in las.header.vlrs:
        if v.user_id == PROVENANCE_USER_ID:
            return json.loads(bytes(v.record_data).decode())
    return None


def _check_poses(prov: dict | None, path: Path, inp: MergeInputs, warnings: list[str]) -> None:
    if prov is None or "poses_hash" not in prov:
        return
    if prov["poses_hash"] != inp.poses_hash:
        msg = f"{path}: poses_hash {prov['poses_hash']} != current poses {inp.poses_hash} ({inp.poses.source})"
        if not inp.allow_mixed_poses:
            raise ValueError(msg + " (pass --allow-mixed-poses to proceed anyway)")
        warnings.append(msg)


# --------------------------------------------------------------------------------------- per-tile merge
def merge_tile(name: str, store: CloudStore, inp: MergeInputs, s: "Settings", log=print) -> dict:
    import laspy

    t0 = time.time()
    td = store.tile(name)
    n = len(td)
    inv = np.asarray(td.orig_index)
    warnings: list[str] = []
    inputs_used: dict[str, str | None] = {}

    xyz = registered_xyz_int(td)
    src_class = np.asarray(td.classification).copy()

    # ------------------------------------------------------------------------------------- colour (tw45)
    rgb = np.asarray(td.rgb).copy()
    ref_rgb = np.asarray(td.rgb).copy()
    nt_rgb = np.zeros((n, 3), np.uint8)
    n_views = np.zeros(n, np.uint8)
    col_conf = np.zeros(n, np.uint8)
    de00_med = np.full(n, np.nan, np.float32)
    de00_nt = np.full(n, np.nan, np.float32)
    de00_nt_noocc = np.full(n, np.nan, np.float32)
    src_image = np.zeros(n, np.uint16)
    cam_dist = np.zeros(n, np.float32)
    inc_angle = np.full(n, 255, np.uint8)  # 255 = unknown, per COLOUR_DIMS
    img_grad = np.zeros(n, np.float32)
    if inp.tw45_tiles is not None:
        p = Path(inp.tw45_tiles) / _out_name(s, name, "_colored")
        if p.exists():
            las = laspy.read(str(p))
            check_same_points(np.stack([np.asarray(las.X), np.asarray(las.Y), np.asarray(las.Z)], 1)[inv], np.asarray(td.xyz), f"tw45 {p.name}")
            _check_poses(_read_provenance(las), p, inp, warnings)
            fused = np.stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)], 1) >> 8
            rgb = _reorder_to_store(fused, inv).astype(np.uint8)
            ref_rgb = _reorder_to_store(np.stack([np.asarray(las.ref_r), np.asarray(las.ref_g), np.asarray(las.ref_b)], 1), inv).astype(np.uint8)
            nt_rgb = _reorder_to_store(np.stack([np.asarray(las.nt_r), np.asarray(las.nt_g), np.asarray(las.nt_b)], 1), inv).astype(np.uint8)
            n_views = _reorder_to_store(np.asarray(las.n_views), inv).astype(np.uint8)
            col_conf = _reorder_to_store(np.asarray(las.col_conf), inv).astype(np.uint8)
            de00_med = _reorder_to_store(np.asarray(las.dE00_med), inv).astype(np.float32)
            de00_nt = _reorder_to_store(np.asarray(las.dE00_nt), inv).astype(np.float32)
            de00_nt_noocc = _reorder_to_store(np.asarray(las.dE00_nt_noocc), inv).astype(np.float32)
            src_image = _reorder_to_store(np.asarray(las.src_image), inv).astype(np.uint16)
            cam_dist = _reorder_to_store(np.asarray(las.cam_dist), inv).astype(np.float32)
            inc_angle = _reorder_to_store(np.asarray(las.inc_angle), inv).astype(np.uint8)
            img_grad = _reorder_to_store(np.asarray(las.img_grad), inv).astype(np.float32)
            inputs_used["tw45"] = str(p)
            if inp.rgb_fallback:
                no_col = n_views == 0
                rgb[no_col] = np.asarray(td.rgb)[no_col]
        else:
            warnings.append(f"tw45 tile missing: {p}")
    else:
        warnings.append("tw45 tiles dir not found: colour dimensions filled with neutral defaults")

    # ------------------------------------------------------------------------------------- seg labels
    classification = np.full(n, T.IGNORE, np.uint8)
    seg_conf = np.zeros(n, np.uint8)
    seg_n_views = np.zeros(n, np.uint8)
    seg_src_frame = np.zeros(n, np.uint16)
    if inp.seg_labels is not None:
        lp = Path(inp.seg_labels) / f"{name}.npy"
        if lp.exists():
            label = np.load(lp)
            if len(label) != n:
                raise ValueError(f"seg labels {lp}: n={len(label)} != store n={n}")
            classification = label.astype(np.uint8)
            cp, np_ = Path(inp.seg_labels) / f"{name}_conf.npy", Path(inp.seg_labels) / f"{name}_nviews.npy"
            if cp.exists():
                seg_conf = np.load(cp).astype(np.uint8)
            if np_.exists():
                seg_n_views = np.load(np_).astype(np.uint8)
            sf = Path(inp.seg_labels) / f"{name}_src_frame.npy"
            if sf.exists():
                seg_src_frame = np.load(sf).astype(np.uint16)
            else:
                # The seg product (`mapping.seg.project`/its successor) writes `src_frame` only into
                # the per-tile LAS it emits, not as a standalone `.npy` alongside label/conf/nviews --
                # so this is the ordinary case, not a missing-product warning.
                warnings.append(f"seg_src_frame: no {name}_src_frame.npy in {inp.seg_labels}; filled zeros")
            inputs_used["seg_labels"] = str(lp)
            mp = Path(inp.seg_labels) / f"{name}_meta.json"
            if mp.exists():
                meta = json.loads(mp.read_text())
                if meta.get("n") not in (None, n):
                    warnings.append(f"seg meta {mp}: n={meta.get('n')} != store n={n}")
                _check_poses(meta, mp, inp, warnings)
        else:
            raise FileNotFoundError(f"seg labels missing for tile {name}: {lp} (labels dir {inp.seg_labels} resolved but has no file for this tile)")
    else:
        warnings.append("seg labels dir not found: classification left all-unlabelled (255)")

    # ------------------------------------------------------------------------------------- clusters
    # Optional branch: `stages.objects.cluster` is `optional=True` and a dataset may never run it.
    # A missing cluster product must NOT fail the merge.
    cluster_id = np.full(n, -1, np.int32)
    obj_class = np.zeros(n, np.uint8)
    hag = np.zeros(n, np.float32)
    cp = Path(inp.clusters_src) / s.tiles.out_name(td.info.id, variant="cluster")
    if cp.exists():
        las = laspy.read(str(cp))
        check_same_points(np.stack([np.asarray(las.X), np.asarray(las.Y), np.asarray(las.Z)], 1)[inv], np.asarray(td.xyz), f"clusters {cp.name}")
        cluster_id = _reorder_to_store(np.asarray(las.cluster_id), inv).astype(np.int32)
        obj_class = _reorder_to_store(np.asarray(las.obj_class), inv).astype(np.uint8)
        hag = _reorder_to_store(np.asarray(las.hag), inv).astype(np.float32)
        inputs_used["clusters"] = str(cp)
    else:
        warnings.append(f"clusters tile missing: {cp} (clustering is optional; cluster_id left at -1)")

    provenance = {
        "product": "consolidated", "frame": "registered", "poses_source": inp.poses.source, "poses_hash": inp.poses_hash,
        "registration": str(inp.poses.registration) if getattr(inp.poses, "registration", None) else None,
        "inputs": inputs_used, "warnings": warnings, "git": _git_rev(),
    }

    extras = {
        "seg_conf": seg_conf, "seg_n_views": seg_n_views, "seg_src_frame": seg_src_frame, "src_class": src_class,
        "cluster_id": cluster_id, "obj_class": obj_class, "hag": hag,
        "ref_r": ref_rgb[:, 0], "ref_g": ref_rgb[:, 1], "ref_b": ref_rgb[:, 2],
        "nt_r": nt_rgb[:, 0], "nt_g": nt_rgb[:, 1], "nt_b": nt_rgb[:, 2],
        "dE00_med": de00_med, "dE00_nt": de00_nt, "dE00_nt_noocc": de00_nt_noocc,
        "src_image": src_image, "n_views": n_views, "col_conf": col_conf,
        "cam_dist": cam_dist, "inc_angle": inc_angle, "img_grad": img_grad,
    }
    assert set(extras) == {n for n, _dt, _d in CONSOLIDATED_DIMS}, sorted(set(extras) ^ {n for n, _dt, _d in CONSOLIDATED_DIMS})

    tiles_dir = Path(inp.out_dir) / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_out = tiles_dir / _out_name(s, name)
    tile_tmp = tile_out.with_suffix(".laz.tmp")
    write_tile(td.info.laz, inv, tile_tmp, rgb, extras, provenance, classification=classification, xyz=xyz,
               description="consolidated provenance")
    os.replace(tile_tmp, tile_out)

    v_tile = verify_tile(tile_out, td, expect_class_exact=False, xyz_mode="registered")

    counts = np.bincount(classification, minlength=256)
    de_valid = de00_med[n_views > 0]
    meta = {
        "n": n, "seconds": time.time() - t0, "counts": counts[: len(T.COMMON)].tolist(), "unlabelled": int(counts[T.IGNORE]),
        "n_clustered": int((cluster_id >= 0).sum()), "n_clusters": int(len(np.unique(cluster_id[cluster_id >= 0]))),
        "dE00_med_median": float(np.median(de_valid)) if len(de_valid) else None,
        "provenance": provenance, "verify": {"tile": v_tile}, "warnings": warnings,
    }
    (tiles_dir / f"{name}_meta.json").write_text(json.dumps(meta))  # last: resume marker
    log(f"tile {name}: {n} pts, {res_summary(meta)}, {meta['seconds']:.0f}s")
    return meta


def res_summary(meta: dict) -> str:
    return f"unlabelled={meta['unlabelled']}, clustered={meta['n_clustered']}, dE00_med={meta['dE00_med_median']}"


def _git_rev() -> str | None:
    from geovap.runtime import manifest

    return manifest.git_rev()


# --------------------------------------------------------------------------------------- driver
_G: dict = {}


def _init(dataset_env: dict[str, str], inp_kwargs: dict) -> None:
    """Re-resolve `Settings` and `MergeInputs` in the child, from plain (picklable) data -- the pool
    may not use `fork`, so the parent's in-memory `Settings`/adapters would otherwise be invisible
    or unpicklable here. Mirrors `geovap.stages.prepare.products._init_worker`."""
    import os

    os.environ.update(dataset_env)
    from geovap.runtime import settings
    from geovap.runtime.store import CloudStore

    settings.reset()
    s = settings.get()
    inp = locate_inputs(s, poses_source=inp_kwargs["poses_source"], out_dir=inp_kwargs["out_dir"],
                         tw45_tiles=inp_kwargs["tw45_tiles"], seg_labels=inp_kwargs["seg_labels"],
                         clusters_src=inp_kwargs["clusters_src"], rgb_fallback=inp_kwargs["rgb_fallback"],
                         allow_mixed_poses=inp_kwargs["allow_mixed_poses"])
    _G["s"] = s
    _G["inp"] = inp
    _G["store"] = CloudStore(s.workspace.store, registration=inp.poses.registration if inp.poses is not None else None)


def _run_tile(name: str) -> dict:
    return merge_tile(name, _G["store"], _G["inp"], _G["s"])


def run(tiles: list[str] | None, inp: MergeInputs, s: "Settings", workers: int = 6, force: bool = False) -> dict:
    """Merges `tiles` (default: all tiles in the store), largest-first, resuming from
    `out_dir/tiles/NNN_meta.json` unless `force`. Writes `out_dir/summary.json`."""
    from geovap.runtime.procs import pool_context
    from geovap.runtime.store import CloudStore

    store = CloudStore(s.workspace.store, registration=inp.poses.registration if inp.poses is not None else None)
    names = [t.name for t in store.tiles] if tiles is None else list(tiles)
    names = sorted(names, key=lambda nm: -store.by_name[nm].n)
    meta_dir = Path(inp.out_dir) / "tiles"
    meta_dir.mkdir(parents=True, exist_ok=True)

    todo = names if force else [nm for nm in names if not (meta_dir / f"{nm}_meta.json").exists()]
    skipped = [nm for nm in names if nm not in todo]
    results: dict[str, dict] = {}
    for nm in skipped:
        results[nm] = json.loads((meta_dir / f"{nm}_meta.json").read_text())

    if todo:
        inp_kwargs = {
            "poses_source": inp.poses_source, "out_dir": str(inp.out_dir),
            "tw45_tiles": str(inp.tw45_tiles) if inp.tw45_tiles else None,
            "seg_labels": str(inp.seg_labels) if inp.seg_labels else None,
            "clusters_src": str(inp.clusters_src) if inp.clusters_src else None,
            "rgb_fallback": inp.rgb_fallback, "allow_mixed_poses": inp.allow_mixed_poses,
        }
        if workers <= 1:
            _init(s.env(), inp_kwargs)
            for nm in todo:
                results[nm] = _run_tile(nm)
        else:
            with pool_context().Pool(workers, initializer=_init, initargs=(s.env(), inp_kwargs)) as pool:
                for nm, meta in zip(todo, pool.map(_run_tile, todo)):
                    results[nm] = meta

    total_n = sum(m["n"] for m in results.values())
    from geovap.runtime.manifest import RunManifest

    manifest = RunManifest.load(s.workspace)
    expected_total_points = manifest.total_points if manifest is not None else None
    summary = {
        "n_tiles": len(results), "total_points": total_n, "expected_total_points": expected_total_points,
        "matches_expected": (expected_total_points is None or total_n == expected_total_points) and len(results) == len(store.tiles),
        "poses_source": inp.poses.source, "poses_hash": inp.poses_hash, "tiles": sorted(results),
    }
    (Path(inp.out_dir) / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# ================================================================================================ stage
class Merge:
    spec = StageSpec(
        name="merge", after=("colorize", "seg-project", "cluster"), est_min=25,
        summary="consolidated per-tile LAZ: colour + semantic label + cluster id + reference RGB in one file",
    )

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        return {
            "poses_corrected": s.workspace.poses / "poses_corrected.csv",
            "tw45_tiles": s.workspace.out / "colour" / "tiles",
        }

    def outputs(self, s: "Settings") -> list[Path]:
        return [s.workspace.consolidated / "summary.json"]

    def metrics(self, s: "Settings") -> dict[str, Any]:
        try:
            return json.loads((s.workspace.consolidated / "summary.json").read_text())
        except Exception as e:  # noqa: BLE001 - metrics must always be reportable
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", *, tiles: list[str] | None = None, workers: int = 8, force: bool = False,
            poses_source: str | None = "corrected", colour_tag: str = "colour", allow_mixed_poses: bool = False) -> None:
        inp = locate_inputs(s, poses_source=poses_source, colour_tag=colour_tag, allow_mixed_poses=allow_mixed_poses)
        run(tiles, inp, s, workers=workers, force=force)


STAGE = registry.add(Merge())


def _add_options(p) -> None:
    p.add_argument("--tiles", nargs="*", default=None, help="tile names, e.g. 037 001 (default: all)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p.add_argument("--poses-source", default="corrected", help="pose table: 'export' | 'corrected' | a CSV path")
    p.add_argument("--colour-tag", default="colour", help="the colorize stage's --tag, if not the default 'colour'")
    p.add_argument("--allow-mixed-poses", action="store_true", help="proceed even if an input's poses_hash != the current poses")


def _to_opts(args) -> dict:
    return {"tiles": args.tiles, "workers": args.workers, "force": args.force, "colour_tag": args.colour_tag,
            "poses_source": args.poses_source, "allow_mixed_poses": args.allow_mixed_poses}


def main(argv=None) -> int:
    return stage_main(STAGE, argv, add_options=_add_options, to_opts=_to_opts)


if __name__ == "__main__":
    raise SystemExit(main())
