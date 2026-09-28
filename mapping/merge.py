"""C2: consolidated 3D product. Merges, per tile, the store's own reference columns with three
per-tile products (tw45 colorization, projected semantic labels, object clusters) that all share the
store's row count and per-row point identity (see `las_out` and `geovap.runtime.store.TileData.orig_index`)
into two LAS 1.4 PF7 files:

    out_dir/tiles/<tile out_name>.laz   rgb = tw45 fused colour (TerraScan reference where n_views==0),
                                         classification = common15 label (255 = unlabelled),
                                         extras = las_out.CONS_EXTRA_DIMS, xyz = registered (S5)
    out_dir/objects/<tile out_name>.laz rgb = cluster palette, extras = las_out.OBJ_EXTRA_DIMS, xyz = registered

Every input is checked against the store's own points (`check_same_points`) before use, and against
the current pose table's hash where it carries one (tw45's `geovap_map` VLR); a mismatch aborts unless
`--allow-mixed-poses` (the seg-labels/clusters products predate S5b and may only exist under an older,
or the export, pose table).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import las_out
from geovap.runtime.store import SCALE, CloudStore, TileData
from .config import EXPECTED_TOTAL_POINTS, OUT_DIR, POTREE_OUTPUT_DIR, CONSOLIDATED_DIR, STORE_DIR, source_dir
from geovap.runtime.pose_tables import load as load_poses
from geovap.domain.scheme import taxonomy as T


def _out_name(name: str, kind: str = "", *, variant: str | None = None) -> str:
    """Product filename for a tile, from the dataset descriptor. Replaces the literal `ID3432_000`
    prefix, which was Dražkov's and was duplicated across ten modules -- on another dataset it wrote
    every product under the wrong name, with no error anywhere."""
    from geovap.domain.model.tiles import TileId
    from geovap.runtime import settings

    return settings.get().tiles.out_name(TileId(name), kind, variant=variant)

CLUSTERS_SRC_DEFAULT = POTREE_OUTPUT_DIR / "clusters" / "src"


# --------------------------------------------------------------------------------------- locating inputs
def _unique_glob(pattern: str, what: str) -> Path:
    hits = sorted(Path(p) for p in __import__("glob").glob(str(pattern)))
    if len(hits) == 0:
        raise FileNotFoundError(f"no {what} found matching {pattern}")
    if len(hits) > 1:
        raise FileNotFoundError(f"ambiguous {what}: {len(hits)} matches for {pattern}: {hits}")
    return hits[0]


def _non_empty(d: Path) -> bool:
    return d.is_dir() and any(d.iterdir())


def _default_dir(base: Path, tail: str, what: str, poses=None) -> Path:
    """Input directory for a per-tile product. Preference: the pose-aware sibling
    `config.source_dir(base, poses)/tail` (e.g. `out/seg_eomt_<hash6>/labels`), then `base/tail`, then the
    unique `<base>_*/tail` glob -- but only NON-EMPTY directories count: `out/seg_eomt/labels` (the
    export-era dir) may exist empty and must not shadow the corrected run's output."""
    cands = []
    if poses is not None:
        try:
            cands.append(source_dir(base, poses) / tail)
        except Exception:  # noqa: BLE001
            pass
    cands.append(base / tail)
    for d in cands:
        if _non_empty(d):
            return d
    hits = [Path(h) for h in __import__("glob").glob(f"{base}_*/{tail}") if _non_empty(Path(h))]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise FileNotFoundError(f"ambiguous {what}: {len(hits)} non-empty matches for {base}_*/{tail}: {hits}")
    raise FileNotFoundError(f"no non-empty {what} found (tried {cands} and {base}_*/{tail})")


@dataclass
class MergeInputs:
    """Resolved inputs for one `run`/`merge_tile` call. `poses` is the loaded `Poses` table the
    consolidated product is registered against (see `geovap.runtime.store.open_store`); `poses_hash` is a
    convenience for provenance."""

    poses: object
    tw45_tiles: Path | None
    seg_labels: Path | None
    clusters_src: Path
    out_dir: Path
    rgb_fallback: bool = True
    allow_mixed_poses: bool = False

    @property
    def poses_hash(self) -> str:
        return self.poses.hash()


def locate_inputs(poses_source: str | None = None, out_dir: Path = CONSOLIDATED_DIR, tw45_tiles: Path | str | None = None,
                   seg_labels: Path | str | None = None, clusters_src: Path | str | None = None, rgb_fallback: bool = True,
                   allow_mixed_poses: bool = False) -> MergeInputs:
    """Resolves the default input directories documented in the module docstring. Missing tw45/seg
    inputs are tolerated here (the tile falls back to reference RGB / all-unlabelled); pass explicit
    paths to override any of them."""
    poses = load_poses(poses_source)
    try:
        tw45 = Path(tw45_tiles) if tw45_tiles is not None else _default_dir(OUT_DIR / "tw45", "tiles", "tw45 tiles dir", poses)
    except FileNotFoundError:
        tw45 = None
    try:
        seg = Path(seg_labels) if seg_labels is not None else _default_dir(OUT_DIR / "seg_eomt", "labels", "seg_eomt labels dir", poses)
    except FileNotFoundError:
        seg = None
    clusters = Path(clusters_src) if clusters_src is not None else CLUSTERS_SRC_DEFAULT
    return MergeInputs(poses=poses, tw45_tiles=tw45, seg_labels=seg, clusters_src=clusters, out_dir=Path(out_dir),
                        rgb_fallback=rgb_fallback, allow_mixed_poses=allow_mixed_poses)


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
        if v.user_id == las_out.PROVENANCE_USER_ID:
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
def merge_tile(name: str, store: CloudStore, inp: MergeInputs, log=print) -> dict:
    import laspy

    t0 = time.time()
    td = store.tile(name)
    n = len(td)
    inv = np.asarray(td.orig_index)
    warnings: list[str] = []
    inputs_used: dict[str, str | None] = {}

    xyz = registered_xyz_int(td)
    src_class = np.asarray(td.classification).copy()

    # ------------------------------------------------------------------------------------- tw45
    rgb = np.asarray(td.rgb).copy()
    ref_rgb = np.asarray(td.rgb).copy()
    n_views = np.zeros(n, np.uint8)
    col_conf = np.zeros(n, np.uint8)
    de00_med = np.full(n, np.nan, np.float32)
    if inp.tw45_tiles is not None:
        p = Path(inp.tw45_tiles) / _out_name(name, "_colored")
        if p.exists():
            las = laspy.read(str(p))
            check_same_points(np.stack([np.asarray(las.X), np.asarray(las.Y), np.asarray(las.Z)], 1)[inv], np.asarray(td.xyz), f"tw45 {p.name}")
            _check_poses(_read_provenance(las), p, inp, warnings)
            fused = np.stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)], 1) >> 8
            rgb = _reorder_to_store(fused, inv).astype(np.uint8)
            ref_rgb = _reorder_to_store(np.stack([np.asarray(las.ref_r), np.asarray(las.ref_g), np.asarray(las.ref_b)], 1), inv).astype(np.uint8)
            n_views = _reorder_to_store(np.asarray(las.n_views), inv).astype(np.uint8)
            col_conf = _reorder_to_store(np.asarray(las.col_conf), inv).astype(np.uint8)
            de00_med = _reorder_to_store(np.asarray(las.dE00_med), inv).astype(np.float32)
            inputs_used["tw45"] = str(p)
            if inp.rgb_fallback:
                no_col = n_views == 0
                rgb[no_col] = np.asarray(td.rgb)[no_col]
        else:
            warnings.append(f"tw45 tile missing: {p}")

    # ------------------------------------------------------------------------------------- seg labels
    classification = np.full(n, T.IGNORE, np.uint8)
    seg_conf = np.zeros(n, np.uint8)
    seg_n_views = np.zeros(n, np.uint8)
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
            inputs_used["seg_labels"] = str(lp)
            mp = Path(inp.seg_labels) / f"{name}_meta.json"
            if mp.exists():
                meta = json.loads(mp.read_text())
                if meta.get("n") not in (None, n):
                    warnings.append(f"seg meta {mp}: n={meta.get('n')} != store n={n}")
                _check_poses(meta, mp, inp, warnings)
        else:
            raise FileNotFoundError(f"seg labels missing for tile {name}: {lp} (labels dir {inp.seg_labels} resolved but has no file for this tile)")

    # ------------------------------------------------------------------------------------- clusters
    cluster_id = np.full(n, -1, np.int32)
    obj_class = np.zeros(n, np.uint8)
    hag = np.zeros(n, np.float32)
    cluster_rgb = np.zeros((n, 3), np.uint16)
    cp = Path(inp.clusters_src) / f"objects_t000{name}.laz"
    if cp.exists():
        las = laspy.read(str(cp))
        check_same_points(np.stack([np.asarray(las.X), np.asarray(las.Y), np.asarray(las.Z)], 1)[inv], np.asarray(td.xyz), f"clusters {cp.name}")
        cluster_id = _reorder_to_store(np.asarray(las.cluster_id), inv).astype(np.int32)
        obj_class = _reorder_to_store(np.asarray(las.obj_class), inv).astype(np.uint8)
        hag = _reorder_to_store(np.asarray(las.hag), inv).astype(np.float32)
        cluster_rgb = _reorder_to_store(np.stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)], 1), inv).astype(np.uint16)
        inputs_used["clusters"] = str(cp)
    else:
        warnings.append(f"clusters tile missing: {cp}")

    provenance = {
        "product": "consolidated", "frame": "registered", "poses_source": inp.poses.source, "poses_hash": inp.poses_hash,
        "registration": str(inp.poses.registration) if getattr(inp.poses, "registration", None) else None,
        "inputs": inputs_used, "warnings": warnings, "git": _git_rev(),
    }

    out_dir = Path(inp.out_dir)
    tiles_dir, objects_dir, meta_dir = out_dir / "tiles", out_dir / "objects", out_dir / "tiles"
    for d in (tiles_dir, objects_dir, out_dir / "vendor"):
        d.mkdir(parents=True, exist_ok=True)

    cons_extras = {
        "src_class": src_class, "seg_conf": seg_conf, "seg_n_views": seg_n_views, "cluster_id": cluster_id,
        "obj_class": obj_class, "hag": hag, "ref_r": ref_rgb[:, 0], "ref_g": ref_rgb[:, 1], "ref_b": ref_rgb[:, 2],
        "dE00_med": de00_med, "n_views": n_views, "col_conf": col_conf,
    }
    tile_out = tiles_dir / _out_name(name)
    tile_tmp = tile_out.with_suffix(".laz.tmp")
    las_out.write_tile(td, tile_tmp, rgb, cons_extras, provenance, extra_dims=las_out.CONS_EXTRA_DIMS,
                        classification=classification, description="consolidated provenance", xyz=xyz)
    os.replace(tile_tmp, tile_out)

    obj_extras = {"cluster_id": cluster_id, "obj_class": obj_class}
    obj_out = objects_dir / _out_name(name)
    obj_tmp = obj_out.with_suffix(".laz.tmp")
    las_out.write_tile(td, obj_tmp, cluster_rgb, obj_extras, provenance, extra_dims=las_out.OBJ_EXTRA_DIMS,
                        classification=None, description="consolidated objects provenance", xyz=xyz)
    os.replace(obj_tmp, obj_out)

    vendor_out = write_vendor_tile(td, out_dir, provenance, xyz)

    v_tile = las_out.verify(tile_out, td, expect_class_exact=False, xyz_mode="registered")
    v_obj = las_out.verify(obj_out, td, expect_class_exact=True, xyz_mode="registered")
    v_vendor = las_out.verify(vendor_out, td, expect_class_exact=True, xyz_mode="registered")

    counts = np.bincount(classification, minlength=256)
    de_valid = de00_med[n_views > 0]
    meta = {
        "n": n, "seconds": time.time() - t0, "counts": counts[: len(T.COMMON)].tolist(), "unlabelled": int(counts[T.IGNORE]),
        "n_clustered": int((cluster_id >= 0).sum()), "n_clusters": int(len(np.unique(cluster_id[cluster_id >= 0]))),
        "dE00_med_median": float(np.median(de_valid)) if len(de_valid) else None,
        "provenance": provenance, "verify": {"tile": v_tile, "objects": v_obj, "vendor": v_vendor}, "warnings": warnings,
    }
    (meta_dir / f"{name}_meta.json").write_text(json.dumps(meta))  # last: resume marker
    log(f"tile {name}: {n} pts, {res_summary(meta)}, {meta['seconds']:.0f}s")
    return meta


def write_vendor_tile(td: TileData, out_dir: Path, provenance: dict, xyz: np.ndarray) -> Path:
    """Third product: the vendor's (TerraScan) RGB from the store, registered xyz, no extra dims -- lets the
    viewer show the original colouring on the same registered points (Potree cannot render three scalar
    extra dims `ref_r/g/b` as a colour, and the old `clusters/rgb` octree is in the unregistered frame)."""
    vendor_dir = Path(out_dir) / "vendor"
    vendor_dir.mkdir(parents=True, exist_ok=True)
    out = vendor_dir / _out_name(td.info.name)
    tmp = out.with_suffix(".laz.tmp")
    prov = {**provenance, "product": "vendor_rgb", "rgb": "TerraScan reference (store rgb)"}
    las_out.write_tile(td, tmp, np.asarray(td.rgb).astype(np.uint8), {}, prov, extra_dims=[], classification=None,
                       description="vendor rgb, registered", xyz=xyz)
    os.replace(tmp, out)
    return out


def _run_vendor_tile(name: str) -> dict:
    """Vendor product only (for a consolidated run made before this product existed)."""
    store, inp = _G["store"], _G["inp"]
    td = store.tile(name)
    xyz = registered_xyz_int(td)
    prov = {"product": "vendor_rgb", "frame": "registered", "poses_source": inp.poses.source, "poses_hash": inp.poses_hash,
            "registration": str(inp.poses.registration) if getattr(inp.poses, "registration", None) else None, "git": _git_rev()}
    out = write_vendor_tile(td, inp.out_dir, prov, xyz)
    v = las_out.verify(out, td, expect_class_exact=True, xyz_mode="registered")
    mp = Path(inp.out_dir) / "tiles" / f"{name}_meta.json"
    if mp.exists():  # fold into the existing marker so check_products sees it
        meta = json.loads(mp.read_text())
        meta.setdefault("verify", {})["vendor"] = v
        mp.write_text(json.dumps(meta))
    td.release() if hasattr(td, "release") else None
    return {"tile": name, "verify": v}


def run_vendor(tiles: list[str] | None, inp: MergeInputs, workers: int = 6, force: bool = False, root: Path = STORE_DIR) -> list[dict]:
    from multiprocessing import Pool

    store = CloudStore(root, registration=inp.poses.registration if inp.poses is not None else None)
    names = [t.name for t in store.tiles] if tiles is None else list(tiles)
    names = sorted(names, key=lambda nm: -store.by_name[nm].n)
    if not force:
        names = [nm for nm in names if not (Path(inp.out_dir) / "vendor" / _out_name(nm)).exists()]
    if not names:
        return []
    with Pool(workers, initializer=_init, initargs=(inp, root)) as pool:
        return list(pool.map(_run_vendor_tile, names))


def res_summary(meta: dict) -> str:
    return f"unlabelled={meta['unlabelled']}, clustered={meta['n_clustered']}, dE00_med={meta['dE00_med_median']}"


def _git_rev() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1]).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------------------- driver
_G: dict = {}


def _init(inp: MergeInputs, root: Path):
    _G["store"] = CloudStore(root, registration=inp.poses.registration if inp.poses is not None else None)
    _G["inp"] = inp


def _run_tile(name: str) -> dict:
    return merge_tile(name, _G["store"], _G["inp"])


def run(tiles: list[str] | None, inp: MergeInputs, workers: int = 6, force: bool = False, root: Path = STORE_DIR) -> dict:
    """Merges `tiles` (default: all tiles in the store), largest-first, resuming from
    `out_dir/tiles/NNN_meta.json` unless `force`. Writes `out_dir/summary.json`."""
    from multiprocessing import Pool

    store = CloudStore(root, registration=inp.poses.registration if inp.poses is not None else None)
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
        if workers <= 1:
            _init(inp, root)
            for nm in todo:
                results[nm] = _run_tile(nm)
        else:
            with Pool(workers, initializer=_init, initargs=(inp, root)) as pool:
                for nm, meta in zip(todo, pool.map(_run_tile, todo)):
                    results[nm] = meta

    total_n = sum(m["n"] for m in results.values())
    summary = {
        "n_tiles": len(results), "total_points": total_n, "expected_total_points": EXPECTED_TOTAL_POINTS,
        "matches_expected": total_n == EXPECTED_TOTAL_POINTS and len(results) == len(store.tiles),
        "poses_source": inp.poses.source, "poses_hash": inp.poses_hash, "tiles": sorted(results),
    }
    (Path(inp.out_dir) / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
