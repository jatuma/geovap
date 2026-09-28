#!/usr/bin/env python3
"""Cross-check the consolidated point-cloud product against its upstream inputs.

Runs the checks from plan Part C6 without a browser:

  (i)   cluster ids: the set of cluster_id >= 0 in the delivered tiles == unique(global_labels.npy),
        and per-tile agreement with clusters/<tile>.laz (the clustering stage's own per-tile output).
  (ii)  classification histogram in the delivered tiles == labels/NNN.npy histogram (per tile).
  (iii) median dE00_med (n_views > 0) in the delivered tiles == tw45 stats/*_meta.json medians (tol
        one histogram bin).
  (iv)  the consolidated octree's metadata.json: point count, attribute list, bbox shift vs the old
        (pre-registration) source-frame octree.
  (v)   verify flags recorded in each tiles/NNN_meta.json marker (the `merge` stage's own
        `geovap.runtime.las_check.verify_tile` result).
  (vi)  every dimension in `geovap.io.las_writer.CONSOLIDATED_DIMS` is present and non-degenerate in
        a delivered tile -- the acceptance test for the 2026-09-28 collapse of `tiles/`+`objects/`+
        `vendor/` into one LAZ per tile (see `merge.py`'s module docstring): a dimension that is
        present but constant (e.g. `cluster_id` all -1, `seg_conf` all 0) means its producing stage
        silently did not run, which the old three-LAZ-set layout could not hide this way.

Writes `<workspace>/out/consolidated/validation/checks.json` (list of {check, tile, ok, detail, ...})
and exits 1 if any check that could actually run failed or reported ok=False; missing inputs are
recorded as skipped (ok=None) and do not by themselves fail the run -- the pipeline stage that
produces them is expected to run first.

Usage:
    uv run python -m geovap.stages.verify.checks [--dataset drazkov]
        [--out FILE] [--potree-dir DIR] [--clusters-src DIR] [--seg-labels DIR]
        [--tw45-stats DIR] [--pass-transforms FILE] [--old-octree-meta FILE]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geovap.stages.base.cli import stage_main
from geovap.stages.base.spec import StageSpec, registry

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings


def _out_name(s: "Settings", name: str, kind: str = "", *, variant: str | None = None) -> str:
    """Product filename for a tile, from the dataset descriptor. `name[-3:]` used to slice three
    characters off a tile id that only happens to BE three characters on Dražkov."""
    from geovap.domain.model.tiles import TileId

    return s.tiles.out_name(TileId(name), kind, variant=variant)


def _load_meta(tiles_dir: Path, name: str) -> dict | None:
    p = tiles_dir / f"{name}_meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def check_cluster_ids(s: "Settings", clusters_src: Path, consolidated_tiles: Path, tile_metas: dict[str, dict]) -> list[dict]:
    """(i) Read cluster_id from every delivered tile (there is only one LAZ per tile now -- see the
    module docstring) and check (a) per tile: identical to the clustering stage's own per-tile
    output, and (b) globally: the union of ids >= 0 equals unique(global_labels.npy) (the global id
    set produced by `geovap.stages.objects.merge_tiles`). Reads ~13 GB of LAZ, so it is the slowest
    check (minutes)."""
    checks = []
    gl_path = clusters_src / "global_labels.npy"
    if not gl_path.exists():
        return [{"check": "cluster_ids", "tile": None, "ok": None, "detail": f"missing {gl_path}"}]
    if not tile_metas:
        return [{"check": "cluster_ids", "tile": None, "ok": None, "detail": "no tiles/*_meta.json markers found (merge stage not run yet)"}]
    import laspy
    import numpy as np

    expected = set(int(x) for x in np.unique(np.load(gl_path)) if x >= 0)
    union: set[int] = set()
    for name in sorted(tile_metas):
        c = {"check": "cluster_ids", "tile": name, "ok": None, "detail": ""}
        out_p = consolidated_tiles / _out_name(s, name)
        in_p = clusters_src / _out_name(s, name, variant="cluster")
        if not out_p.exists() or not in_p.exists():
            c["detail"] = f"missing {out_p if not out_p.exists() else in_p}"
            checks.append(c)
            continue
        cid_out = np.asarray(laspy.read(str(out_p)).cluster_id)
        cid_in = np.asarray(laspy.read(str(in_p)).cluster_id)
        same = cid_out.shape == cid_in.shape and bool(np.array_equal(cid_out, cid_in))
        ids = set(int(x) for x in np.unique(cid_out) if x >= 0)
        union |= ids
        c["ok"] = same
        c["detail"] = f"{len(ids)} clusters, per-point cluster_id {'identical to' if same else 'DIFFERS from'} {in_p.name}"
        checks.append(c)
    g = {"check": "cluster_ids_global", "tile": None, "ok": None, "detail": ""}
    if union:
        missing, extra = expected - union, union - expected
        g["ok"] = not missing and not extra
        g["detail"] = f"union of merged ids={len(union)} vs global_labels.npy unique={len(expected)}; missing={len(missing)}, extra={len(extra)}"
    else:
        g["detail"] = "no delivered tile LAZ readable"
    checks.append(g)
    return checks


def check_classification_hist(seg_labels_dir: Path | None, tile_metas: dict[str, dict]) -> list[dict]:
    """`stages.deliver.merge` writes 'counts' (a list indexed by COMMON class id, from
    np.bincount(classification, minlength=256)[: len(T.COMMON)]) and 'unlabelled' (count of
    T.IGNORE=255), not a 'classification_hist' dict. Compare those against the per-tile labels
    NNN.npy histogram directly, bucket by bucket."""
    checks = []
    if seg_labels_dir is None or not seg_labels_dir.exists():
        return [{"check": "classification_hist", "tile": None, "ok": None, "detail": f"missing seg labels dir {seg_labels_dir}"}]
    import numpy as np

    from geovap.domain.scheme import taxonomy as T

    for name, meta in sorted(tile_metas.items()):
        c = {"check": "classification_hist", "tile": name, "ok": None, "detail": ""}
        labels_path = seg_labels_dir / f"{name}.npy"
        counts = meta.get("counts")
        unlabelled = meta.get("unlabelled")
        if counts is None or not labels_path.exists():
            c["detail"] = f"missing {'meta counts' if counts is None else labels_path}"
            checks.append(c)
            continue
        labels = np.load(labels_path)
        ref_hist: dict[int, int] = {}
        for v, n in zip(*np.unique(labels, return_counts=True)):
            ref_hist[int(v)] = int(n)
        merged_hist = {i: int(n) for i, n in enumerate(counts) if int(n) > 0}
        if unlabelled:
            merged_hist[T.IGNORE] = int(unlabelled)
        c["ok"] = merged_hist == ref_hist
        c["detail"] = "match" if c["ok"] else f"merged {merged_hist} vs labels {ref_hist}"
        checks.append(c)
    return checks


def check_de00_medians(tw45_stats_dir: Path | None, tile_metas: dict[str, dict], tol: float) -> list[dict]:
    checks = []
    if tw45_stats_dir is None or not tw45_stats_dir.exists():
        return [{"check": "de00_median", "tile": None, "ok": None, "detail": f"missing tw45 stats dir {tw45_stats_dir}"}]
    from geovap.domain.math import colour_metrics as metrics

    for name, meta in sorted(tile_metas.items()):
        c = {"check": "de00_median", "tile": name, "ok": None, "detail": ""}
        npz_path = tw45_stats_dir / f"{name}_med.npz"
        merged_med = meta.get("dE00_med_median")
        if merged_med is None or not npz_path.exists():
            c["detail"] = f"missing {'meta de00_med_median' if merged_med is None else npz_path}"
            checks.append(c)
            continue
        h = metrics.StrataHist.load(npz_path)
        ref_stats = metrics.hist_stats(h.total_hist())
        ref_med = ref_stats.get("median")
        if ref_med is None:
            c["detail"] = "empty tw45 histogram (n=0)"
            checks.append(c)
            continue
        c["ok"] = abs(merged_med - ref_med) <= tol
        c["detail"] = f"merged {merged_med:.3f} vs tw45 {ref_med:.3f} (tol {tol})"
        checks.append(c)
    return checks


def _registration_bound_m(pass_transforms: Path) -> float | None:
    """Upper bound on how far registration can move any point: max |t| + max |yaw| x 1 km
    (rotation about the pass centroid; the site is < 1 km across). `pass_transforms.json` layout:
    {"passes": {pid: {"t": [dE, dN, dH], "yaw_deg": ..., "centre": [cx, cy]}}, "summary": {...}}."""
    if not pass_transforms.exists():
        return None
    import math

    pt = json.loads(pass_transforms.read_text())
    passes = pt.get("passes", {})
    if not passes:
        return None
    max_t = max(max(abs(float(v)) for v in p.get("t", [0, 0, 0])) for p in passes.values())
    max_yaw = max(abs(float(p.get("yaw_deg", 0.0))) for p in passes.values())
    return max_t + math.radians(max_yaw) * 1000.0 + 0.01


def check_octree_metadata(potree_dir: Path, pass_transforms: Path, expected_points: int | None, old_octree_meta: Path | None) -> list[dict]:
    """(iv) The consolidated octree (there is only one now -- `cloud`, no `objects`/`vendor`
    siblings; see `merge.py`'s module docstring) holds every point. Its bbox is compared with the
    OLD source-frame octree (pre-registration, `clusters/rgb/metadata.json`): the shift must be > 0
    (registration was applied) and <= the registration bound from pass_transforms.json."""
    checks = []
    meta_path = potree_dir / "cloud" / "metadata.json"
    c = {"check": "octree_metadata", "tile": "cloud", "ok": None, "detail": ""}
    meta = None
    if not meta_path.exists():
        c["detail"] = f"missing {meta_path}"
        checks.append(c)
    else:
        meta = json.loads(meta_path.read_text())
        points = meta.get("points")
        attrs = [a.get("name") for a in meta.get("attributes", [])]
        c["ok"] = expected_points is None or points == expected_points
        c["detail"] = f"points={points} (expected {expected_points}), attributes={attrs}"
        checks.append(c)

    c2 = {"check": "octree_bbox_shift", "tile": "cloud vs old source-frame octree", "ok": None, "detail": ""}
    if meta is not None and old_octree_meta is not None and old_octree_meta.exists():
        old = json.loads(old_octree_meta.read_text())
        cb, ob = meta.get("boundingBox", {}), old.get("boundingBox", {})
        if "min" in cb and "min" in ob:
            shift = max(max(abs(a - b) for a, b in zip(cb["min"], ob["min"])), max(abs(a - b) for a, b in zip(cb["max"][:2], ob["max"][:2])))
            bound = _registration_bound_m(pass_transforms)
            if bound is None:
                c2["detail"] = f"bbox shift {shift:.4f} m (no pass_transforms.json to bound it)"
            else:
                c2["ok"] = 0.0 < shift <= bound
                c2["detail"] = f"bbox shift {shift:.4f} m vs registration bound {bound:.4f} m (must be > 0: registered frame, and <= bound)"
        else:
            c2["detail"] = "boundingBox missing in one of the metadata.json files"
    else:
        c2["detail"] = f"missing {meta_path} or {old_octree_meta}"
    checks.append(c2)
    return checks


def _sub_verify_ok(sub: dict) -> bool:
    """`geovap.runtime.las_check.verify_tile` returns keys xyz_exact/class_exact/provenance (plus
    n/n_src, and shift_mm/n_moved when xyz_mode='registered'). A sub-verify passes when all three
    flags are true, and -- when it recorded n_moved (registered mode) -- also n_moved == 0."""
    ok = bool(sub.get("xyz_exact")) and bool(sub.get("class_exact")) and bool(sub.get("provenance"))
    if "n_moved" in sub:
        ok = ok and int(sub["n_moved"]) == 0
    return ok


def check_verify_flags(tile_metas: dict[str, dict]) -> list[dict]:
    """`stages.deliver.merge` writes meta['verify'] = {'tile': v_tile}, the dict returned by
    `geovap.runtime.las_check.verify_tile` for the one delivered LAZ -- there is no 'objects' or
    'vendor' sibling any more (see the module docstring), and no top-level 'ok'/'n_moved' key on
    meta['verify'] itself."""
    checks = []
    if not tile_metas:
        return [{"check": "verify_flags", "tile": None, "ok": None, "detail": "no tiles/*_meta.json markers found"}]
    for name, meta in sorted(tile_metas.items()):
        c = {"check": "verify_flags", "tile": name, "ok": None, "detail": ""}
        verify = meta.get("verify")
        if verify is None or "tile" not in verify:
            c["detail"] = "no 'verify': {'tile':..} field in marker"
            checks.append(c)
            continue
        ok_tile = _sub_verify_ok(verify["tile"])
        c["ok"] = ok_tile
        c["detail"] = json.dumps({"tile_ok": ok_tile, "verify": verify})
        checks.append(c)
    return checks


def check_consolidated_dims(s: "Settings", consolidated_tiles: Path, tile_metas: dict[str, dict]) -> list[dict]:
    """(vi) Acceptance test for the objects/vendor collapse (2026-09-28, see the module docstring):
    reads the first delivered tile (by name) and checks that every dimension declared in
    `geovap.io.las_writer.CONSOLIDATED_DIMS` both exists and is non-degenerate (more than one
    distinct value across the tile) -- a dimension present but constant (all -1, all 0, ...) means
    its producing stage silently did not run, which the old three-parallel-LAZ layout could not hide
    this way (each product had its own file, so a missing one was simply a missing file)."""
    from geovap.io.las_writer import CONSOLIDATED_DIMS

    if not tile_metas:
        return [{"check": "consolidated_dims", "tile": None, "ok": None, "detail": "no tiles/*_meta.json markers found"}]
    name = sorted(tile_metas)[0]
    path = consolidated_tiles / _out_name(s, name)
    if not path.exists():
        return [{"check": "consolidated_dims", "tile": name, "ok": None, "detail": f"missing {path}"}]
    import laspy
    import numpy as np

    las = laspy.read(str(path))
    checks = []
    for dim_name, _dt, _desc in CONSOLIDATED_DIMS:
        c = {"check": "consolidated_dims", "tile": f"{name}/{dim_name}", "ok": None, "detail": ""}
        try:
            arr = np.asarray(getattr(las, dim_name))
        except AttributeError:
            c["ok"] = False
            c["detail"] = "dimension missing from delivered tile"
            checks.append(c)
            continue
        n_unique = int(len(np.unique(arr)))
        c["ok"] = n_unique > 1
        c["detail"] = f"{n_unique} distinct value(s)" + ("" if c["ok"] else " (degenerate: producing stage may not have run)")
        checks.append(c)
    return checks


def _pick(s: "Settings", pattern: str, explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    # the delivered tiles' provenance names the exact input dirs used -- trust that first
    cands = [p for p in sorted(s.workspace.out.glob(pattern)) if p.is_dir() and any(p.iterdir())]
    return cands[-1] if cands else None  # hash-suffixed (corrected) dirs sort after the bare export dir


def run_checks(
    s: "Settings",
    *,
    out: str | None = None,
    potree_dir: str | None = None,
    clusters_src: str | None = None,
    seg_labels: str | None = None,
    tw45_stats: str | None = None,
    pass_transforms: str | None = None,
    old_octree_meta: str | None = None,
    tol: float | None = None,
) -> dict:
    consolidated_tiles = s.workspace.consolidated_tiles
    potree = Path(potree_dir) if potree_dir else s.paths.publish / "consolidated"
    clusters = Path(clusters_src) if clusters_src else s.workspace.out / "clusters"

    seg_labels_dir = _pick(s, "seg_eomt*/labels", seg_labels)
    tw45_stats_dir = _pick(s, "tw45*/stats", tw45_stats)

    transforms = Path(pass_transforms) if pass_transforms else s.workspace.out / "pass_reg" / "pass_transforms.json"
    old_meta = Path(old_octree_meta) if old_octree_meta else s.paths.publish / "clusters" / "rgb" / "metadata.json"

    tile_metas = {}
    if consolidated_tiles.exists():
        for p in sorted(consolidated_tiles.glob("*_meta.json")):
            tile_metas[p.stem[: -len("_meta")]] = json.loads(p.read_text())

    from geovap.runtime.manifest import RunManifest

    manifest = RunManifest.load(s.workspace)
    expected_total_points = manifest.total_points if manifest is not None else None

    checks: list[dict] = []
    checks += check_cluster_ids(s, clusters, consolidated_tiles, tile_metas)
    checks += check_classification_hist(seg_labels_dir, tile_metas)
    from geovap.domain.math import colour_metrics as _metrics

    tol = tol if tol is not None else float(_metrics.DE_BIN)
    checks += check_de00_medians(tw45_stats_dir, tile_metas, tol=tol)
    checks += check_octree_metadata(potree, transforms, expected_total_points, old_meta)
    checks += check_verify_flags(tile_metas)
    checks += check_consolidated_dims(s, consolidated_tiles, tile_metas)

    n_fail = sum(1 for c in checks if c["ok"] is False)
    n_skip = sum(1 for c in checks if c["ok"] is None)
    n_ok = sum(1 for c in checks if c["ok"] is True)
    summary = {"n_ok": n_ok, "n_fail": n_fail, "n_skip": n_skip, "checks": checks}

    out_path = Path(out) if out else s.workspace.consolidated / "validation" / "checks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"wrote {out_path}: {n_ok} ok, {n_fail} failed, {n_skip} skipped (inputs not present yet)")
    for c in checks:
        if c["ok"] is False:
            print(f"  FAIL [{c['check']}/{c.get('tile')}] {c['detail']}")

    return summary


# ================================================================================================ stage
class Checks:
    spec = StageSpec(
        name="checks", after=("merge",), est_min=5,
        summary="cross-check the delivered consolidated tiles against their upstream inputs",
    )
    cli_args: tuple[str, ...] = ()

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        return {"consolidated_summary": s.workspace.consolidated / "summary.json"}

    def outputs(self, s: "Settings") -> list[Path]:
        return [s.workspace.consolidated / "validation" / "checks.json"]

    def metrics(self, s: "Settings") -> dict[str, Any]:
        try:
            return json.loads((s.workspace.consolidated / "validation" / "checks.json").read_text())
        except Exception as e:  # noqa: BLE001 - metrics must always be reportable
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", **opts: Any) -> None:
        summary = run_checks(s, **opts)
        if summary["n_fail"] > 0:
            raise SystemExit(1)


STAGE = registry.add(Checks())


def _add_options(p) -> None:
    p.add_argument("--out", default=None, help="output checks.json path (default: <workspace>/out/consolidated/validation/checks.json)")
    p.add_argument("--potree-dir", default=None, help="Potree output .../consolidated (default: <publish>/consolidated)")
    p.add_argument("--clusters-src", default=None, help="clusters dir with global_labels.npy (default: <workspace>/out/clusters)")
    p.add_argument("--seg-labels", default=None, help="seg_eomt labels dir with NNN.npy (default: autodetect under <workspace>/out)")
    p.add_argument("--tw45-stats", default=None, help="tw45 run's stats/ dir (default: autodetect tw45*/stats under <workspace>/out)")
    p.add_argument("--pass-transforms", default=None, help="pass_transforms.json (default: <workspace>/out/pass_reg/pass_transforms.json)")
    p.add_argument("--old-octree-meta", default=None, help="source-frame octree metadata.json to measure the registration shift against (default <publish>/clusters/rgb/metadata.json)")
    p.add_argument("--tol", type=float, default=None, help="tolerance for dE00 median comparison (default: one histogram bin, geovap.domain.math.colour_metrics.DE_BIN)")


def _to_opts(args) -> dict:
    return {
        "out": args.out, "potree_dir": args.potree_dir, "clusters_src": args.clusters_src,
        "seg_labels": args.seg_labels, "tw45_stats": args.tw45_stats,
        "pass_transforms": args.pass_transforms, "old_octree_meta": args.old_octree_meta, "tol": args.tol,
    }


def main(argv=None) -> int:
    return stage_main(STAGE, argv, add_options=_add_options, to_opts=_to_opts)


if __name__ == "__main__":
    raise SystemExit(main())
