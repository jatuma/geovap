#!/usr/bin/env python3
"""Cross-check the consolidated point-cloud product against its upstream inputs.

Runs the checks from plan Part C6 without a browser:

  (i)   cluster ids: the set of cluster_id >= 0 in the merged tiles == unique(global_labels.npy),
        and per-tile agreement with clusters/src/objects_t*.laz.
  (ii)  classification histogram in the merged tiles == labels/NNN.npy histogram (per tile).
  (iii) median dE00_med (n_views > 0) in the merged tiles == tw45 stats/*_meta.json medians (tol 0.05).
  (iv)  both octree metadata.json (cloud, objects): point counts, attribute lists, bbox shift vs
        pass_transforms.json.
  (v)   verify flags recorded in each tiles/NNN_meta.json marker (mapping.merge / mapping.las_out.verify).

Writes validation/checks.json (list of {check, tile, ok, detail, ...}) and exits 1 if any check that
could actually run failed or reported ok=False; missing inputs are recorded as skipped (ok=None) and
do not by themselves fail the run - the pipeline stage that produces them is expected to run first.

Usage:
    uv run python pointcloud-tools/validate/check_products.py [--out validation/checks.json]
        [--consolidated-dir DIR] [--potree-dir DIR] [--clusters-src DIR] [--seg-labels DIR]
        [--tw45-stats DIR] [--pass-transforms FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _try_import_mapping():
    """Import mapping lazily so --help works even if torch/laspy/etc. are unavailable."""
    from mapping import config  # noqa: F401

    return config


def _default_dirs(config):
    return dict(
        consolidated_dir=config.CONSOLIDATED_DIR,
        potree_dir=config.POTREE_OUTPUT_DIR / "consolidated",
        clusters_src=config.POTREE_OUTPUT_DIR / "clusters" / "src",
        seg_labels=None,  # resolved after poses/hash are known; see main()
        tw45_stats=None,  # resolved via glob("tw45*/stats") under OUT_DIR; see main()
        pass_transforms=config.OUT_DIR / "pass_reg" / "pass_transforms.json",
    )


def _load_meta(tiles_dir: Path, name: str) -> dict | None:
    p = tiles_dir / f"{name}_meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def check_cluster_ids(clusters_src: Path, consolidated_dir: Path, tile_metas: dict[str, dict]) -> list[dict]:
    """(i) Read cluster_id from every merged objects/ID3432_000NNN.laz (source order, like the input
    objects_t000NNN.laz) and check (a) per tile: identical to the input's cluster_id column, and
    (b) globally: the union of ids >= 0 equals unique(global_labels.npy) (the global id set produced
    by clusters/merge_tiles.py). Reads ~13 GB of LAZ, so it is the slowest check (minutes)."""
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
        out_p = consolidated_dir / "objects" / f"ID3432_000{name[-3:]}.laz"
        in_p = clusters_src / f"objects_t000{name[-3:]}.laz"
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
        g["detail"] = "no merged objects LAZ readable"
    checks.append(g)
    return checks


def check_classification_hist(seg_labels_dir: Path | None, tile_metas: dict[str, dict]) -> list[dict]:
    """mapping.merge.py writes 'counts' (a list indexed by COMMON class id, from
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
        labels_path = seg_labels_dir / f"{name[-3:]}.npy"
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


def check_de00_medians(tw45_stats_dir: Path | None, tile_metas: dict[str, dict], tol: float = 0.05) -> list[dict]:
    checks = []
    if tw45_stats_dir is None or not tw45_stats_dir.exists():
        return [{"check": "de00_median", "tile": None, "ok": None, "detail": f"missing tw45 stats dir {tw45_stats_dir}"}]
    from mapping import metrics

    for name, meta in sorted(tile_metas.items()):
        c = {"check": "de00_median", "tile": name, "ok": None, "detail": ""}
        num = name[-3:]
        npz_path = tw45_stats_dir / f"{num}_med.npz"
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


def check_octree_metadata(potree_dir: Path, pass_transforms: Path, expected_points: int, old_octree_meta: Path | None) -> list[dict]:
    """(iv) Both consolidated octrees (cloud, objects) hold every point (the objects LAZ carries
    cluster_id = -1 for unclustered points), so both must report expected_points. The cloud's bbox
    is compared with the OLD source-frame octree (clusters/rgb/metadata.json): the shift must be
    > 0 (registration was applied) and <= the registration bound from pass_transforms.json."""
    checks = []
    metas = {}
    for name in ("cloud", "objects", "vendor"):
        c = {"check": "octree_metadata", "tile": name, "ok": None, "detail": ""}
        meta_path = potree_dir / name / "metadata.json"
        if not meta_path.exists():
            c["detail"] = f"missing {meta_path}"
            checks.append(c)
            continue
        meta = json.loads(meta_path.read_text())
        metas[name] = meta
        points = meta.get("points")
        attrs = [a.get("name") for a in meta.get("attributes", [])]
        c["ok"] = points == expected_points
        c["detail"] = f"points={points} (expected {expected_points}), attributes={attrs}"
        checks.append(c)

    c = {"check": "octree_bbox_shift", "tile": "cloud vs old source-frame octree", "ok": None, "detail": ""}
    if "cloud" in metas and old_octree_meta is not None and old_octree_meta.exists():
        old = json.loads(old_octree_meta.read_text())
        cb, ob = metas["cloud"].get("boundingBox", {}), old.get("boundingBox", {})
        if "min" in cb and "min" in ob:
            shift = max(max(abs(a - b) for a, b in zip(cb["min"], ob["min"])), max(abs(a - b) for a, b in zip(cb["max"][:2], ob["max"][:2])))
            bound = _registration_bound_m(pass_transforms)
            if bound is None:
                c["detail"] = f"bbox shift {shift:.4f} m (no pass_transforms.json to bound it)"
            else:
                c["ok"] = 0.0 < shift <= bound
                c["detail"] = f"bbox shift {shift:.4f} m vs registration bound {bound:.4f} m (must be > 0: registered frame, and <= bound)"
        else:
            c["detail"] = "boundingBox missing in one of the metadata.json files"
    else:
        c["detail"] = f"missing {potree_dir / 'cloud' / 'metadata.json'} or {old_octree_meta}"
    checks.append(c)
    if "cloud" in metas and "objects" in metas:
        c2 = {"check": "octree_bbox_shift", "tile": "cloud vs objects", "ok": None, "detail": ""}
        cb, ob = metas["cloud"].get("boundingBox", {}), metas["objects"].get("boundingBox", {})
        if "min" in cb and "min" in ob:
            shift = max(abs(a - b) for a, b in zip(cb["min"], ob["min"]))
            c2["ok"] = shift <= 0.001
            c2["detail"] = f"bbox min shift {shift:.4f} m (same points, same frame: must be 0)"
        checks.append(c2)
    return checks


def _sub_verify_ok(sub: dict) -> bool:
    """mapping.las_out.verify() returns keys xyz_exact/class_exact/provenance (plus n/n_src, and
    shift_mm/n_moved when xyz_mode='registered'). A sub-verify passes when all three flags are
    true, and - when it recorded n_moved (registered mode) - also n_moved == 0."""
    ok = bool(sub.get("xyz_exact")) and bool(sub.get("class_exact")) and bool(sub.get("provenance"))
    if "n_moved" in sub:
        ok = ok and int(sub["n_moved"]) == 0
    return ok


def check_verify_flags(tile_metas: dict[str, dict]) -> list[dict]:
    """mapping.merge.py writes meta['verify'] = {'tile': v_tile, 'objects': v_obj}, the nested
    dicts returned by mapping.las_out.verify() for the tile and objects LAZ respectively - there
    is no top-level 'ok' or 'n_moved' key on meta['verify'] itself."""
    checks = []
    if not tile_metas:
        return [{"check": "verify_flags", "tile": None, "ok": None, "detail": "no tiles/*_meta.json markers found"}]
    for name, meta in sorted(tile_metas.items()):
        c = {"check": "verify_flags", "tile": name, "ok": None, "detail": ""}
        verify = meta.get("verify")
        if verify is None or "tile" not in verify or "objects" not in verify:
            c["detail"] = "no 'verify': {'tile':..,'objects':..} field in marker"
            checks.append(c)
            continue
        ok_tile = _sub_verify_ok(verify["tile"])
        ok_obj = _sub_verify_ok(verify["objects"])
        ok_vendor = _sub_verify_ok(verify["vendor"]) if "vendor" in verify else None
        c["ok"] = ok_tile and ok_obj and (ok_vendor is not False)
        c["detail"] = json.dumps({"tile_ok": ok_tile, "objects_ok": ok_obj, "vendor_ok": ok_vendor, "verify": verify})
        checks.append(c)
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output checks.json path (default: <consolidated-dir>/validation/checks.json)")
    ap.add_argument("--consolidated-dir", default=None, help="out/consolidated (default: mapping.config.CONSOLIDATED_DIR)")
    ap.add_argument("--potree-dir", default=None, help="Potree output .../consolidated (default: POTREE_OUTPUT_DIR/consolidated)")
    ap.add_argument("--clusters-src", default=None, help="clusters/src dir with global_labels.npy (default: POTREE_OUTPUT_DIR/clusters/src)")
    ap.add_argument("--seg-labels", default=None, help="seg_eomt labels dir with NNN.npy (default: autodetect under OUT_DIR)")
    ap.add_argument("--tw45-stats", default=None, help="tw45 run's stats/ dir (default: autodetect tw45*/stats under OUT_DIR)")
    ap.add_argument("--pass-transforms", default=None, help="pass_transforms.json (default: OUT_DIR/pass_reg/pass_transforms.json)")
    ap.add_argument("--tol", type=float, default=None, help="tolerance for dE00 median comparison (default: one histogram bin, geovap.domain.math.colour_metrics.DE_BIN)")
    ap.add_argument("--old-octree-meta", default=None, help="source-frame octree metadata.json to measure the registration shift against (default POTREE_OUTPUT_DIR/clusters/rgb/metadata.json)")
    args = ap.parse_args(argv)

    config = _try_import_mapping()
    d = _default_dirs(config)

    consolidated_dir = Path(args.consolidated_dir) if args.consolidated_dir else d["consolidated_dir"]
    potree_dir = Path(args.potree_dir) if args.potree_dir else d["potree_dir"]
    clusters_src = Path(args.clusters_src) if args.clusters_src else d["clusters_src"]

    def _pick(pattern: str, explicit: str | None) -> Path | None:
        if explicit:
            return Path(explicit)
        # the merged tiles' provenance names the exact input dirs used -- trust that first
        cands = [Path(p) for p in sorted(config.OUT_DIR.glob(pattern)) if Path(p).is_dir() and any(Path(p).iterdir())]
        return cands[-1] if cands else None  # hash-suffixed (corrected) dirs sort after the bare export dir

    seg_labels_dir = _pick("seg_eomt*/labels", args.seg_labels)
    tw45_stats_dir = _pick("tw45*/stats", args.tw45_stats)

    pass_transforms = Path(args.pass_transforms) if args.pass_transforms else d["pass_transforms"]

    tiles_dir = consolidated_dir / "tiles"
    tile_metas = {}
    if tiles_dir.exists():
        for p in sorted(tiles_dir.glob("*_meta.json")):
            tile_metas[p.stem[: -len("_meta")]] = json.loads(p.read_text())

    checks: list[dict] = []
    checks += check_cluster_ids(clusters_src, consolidated_dir, tile_metas)
    checks += check_classification_hist(seg_labels_dir, tile_metas)
    from mapping import metrics as _metrics

    tol = args.tol if args.tol is not None else float(_metrics.DE_BIN)
    old_meta = Path(args.old_octree_meta) if args.old_octree_meta else config.POTREE_OUTPUT_DIR / "clusters" / "rgb" / "metadata.json"
    checks += check_de00_medians(tw45_stats_dir, tile_metas, tol=tol)
    checks += check_octree_metadata(potree_dir, pass_transforms, config.EXPECTED_TOTAL_POINTS, old_meta)
    checks += check_verify_flags(tile_metas)

    n_fail = sum(1 for c in checks if c["ok"] is False)
    n_skip = sum(1 for c in checks if c["ok"] is None)
    n_ok = sum(1 for c in checks if c["ok"] is True)
    summary = {"n_ok": n_ok, "n_fail": n_fail, "n_skip": n_skip, "checks": checks}

    out_path = Path(args.out) if args.out else consolidated_dir / "validation" / "checks.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"wrote {out_path}: {n_ok} ok, {n_fail} failed, {n_skip} skipped (inputs not present yet)")
    for c in checks:
        if c["ok"] is False:
            print(f"  FAIL [{c['check']}/{c.get('tile')}] {c['detail']}")

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
