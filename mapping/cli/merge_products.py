"""uv run python -m mapping.cli.merge_products run|verify|classes [options]

run       merge every input product into out_dir/{tiles,objects,vendor}/<tile out_name>.laz + summary.json
vendor    (re)write only out_dir/vendor/ (TerraScan RGB on registered points) for an existing consolidated run
verify    re-check every merged tile's registered XYZ / provenance against the store
classes   write out_dir/classes.json (Potree `classes.json`, common15 palette) via
          mapping.seg.project_report.write_potree_classes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import CONSOLIDATED_DIR
from ..merge import MergeInputs, locate_inputs, run as run_merge, run_vendor


def _add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--tiles", nargs="*", default=None, help="tile names, e.g. 037 001 (default: all)")
    ap.add_argument("--poses", default=None, help="pose table: 'export' | 'corrected' | None -> env GEOVAP_POSES / export")
    ap.add_argument("--out", default=str(CONSOLIDATED_DIR))
    ap.add_argument("--tw45-dir", default=None, help="tw45 tiles dir (default: OUT_DIR/tw45/tiles or unique OUT_DIR/tw45_*/tiles)")
    ap.add_argument("--seg-labels", default=None, help="seg_eomt labels dir (default: OUT_DIR/seg_eomt/labels or unique OUT_DIR/seg_eomt_*/labels)")
    ap.add_argument("--clusters-src", default=None, help="clusters src dir (default: POTREE_OUTPUT_DIR/clusters/src)")
    ap.add_argument("--no-rgb-fallback", action="store_true", help="don't fall back to reference RGB where n_views==0")
    ap.add_argument("--allow-mixed-poses", action="store_true", help="proceed even if an input's poses_hash != the current poses")


def _inputs_from_args(a) -> MergeInputs:
    return locate_inputs(poses_source=a.poses, out_dir=Path(a.out), tw45_tiles=a.tw45_dir, seg_labels=a.seg_labels,
                          clusters_src=a.clusters_src, rgb_fallback=not a.no_rgb_fallback, allow_mixed_poses=a.allow_mixed_poses)


def cmd_run(a) -> int:
    inp = _inputs_from_args(a)
    summary = run_merge(a.tiles, inp, workers=a.workers, force=a.force)
    print(json.dumps(summary, indent=1))
    return 0 if summary["matches_expected"] or a.tiles else 1


def cmd_vendor(a) -> int:
    inp = _inputs_from_args(a)
    res = run_vendor(a.tiles, inp, workers=a.workers, force=a.force)
    bad = [r["tile"] for r in res if not r["verify"].get("xyz_exact")]
    print(f"vendor tiles written: {len(res)}, xyz mismatches: {bad}")
    return 0 if not bad else 1


def cmd_verify(a) -> int:
    out = Path(a.out)
    meta_dir = out / "tiles"
    metas = sorted(meta_dir.glob("*_meta.json"))
    if not metas:
        print(f"no merged tiles found under {meta_dir}", file=sys.stderr)
        return 1
    poses = None
    ok = True
    for mp in metas:
        name = mp.name.split("_meta.json")[0]
        meta = json.loads(mp.read_text())
        v_tile = meta.get("verify", {}).get("tile", {})
        v_obj = meta.get("verify", {}).get("objects", {})
        this_ok = v_tile.get("xyz_exact") and v_obj.get("xyz_exact")
        ok &= bool(this_ok)
        print(f"tile {name}: tile.shift_mm={v_tile.get('shift_mm')} objects.shift_mm={v_obj.get('shift_mm')} ok={this_ok}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


def cmd_classes(a) -> int:
    from ..seg.project_report import write_potree_classes

    p = write_potree_classes(Path(a.out))
    print(f"wrote {p}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run")
    _add_common(ap_run)
    ap_run.add_argument("--workers", type=int, default=6)
    ap_run.add_argument("--force", action="store_true")
    ap_run.set_defaults(func=cmd_run)

    ap_vendor = sub.add_parser("vendor")
    _add_common(ap_vendor)
    ap_vendor.add_argument("--workers", type=int, default=6)
    ap_vendor.add_argument("--force", action="store_true")
    ap_vendor.set_defaults(func=cmd_vendor)

    ap_verify = sub.add_parser("verify")
    ap_verify.add_argument("--out", default=str(CONSOLIDATED_DIR))
    ap_verify.set_defaults(func=cmd_verify)

    ap_classes = sub.add_parser("classes")
    ap_classes.add_argument("--out", default=str(CONSOLIDATED_DIR))
    ap_classes.set_defaults(func=cmd_classes)

    a = ap.parse_args()
    sys.exit(a.func(a))


if __name__ == "__main__":
    main()
