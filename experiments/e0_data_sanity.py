"""E0 - sanity check dat pred cimkoli dalsim.

Overuje: pocty souboru/radku, jestli JPG odpovidaji export.csv, rozlozeni JVF trid
v geojsonu, a jestli LAZ dlazdice sedi souradnicove s export.csv/geojsonem.
"""
from __future__ import annotations

import glob
import os
from collections import Counter

import laspy
import numpy as np

from common import class_map, io_data


def main() -> None:
    print("=== export.csv ===")
    frames = io_data.load_frames()
    print(f"pocet snimku v export.csv: {len(frames)}")

    jpgs = sorted(glob.glob(os.path.join(io_data.PANO_DIR, "*.jpg")))
    print(f"pocet .jpg souboru na disku: {len(jpgs)}")
    jpg_names = {os.path.basename(p) for p in jpgs}
    csv_names = set(frames.filename.tolist())
    missing_on_disk = csv_names - jpg_names
    missing_in_csv = jpg_names - csv_names
    print(f"v csv, chybi na disku: {len(missing_on_disk)}")
    print(f"na disku, chybi v csv: {len(missing_in_csv)}")
    if missing_on_disk:
        print("  priklady:", list(missing_on_disk)[:5])
    if missing_in_csv:
        print("  priklady:", list(missing_in_csv)[:5])

    print(f"roll [deg]  min/max/mean: {frames.roll_deg.min():.3f} / {frames.roll_deg.max():.3f} / {frames.roll_deg.mean():.3f}")
    print(f"pitch [deg] min/max/mean: {frames.pitch_deg.min():.3f} / {frames.pitch_deg.max():.3f} / {frames.pitch_deg.mean():.3f}")
    e_min, n_min, h_min = frames.origin_enh.min(axis=0)
    e_max, n_max, h_max = frames.origin_enh.max(axis=0)
    print(f"origin E: [{e_min:.1f}, {e_max:.1f}]  N: [{n_min:.1f}, {n_max:.1f}]  H: [{h_min:.1f}, {h_max:.1f}]")

    print("\n=== 1_ZPS_GAD.geojson ===")
    objects = io_data.load_jvf_objects()
    print(f"pocet objektu s jvfcode: {len(objects)}")
    by_code = Counter((o.jvfcode, o.geom_type) for o in objects)
    known, unknown = 0, 0
    for (code, gtype), n in by_code.most_common():
        info = class_map.lookup(code)
        tag = f"{info.name_cz} -> {info.public}" if info else "!! NEZNAMY KOD (doplnit do class_map.py)"
        if info:
            known += n
        else:
            unknown += n
        print(f"  {n:5d}  {code} {gtype:10s} {tag}")
    print(f"pokryto class_map.py: {known}/{known+unknown} objektu ({100*known/(known+unknown):.1f} %)")

    with_z = sum(1 for o in objects if not np.isnan(o.coords[:, 2]).any())
    print(f"objekty s kompletni Z souradnici: {with_z}/{len(objects)}")

    geo_e = np.concatenate([o.coords[:, 0] for o in objects])
    geo_n = np.concatenate([o.coords[:, 1] for o in objects])
    print(f"geojson E: [{geo_e.min():.1f}, {geo_e.max():.1f}]  N: [{geo_n.min():.1f}, {geo_n.max():.1f}]")
    overlap_e = not (geo_e.max() < e_min or geo_e.min() > e_max)
    overlap_n = not (geo_n.max() < n_min or geo_n.min() > n_max)
    print(f"prekryv rozsahu E/N s export.csv: {overlap_e and overlap_n}")

    print("\n=== LAZ dlazdice ===")
    tiles = io_data.list_laz_tiles()
    print(f"pocet .laz souboru: {len(tiles)}")
    for path in tiles[:3]:
        with laspy.open(path) as f:
            header = f.header
            print(
                f"  {os.path.basename(path)}: {header.point_count} bodu, "
                f"mins={header.mins}, maxs={header.maxs}"
            )
    if tiles:
        with laspy.open(tiles[0]) as f:
            las = f.read()
            classes = Counter(np.asarray(las.classification).tolist())
            print(f"  klasifikace v prvni dlazdici: {dict(classes)}")


if __name__ == "__main__":
    main()
