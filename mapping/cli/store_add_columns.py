"""Add S1 columns (user_data, scan_angle_rank, return_number) + per-tile time index to an
existing store, without touching row order or the columns `build_store` already wrote.

uv run python -m mapping.cli.store_add_columns [--workers 8] [--force]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..cloud_store import TIME_BUCKET_S, CloudStore, EXTRA_COLUMNS
from ..config import STORE_DIR


def _done(tile_dir: Path) -> bool:
    return all((tile_dir / f"{name}.npy").exists() for name in EXTRA_COLUMNS) and (tile_dir / "time_meta.json").exists()


def add_columns_tile(laz_file: str, tile_dir: str, n_expected: int, force: bool) -> dict:
    """Read one LAZ once, permute its columns with the tile's existing orig_index (row order fixed at
    build_store time), and write the S1 extra columns + time index. Returns a small check summary."""
    import laspy

    d = Path(tile_dir)
    if not force and _done(d):
        return {"name": d.name, "skipped": True}

    order = np.load(d / "orig_index.npy")  # local row -> position in the source LAZ
    assert len(order) == n_expected, (d.name, len(order), n_expected)

    with laspy.open(laz_file) as reader:
        las = reader.read()
    assert las.header.point_count == n_expected, (d.name, las.header.point_count, n_expected)

    user_data = np.asarray(las.user_data, dtype=np.uint8)[order]
    scan_angle_rank = np.asarray(las.scan_angle_rank, dtype=np.int8)[order]
    return_number = (np.asarray(las.return_number, dtype=np.uint8) | (np.asarray(las.number_of_returns, dtype=np.uint8) << 4))[order]
    for a in (user_data, scan_angle_rank, return_number):
        assert len(a) == n_expected, (d.name, len(a), n_expected)

    np.save(d / "user_data.npy", user_data)
    np.save(d / "scan_angle_rank.npy", scan_angle_rank)
    np.save(d / "return_number.npy", return_number)

    # time index: CSR over 10 ms gps_time buckets, rows within a bucket kept in time_order.
    gps_time = np.load(d / "gps_time.npy", mmap_mode="r")
    gps_time = np.asarray(gps_time, dtype=np.float64)
    t_min = float(gps_time.min())
    t_max = float(gps_time.max())
    n_buckets = int(np.floor((t_max - t_min) / TIME_BUCKET_S)) + 1
    bucket = np.floor((gps_time - t_min) / TIME_BUCKET_S).astype(np.int64)
    bucket = np.clip(bucket, 0, n_buckets - 1)
    time_order = np.argsort(bucket, kind="stable").astype(np.uint32)
    bucket_sorted = bucket[time_order]
    bucket_starts = np.searchsorted(bucket_sorted, np.arange(n_buckets + 1)).astype(np.uint32)

    np.save(d / "time_order.npy", time_order)
    np.save(d / "time_bucket_starts.npy", bucket_starts)
    (d / "time_meta.json").write_text(json.dumps({"t_min": t_min, "bucket_s": TIME_BUCKET_S, "n_buckets": n_buckets}))

    vals, counts = np.unique(user_data, return_counts=True)
    hist = {int(v): int(c) for v, c in zip(vals, counts)}
    return {"name": d.name, "skipped": False, "n": int(n_expected), "user_data_hist": hist}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--root", type=str, default=str(STORE_DIR))
    a = ap.parse_args()

    from multiprocessing import Pool

    root = Path(a.root)
    store = CloudStore(root)
    jobs = [(t.laz, str(t.dir), t.n, a.force) for t in store.tiles]
    # largest first for better packing across workers
    jobs.sort(key=lambda j: -Path(j[0]).stat().st_size)

    t0 = time.time()
    with Pool(a.workers) as pool:
        results = pool.starmap(add_columns_tile, jobs)
    dt = time.time() - t0

    n_skipped = sum(1 for r in results if r["skipped"])
    n_done = len(results) - n_skipped
    total_pts = sum(r.get("n", 0) for r in results if not r["skipped"])
    print(f"store_add_columns: {n_done} tiles processed, {n_skipped} skipped, {total_pts} points, {dt:.0f} s")
    for r in results:
        if r["skipped"]:
            continue
        h = r["user_data_hist"]
        total = sum(h.values())
        frac = {k: round(v / total, 3) for k, v in sorted(h.items())}
        print(f"  {r['name']}: n={r['n']} user_data_hist(frac)={frac}")


if __name__ == "__main__":
    main()
