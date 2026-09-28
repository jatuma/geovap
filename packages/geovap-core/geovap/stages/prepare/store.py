"""`store` / `store-columns`: build the columnar point store, then extend it.

Two stages live in this one module because they are two passes over the SAME store, in the order a
CLI needs to run them in:

  - `store` (`after=("ingest",)`) is `geovap.runtime.store.build_store`: converts every tile's LAZ
    into the columnar/cell-indexed layout `runtime.store` documents, one directory per tile. This is
    the one-off conversion the whole `CloudStore` design exists for (see `runtime/store.py`'s module
    docstring): it lets a worker fork without copying hundreds of millions of points, and it lets a
    z-buffer gather every point within `r_max` regardless of which tile it happened to land in.

  - `store-columns` (`after=("store",)`) is the S1 second pass, ported unchanged from
    `mapping/cli/store_add_columns.py` and `mapping/cli/pass_psid.py`: it reads each tile's LAZ a
    SECOND time to pull out three extra columns build_store does not write by default
    (`user_data`/`scan_angle_rank`/`return_number` -- useful for a few later analyses but not needed
    by every stage, hence optional and added lazily rather than bloating every store by default),
    plus a per-tile TIME INDEX over `gps_time` (a CSR-style bucket index -- see below for why it
    exists), and finally the pass/point-source-id crosstab that answers whether a LAS `psid` is the
    same partition as a photo "pass", a refinement of it, or something coarser.

`store-columns` is deliberately its own stage rather than folded into `store`: a store already built
without these columns must keep opening (`runtime.store.TileData` treats every one of them as
optional, `None` when the `.npy` is missing), and re-running `store-columns` against an old store
must not touch the columns `build_store` already wrote or reorder any row.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from geovap.stages.base.cli import stage_main
from geovap.stages.base.spec import StageSpec, registry

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings

MARGIN_S = 3.0  # pass_psid: gather every scan point within this margin of a pass's [t_min, t_max]


# ============================================================================================ store
class BuildStore:
    spec = StageSpec(
        name="store", after=("ingest",), est_min=25,
        summary="columnar point store, one directory per tile",
    )

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        # `Stage.inputs()` is `dict[str, Path]` (see `describe()` in `stages.base.cli`, which treats
        # every value as one) -- so `cell_size` (a scalar, not a path) cannot be declared here even
        # though it does affect the built store; a change to it is caught the same way a change to
        # `s.sensor` generally is, by the descriptor file itself being one of `ingest`'s inputs.
        return {ref.id.value: ref.path for ref in s.tiles.tiles()}

    def outputs(self, s: "Settings") -> list[Path]:
        return [s.workspace.store / "tiles.json"]

    def metrics(self, s: "Settings") -> dict:
        try:
            from geovap.runtime import store as store_mod

            root = s.workspace.store
            if not (root / "tiles.json").exists():
                return {"error": "store not built yet"}
            cs = store_mod.CloudStore(root)
            return {"n_tiles": len(cs.tiles), "total_points": cs.total, "cell_size": s.sensor.cell_size}
        except Exception as e:  # noqa: BLE001 - metrics must always be reportable
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", *, workers: int = 8) -> None:
        from geovap.runtime import store as store_mod

        store_mod.build_store(s, workers=workers)


STAGE = registry.add(BuildStore())


# ===================================================================================== store-columns
def _done(tile_dir: Path, extra_columns: tuple[str, ...]) -> bool:
    return all((tile_dir / f"{name}.npy").exists() for name in extra_columns) and (tile_dir / "time_meta.json").exists()


def add_columns_tile(laz_file: str, tile_dir: str, n_expected: int, force: bool) -> dict:
    """Read one LAZ once, permute its columns with the tile's existing `orig_index` (row order fixed
    at `build_store` time, never touched here), and write the S1 extra columns + time index. Runs in
    a worker process (module-level function, so `multiprocessing.Pool.starmap` can pickle it).
    Returns a small check summary."""
    import laspy

    from geovap.runtime.store import EXTRA_COLUMNS, TIME_BUCKET_S

    d = Path(tile_dir)
    if not force and _done(d, EXTRA_COLUMNS):
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

    # Why a time index: `pass_psid` (and anything else that wants "every point seen during pose
    # window [t0, t1]") needs points by `gps_time`, but the store's only spatial index is the dense
    # cell grid -- there is no way to ask "which rows have gps_time in this range" without scanning
    # every row, once per query, across every tile. So we sort rows into 10 ms buckets ONCE here: a
    # CSR-style `time_bucket_starts` over `time_order` (`time_order` sorted by `gps_time`, ties kept
    # in that order), so `rows_in_time(t0, t1)` (`runtime.store.TileData.rows_in_time`) is a
    # searchsorted + a small in-bucket refine, not an O(n) scan. This column is optional (S1): a
    # store built before this stage existed, or one this stage hasn't reached yet, keeps opening --
    # `rows_in_time` simply raises if a caller asks for it before it's built.
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


def gather_pass_psid(store, t0: float, t1: float) -> dict:
    """psid and user_data histograms + n_points for all store points with gps_time in [t0, t1].
    Requires the time index built by `add_columns_tile` above (via `CloudStore.query_time`)."""
    from collections import Counter

    parts = store.query_time(t0, t1)
    psid_counts: Counter = Counter()
    ud_counts: Counter = Counter()
    n = 0
    for t, rows in parts:
        td = store.tile(t.name)
        psid = np.asarray(td.psid[rows])
        n += len(rows)
        for v, c in zip(*np.unique(psid, return_counts=True)):
            psid_counts[int(v)] += int(c)
        if td.user_data is not None:
            ud = np.asarray(td.user_data[rows])
            for v, c in zip(*np.unique(ud, return_counts=True)):
                ud_counts[int(v)] += int(c)
        td.release()
    return {"n_points": n, "psid_hist": dict(psid_counts), "user_data_hist": dict(ud_counts)}


class StoreColumns:
    spec = StageSpec(
        name="store-columns", after=("store",), est_min=20,
        summary="S1 extra columns (user_data/scan_angle_rank/return_number) + time index + pass/psid crosstab",
    )

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        root = s.workspace.store
        d: dict[str, Path] = {"tiles_json": root / "tiles.json"}
        try:
            from geovap.runtime import store as store_mod

            cs = store_mod.CloudStore(root)
            for t in cs.tiles:
                d[f"laz_{t.name}"] = Path(t.laz)
        except Exception:  # noqa: BLE001 - store not built yet; the tiles_json entry alone marks this stale
            pass
        return d

    def outputs(self, s: "Settings") -> list[Path]:
        from geovap.runtime import store as store_mod

        cs = store_mod.CloudStore(s.workspace.store)  # raises if `store` hasn't run yet
        outs: list[Path] = []
        for t in cs.tiles:
            for name in store_mod.EXTRA_COLUMNS:
                outs.append(t.dir / f"{name}.npy")
            outs.append(t.dir / "time_meta.json")
        outs.append(s.workspace.poses / "pass_psid.json")
        return outs

    def metrics(self, s: "Settings") -> dict:
        try:
            from geovap.runtime import store as store_mod

            cs = store_mod.CloudStore(s.workspace.store)
            built = sum(1 for t in cs.tiles if (t.dir / "user_data.npy").exists())
            out = {"n_tiles": len(cs.tiles), "tiles_with_columns": built}
            pj = s.workspace.poses / "pass_psid.json"
            if pj.exists():
                data = json.loads(pj.read_text())
                out["n_passes"] = len(data.get("passes", {}))
                out["n_distinct_psid"] = len(data.get("psid_to_passes", {}))
            return out
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", *, workers: int = 8, force: bool = False, margin_s: float = MARGIN_S) -> None:
        from geovap.runtime.procs import pool_context

        from geovap.runtime import store as store_mod

        root = s.workspace.store
        store = store_mod.CloudStore(root)
        jobs = [(t.laz, str(t.dir), t.n, force) for t in store.tiles]
        # largest first for better packing across workers
        jobs.sort(key=lambda j: -Path(j[0]).stat().st_size)

        t0 = time.time()
        with pool_context().Pool(workers) as pool:
            results = pool.starmap(add_columns_tile, jobs)
        dt = time.time() - t0

        n_skipped = sum(1 for r in results if r["skipped"])
        n_done = len(results) - n_skipped
        total_pts = sum(r.get("n", 0) for r in results if not r["skipped"])
        print(f"store-columns: {n_done} tiles processed, {n_skipped} skipped, {total_pts} points, {dt:.0f} s")
        for r in results:
            if r["skipped"]:
                continue
            h = r["user_data_hist"]
            total = sum(h.values())
            frac = {k: round(v / total, 3) for k, v in sorted(h.items())}
            print(f"  {r['name']}: n={r['n']} user_data_hist(frac)={frac}")

        # Re-open: the `store` object above cached each `TileData` before its time index existed, so
        # a fresh `CloudStore` is needed for `query_time` to see the columns just written.
        store = store_mod.CloudStore(root)
        poses = s.poses.load()

        t0_run = time.time()
        passes: dict = {}
        psid_to_passes: dict[int, dict] = {}
        for p in np.unique(poses.pass_id).tolist():
            sel = poses.pass_id == p
            pt0 = float(poses.t[sel].min()) - margin_s
            pt1 = float(poses.t[sel].max()) + margin_s
            res = gather_pass_psid(store, pt0, pt1)
            res["t0"], res["t1"] = pt0, pt1
            res["n_frames"] = int(sel.sum())
            passes[str(p)] = res
            for psid_str, c in res["psid_hist"].items():
                psid_to_passes.setdefault(int(psid_str), {}).setdefault(p, 0)
                psid_to_passes[int(psid_str)][p] += c
        dt_run = time.time() - t0_run

        reverse = {str(psid): dict(cnt) for psid, cnt in sorted(psid_to_passes.items())}
        out = {"passes": passes, "psid_to_passes": reverse, "margin_s": margin_s}
        out_path = s.workspace.poses / "pass_psid.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=1))

        n_pass_multi_psid = sum(1 for r in passes.values() if len(r["psid_hist"]) > 1)
        n_psid_multi_pass = sum(1 for c in psid_to_passes.values() if len(c) > 1)
        print(f"pass_psid: {len(passes)} passes, {len(psid_to_passes)} distinct psid, {dt_run:.0f} s -> {out_path}")
        print(f"  passes spanning >1 psid: {n_pass_multi_psid}/{len(passes)}")
        print(f"  psid spanning >1 pass:   {n_psid_multi_pass}/{len(psid_to_passes)}")


STORE_COLUMNS = registry.add(StoreColumns())


# ================================================================================================ cli
def _add_options_store(p) -> None:
    p.add_argument("--workers", type=int, default=8)


def _to_opts_store(args) -> dict:
    return {"workers": args.workers}


def _add_options_columns(p) -> None:
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="reprocess tiles even if already done")
    p.add_argument("--margin-s", type=float, default=MARGIN_S, help="pass/psid gather margin, seconds")


def _to_opts_columns(args) -> dict:
    return {"workers": args.workers, "force": args.force, "margin_s": args.margin_s}


def main(argv=None) -> int:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    # `--stage {store,columns}` picks which of this module's two `Stage` objects the rest of argv
    # applies to (its own `available()`/`--status`/options), so it has to be resolved BEFORE
    # `stage_main` builds that stage's parser -- it is consumed here, not registered as one more
    # option on either stage's own CLI.
    stage_choice = "store"
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--stage" and i + 1 < len(argv):
            stage_choice = argv[i + 1]
            i += 2
            continue
        if a.startswith("--stage="):
            stage_choice = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1

    if stage_choice not in ("store", "columns"):
        raise SystemExit(f"--stage must be 'store' or 'columns', got {stage_choice!r}")

    if stage_choice == "store":
        return stage_main(STAGE, rest, add_options=_add_options_store, to_opts=_to_opts_store)
    return stage_main(STORE_COLUMNS, rest, add_options=_add_options_columns, to_opts=_to_opts_columns)


if __name__ == "__main__":
    raise SystemExit(main())
