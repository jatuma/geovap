"""Crosstab photo pass <-> LAS point_source_id (psid), via the store's time index (S1).

For each of the 30 photo passes (mapping.poses.load_poses, read-only), gather every scan point whose
gps_time falls in [min frame t, max frame t] +/- 3 s margin (CloudStore.query_time over all tiles),
and record psid / user_data histograms. Answers whether psid == photo pass, or coarser/finer.

uv run python -m mapping.cli.pass_psid [--margin-s 3.0] [--out Geovap_cache/out/poses/pass_psid.json]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..cloud_store import CloudStore
from ..config import POSES_DIR
from ..poses import load_poses

MARGIN_S = 3.0


def gather_pass_psid(store: CloudStore, t0: float, t1: float) -> dict:
    """psid and user_data histograms + n_points for all store points with gps_time in [t0, t1]."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-s", type=float, default=MARGIN_S)
    ap.add_argument("--out", type=str, default=str(POSES_DIR / "pass_psid.json"))
    a = ap.parse_args()

    poses = load_poses()
    store = CloudStore()

    t0_run = time.time()
    passes = {}
    psid_to_passes: dict[int, Counter] = {}
    for p in np.unique(poses.pass_id).tolist():
        sel = poses.pass_id == p
        t0 = float(poses.t[sel].min()) - a.margin_s
        t1 = float(poses.t[sel].max()) + a.margin_s
        res = gather_pass_psid(store, t0, t1)
        res["t0"], res["t1"] = t0, t1
        res["n_frames"] = int(sel.sum())
        passes[str(p)] = res
        for psid_str, c in res["psid_hist"].items():
            psid_to_passes.setdefault(int(psid_str), Counter())[p] += c
    dt = time.time() - t0_run

    reverse = {str(psid): dict(cnt) for psid, cnt in sorted(psid_to_passes.items())}
    out = {"passes": passes, "psid_to_passes": reverse, "margin_s": a.margin_s}
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))

    # summary: is psid a refinement of pass, coarser, or the same partition?
    n_pass_multi_psid = sum(1 for r in passes.values() if len(r["psid_hist"]) > 1)
    n_psid_multi_pass = sum(1 for c in psid_to_passes.values() if len(c) > 1)
    print(f"pass_psid: {len(passes)} passes, {len(psid_to_passes)} distinct psid, {dt:.0f} s -> {out_path}")
    print(f"  passes spanning >1 psid: {n_pass_multi_psid}/{len(passes)}")
    print(f"  psid spanning >1 pass:   {n_psid_multi_pass}/{len(psid_to_passes)}")
    for p, r in passes.items():
        print(f"  pass {p}: n={r['n_points']} psid={r['psid_hist']} user_data={r['user_data_hist']}")


if __name__ == "__main__":
    main()
