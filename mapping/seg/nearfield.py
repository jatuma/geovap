"""Near-field alignment check for the labels: projected JVF road-boundary lines vs photo edges.

The `clean` criterion of `mapping.quality` (04_cisty_dataset.md) uses depth/sky silhouettes, which sit at 10-40 m
and are insensitive to camera position / height / timing errors that move the near ground (3-10 m) by degrees.
Here the centre line of the visible `hranice dopravní stavby` band (bands_erp, band id 1, a geodetic curb / road
edge) is compared with the nearest Canny edge of the photo in the ground rows; the per-frame median distance
(px at 2000x1000, 1 px = 0.18 deg) flags frames whose near field is displaced. Soft edges (grass/gravel
transitions) inflate the absolute value, so the number is a relative indicator: aligned frames 4-12 px,
displaced frames > 20 px (frame 1401: 33 px, visible as labels sitting beside the pavement).

Writes segds/nearfield.json.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from ..config import ZB_H, ZB_W
from ..poses import load_poses
from .areas import SEGDS_DIR
from .render_labels import BANDS_DIR, clean_frames

ROWS = (500, 850)  # elevation 0 .. -63 deg
BAND_ID = 1  # road boundary
MIN_PX = 200
FLAG_PX = 20.0  # median band-to-edge distance above which the frame is flagged `nearfield_bad`
OUT = SEGDS_DIR / "nearfield.json"
_G: dict = {}


def band_edge_distance(photo_bgr: np.ndarray, bands: np.ndarray, band_id: int = BAND_ID, rows=ROWS) -> dict:
    photo = cv2.resize(photo_bgr, (ZB_W, ZB_H), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(g, 40, 100) > 0
    dist = ndimage.distance_transform_edt(~edges)
    sel = np.zeros(bands.shape, bool)
    sel[rows[0] : rows[1]] = bands[rows[0] : rows[1]] == band_id
    n = int(sel.sum())
    if n < MIN_PX:
        return {"n": n, "median_px": None, "p75_px": None, "within3": None}
    inside = ndimage.distance_transform_edt(sel)
    centre = sel & (inside >= np.maximum(1, ndimage.maximum_filter(inside, 5) - 0.5))
    d = dist[centre]
    return {"n": int(centre.sum()), "median_px": round(float(np.median(d)), 1), "p75_px": round(float(np.percentile(d, 75)), 1), "within3": round(float((d <= 3).mean()), 3)}


def _init():
    _G["poses"] = load_poses()


def _work(k: int):
    photo = cv2.imread(_G["poses"].path(k))
    bands = cv2.imread(str(BANDS_DIR / f"f{k:04d}.png"), 0)
    return k, band_edge_distance(photo, bands)


def run(frames: list[int] | None = None, workers: int = 8, out: Path = OUT) -> dict:
    frames = frames if frames is not None else clean_frames()
    res = {}
    with Pool(workers, initializer=_init) as pool:
        for i, (k, r) in enumerate(pool.imap_unordered(_work, frames, chunksize=8)):
            res[k] = r
            if i % 200 == 0:
                print(f"[{i + 1}/{len(frames)}] frame {k}: {r}")
    med = np.array([v["median_px"] for v in res.values() if v["median_px"] is not None])
    summary = {"n_measured": int(len(med)), "median_of_medians_px": round(float(np.median(med)), 1), "p90_px": round(float(np.percentile(med, 90)), 1), "flag_px": FLAG_PX, "n_flagged": int((med > FLAG_PX).sum())}
    out.write_text(json.dumps({"rows": ROWS, "band_id": BAND_ID, "px_deg": 0.18, "summary": summary, "frames": {str(k): v for k, v in sorted(res.items())}}, indent=0))
    print(summary)
    return res


def load() -> dict[int, dict]:
    if not OUT.exists():
        return {}
    return {int(k): v for k, v in json.loads(OUT.read_text())["frames"].items()}


def is_bad(entry: dict | None) -> bool | None:
    """True = flagged, False = ok, None = not measurable (no visible road boundary in the ground rows)."""
    if entry is None or entry.get("median_px") is None:
        return None
    return entry["median_px"] > FLAG_PX


if __name__ == "__main__":
    run()
