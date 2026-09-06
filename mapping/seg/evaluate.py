"""Evaluate fused model predictions against the JVF-derived ERP labels.

Per model: confusion matrix in the common taxonomy over the operational band phi in [-55, +45] deg (rows 250..805
of the 2000x1000 ERP), GT != 255, model coverage; plain and cos(latitude)-weighted; per-class IoU, mIoU_core /
mIoU_ext (only over classes the model can predict); boundary IoU at 2/4/8 px on GT boundaries; band metrics on
the visible JVF line bands (fence / wall / rail bands: fraction predicted as that class; road- and building-edge
bands: median distance to the nearest predicted transition); composition of predictions inside the diagnostic GT
classes (verge, road_or_verge, paved_other). Writes CSVs into dataset/seg/bench/ and tables for the report.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from ..config import ZB_H, ZB_W
from ..products import FrameProducts
from . import classes as C
from . import taxonomy as T
from .bench import BENCH_DIR, DATASET_SEG_DIR
from .dataset import BAND_ROWS
from .models import SPECS
from .render_labels import BANDS_DIR, BAND_NAMES, LABELS_DIR

OUT_DIR = DATASET_SEG_DIR / "bench"
NC = len(T.COMMON)
BOUNDARY_DIL = (2, 4, 8)
BOUNDARY_CLASSES = ("road", "sidewalk", "building", "fence", "terrain", "vegetation")
TRANSITION_BANDS = {"road_boundary": ("road", "sidewalk"), "building_edge": ("building",)}
CLASS_BANDS = {"fence": "fence", "wall": "wall", "guard_rail": "guard_rail"}


def lat_weights() -> np.ndarray:
    v = (np.arange(ZB_H) + 0.5) / ZB_H
    return np.cos(np.pi / 2 - v * np.pi)[:, None].repeat(ZB_W, 1).astype(np.float32)


def confusion(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    g = gt[mask].astype(np.int64)
    p = pred[mask].astype(np.int64)
    idx = g * NC + p
    if w is None:
        return np.bincount(idx, minlength=NC * NC).reshape(NC, NC).astype(np.float64)
    return np.bincount(idx, weights=w[mask], minlength=NC * NC).reshape(NC, NC)


def iou_from_conf(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, tp / denom, np.nan)


def _near(mask: np.ndarray, dil: int) -> np.ndarray:
    """Pixels within `dil` px (Euclidean) of a True pixel, with horizontal wrap."""
    pad = dil + 1
    mp = np.pad(mask, ((0, 0), (pad, pad)), mode="wrap")
    d = ndimage.distance_transform_edt(~mp)
    return (d <= dil)[:, pad:-pad]


def boundary_iou(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, cls: int, dil: int) -> tuple[float, float]:
    """Boundary IoU (Cheng et al. 2021) of class `cls`: boundary region = pixels of the class within `dil` px of a
    pixel of another *labelled* class. Ignore pixels are neither inside nor outside, so lidar holes in the GT do not
    create boundaries. Returns (intersection, union)."""
    other_g = valid & (gt != cls)
    other_p = valid & (pred != cls)
    bg = valid & (gt == cls) & _near(other_g, dil)
    bp = valid & (pred == cls) & _near(other_p, dil)
    return float((bg & bp).sum()), float((bg | bp).sum())


def transition_distance(pred: np.ndarray, band: np.ndarray, classes: tuple[int, ...], valid: np.ndarray) -> np.ndarray:
    """For band pixels: distance (px) to the nearest predicted transition of any of `classes` against anything
    else (4-neighbour label changes between valid pixels; image borders are not transitions)."""
    edge = np.zeros(pred.shape, bool)
    for c in classes:
        inside = pred == c
        dh = (inside[:, 1:] != inside[:, :-1]) & valid[:, 1:] & valid[:, :-1]
        dv = (inside[1:, :] != inside[:-1, :]) & valid[1:, :] & valid[:-1, :]
        edge[:, 1:] |= dh
        edge[:, :-1] |= dh
        edge[1:, :] |= dv
        edge[:-1, :] |= dv
    if not edge.any():
        return np.full(int(band.sum()), np.inf)
    dist = ndimage.distance_transform_edt(~edge)
    return dist[band]


def evaluate_model(tag: str, frames: list[int], lut: np.ndarray, W: np.ndarray) -> dict:
    can = set(np.unique(_model_common_ids(tag)).tolist())  # common ids the model can output
    cm = np.zeros((NC, NC))
    cmw = np.zeros((NC, NC))
    b_int = {(c, d): 0.0 for c in BOUNDARY_CLASSES for d in BOUNDARY_DIL}
    b_uni = {(c, d): 0.0 for c in BOUNDARY_CLASSES for d in BOUNDARY_DIL}
    class_band = {k: [0, 0] for k in CLASS_BANDS}
    trans = {k: [] for k in TRANSITION_BANDS}
    trans_tol = {k: [0, 0] for k in TRANSITION_BANDS}
    diag_comp = {n: np.zeros(NC) for n in ("verge", "road_or_verge", "paved_other")}
    n_frames = 0
    r0, r1 = BAND_ROWS
    for k in frames:
        pp = BENCH_DIR / tag / f"f{k:04d}_common.png"
        if not pp.exists():
            continue
        n_frames += 1
        pred = cv2.imread(str(pp), 0)
        gt_raw = cv2.imread(str(LABELS_DIR / f"f{k:04d}.png"), 0)
        bands = cv2.imread(str(BANDS_DIR / f"f{k:04d}.png"), 0)
        gt = lut[gt_raw]
        band_rows = np.zeros((ZB_H, ZB_W), bool)
        band_rows[r0:r1] = True
        cov = pred != 255
        mask = (gt != 255) & cov & band_rows
        cm += confusion(gt, pred, mask)
        cmw += confusion(gt, pred, mask, W)
        valid = mask
        for c in BOUNDARY_CLASSES:
            cid = T.COMMON_ID[c]
            if cid not in can or not (gt[valid] == cid).any():
                continue
            for d in BOUNDARY_DIL:
                i, u = boundary_iou(gt, pred, valid, cid, d)
                b_int[(c, d)] += i
                b_uni[(c, d)] += u
        # bands (visible JVF lines) within the operational band and model coverage
        bvalid = cov & band_rows
        for bname, cname in CLASS_BANDS.items():
            bid = [i for i, n in BAND_NAMES.items() if n == bname][0]
            bm = (bands == bid) & bvalid
            if T.COMMON_ID[cname] in can and bm.any():
                class_band[bname][0] += int((pred[bm] == T.COMMON_ID[cname]).sum())
                class_band[bname][1] += int(bm.sum())
        depth = None
        for bname, cls_names in TRANSITION_BANDS.items():
            bid = [i for i, n in BAND_NAMES.items() if n == bname][0]
            bm = (bands == bid) & bvalid
            cls_ids = tuple(T.COMMON_ID[n] for n in cls_names if T.COMMON_ID[n] in can)
            if not cls_ids or not bm.any():
                continue
            d = transition_distance(pred, bm, cls_ids, cov)
            trans[bname].append(d)
            if depth is None:
                depth = FrameProducts.load(k).depth_m
            r = np.maximum(depth[bm], 1.0)
            tol_px = 0.14 / r * ZB_W / (2 * np.pi)  # 14 cm at the pixel's range, in ERP px
            trans_tol[bname][0] += int((d <= np.maximum(tol_px, 1.0)).sum())
            trans_tol[bname][1] += int(bm.sum())
        for n in diag_comp:
            m = (gt_raw == C.BY_NAME[n].id) & cov & band_rows
            if m.any():
                diag_comp[n] += np.bincount(pred[m], minlength=NC)[:NC]

    iou = iou_from_conf(cm)
    iouw = iou_from_conf(cmw)
    per_class = {}
    for n in T.CORE + T.EXT:
        i = T.COMMON_ID[n]
        per_class[n] = {"iou": None if i not in can else _f(iou[i]), "iou_cosw": None if i not in can else _f(iouw[i]), "gt_px": int(cm[i].sum()), "predictable": i in can}
    core = [per_class[n]["iou_cosw"] for n in T.CORE if per_class[n]["iou_cosw"] is not None]
    ext = [per_class[n]["iou_cosw"] for n in T.CORE + T.EXT if per_class[n]["iou_cosw"] is not None and per_class[n]["gt_px"] > 0]
    total = cm.sum()
    res = {
        "tag": tag,
        "n_frames": n_frames,
        "pixel_acc": _f(np.diag(cm).sum() / total) if total else None,
        "pixel_acc_cosw": _f(np.diag(cmw).sum() / cmw.sum()) if total else None,
        "mIoU_core": _f(np.nanmean(core)) if core else None,
        "mIoU_core_plain": _f(np.nanmean([per_class[n]["iou"] for n in T.CORE if per_class[n]["iou"] is not None])),
        "mIoU_ext": _f(np.nanmean(ext)) if ext else None,
        "per_class": per_class,
        "boundary_iou": {f"{c}@{d}": (_f(b_int[(c, d)] / b_uni[(c, d)]) if b_uni[(c, d)] > 0 else None) for c in BOUNDARY_CLASSES for d in BOUNDARY_DIL},
        "class_bands": {k: (_f(v[0] / v[1]) if v[1] else None) for k, v in class_band.items()},
        "transition_median_px": {k: (_f(np.median(np.concatenate(v))) if v else None) for k, v in trans.items()},
        "transition_within_14cm": {k: (_f(v[0] / v[1]) if v[1] else None) for k, v in trans_tol.items()},
        "diag_composition": {n: {T.COMMON[i]: _f(v[i] / v.sum()) for i in np.argsort(-v)[:4] if v.sum() > 0} for n, v in diag_comp.items()},
        "confusion": cm.tolist(),
    }
    return res


def _model_common_ids(tag: str) -> np.ndarray:
    meta = json.loads((DATASET_SEG_DIR / f"id2label_{tag}.json").read_text())
    return T.native_to_common(meta["id2label"], meta["taxonomy"])


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def timing(tag: str) -> dict:
    p = BENCH_DIR / tag / "timing.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(p.open()))
    if not rows:
        return {}
    secs = np.array([float(r["seconds"]) for r in rows])
    return {"s_per_pano_median": _f(np.median(secs[1:] if len(secs) > 1 else secs)), "peak_mib": int(max(float(r["peak_mib"]) for r in rows))}


def run(tags: list[str], frames: list[int]) -> dict:
    """Evaluate on all frames and on the near-field-clean subset (labels not displaced by pass registration)."""
    from . import nearfield

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lut = T.gt_to_common_lut()
    W = lat_weights()
    nf = nearfield.load()
    subsets = {"all": frames, "nf_ok": [k for k in frames if nearfield.is_bad(nf.get(k)) is False]}
    results = {}
    for tag in tags:
        if not (BENCH_DIR / tag).exists():
            print(f"[{tag}] no predictions, skipped")
            continue
        r = evaluate_model(tag, subsets["all"], lut, W)
        r["nf_ok"] = evaluate_model(tag, subsets["nf_ok"], lut, W)
        r["nf_ok"].pop("confusion", None)
        r["timing"] = timing(tag)
        r["spec"] = {"checkpoint": SPECS[tag].checkpoint, "taxonomy": SPECS[tag].taxonomy, "licence": SPECS[tag].licence, "note": SPECS[tag].note}
        results[tag] = r
        print(f"[{tag}] {r['n_frames']} frames: mIoU_core {r['mIoU_core']} (nf_ok {r['nf_ok']['n_frames']} frames: {r['nf_ok']['mIoU_core']}), acc {r['pixel_acc_cosw']}, fence band {r['class_bands']['fence']}, {r['timing']}")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1))
    write_csvs(results)
    return results


def write_csvs(results: dict) -> None:
    tags = list(results)
    with (OUT_DIR / "per_class_iou.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "gt_px"] + tags)
        for n in T.CORE + T.EXT:
            w.writerow([n, results[tags[0]]["per_class"][n]["gt_px"]] + [_cell(results[t]["per_class"][n]) for t in tags])
    with (OUT_DIR / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "checkpoint", "taxonomy", "frames", "mIoU_core_cosw", "mIoU_core_plain", "mIoU_ext", "pixel_acc_cosw", "frames_nf_ok", "mIoU_core_nf_ok", "s_per_pano", "peak_MiB", "licence"])
        for t in tags:
            r = results[t]
            w.writerow([t, r["spec"]["checkpoint"], r["spec"]["taxonomy"], r["n_frames"], r["mIoU_core"], r["mIoU_core_plain"], r["mIoU_ext"], r["pixel_acc_cosw"], r["nf_ok"]["n_frames"], r["nf_ok"]["mIoU_core"], r["timing"].get("s_per_pano_median"), r["timing"].get("peak_mib"), r["spec"]["licence"]])
    with (OUT_DIR / "boundary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        keys = list(results[tags[0]]["boundary_iou"])
        w.writerow(["boundary"] + tags)
        for k in keys:
            w.writerow([k] + [results[t]["boundary_iou"][k] for t in tags])
    with (OUT_DIR / "bands.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric"] + tags)
        for k in CLASS_BANDS:
            w.writerow([f"band_{k}_frac_correct"] + [results[t]["class_bands"][k] for t in tags])
        for k in TRANSITION_BANDS:
            w.writerow([f"band_{k}_median_px"] + [results[t]["transition_median_px"][k] for t in tags])
            w.writerow([f"band_{k}_within_14cm"] + [results[t]["transition_within_14cm"][k] for t in tags])


def _cell(pc: dict):
    return "n/a" if not pc["predictable"] else pc["iou_cosw"]


# ------------------------------------------------------------------------------------- report tables
def md_table(header: list[str], rows: list[list]) -> str:
    fmt = lambda x: "–" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(fmt(x) for x in r) + " |")
    return "\n".join(out)


def report() -> str:
    results = json.loads((OUT_DIR / "results.json").read_text())
    tags = sorted(results, key=lambda t: -(results[t]["mIoU_core"] or 0))
    parts = []
    n_all = results[tags[0]]["n_frames"]
    n_ok = results[tags[0]]["nf_ok"]["n_frames"]
    parts.append(md_table(["model", "taxonomie", f"mIoU_core cos ({n_all} sn.)", f"mIoU_core cos, blízké pole ok ({n_ok} sn.)", "mIoU_core plain", "mIoU_ext", "pixel acc (cos)", "s/pano", "GPU MiB", "licence"],
                          [[t, results[t]["spec"]["taxonomy"], results[t]["mIoU_core"], results[t]["nf_ok"]["mIoU_core"], results[t]["mIoU_core_plain"], results[t]["mIoU_ext"], results[t]["pixel_acc_cosw"], results[t]["timing"].get("s_per_pano_median"), results[t]["timing"].get("peak_mib"), results[t]["spec"]["licence"]] for t in tags]))
    for key, label in (("per_class", "všechny snímky"), ("nf_ok", "blízké pole ok")):
        rows = []
        for n in T.CORE + T.EXT:
            src = results[tags[0]] if key == "per_class" else results[tags[0]]["nf_ok"]
            gt = src["per_class"][n]["gt_px"]
            rows.append([n, f"{gt / 1e6:.1f} M"] + [("n/a" if not (results[t] if key == "per_class" else results[t]["nf_ok"])["per_class"][n]["predictable"] else (results[t] if key == "per_class" else results[t]["nf_ok"])["per_class"][n]["iou_cosw"]) for t in tags])
        parts.append(f"IoU po třídách (cos), {label}:\n\n" + md_table(["třída", "GT px", *tags], rows))
    keys = list(results[tags[0]]["boundary_iou"])
    parts.append(md_table(["boundary IoU", *tags], [[k] + [results[t]["boundary_iou"][k] for t in tags] for k in keys]))
    brows = [[f"{k} band: podíl správně"] + [results[t]["class_bands"][k] for t in tags] for k in CLASS_BANDS]
    brows += [[f"{k}: medián px k přechodu"] + [results[t]["transition_median_px"][k] for t in tags] for k in TRANSITION_BANDS]
    brows += [[f"{k}: podíl do 14 cm"] + [results[t]["transition_within_14cm"][k] for t in tags] for k in TRANSITION_BANDS]
    parts.append(md_table(["pásy JVF linií", *tags], brows))
    drows = []
    for n in ("road_or_verge", "verge", "paved_other"):
        for t in tags:
            comp = results[t]["diag_composition"][n]
            drows.append([n, t, ", ".join(f"{c} {v:.2f}" for c, v in comp.items())])
    parts.append(md_table(["diagnostická třída", "model", "složení predikcí"], drows))
    text = "\n\n".join(parts)
    (OUT_DIR / "tables.md").write_text(text)
    print(text)
    return text
