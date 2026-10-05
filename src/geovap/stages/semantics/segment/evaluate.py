"""Evaluate fused model predictions against the JVF-derived ERP labels.

Per model: confusion matrix in the common taxonomy over the operational band phi in [-55, +45] deg (rows 250..805
of the 2000x1000 ERP), GT != 255, model coverage; plain and cos(latitude)-weighted; per-class IoU, mIoU_core /
mIoU_ext (only over classes the model can predict); boundary IoU at 2/4/8 px on GT boundaries; band metrics on
the visible JVF line bands (fence / wall / rail bands: fraction predicted as that class; road- and building-edge
bands: median distance to the nearest predicted transition); composition of predictions inside the diagnostic GT
classes (verge, road_or_verge, paved_other). Writes CSVs into the workspace's `dataset/seg/bench/` and tables
for the report.

Also declares and folds `mapping/cli/seg_bench.py`'s CLI in as this module's `main()`, and the
`seg-eval` `StageSpec` -- named for the marker file the old `mapping.cli.pipeline` driver already
wrote (`seg-eval.json`), so a run started under that driver resumes here unchanged.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from geovap.runtime.pose_tables import load as load_poses
from geovap.stages.prepare.products import load_products
from geovap.domain.scheme import classes as C
from geovap.domain.scheme import taxonomy as T
from geovap.stages.base.spec import StageSpec, registry
from .bench import bench_dir, dataset_seg_dir, missing_predictions
from .models import SPECS
from geovap.stages.semantics.pseudogt.erp import bands_dir, BAND_NAMES, labels_dir, frames_arg
from geovap.stages.semantics.pseudogt.dataset import band_rows

NC = len(T.COMMON)
BOUNDARY_DIL = (2, 4, 8)
BOUNDARY_CLASSES = ("road", "sidewalk", "building", "fence", "terrain", "vegetation")
TRANSITION_BANDS = {"road_boundary": ("road", "sidewalk"), "building_edge": ("building",)}
CLASS_BANDS = {"fence": "fence", "wall": "wall", "guard_rail": "guard_rail"}


def out_dir_default(s=None) -> Path:
    return dataset_seg_dir(s) / "bench"


def lat_weights(zb_h: int, zb_w: int) -> np.ndarray:
    v = (np.arange(zb_h) + 0.5) / zb_h
    return np.cos(np.pi / 2 - v * np.pi)[:, None].repeat(zb_w, 1).astype(np.float32)


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


def evaluate_model(tag: str, frames: list[int], lut: np.ndarray, W: np.ndarray, zb_h: int, zb_w: int, band_rows_: tuple[int, int], poses=None) -> dict:
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
    r0, r1 = band_rows_
    n_no_gt = 0
    for k in frames:
        pp = bench_dir() / tag / f"f{k:04d}_common.png"
        if not pp.exists():
            continue
        if not (labels_dir() / f"f{k:04d}.png").exists() or not (bands_dir() / f"f{k:04d}.png").exists():
            n_no_gt += 1  # frame not in this pose source's clean set -> no pseudo-GT rendered for it
            continue
        n_frames += 1
        pred = cv2.imread(str(pp), 0)
        gt_raw = cv2.imread(str(labels_dir() / f"f{k:04d}.png"), 0)
        bands = cv2.imread(str(bands_dir() / f"f{k:04d}.png"), 0)
        gt = lut[gt_raw]
        band_mask = np.zeros((zb_h, zb_w), bool)
        band_mask[r0:r1] = True
        cov = pred != 255
        mask = (gt != 255) & cov & band_mask
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
        bvalid = cov & band_mask
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
                depth = load_products(k, poses or load_poses()).depth_m
            r = np.maximum(depth[bm], 1.0)
            tol_px = 0.14 / r * zb_w / (2 * np.pi)  # 14 cm at the pixel's range, in ERP px
            trans_tol[bname][0] += int((d <= np.maximum(tol_px, 1.0)).sum())
            trans_tol[bname][1] += int(bm.sum())
        for n in diag_comp:
            m = (gt_raw == C.BY_NAME[n].id) & cov & band_mask
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
        "n_frames": n_frames, "n_no_gt": n_no_gt,
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
    meta = json.loads((dataset_seg_dir() / f"id2label_{tag}.json").read_text())
    return T.native_to_common(meta["id2label"], meta["taxonomy"])


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def timing(tag: str) -> dict:
    p = bench_dir() / tag / "timing.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(p.open()))
    if not rows:
        return {}
    secs = np.array([float(r["seconds"]) for r in rows])
    return {"s_per_pano_median": _f(np.median(secs[1:] if len(secs) > 1 else secs)), "peak_mib": int(max(float(r["peak_mib"]) for r in rows))}


def run(tags: list[str], frames: list[int], poses_source: str | None = None, out_dir: Path | None = None) -> dict:
    """Evaluate on all frames and on the near-field-clean subset (labels not displaced by pass registration).
    `out_dir` (default the workspace's `dataset/seg/bench`) receives results.json + the CSVs; pass another
    dir to keep a second evaluation (e.g. on a fixed baseline frame list) from overwriting the main one."""
    from geovap.runtime import settings
    from geovap.stages.semantics.pseudogt import nearfield

    s = settings.get()
    out_dir = out_dir_default(s) if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = load_poses(poses_source)
    lut = T.gt_to_common_lut()
    zb_h, zb_w = s.sensor.zb_h, s.sensor.zb_w
    W = lat_weights(zb_h, zb_w)
    rows = band_rows(zb_h)
    nf = nearfield.load()
    subsets = {"all": frames, "nf_ok": [k for k in frames if nearfield.is_bad(nf.get(k)) is False]}
    results = {}
    for tag in tags:
        if not (bench_dir() / tag).exists():
            print(f"[{tag}] no predictions, skipped")
            continue
        r = evaluate_model(tag, subsets["all"], lut, W, zb_h, zb_w, rows, poses)
        r["nf_ok"] = evaluate_model(tag, subsets["nf_ok"], lut, W, zb_h, zb_w, rows, poses)
        r["nf_ok"].pop("confusion", None)
        r["timing"] = timing(tag)
        r["spec"] = {"checkpoint": SPECS[tag].checkpoint, "taxonomy": SPECS[tag].taxonomy, "licence": SPECS[tag].licence, "note": SPECS[tag].note}
        results[tag] = r
        print(f"[{tag}] {r['n_frames']} frames: mIoU_core {r['mIoU_core']} (nf_ok {r['nf_ok']['n_frames']} frames: {r['nf_ok']['mIoU_core']}), acc {r['pixel_acc_cosw']}, fence band {r['class_bands']['fence']}, {r['timing']}")
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    write_csvs(results, out_dir)
    return results


def write_csvs(results: dict, out_dir: Path) -> None:
    tags = list(results)
    with (out_dir / "per_class_iou.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "gt_px"] + tags)
        for n in T.CORE + T.EXT:
            w.writerow([n, results[tags[0]]["per_class"][n]["gt_px"]] + [_cell(results[t]["per_class"][n]) for t in tags])
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "checkpoint", "taxonomy", "frames", "mIoU_core_cosw", "mIoU_core_plain", "mIoU_ext", "pixel_acc_cosw", "frames_nf_ok", "mIoU_core_nf_ok", "s_per_pano", "peak_MiB", "licence"])
        for t in tags:
            r = results[t]
            w.writerow([t, r["spec"]["checkpoint"], r["spec"]["taxonomy"], r["n_frames"], r["mIoU_core"], r["mIoU_core_plain"], r["mIoU_ext"], r["pixel_acc_cosw"], r["nf_ok"]["n_frames"], r["nf_ok"]["mIoU_core"], r["timing"].get("s_per_pano_median"), r["timing"].get("peak_mib"), r["spec"]["licence"]])
    with (out_dir / "boundary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        keys = list(results[tags[0]]["boundary_iou"])
        w.writerow(["boundary"] + tags)
        for k in keys:
            w.writerow([k] + [results[t]["boundary_iou"][k] for t in tags])
    with (out_dir / "bands.csv").open("w", newline="") as fh:
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


def report(out_dir: Path | None = None) -> str:
    out_dir = out_dir_default() if out_dir is None else Path(out_dir)
    results = json.loads((out_dir / "results.json").read_text())
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
    (out_dir / "tables.md").write_text(text)
    print(text)
    return text


# ================================================================================================ stage
def _base_frames(spec: str, s=None) -> list[int]:
    if spec == "bench":
        return json.loads((dataset_seg_dir(s) / "bench_frames.json").read_text())["frames"]
    p = Path(spec)
    if p.suffix == ".json" and p.exists():
        data = json.loads(p.read_text())
        return data["frames"] if isinstance(data, dict) else list(data)
    return frames_arg(spec)


def _frames_for_model(spec: str, tag: str) -> list[int]:
    """Resolve `spec` to a frame list for model `tag`; `missing:<spec>` filters to frames lacking
    a prediction for `tag` (see `missing_predictions`)."""
    if spec.startswith("missing:"):
        base = _base_frames(spec[len("missing:") :])
        return missing_predictions(tag, base)
    return _base_frames(spec)


class SegEval:
    spec = StageSpec(
        name="seg-eval", after=("segds",), optional=True, est_min=17,
        summary="zero-shot models over clean+bench frames, evaluated against the JVF pseudo-GT",
    )
    cli_args: tuple[str, ...] = ()

    def available(self, s) -> bool:
        """Same precondition as `segds`: no `[reference]` table means no pseudo-GT to evaluate
        against, so this stage is skipped rather than failed."""
        return s.reference is not None

    def inputs(self, s) -> dict[str, Path]:
        return {"clean_frames": s.workspace.clean_frames_json}

    def outputs(self, s) -> list[Path]:
        d = out_dir_default(s)
        return [d / "results.json", d / "tables.md"]

    def metrics(self, s) -> dict:
        try:
            return json.loads((out_dir_default(s) / "results.json").read_text())
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s, *, primary_tag: str = "eomt_city") -> None:
        """Mirrors the old `mapping.cli.pipeline` `seg-eval` stage: `primary_tag` (the model
        `stages.semantics.label` projects into the cloud) over every missing CLEAN frame -- that is
        what `label`'s stage depends on -- then every registered model over the missing BENCH
        frames, evaluated against the JVF pseudo-GT and reported.

        (The old driver also re-evaluated the fixed export-era 100-frame bench list into a second
        `export_frames` output, purely so a regression run stays comparable with the documented
        pre-refactor numbers. That comparison-only branch is not reproduced here.)"""
        frames = _frames_for_model("missing:clean", primary_tag)
        if frames:
            run_bench([primary_tag], frames)
        for tag in SPECS:
            frames = _frames_for_model("missing:bench", tag)
            if frames:
                run_bench([tag], frames)
        run(list(SPECS), _base_frames("bench", s), out_dir=out_dir_default(s))
        report(out_dir_default(s))


def run_bench(tags: list[str], frames: list[int]) -> None:
    from .bench import run as bench_run

    bench_run(tags, frames)


STAGE = registry.add(SegEval())


# ================================================================================================ cli
def main(argv=None) -> int:
    """Folds `mapping/cli/seg_bench.py` in.

    uv run python -m geovap.stages.semantics.segment.evaluate run --models all|tag,tag --frames bench|clean|1,2,3|missing:<spec>|<path.json> [--no-amp]
    uv run python -m geovap.stages.semantics.segment.evaluate evaluate [--models ...] [--frames bench|clean|1,2,3|<path.json>]
    uv run python -m geovap.stages.semantics.segment.evaluate report

    `--frames missing:<spec>` resolves `<spec>` the usual way and then, per model, drops any frame
    that already has a prediction, so `run` only recomputes what's missing.
    """
    import argparse

    from geovap.stages.base.cli import add_dataset_flags, configure_from

    ap = argparse.ArgumentParser(description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", default="all")
    r.add_argument("--frames", default="bench")
    r.add_argument("--no-amp", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--models", default="all")
    e.add_argument("--frames", default="bench")
    e.add_argument("--out", default=None, help="output dir for results.json + CSVs (default the workspace's dataset/seg/bench); use another dir to keep a second evaluation separate")
    sub.add_parser("report")
    add_dataset_flags(ap)
    a = ap.parse_args(argv)
    s = configure_from(a)

    if a.cmd == "run":
        from .bench import run as bench_run

        tags = list(SPECS) if a.models == "all" else a.models.split(",")
        for tag in tags:
            frames = _frames_for_model(a.frames, tag)
            if not frames:
                print(f"[{tag}] nothing to run ({a.frames}: 0 frames)")
                continue
            bench_run([tag], frames, amp=not a.no_amp)
    elif a.cmd == "evaluate":
        tags = list(SPECS) if a.models == "all" else a.models.split(",")
        run(tags, _base_frames(a.frames, s), out_dir=Path(a.out) if a.out else out_dir_default(s))
    elif a.cmd == "report":
        report(out_dir_default(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
