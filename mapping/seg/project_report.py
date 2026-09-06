"""Evaluation and renders for the 2D->3D label projection (mapping.seg.project).

evaluate()             3D confusion of the projected labels vs the JVF pseudo-GT point labels -> eval.json / eval.md
render_all()           ERP round-trip panels, bird's-eye and oblique renders -> out_dir/report/*.jpg
write_potree_classes() classes.json (names + colours of the common ids) next to the Potree octree
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .. import render
from ..cloud_store import CloudStore
from ..config import NO_POINT
from ..frame_select import FrameIndex
from ..poses import load_poses
from ..products import FrameProducts
from ..vehicle_mask import VehicleMask
from . import taxonomy as T
from .bench import DATASET_SEG_DIR
from .point_labels import LABEL_DIR as GT_LABEL_DIR
from .project import LAS_DIR, N_CLASSES, SEG_OUT_DIR, load_mask, mask_paths
from .render_labels import LABELS_DIR as GT_ERP_DIR, vehicle_cells

REPORT_DIR = SEG_OUT_DIR / "report"
POTREE_DIR = LAS_DIR.parents[1] / "eomt_city_seg"
PAL = T.common_palette()
CORE_IDS = [T.COMMON_ID[n] for n in T.CORE]
EXT_IDS = [T.COMMON_ID[n] for n in T.EXT]


class SegLabels:
    """Store-aligned reader of the projected labels; tiles without output read as 255."""

    def __init__(self, store: CloudStore, root: Path = SEG_OUT_DIR / "labels", suffix: str = ""):
        self.store = store
        self.arrays = []
        for t in store.tiles:
            p = Path(root) / f"{t.name}{suffix}.npy"
            self.arrays.append(np.load(p, mmap_mode="r") if p.exists() else None)

    def at(self, point_id: np.ndarray) -> np.ndarray:
        ti, local = self.store.locate(point_id)
        out = np.full(len(point_id), T.IGNORE, np.uint8)
        for t in np.unique(ti):
            a = self.arrays[t]
            if a is None:
                continue
            m = ti == t
            loc = local[m]
            order = np.argsort(loc)
            out[np.flatnonzero(m)[order]] = a[loc[order]]
        return out


# ------------------------------------------------------------------------------------------ evaluation
def _iou(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, tp / den, np.nan)


def evaluate(out_dir: Path = SEG_OUT_DIR) -> dict:
    store = CloudStore()
    lut = T.gt_to_common_lut()
    cm = np.zeros((N_CLASSES, N_CLASSES), np.float64)
    cm_ground = np.zeros_like(cm)
    counts = np.zeros(256, np.int64)
    gt_counts = np.zeros(256, np.int64)
    nviews = np.zeros(256, np.int64)
    conf_hist = np.zeros(256, np.int64)
    conf_correct = np.zeros(256, np.int64)  # correct predictions per conf bin (calibration)
    conf_eval = np.zeros(256, np.int64)
    n_total = 0
    tiles = []
    for t in store.tiles:
        p = Path(out_dir) / "labels" / f"{t.name}.npy"
        if not p.exists():
            continue
        pred = np.load(p)
        conf = np.load(Path(out_dir) / "labels" / f"{t.name}_conf.npy")
        nv = np.load(Path(out_dir) / "labels" / f"{t.name}_nviews.npy")
        gt = lut[np.load(GT_LABEL_DIR / f"{t.name}.npy", mmap_mode="r")]
        td = store.tile(t.name)
        src = np.asarray(td.classification)
        store.release()
        n_total += len(pred)
        counts += np.bincount(pred, minlength=256)
        gt_counts += np.bincount(gt, minlength=256)
        nviews += np.bincount(nv, minlength=256)
        conf_hist += np.bincount(conf[pred != T.IGNORE], minlength=256)
        m = (gt != T.IGNORE) & (pred != T.IGNORE)
        idx = gt[m].astype(np.int64) * N_CLASSES + pred[m]
        cm_t = np.bincount(idx, minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
        cm += cm_t
        mg = m & (src == 2)
        cm_ground += np.bincount(gt[mg].astype(np.int64) * N_CLASSES + pred[mg], minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)
        ok = gt[m] == pred[m]
        conf_eval += np.bincount(conf[m], minlength=256)
        conf_correct += np.bincount(conf[m][ok], minlength=256)
        iou_t = _iou(cm_t)
        tiles.append({"tile": t.name, "n": int(len(pred)), "coverage": float((pred != T.IGNORE).mean()), "n_eval": int(m.sum()),
                      "acc": float(ok.mean()) if m.any() else None, "miou_core": float(np.nanmean(iou_t[CORE_IDS])) if m.any() else None})
        print(f"tile {t.name}: coverage {tiles[-1]['coverage']:.3f}, acc {tiles[-1]['acc']}, mIoU_core {tiles[-1]['miou_core']}")
    iou = _iou(cm)
    iou_g = _iou(cm_ground)
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    res = {
        "n_points": int(n_total), "n_tiles": len(tiles), "coverage": float(counts[:N_CLASSES].sum() / max(n_total, 1)),
        "n_eval": float(cm.sum()), "pixel_acc": acc,
        "miou_core": float(np.nanmean(iou[CORE_IDS])), "miou_ext": float(np.nanmean(iou[EXT_IDS])),
        "miou_core_ground": float(np.nanmean(iou_g[CORE_IDS])), "acc_ground": float(np.trace(cm_ground) / max(cm_ground.sum(), 1)),
        "iou": {T.COMMON[i]: (None if np.isnan(iou[i]) else float(iou[i])) for i in range(N_CLASSES)},
        "iou_ground": {T.COMMON[i]: (None if np.isnan(iou_g[i]) else float(iou_g[i])) for i in range(N_CLASSES)},
        "pred_counts": {T.COMMON[i]: int(counts[i]) for i in range(N_CLASSES)}, "unlabelled": int(counts[T.IGNORE]),
        "gt_counts": {T.COMMON[i]: int(gt_counts[i]) for i in range(N_CLASSES)}, "gt_ignore": int(gt_counts[T.IGNORE]),
        "confusion": cm.astype(np.int64).tolist(), "classes": T.COMMON,
        "nviews_hist": nviews[:64].tolist(), "conf_hist": conf_hist.tolist(),
        "conf_calibration": {"n": conf_eval.tolist(), "correct": conf_correct.tolist()},
        "tiles": tiles,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "eval.json").write_text(json.dumps(res, indent=1))
    md = ["| class | IoU (all) | IoU (ground pts) | pred points | GT points |", "|---|---|---|---|---|"]
    for i, n in enumerate(T.COMMON):
        f = lambda x: "–" if x is None or np.isnan(x) else f"{x:.3f}"
        md.append(f"| {n} | {f(iou[i])} | {f(iou_g[i])} | {counts[i]:,} | {gt_counts[i]:,} |")
    md.append(f"\ncoverage {res['coverage']:.3f}, pixel acc {acc:.3f}, mIoU_core {res['miou_core']:.3f}, mIoU_ext {res['miou_ext']:.3f}, evaluated points {cm.sum():,.0f}")
    (Path(out_dir) / "eval.md").write_text("\n".join(md))
    print("\n".join(md))
    return res


# ------------------------------------------------------------------------------------------ renders
def _overlay(photo_bgr: np.ndarray, lab: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    valid = lab != T.IGNORE
    return render.overlay_on_photo(photo_bgr, PAL[lab][..., ::-1], valid, alpha)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (12 + 17 * len(text), 36), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def erp_roundtrip(k: int, store: CloudStore, sl: SegLabels, fi: FrameIndex, poses, vm_cells: np.ndarray, tag: str, out: Path, crop: tuple[float, float] = (0.2, 0.8)) -> Path:
    """photo | EoMT mask | labels re-rendered from the cloud | JVF pseudo-GT, cropped to the elevation band `crop`."""
    from .render_labels import gather_labels

    photo = cv2.imread(poses.path(k))
    fp = FrameProducts.load(k)
    common, _conf, _ = load_mask(tag, k, 0)
    cloud_lab, valid = gather_labels(store, sl, fp)
    cloud_lab[~valid] = T.IGNORE
    cloud_lab[vm_cells] = T.IGNORE
    cloud_lab[valid & (fp.depth_m > 40)] = T.IGNORE
    gt_p = GT_ERP_DIR / f"f{k:04d}.png"
    gt = T.gt_to_common_lut()[cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE)] if gt_p.exists() else np.full(common.shape, T.IGNORE, np.uint8)
    h = common.shape[0]
    a, b = int(crop[0] * h), int(crop[1] * h)
    small = cv2.resize(photo, (common.shape[1], h), interpolation=cv2.INTER_AREA)
    panels = [
        _label(small[a:b].copy(), f"frame {k}: photo"),
        _label(_overlay(photo, common)[a:b], "EoMT-L prediction (2D)"),
        _label(_overlay(photo, cloud_lab)[a:b], "labels projected to cloud, re-rendered"),
        _label(_overlay(photo, gt)[a:b], "JVF pseudo-GT (points)"),
    ]
    grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    p = out / f"erp_f{k:04d}.jpg"
    cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
    store.release()
    return p


def _splat(img: np.ndarray, ix: np.ndarray, iy: np.ndarray, col: np.ndarray, order: np.ndarray, size: int = 2) -> None:
    h, w = img.shape[:2]
    for dy in range(size):
        for dx in range(size):
            x = ix[order] + dx
            y = iy[order] + dy
            m = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            img[y[m], x[m]] = col[order][m]


def shade(label_rgb: np.ndarray, ref_rgb: np.ndarray, k: float = 0.45) -> np.ndarray:
    """Modulate class colours by the reference luminance so geometry stays readable."""
    lum = (0.299 * ref_rgb[:, 0] + 0.587 * ref_rgb[:, 1] + 0.114 * ref_rgb[:, 2]) / 255.0
    f = (1 - k) + k * (lum[:, None] * 1.6)
    return np.clip(label_rgb.astype(np.float32) * f, 0, 255).astype(np.uint8)


def bev(store: CloudStore, sl_root: Path, tile_name: str, out: Path, res_m: float = 0.1) -> Path:
    """Bird's-eye view: prediction | JVF pseudo-GT | confidence, highest point wins."""
    td = store.tile(tile_name)
    xyz = td.xyz_m()
    ref = np.asarray(td.rgb)
    pred = np.load(sl_root / f"{tile_name}.npy")
    conf = np.load(sl_root / f"{tile_name}_conf.npy")
    gt = T.gt_to_common_lut()[np.load(GT_LABEL_DIR / f"{tile_name}.npy", mmap_mode="r")]
    store.release()
    minE, minN, maxE, maxN = td.info.bbox
    w, h = int((maxE - minE) / res_m) + 1, int((maxN - minN) / res_m) + 1
    ix = ((xyz[:, 0] - minE) / res_m).astype(np.int64)
    iy = ((maxN - xyz[:, 1]) / res_m).astype(np.int64)
    order = np.argsort(xyz[:, 2])  # highest last -> wins
    panels = []
    for name, col in (("EoMT-L labels in 3D", shade(PAL[pred], ref)), ("JVF pseudo-GT", shade(PAL[gt], ref)),
                      ("vote share (dark = uncertain)", cv2.applyColorMap(conf, cv2.COLORMAP_VIRIDIS)[:, 0, ::-1] * (pred != T.IGNORE)[:, None])):
        img = np.full((h, w, 3), 24, np.uint8)
        _splat(img, ix, iy, col, order, 1)
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))
        panels.append(_label(img[..., ::-1].copy(), f"tile {tile_name}: {name}"))
    grid = np.hstack(panels)
    if grid.shape[1] > 3600:
        s = 3600 / grid.shape[1]
        grid = cv2.resize(grid, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    p = out / f"bev_{tile_name}.jpg"
    cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return p


def oblique(xyz: np.ndarray, col_rgb: np.ndarray, center: np.ndarray, yaw_deg: float, tilt_deg: float, radius: float, size=(1600, 900), splat: int = 2) -> np.ndarray:
    """Orthographic painter's render looking at `center` from azimuth `yaw_deg`, `tilt_deg` above the horizon."""
    d2 = (xyz[:, 0] - center[0]) ** 2 + (xyz[:, 1] - center[1]) ** 2
    m = d2 <= radius**2
    P = xyz[m] - center
    c = col_rgb[m]
    ya, ti = np.deg2rad(yaw_deg), np.deg2rad(tilt_deg)
    # view axes: forward f (towards the scene), right r, up u
    f = np.array([np.cos(ya) * np.cos(ti), np.sin(ya) * np.cos(ti), -np.sin(ti)])
    r = np.array([-np.sin(ya), np.cos(ya), 0.0])
    u = np.cross(r, f)
    x = P @ r
    y = P @ u
    z = P @ f  # depth along the view
    W, H = size
    scale = min(W / (2.2 * radius), H / (1.4 * radius))
    ix = (W / 2 + x * scale).astype(np.int64)
    iy = (H * 0.62 - y * scale).astype(np.int64)
    order = np.argsort(-z)  # far first, near overwrites
    img = np.full((H, W, 3), 24, np.uint8)
    _splat(img, ix, iy, c, order, splat)
    return img


def _crop_pair(a: np.ndarray, b: np.ndarray, margin: int = 24, bg: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Crop both renders to the bounding box of the drawn points (same box for both)."""
    drawn = (a != bg).any(2) | (b != bg).any(2)
    ys, xs = np.nonzero(drawn)
    if len(ys) == 0:
        return a, b
    y0, y1 = max(0, ys.min() - margin), min(a.shape[0], ys.max() + margin)
    x0, x1 = max(0, xs.min() - margin), min(a.shape[1], xs.max() + margin)
    y0 = min(y0, y1 - 60)  # keep room for the caption
    return a[y0:y1, x0:x1], b[y0:y1, x0:x1]


def oblique_views(store: CloudStore, sl_root: Path, fi: FrameIndex, tile_name: str, out: Path, frames: list[int], radius: float = 45.0) -> list[Path]:
    td = store.tile(tile_name)
    xyz = td.xyz_m()
    ref = np.asarray(td.rgb)
    pred = np.load(sl_root / f"{tile_name}.npy")
    store.release()
    col = shade(PAL[pred], ref)
    paths = []
    for k in frames:
        C = fi.C[k]
        yaw = float(fi.poses.yaw[k]) + 35.0  # look roughly along the road, offset to the side
        img = oblique(xyz, col, C, yaw, 32.0, radius)
        img_ref = oblique(xyz, ref, C, yaw, 32.0, radius)
        img, img_ref = _crop_pair(img, img_ref)
        grid = np.vstack([_label(img[..., ::-1].copy(), f"tile {tile_name}, around frame {k}: EoMT-L labels"), _label(img_ref[..., ::-1].copy(), "reference RGB")])
        p = out / f"oblique_{tile_name}_f{k:04d}.jpg"
        cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(p)
    return paths


def _pick_frames(tag: str, n_pick: int = 6) -> list[int]:
    """Bench frames with the most building/wall/fence pixels + a few evenly spaced others."""
    frames = json.loads((DATASET_SEG_DIR / "bench_frames.json").read_text())["frames"]
    ids = [T.COMMON_ID[x] for x in ("building", "wall", "fence", "sidewalk")]
    score = []
    for k in frames:
        pc, _ = mask_paths(tag, k)
        if not pc.exists():
            continue
        m = cv2.imread(str(pc), cv2.IMREAD_GRAYSCALE)
        score.append((float(np.isin(m, ids).mean()), k))
    score.sort(reverse=True)
    top = [k for _, k in score[: n_pick // 2]]
    rest = [k for _, k in score[n_pick // 2 :]]
    step = max(1, len(rest) // (n_pick - len(top)))
    return sorted(set(top + rest[::step][: n_pick - len(top)]))


def _pick_tiles(sl_root: Path, n_pick: int = 3) -> list[str]:
    metas = []
    for p in sorted(sl_root.glob("*_meta.json")):
        m = json.loads(p.read_text())
        c = np.array(m["counts"], float)
        metas.append((p.name[:3], c))
    if not metas:
        return []
    bid = T.COMMON_ID["building"]
    by_building = sorted(metas, key=lambda x: -x[1][bid])
    picks = [nm for nm, _ in by_building[: n_pick - 1]]
    by_road = sorted(metas, key=lambda x: -x[1][T.COMMON_ID["road"]])
    for nm, _ in by_road:
        if nm not in picks:
            picks.append(nm)
            break
    return picks[:n_pick]


def render_all(frames: list[int] | None = None, tiles: list[str] | None = None, tag: str = "eomt_city", out_dir: Path = SEG_OUT_DIR) -> dict:
    out = Path(out_dir) / "report"
    out.mkdir(parents=True, exist_ok=True)
    sl_root = Path(out_dir) / "labels"
    store = CloudStore()
    poses = load_poses()
    fi = FrameIndex(poses)
    sl = SegLabels(store, sl_root)
    vm_cells = vehicle_cells(VehicleMask())
    frames = frames or _pick_frames(tag)
    tiles = tiles or _pick_tiles(sl_root)
    made = {"erp": [], "bev": [], "oblique": []}
    for k in frames:
        made["erp"].append(str(erp_roundtrip(k, store, sl, fi, poses, vm_cells, tag, out)))
        print("erp", k)
    for tn in tiles:
        made["bev"].append(str(bev(store, sl_root, tn, out)))
        print("bev", tn)
        info = store.by_name[tn]
        ks = [int(k) for k in fi.frames_in_bbox(info.bbox, margin=-20.0) if int(k) in set(frames)] or [int(k) for k in fi.frames_in_bbox(info.bbox, margin=-20.0)[::max(1, len(fi.frames_in_bbox(info.bbox, margin=-20.0)) // 2)][:2]]
        made["oblique"] += [str(p) for p in oblique_views(store, sl_root, fi, tn, out, ks[:2])]
        print("oblique", tn, ks[:2])
    (out / "index.json").write_text(json.dumps({"frames": frames, "tiles": tiles, **made}, indent=1))
    return made


# ------------------------------------------------------------------------------------------ potree
def write_potree_classes(potree_dir: Path | None = None) -> Path:
    d = Path(potree_dir) if potree_dir else POTREE_DIR
    d.mkdir(parents=True, exist_ok=True)
    cls = {str(i): {"name": n, "color": [*T.COMMON_COLOURS[n], 255], "visible": n != "sky"} for i, n in enumerate(T.COMMON)}
    cls["255"] = {"name": "unlabelled", "color": [60, 60, 60, 255], "visible": True}
    cls["DEFAULT"] = {"name": "other id", "color": [0, 0, 0, 255], "visible": True}
    p = d / "classes.json"
    p.write_text(json.dumps({"attribute": "classification", "classes": cls}, indent=1))
    return p
