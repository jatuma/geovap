"""Dataset metadata: per-frame class statistics, spatial splits, benchmark subset, classes.json, README.

Small JSON files are versioned in geovap/dataset/seg/, the README next to the data in Geovap_cache/segds/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ..cloud_store import CloudStore
from ..config import ZB_H
from ..poses import load_poses
from ..products import git_rev
from . import classes as C
from .areas import SEGDS_DIR
from .render_labels import BANDS_DIR, LABELS_DIR, clean_frames
from .views import VIEW_SIZE, VIEWS
from . import nearfield

DATASET_SEG_DIR = Path(__file__).resolve().parents[2] / "dataset" / "seg"
QUALITY_CSV = Path(__file__).resolve().parents[2] / "dataset" / "frame_quality.csv"
BAND_ROWS = (int(round((90 - 45) / 180 * ZB_H)), int(round((90 + 55) / 180 * ZB_H)))  # phi in [-55, +45] deg -> rows 250..805
N_BENCH = 100
RARE_MIN_FRAMES = 40
RARE_MIN_PX = 2000
MIN_CAM_DIST_M = 15.0
MIN_LABELLED_FRAC = 0.15


def frame_stats(frames: list[int]) -> dict[int, dict]:
    out = {}
    for k in frames:
        lab = cv2.imread(str(LABELS_DIR / f"f{k:04d}.png"), 0)
        band = lab[BAND_ROWS[0] : BAND_ROWS[1]]
        c_full = np.bincount(lab.ravel(), minlength=256)
        c_band = np.bincount(band.ravel(), minlength=256)
        bands = cv2.imread(str(BANDS_DIR / f"f{k:04d}.png"), 0)
        out[k] = {
            "counts_full": c_full[: C.N_CLASSES].tolist(),
            "counts_band": c_band[: C.N_CLASSES].tolist(),
            "labelled_frac_band": float(1 - c_band[255] / band.size),
            "band_px": np.bincount(bands.ravel(), minlength=8)[1:].tolist(),
        }
    return out


def camera_tiles(poses, frames: list[int]) -> dict[int, str]:
    store = CloudStore()
    tiles = {}
    for k in frames:
        e, n = poses.origin[k][:2]
        best, bd = None, np.inf
        for t in store.tiles:
            minE, minN, maxE, maxN = t.bbox
            d = max(minE - e, 0, e - maxE) ** 2 + max(minN - n, 0, n - maxN) ** 2
            if d < bd:
                best, bd = t.name, d
        tiles[k] = best
    return tiles


def make_splits(poses, frames: list[int], tiles: dict[int, str], seed: int = 0, k: int = 10, buffer_m: float = 20.0) -> dict:
    """Cameras clustered spatially (k-means on camera E,N) -> contiguous road segments; clusters are assigned to
    test and val until each holds ~15 % of the frames, the rest is train. Frames closer than `buffer_m` to a
    camera of another split become `buffer` (excluded from val/test evaluation)."""
    from scipy.cluster.vq import kmeans2

    xy = np.array([poses.origin[k_][:2] for k_ in frames])
    _, lab = kmeans2(xy, k, seed=seed, minit="++")
    sizes = {c: int((lab == c).sum()) for c in range(k)}
    order = sorted(sizes, key=lambda c: sizes[c])
    target = 0.15 * len(frames)
    role_of_cluster = {}
    for role in ("test", "val"):
        acc = 0
        while order and acc < target:
            c = order.pop(0)
            role_of_cluster[c] = role
            acc += sizes[c]
    for c in order:
        role_of_cluster[c] = "train"
    role = np.array([role_of_cluster[int(c)] for c in lab])
    split = {}
    for i, k_ in enumerate(frames):
        d = np.hypot(*(xy - xy[i]).T)
        foreign = (role != role[i]) & (d < buffer_m)
        split[k_] = "buffer" if foreign.any() else str(role[i])
    clusters = {int(c): {"role": role_of_cluster[c], "n": sizes[c], "centre": xy[lab == c].mean(0).round(1).tolist()} for c in range(k)}
    return {"clusters": clusters, "frames": split}


def select_bench_frames(poses, frames: list[int], stats: dict, tiles: dict[int, str], n: int = N_BENCH, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    ok = [k for k in frames if stats[k]["labelled_frac_band"] >= MIN_LABELLED_FRAC]
    chosen: list[dict] = []
    pos = lambda k: poses.origin[k][:2]

    def far_enough(k):
        return all(np.hypot(*(pos(k) - pos(c["frame"]))) >= MIN_CAM_DIST_M for c in chosen)

    # rare classes: fewer than RARE_MIN_FRAMES frames with >= RARE_MIN_PX band pixels
    has = {c.id: [k for k in ok if stats[k]["counts_band"][c.id] >= RARE_MIN_PX] for c in C.CLASSES if c.eval != "diag"}
    rare = [cid for cid, ks in has.items() if 0 < len(ks) < RARE_MIN_FRAMES]
    for cid in rare:
        cands = sorted(has[cid], key=lambda k: -stats[k]["counts_band"][cid])
        taken = 0
        for k in cands:
            if taken >= 3:
                break
            if k not in {c["frame"] for c in chosen} and far_enough(k):
                chosen.append({"frame": k, "reason": f"rare:{C.BY_ID[cid].name}"})
                taken += 1
    # fill by tile allocation ~ sqrt(n_clean_tile)
    per_tile = {}
    for k in ok:
        per_tile.setdefault(tiles[k], []).append(k)
    weights = {t: np.sqrt(len(ks)) for t, ks in per_tile.items() if len(ks) >= 5}
    wsum = sum(weights.values())
    remaining = n - len(chosen)
    alloc = {t: max(1, int(round(remaining * w / wsum))) for t, w in weights.items()}
    for t in sorted(alloc, key=lambda t: -alloc[t]):
        ks = list(per_tile[t])
        rng.shuffle(ks)
        got = 0
        for k in ks:
            if len(chosen) >= n or got >= alloc[t]:
                break
            if k not in {c["frame"] for c in chosen} and far_enough(k):
                chosen.append({"frame": k, "reason": f"tile:{t}"})
                got += 1
    # top up if rounding left a gap
    pool = [k for k in ok if k not in {c["frame"] for c in chosen}]
    rng.shuffle(pool)
    for k in pool:
        if len(chosen) >= n:
            break
        if far_enough(k):
            chosen.append({"frame": k, "reason": "fill"})
    chosen.sort(key=lambda c: c["frame"])
    return chosen


def build(out_dir: Path = DATASET_SEG_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = load_poses()
    frames = [k for k in clean_frames() if (LABELS_DIR / f"f{k:04d}.png").exists()]
    quality = {int(r["frame"]): r for r in csv.DictReader(QUALITY_CSV.open())}
    stats = frame_stats(frames)
    tiles = camera_tiles(poses, frames)
    splits = make_splits(poses, frames, tiles)
    bench = select_bench_frames(poses, frames, stats, tiles)
    nf_all = nearfield.load()
    for b in bench:
        b["nearfield_px"] = (nf_all.get(b["frame"]) or {}).get("median_px")
        b["nearfield_bad"] = nearfield.is_bad(nf_all.get(b["frame"]))
        b["tile"] = tiles[b["frame"]]
        b["pass"] = int(quality[b["frame"]]["pass_id"])
        b["split"] = splits["frames"][b["frame"]]

    classes_json = {
        "ignore": C.IGNORE,
        "classes": [{"id": c.id, "name": c.name, "czech": c.czech, "colour": list(c.colour), "eval": c.eval} for c in C.CLASSES],
        "area_class": C.AREA_CLASS,
        "rules": C.RULES,
        "rules_hash": C.rules_hash(),
    }
    (out_dir / "classes.json").write_text(json.dumps(classes_json, indent=1, ensure_ascii=False))
    (out_dir / "splits.json").write_text(json.dumps({"method": "k-means(10) on camera positions; test/val ~15 % of clean frames each; 20 m camera buffer", "clusters": splits["clusters"], "frames": {str(k): v for k, v in splits["frames"].items()}}, indent=0))
    (out_dir / "bench_frames.json").write_text(json.dumps({"n": len(bench), "frames": [b["frame"] for b in bench], "detail": bench, "band_rows": BAND_ROWS}, indent=0))
    nf = nearfield.load()
    nf_flag = {k: nearfield.is_bad(nf.get(k)) for k in frames}
    tot_band = np.sum([stats[k]["counts_band"] for k in frames], axis=0)
    tot_full = np.sum([stats[k]["counts_full"] for k in frames], axis=0)
    summary = {
        "n_frames": len(frames),
        "n_views_per_frame": len(VIEWS),
        "view_size": VIEW_SIZE,
        "band_rows": BAND_ROWS,
        "split_counts": {s: sum(1 for v in splits["frames"].values() if v == s) for s in ("train", "val", "test", "buffer")},
        "pixels_band_by_class": {c.name: int(tot_band[c.id]) for c in C.CLASSES},
        "pixels_full_by_class": {c.name: int(tot_full[c.id]) for c in C.CLASSES},
        "frames_with_class_band": {c.name: int(sum(1 for k in frames if stats[k]["counts_band"][c.id] >= RARE_MIN_PX)) for c in C.CLASSES},
        "labelled_frac_band_median": float(np.median([stats[k]["labelled_frac_band"] for k in frames])),
        "frames_with_no_labels": [k for k in frames if stats[k]["labelled_frac_band"] < 0.01],
        "nearfield": {"flag_px": nearfield.FLAG_PX, "bad": sum(1 for v in nf_flag.values() if v is True), "ok": sum(1 for v in nf_flag.values() if v is False), "unmeasured": sum(1 for v in nf_flag.values() if v is None)},
        "git": git_rev(),
        "per_frame": {str(k): {"tile": tiles[k], "pass": int(quality[k]["pass_id"]), "split": splits["frames"][k], "labelled_frac_band": round(stats[k]["labelled_frac_band"], 4), "nearfield_px": (nf.get(k) or {}).get("median_px"), "nearfield_bad": nf_flag[k], "counts_band": stats[k]["counts_band"], "band_px": stats[k]["band_px"]} for k in frames},
    }
    (out_dir / "stats.json").write_text(json.dumps(summary, indent=0))
    write_readme(SEGDS_DIR / "README.md", summary, bench)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frame"}, indent=1)[:3000])
    return summary


def write_readme(path: Path, s: dict, bench: list[dict]) -> None:
    lines = [
        "# segds — sémantická segmentace, Dražkov (clean snímky)",
        "",
        f"Snímků: {s['n_frames']} (třída `clean` z `dataset/clean_frames.json`), {s['n_views_per_frame']} výsečí {s['view_size']}² na snímek.",
        "Zdroj značek: JVF DTM export (plochy z polygonizace hraničních linií + definiční body, 3D pravidla nad mračnem),",
        "vykreslené do panoramatu přes `point_id` produkty (okluze zdarma). Značky se nikdy neinterpolují; 255 = ignore.",
        "",
        "```",
        "areas/          faces.geojson, report.json, map.png        polygonizované JVF plochy a jejich třídy",
        "rasters/        0,1 m rastry ploch a linií, 0,5 m DTM",
        "point_labels/   NN.npy  uint8 značka každého bodu store (pořadí řádků = store)",
        "labels_erp/     f%04d.png  2000x1000 uint8 (ERP, id třídy, 255 ignore)",
        "bands_erp/      f%04d.png  pásy 14 cm kolem VIDITELNÝCH JVF linií (jen pro evaluaci, ne trénink)",
        "qa/             f%04d.jpg  překryv na fotce",
        "views/images/   f%04d_y{yaw}_p{pitch}.jpg  gnómonické výseče 1024², srovnané do horizontu",
        "views/labels/   f%04d_y{yaw}_p{pitch}.png  totéž pro značky (nearest)",
        "bench/<model>/  f%04d_{native,common,conf}.png  fúzované predikce modelů",
        "```",
        "",
        "Metadata (verzovaná v `geovap/dataset/seg/`): `classes.json`, `splits.json`, `bench_frames.json`, `stats.json`.",
        "",
        "## Známé slabiny značek",
        "- Střechy za bezlistými stromy: lidar vidí skrz větve, fotka ne → střecha je označená tam, kde fotka ukazuje větve.",
        "- Vegetace je odvozená (body nad zemí v plochách zeleně/zahrad); sloupy a předměty v zeleni dostanou `vegetation`.",
        "- Hlavní ulice nemá v JVF uzavřenou hranici vozovka/krajnice → třída `road_or_verge` (diagnostická, mimo mIoU).",
        "- Bez značek: obloha, vozidla, lidé, vše dál než 40 m a mimo pokrytí JVF (otevřené pole).",
        f"- Snímků bez značek v pásu φ∈[−55°,+45°]: {len(s['frames_with_no_labels'])}.",
        f"- **Blízké pole**: kritérium `clean` (siluety v 10–40 m) nevidí chyby pozice/času kamery, které posunou zem ve 3–10 m o stupně. `nearfield.json` (medián vzdálenosti promítnuté JVF hranice komunikace k hraně fotky, px @2000): >{s['nearfield']['flag_px']} px = `nearfield_bad` u {s['nearfield']['bad']} snímků, ok {s['nearfield']['ok']}, neměřitelné {s['nearfield']['unmeasured']}. Značky v takových snímcích jsou v blízkém poli posunuté proti fotce.",
        "",
        f"Splity (prostorové bloky dlaždic): {s['split_counts']}. Benchmark: {len(bench)} snímků (`bench_frames.json`).",
        f"Git: {s['git']}.",
    ]
    path.write_text("\n".join(lines) + "\n")
