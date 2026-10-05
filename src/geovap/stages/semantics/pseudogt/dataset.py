"""`segds`: build the pseudo-GT segmentation dataset (JVF areas -> rasters -> point/ERP labels ->
gnomonic views -> dataset metadata) and export it as one stage.

Small JSON files are versioned in `datasets/<name>/baseline/seg/` (the tracked reference for this
dataset); the live, per-run copy of the same metadata -- and the README next to the data -- are
written under the workspace (`s.workspace.derived / "seg"`), never into the repository: a re-run
against a different dataset or a corrected pose table must not overwrite the tracked baseline.

Folds `mapping/cli/seg_build.py` (areas/rasters/points/labels/views/dataset subcommands) into this
module's `main()`, and declares the `segds` `StageSpec` the driver resumes by marker file name --
unchanged from the old `mapping.cli.pipeline` stage of the same name, which chained exactly this
sequence of commands.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from geovap.runtime.store import open_store
from geovap.runtime.pose_tables import load as load_poses
from geovap.runtime.manifest import git_rev
from geovap.domain.scheme import classes as C
from geovap.stages.base.spec import StageSpec, registry
from . import areas, rasters, points, erp, views as views_mod, nearfield
from .areas import segds_dir
from .erp import bands_dir, labels_dir, clean_frames
from .views import VIEW_SIZE, VIEWS

RARE_MIN_FRAMES = 40
RARE_MIN_PX = 2000
MIN_CAM_DIST_M = 15.0
MIN_LABELLED_FRAC = 0.15
N_BENCH = 100


def dataset_dir(s=None) -> Path:
    """Per-run dataset metadata (`classes.json`, `splits.json`, `bench_frames.json`, `stats.json`,
    `id2label_*.json`): an OUTPUT, so it lives in the workspace, not in the tracked
    `datasets/<name>/baseline/seg/` this replaces as the default write target -- see module
    docstring. A function, not a constant: resolving it at import time is exactly what made the old
    code unable to pick a dataset from the command line."""
    from geovap.runtime import settings

    return (s or settings.get()).workspace.derived / "seg"


def quality_csv(s=None) -> Path:
    """`frame_quality.csv` (pass_id per frame): the live per-run copy if the screening stage has
    written one this run, else the dataset's tracked baseline -- same fallback shape as
    `Workspace.clean_frames`."""
    from geovap.runtime import settings

    s = s or settings.get()
    for p in (s.workspace.quality_csv, s.workspace.baseline / "frame_quality.csv"):
        if p.is_file():
            return p
    return s.workspace.quality_csv


def band_rows(zb_h: int) -> tuple[int, int]:
    """phi in [-55, +45] deg -> rows 250..805 at the default 1000-row z-buffer."""
    return (int(round((90 - 45) / 180 * zb_h)), int(round((90 + 55) / 180 * zb_h)))


def frame_stats(frames: list[int], rows: tuple[int, int]) -> dict[int, dict]:
    out = {}
    for k in frames:
        lab = cv2.imread(str(labels_dir() / f"f{k:04d}.png"), 0)
        band = lab[rows[0] : rows[1]]
        c_full = np.bincount(lab.ravel(), minlength=256)
        c_band = np.bincount(band.ravel(), minlength=256)
        bandpx = cv2.imread(str(bands_dir() / f"f{k:04d}.png"), 0)
        out[k] = {
            "counts_full": c_full[: C.N_CLASSES].tolist(),
            "counts_band": c_band[: C.N_CLASSES].tolist(),
            "labelled_frac_band": float(1 - c_band[255] / band.size),
            "band_px": np.bincount(bandpx.ravel(), minlength=8)[1:].tolist(),
        }
    return out


def camera_tiles(poses, frames: list[int]) -> dict[int, str]:
    store = open_store(poses=poses)
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


def build(out_dir: Path | None = None, poses_source: str | None = None) -> dict:
    from geovap.runtime import settings

    s = settings.get()
    out_dir = dataset_dir(s) if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = load_poses(poses_source)
    rows = band_rows(s.sensor.zb_h)
    frames = [k for k in clean_frames() if (labels_dir() / f"f{k:04d}.png").exists()]
    quality = {int(r["frame"]): r for r in csv.DictReader(quality_csv(s).open())}
    stats = frame_stats(frames, rows)
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
    (out_dir / "bench_frames.json").write_text(json.dumps({"n": len(bench), "frames": [b["frame"] for b in bench], "detail": bench, "band_rows": rows}, indent=0))
    nf = nearfield.load()
    nf_flag = {k: nearfield.is_bad(nf.get(k)) for k in frames}
    tot_band = np.sum([stats[k]["counts_band"] for k in frames], axis=0)
    tot_full = np.sum([stats[k]["counts_full"] for k in frames], axis=0)
    summary = {
        "n_frames": len(frames),
        "n_views_per_frame": len(VIEWS),
        "view_size": VIEW_SIZE,
        "band_rows": rows,
        "split_counts": {sn: sum(1 for v in splits["frames"].values() if v == sn) for sn in ("train", "val", "test", "buffer")},
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
    write_readme(segds_dir(s) / "README.md", summary, bench, s.name)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frame"}, indent=1)[:3000])
    return summary


def write_readme(path: Path, s: dict, bench: list[dict], dataset_name: str) -> None:
    lines = [
        f"# segds — sémantická segmentace, {dataset_name} (clean snímky)",
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
        "Metadata (verzovaná v `datasets/<name>/baseline/seg/`): `classes.json`, `splits.json`, `bench_frames.json`, `stats.json`.",
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


# ================================================================================================ stage
class Segds:
    spec = StageSpec(
        name="segds", after=("products",), optional=True, est_min=15,
        summary="JVF pseudo-GT dataset: areas/rasters/point labels/ERP labels/views/dataset metadata + nearfield check",
    )
    cli_args: tuple[str, ...] = ()

    def available(self, s) -> bool:
        """No `[reference]` table -> nothing to derive JVF pseudo-GT from, so the whole segds build
        is skipped (not failed) -- see `stages.base.spec.StageSpec.optional`."""
        return s.reference is not None

    def inputs(self, s) -> dict[str, Path]:
        return {"clean_frames": s.workspace.clean_frames_json}

    def outputs(self, s) -> list[Path]:
        d = segds_dir(s)
        return [d / "areas" / "report.json", d / "nearfield.json"]

    def metrics(self, s) -> dict:
        try:
            d = segds_dir(s)
            areas_rep = json.loads((d / "areas" / "report.json").read_text())
            nf = json.loads((d / "nearfield.json").read_text())
            return {"areas": {"n_faces": areas_rep.get("n_faces"), "all_lines": areas_rep.get("all_lines")}, "nearfield_summary": nf.get("summary")}
        except Exception as e:  # noqa: BLE001 - metrics must always be reportable
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s, *, workers: int = 8) -> None:
        """The exact sequence `mapping.cli.pipeline`'s old `segds` stage ran as separate
        subprocesses (areas -> rasters -> points -> labels -> views -> dataset -> nearfield ->
        dataset again, the last `dataset` re-run so its stats/splits/bench selection see the
        nearfield flags), now as one stage's `run()` -- a stage is already its own subprocess (see
        `stages.base.spec` module docstring), so there is no need to shell out to six more CLIs to
        get that isolation."""
        poses_source = s.pose_table
        faces, rep = areas.build(all_lines=True)
        print(f"areas: {rep['n_faces']} faces, conflicts {len(rep['conflicts'])}")
        rasters.build(poses_source=poses_source)
        points.build(workers=workers, poses_source=poses_source)
        erp.build(frames="clean", workers=min(workers, 10), poses_source=poses_source)
        views_mod.build(workers=min(workers, 10), poses_source=poses_source)
        build(poses_source=poses_source)
        nearfield.run(workers=workers, poses_source=poses_source)
        build(poses_source=poses_source)  # re-run: pick up nearfield flags in stats/bench selection


STAGE = registry.add(Segds())


# ================================================================================================ cli
def main(argv=None) -> int:
    """Folds `mapping/cli/seg_build.py` in: with a subcommand (areas/rasters/points/labels/views/
    dataset) it runs just that step against the resolved dataset flags (`--dataset`, `--poses`, ...);
    with none, it is a normal stage entry point (`stage_main`), i.e. the whole `segds` build."""
    import argparse

    from geovap.stages.base.cli import add_dataset_flags, configure_from, describe, stage_main

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("areas")
    a.add_argument("--hard-only", action="store_true", help="polygonize only hard boundary codes (default: all lines)")
    sub.add_parser("rasters")
    p = sub.add_parser("points")
    p.add_argument("--workers", type=int, default=4)
    l = sub.add_parser("labels")
    l.add_argument("--frames", default="clean", help="clean | all | comma list")
    l.add_argument("--workers", type=int, default=6)
    l.add_argument("--limit", type=int, default=None)
    v = sub.add_parser("views")
    v.add_argument("--workers", type=int, default=8)
    v.add_argument("--limit", type=int, default=None)
    sub.add_parser("dataset")
    add_dataset_flags(ap)
    ap.add_argument("--workers", type=int, default=8, help="workers for a full stage run (no subcommand)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)
    s = configure_from(args)

    if args.cmd is None:
        if not STAGE.available(s):
            if STAGE.spec.optional:
                print(f"{STAGE.spec.name}: unavailable for dataset {s.name!r}; skipping (stage is optional)")
                return 0
            ap.error(f"{STAGE.spec.name}: unavailable for dataset {s.name!r}")
        if args.status:
            print(describe(STAGE, s))
            return 0
        STAGE.run(s, workers=args.workers)
        return 0

    poses_source = args.poses
    if args.cmd == "areas":
        f, rep = areas.build(all_lines=not args.hard_only)
        print(json.dumps({k: v for k, v in rep.items() if k not in ("conflicts", "unresolved_largest")}, indent=1, ensure_ascii=False))
        print(f"conflicts: {len(rep['conflicts'])}, unresolved listed: {len(rep['unresolved_largest'])} -> {areas.areas_dir()}")
    elif args.cmd == "rasters":
        rasters.build(poses_source=poses_source)
    elif args.cmd == "points":
        points.build(workers=args.workers, poses_source=poses_source)
    elif args.cmd == "labels":
        erp.build(frames=args.frames, workers=args.workers, limit=args.limit, poses_source=poses_source)
    elif args.cmd == "views":
        views_mod.build(workers=args.workers, limit=args.limit, poses_source=poses_source)
    elif args.cmd == "dataset":
        build(poses_source=poses_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
