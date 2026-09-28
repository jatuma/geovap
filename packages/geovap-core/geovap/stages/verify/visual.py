#!/usr/bin/env python3
"""Headless screenshot validation of the consolidated Potree viewer (plan Part C6).

For a sample of clean frames, at several yaw angles, takes {cloud-only (opacity 0),
photo-only (opacity 1), blend (opacity 0.5)} screenshots for each panorama variant
(export vs corrected poses, AZ_OFFSET 0 vs 180), scores cloud/photo edge agreement with
`edges.edge_agreement`, builds contact sheets, and writes a numeric verdict on
AZ_OFFSET (and export vs corrected) to summary.json - independent of (and a numeric
second opinion on) `pose.py`'s own interpolation-error verdict.

Prefers the Playwright driver (`driver.js`, shipped by `geovap-deliver` at
`geovap/infra/shots/driver.js`, one page load per frame+variant, several shots via
window.__setCam/__setOpacity); falls back to plain
`chrome --headless=new --screenshot` (one URL load per shot, all params in the query
string) if playwright/node is unavailable.

Usage:
    uv run python -m geovap.stages.verify.visual \\
        --frames 12 --yaws 0,90,180,270 \\
        --variants cloud/panos,panos_corr180,panos_exp0,panos_exp180 \\
        [--base-url http://localhost:8080/pointclouds/consolidated/index.html] \\
        [--out-dir <workspace>/out/consolidated/validation] [--mode rgb] [--seed 0] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geovap.stages.base.cli import add_dataset_flags, configure_from, describe
from geovap.stages.base.spec import StageSpec, registry

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings

OPACITIES = {"cloud": 0.0, "photo": 1.0, "blend": 0.5}


def _driver_js() -> Path | None:
    """Path to the Playwright driver, shipped as data by `geovap-deliver`
    (`packages/geovap-deliver/geovap/infra/shots/driver.js`). Resolved via `importlib.resources`
    rather than a repo-relative `Path(__file__).parents[...]` -- this module has no reason to assume
    it is running from a checkout at all, only that `geovap-deliver` is installed (it is optional:
    a core-only install has no driver and falls back to the plain chrome screenshot below)."""
    try:
        import importlib.resources as res

        p = res.files("geovap.infra") / "shots" / "driver.js"
        return Path(str(p)) if p.is_file() else None
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        return None  # geovap-deliver not installed


def _driver_dir() -> Path | None:
    js = _driver_js()
    return js.parent if js is not None else None


def pick_frames(s: "Settings", n: int, seed: int) -> list[int]:
    frames = s.workspace.clean_frames()
    if n >= len(frames):
        return frames
    rnd = random.Random(seed)
    return sorted(rnd.sample(frames, n))


def build_jobs(frames, yaws, variants, base_url, mode, out_dir: Path, budget: int = 1_000_000, skip_existing: bool = True):
    """One job (page load) per (frame, variant); shots cover every yaw x opacity combo."""
    jobs = []
    shot_index = []  # parallel list of (frame, variant, yaw_deg, kind, rel_png_path)
    for frame in frames:
        for variant in variants:
            url = f"{base_url}{'&' if '?' in base_url else '?'}mode={mode}&panos={variant}&frame={frame}&nogui=1&budget={budget}"
            shots = []
            for yaw_deg in yaws:
                yaw_rad = yaw_deg * 3.141592653589793 / 180.0
                for kind, opacity in OPACITIES.items():
                    vname = variant.strip("/").replace("/", "_") or "panos"  # variants may be absolute URL paths
                    rel = f"{vname}/f{frame:04d}_y{int(yaw_deg):03d}_{kind}.png"
                    shot_index.append((frame, variant, yaw_deg, kind, rel))
                    if skip_existing and (out_dir / rel).exists():
                        continue  # resumable: keep the shot in the index (metrics/sheets) but do not re-shoot it
                    shots.append({"out": rel, "yaw": yaw_rad, "pitch": 0.0, "opacity": opacity})
            if shots:
                jobs.append({"url": url, "shots": shots})
    return jobs, shot_index


def _driver_available() -> bool:
    driver_dir = _driver_dir()
    if driver_dir is None:
        return False
    if not (driver_dir / "node_modules" / "playwright").exists():
        return False
    return shutil.which("node") is not None


def run_playwright(jobs, out_dir: Path, width: int, height: int) -> None:
    driver_js = _driver_js()
    jobs_path = out_dir / "_jobs.json"
    jobs_path.write_text(json.dumps(jobs))
    cmd = ["node", str(driver_js), "--jobs", str(jobs_path), "--out-dir", str(out_dir), "--width", str(width), "--height", str(height)]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, cwd=driver_js.parent, check=True)


def _chrome_binary() -> str | None:
    import os

    for cand in (os.environ.get("CHROME"), "google-chrome", "chromium", "chromium-browser"):
        if cand and shutil.which(cand):
            return shutil.which(cand)
    default = Path.home() / ".cache/ms-playwright/chromium-1234/chrome-linux64/chrome"
    return str(default) if default.exists() else None


def run_chrome_fallback(shot_index, base_url, mode, out_dir: Path, width: int, height: int) -> None:
    chrome = _chrome_binary()
    if chrome is None:
        raise RuntimeError("no chrome/chromium binary found (set CHROME env var)")
    for frame, variant, yaw_deg, kind, rel in shot_index:
        opacity = OPACITIES[kind]
        url = f"{base_url}{'&' if '?' in base_url else '?'}mode={mode}&panos={variant}&frame={frame}&yaw={yaw_deg * 3.141592653589793 / 180.0}&pano_opacity={opacity}&nogui=1"
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            chrome, "--headless=new", "--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
            f"--window-size={width},{height}", f"--screenshot={out_path}", "--virtual-time-budget=8000", url,
        ]
        subprocess.run(cmd, check=True)


def compute_edge_metrics(shot_index, out_dir: Path) -> list[dict]:
    from geovap.stages.verify.edges import edge_agreement, silhouette_agreement

    by_key = {}
    for frame, variant, yaw_deg, kind, rel in shot_index:
        by_key.setdefault((frame, variant, yaw_deg), {})[kind] = out_dir / rel

    rows = []
    for (frame, variant, yaw_deg), kinds in sorted(by_key.items()):
        cloud_png, photo_png = kinds.get("cloud"), kinds.get("photo")
        if not (cloud_png and photo_png and cloud_png.exists() and photo_png.exists()):
            continue
        m = edge_agreement(cloud_png, photo_png)  # raw Canny-vs-Canny (kept for reference; saturates on splat texture)
        sm = silhouette_agreement(cloud_png, photo_png)  # skyline vs photo edges: the decisive number
        rows.append({"frame": frame, "variant": variant, "yaw_deg": yaw_deg,
                     **{f"canny_{k}": v for k, v in m.items()}, **{f"sil_{k}": v for k, v in sm.items()}})
    return rows


def write_contact_sheets(shot_index, out_dir: Path) -> list[Path]:
    import cv2
    import numpy as np

    by_frame_variant = {}
    for frame, variant, yaw_deg, kind, rel in shot_index:
        by_frame_variant.setdefault((frame, variant), {})[(yaw_deg, kind)] = out_dir / rel

    sheets = []
    for (frame, variant), tiles in sorted(by_frame_variant.items()):
        imgs = []
        for key in sorted(tiles):
            p = tiles[key]
            if not p.exists():
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            small = cv2.resize(img, (320, 180))
            cv2.putText(small, f"y{key[0]} {key[1]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            imgs.append(small)
        if not imgs:
            continue
        cols = 3
        rows_of_imgs = [imgs[i : i + cols] for i in range(0, len(imgs), cols)]
        rows_of_imgs[-1] += [np.zeros_like(imgs[0])] * (cols - len(rows_of_imgs[-1]))
        sheet = np.vstack([np.hstack(r) for r in rows_of_imgs])
        sheet_path = out_dir / (variant.strip("/").replace("/", "_") or "panos") / f"contact_f{frame:04d}.png"
        cv2.imwrite(str(sheet_path), sheet)
        sheets.append(sheet_path)
    return sheets


def write_summary(edge_rows: list[dict], out_dir: Path) -> dict:
    import statistics

    per_variant = {}
    for row in edge_rows:
        per_variant.setdefault(row["variant"], []).append(row)

    summary = {"variants": {}}
    for variant, rows in per_variant.items():
        sil = [r["sil_median_px"] for r in rows if r.get("sil_median_px") == r.get("sil_median_px") and r.get("sil_median_px") is not None]
        sil_in = [r["sil_inlier6"] for r in rows if r.get("sil_inlier6") == r.get("sil_inlier6") and r.get("sil_inlier6") is not None]
        canny = [r["canny_median_px"] for r in rows if r.get("canny_median_px") == r.get("canny_median_px") and r.get("canny_median_px") is not None]
        summary["variants"][variant] = {
            "n_shots": len(rows),
            "n_valid": len(sil),
            "silhouette_median_px": statistics.median(sil) if sil else None,
            "silhouette_mean_px": (sum(sil) / len(sil)) if sil else None,
            "silhouette_inlier6": (sum(sil_in) / len(sil_in)) if sil_in else None,
            "canny_median_px": statistics.median(canny) if canny else None,
        }

    ranked = sorted(
        ((name, s["silhouette_median_px"]) for name, s in summary["variants"].items() if s["silhouette_median_px"] is not None),
        key=lambda kv: kv[1],
    )
    summary["az_offset_verdict"] = {
        "ranked_variants_best_first": [{"variant": n, "silhouette_median_px": m} for n, m in ranked],
        "best_variant": ranked[0][0] if ranked else None,
        "note": "silhouette_median_px = median distance (px @1600 wide, cap 40) from the cloud skyline to the nearest photo edge; lower = better. Compare cloud/panos (corrected, AZ 0) vs panos_corr180 (corrected, AZ 180) for the azimuth convention, and vs panos_exp0 (export poses) for export-vs-corrected; sphere.py is the decisive test for the convention.",
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def run_visual_check(
    s: "Settings",
    *,
    frames_arg: str = "12",
    yaws: str = "0,90,180,270",
    variants: str = "cloud/panos,panos_corr180,panos_exp0,panos_exp180",
    base_url: str = "http://localhost:8080/pointclouds/consolidated/index.html",
    out_dir: Path | None = None,
    mode: str = "rgb",
    width: int = 1600,
    height: int = 900,
    seed: int = 0,
    budget: int = 1_000_000,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    out_dir = out_dir if out_dir is not None else s.workspace.consolidated / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "," in frames_arg:
        frames = [int(x) for x in frames_arg.split(",") if x.strip()]
    else:
        frames = pick_frames(s, int(frames_arg), seed)
    yaw_list = [float(x) for x in yaws.split(",") if x.strip()]
    variant_list = [x.strip() for x in variants.split(",") if x.strip()]

    jobs, shot_index = build_jobs(frames, yaw_list, variant_list, base_url, mode, out_dir, budget=budget, skip_existing=not force)
    print(f"{len(frames)} frames x {len(variant_list)} variants x {len(yaw_list)} yaws x {len(OPACITIES)} opacities = {len(shot_index)} shots")

    if dry_run:
        (out_dir / "_jobs.json").write_text(json.dumps(jobs, indent=2))
        print(f"dry run: wrote {out_dir / '_jobs.json'}, no browser launched")
        return {"dry_run": True, "n_jobs": len(jobs)}

    if not jobs:
        print(f"all {len(shot_index)} screenshots already exist under {out_dir} (use --force to re-shoot)")
    elif _driver_available():
        run_playwright(jobs, out_dir, width, height)
    else:
        print("playwright driver not available (run `npm install` in the geovap-deliver `geovap/infra/shots/` dir); falling back to chrome --headless=new --screenshot")
        run_chrome_fallback(shot_index, base_url, mode, out_dir, width, height)

    edge_rows = compute_edge_metrics(shot_index, out_dir)
    csv_path = out_dir / "edge_metric.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(edge_rows[0].keys()) if edge_rows else ["frame", "variant", "yaw_deg"])
        w.writeheader()
        w.writerows(edge_rows)
    print(f"wrote {csv_path} ({len(edge_rows)} rows)")

    write_contact_sheets(shot_index, out_dir)
    summary = write_summary(edge_rows, out_dir)
    print(json.dumps(summary, indent=2))
    return summary


# ================================================================================================ stage
class Visual:
    spec = StageSpec(
        name="visual", after=("merge",), est_min=5,
        summary="headless screenshot cloud/photo edge agreement, AZ_OFFSET + export-vs-corrected verdict",
    )
    cli_args: tuple[str, ...] = ()

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        return {}

    def outputs(self, s: "Settings") -> list[Path]:
        return [s.workspace.consolidated / "validation" / "summary.json"]

    def metrics(self, s: "Settings") -> dict[str, Any]:
        try:
            p = s.workspace.consolidated / "validation" / "summary.json"
            return json.loads(p.read_text()).get("az_offset_verdict", {})
        except Exception as e:  # noqa: BLE001 - metrics must always be reportable
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", **opts: Any) -> None:
        run_visual_check(s, **opts)


STAGE = registry.add(Visual())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_flags(ap)
    ap.add_argument("--status", action="store_true", help="report whether this stage is done, then exit")
    ap.add_argument("--frames", default="12", help="number of random clean frames to sample, or a comma list of frame ids")
    ap.add_argument("--yaws", default="0,90,180,270", help="comma list of yaw angles in degrees")
    ap.add_argument("--variants", default="cloud/panos,panos_corr180,panos_exp0,panos_exp180", help="comma list of ?panos= dir names/paths")
    ap.add_argument("--base-url", default="http://localhost:8080/pointclouds/consolidated/index.html")
    ap.add_argument("--out-dir", default=None, help="default: <workspace>/out/consolidated/validation")
    ap.add_argument("--mode", default="rgb")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=1_000_000, help="Potree point budget for the screenshots")
    ap.add_argument("--force", action="store_true", help="re-shoot screenshots that already exist (default: skip them)")
    ap.add_argument("--dry-run", action="store_true", help="build jobs.json and print the plan without launching a browser")
    args = ap.parse_args(argv)
    s = configure_from(args)

    if args.status:
        print(describe(STAGE, s))
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else None
    run_visual_check(
        s, frames_arg=args.frames, yaws=args.yaws, variants=args.variants, base_url=args.base_url,
        out_dir=out_dir, mode=args.mode, width=args.width, height=args.height, seed=args.seed,
        budget=args.budget, force=args.force, dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
