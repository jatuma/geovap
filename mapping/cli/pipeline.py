"""Consolidated driver: pozy -> obarveni -> dataset -> segmentace -> 3D produkt -> vizualizace.

    uv run python -m mapping.cli.pipeline run [--from S] [--to S] [--only a,b] [--force]
                                               [--poses corrected] [--tag tw45] [--dry-run]
                                               [--detach] [--with-optional]
    uv run python -m mapping.cli.pipeline status
    uv run python -m mapping.cli.pipeline compare
    uv run python -m mapping.cli.pipeline env-check [--fix-symlink]

Each stage is a fixed list of subprocess commands (`Stage.commands(ctx)`), run one at a time as
`["uv","run","python","-u",...]` (cwd=REPO_ROOT, env GEOVAP_CACHE/GEOVAP_POSES/PYTHONUNBUFFERED set)
so a stage's memmaps/CUDA context are released between steps (see plan Part B). Progress is recorded
as `out/pipeline/<stage>.json` markers (inputs_hash/outputs/metrics) and `out/pipeline/pipeline.log` +
`out/pipeline/logs/<stage>_<n>.log`; `is_done` short-circuits a stage whose marker is green, whose
declared inputs still hash the same, and whose declared outputs still exist.

Everything here is a thin, testable orchestration layer: it never computes geometry itself, it only
shells out to the CLIs implemented elsewhere in `mapping/` and `pointcloud-tools/validate/` and reads
their output files back for the marker's `metrics`. Stage bodies are intentionally simple/defensive:
a metrics reader that can't find its inputs yet (e.g. a stage run out of order, or against a checkout
with no data) returns `{"error": str(e)}` rather than raising, so a marker is always writable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .. import config
from ..config import (
    CACHE_ROOT,
    CLEAN_FRAMES_JSON,
    CONSOLIDATED_DIR,
    EXPORT_CSV,
    FRAMES_DIR,
    OUT_DIR,
    PIPELINE_DIR,
    POSES_DIR,
    POTREE_OUTPUT_DIR,
    REPO_ROOT,
    STORE_DIR,
)

DATASET_DIR = config.BASELINE_DIR
LOGS_DIR = PIPELINE_DIR / "logs"
BASELINE_DIR = PIPELINE_DIR / "baseline"
PIPELINE_LOG = PIPELINE_DIR / "pipeline.log"
PID_FILE = PIPELINE_DIR / "pipeline.pid"

# reference numbers from the plan (docs / previous e8f3e1 run) -- used only by `compare`.
# NOTE (2026-09-16): every value below was measured BEFORE the mapping/geometry.py camera-model fix
# (the panorama was displayed under the reflected/old azimuth convention, seam at the front instead
# of the rear). They are kept only as "before fix" comparison points for `compare`, not as ground
# truth to match going forward -- see sphere_check.py / foe_check.py for the post-fix verdicts.
REFERENCE = {
    "clean_classes": {"export": {"clean": 825, "unverified": 163, "usable": 295, "reject": 220},
                      "docs": {"clean": 830, "unverified": 163, "usable": 302, "reject": 208}},
    "eomt_city_mIoU_core": {"docs": 0.358},  # export pseudo-GT, 100 bench frames (05)
    "interp_yaw_deg": {"export": 5.639, "corrected_docs": 0.197},
    "colour_de00_median": {"docs": 4.93},  # tw45 on export poses (02 §13.3)
    "colour_coverage": {"docs": 0.941},
    "tile037_cie76": {"export": 6.076, "docs": 6.088, "unregistered": 7.80},
    "nearfield": {"export": {"n": 755, "median_px": 15.8, "flagged": 328}, "docs": {"n": 757, "median_px": 17.9, "flagged": 336}},
    "seg_project_coverage": {"export": 0.799, "docs": 0.807},
    "seg_project_mIoU_core": {"export": 0.352, "docs": 0.328},
    "pairs_rms": {"docs_before": 0.102, "docs_after": 0.035},
    "pairs_converged": {"docs": "66/72"},
    "expected_total_points": config.EXPECTED_TOTAL_POINTS,
    "baseline_hash": "e8f3e1f2b3",
}


# --------------------------------------------------------------------------------------------- ctx
@dataclass
class Ctx:
    poses: str = "corrected"
    tag: str = "tw45"
    with_optional: bool = False
    force: bool = False
    dry_run: bool = False

    def corrected_hash(self) -> str:
        from ..poses import load_poses

        return load_poses("corrected").hash()[:6]

    def source_dir(self, base: Path) -> Path:
        """`config.source_dir(base, load_poses(self.poses))`, but falls back to a `_pending`
        placeholder (instead of raising) when the corrected pose table hasn't been assembled yet --
        so a `--dry-run` or `status` before stage `assemble` has run can still print/inspect later
        stages' commands and outputs without a real `poses_corrected.csv` on disk."""
        try:
            from ..poses import load_poses

            return config.source_dir(base, load_poses(self.poses))
        except Exception:
            return base if self.poses == "export" else base.with_name(base.name + "_pending")


def _safe(fn: Callable[[Ctx], dict]) -> Callable[[Ctx], dict]:
    def wrapped(ctx: Ctx) -> dict:
        try:
            return fn(ctx)
        except Exception as e:  # noqa: BLE001 - a marker must always be writable
            return {"error": f"{type(e).__name__}: {e}"}

    return wrapped


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _pctl(xs, q):
    import numpy as np

    return float(np.percentile(np.asarray(xs, dtype=float), q)) if len(xs) else float("nan")


# ---------------------------------------------------------------------------------------- commands
PY = ["uv", "run", "python", "-u"]


def mod_cmd(modname: str, *args: str) -> list[str]:
    return [*PY, "-m", modname, *args]


def script_cmd(path: str, *args: str) -> list[str]:
    return [*PY, path, *args]


def sh_cmd(script: str) -> list[str]:
    """A shell one-liner (docker compose / cp), run via `bash -c` so pipes/globs/subshells work."""
    return ["bash", "-c", script]


# --------------------------------------------------------------------------------------- one stage
@dataclass
class Stage:
    name: str
    commands: Callable[[Ctx], list[list[str]]]
    inputs: Callable[[Ctx], dict] = field(default=lambda ctx: {})
    outputs: Callable[[Ctx], list[Path]] = field(default=lambda ctx: [])
    metrics: Callable[[Ctx], dict] = field(default=lambda ctx: {})
    optional: bool = False
    est_min: float = 0.0
    # in-process stages (env-check/baseline/compare) bypass the subprocess runner entirely; called
    # as inprocess(ctx) -> metrics dict, rc is 0 unless it raises.
    inprocess: Callable[[Ctx], dict] | None = None
    # extra file-copy/bookkeeping step run in-process after all commands succeed (seg-project's
    # eval.json -> dataset/seg/project_<tag>.json promotion); not itself a subprocess.
    post: Callable[[Ctx], None] | None = None
    cwd: Path = REPO_ROOT


# ============================================================================ 0: env-check
def _do_env_check(ctx: Ctx) -> dict:
    checks: dict[str, Any] = {}
    checks["cache_root"] = {"path": str(CACHE_ROOT), "exists": CACHE_ROOT.exists()}
    checks["store_tiles_json"] = {"path": str(STORE_DIR / "tiles.json"), "exists": (STORE_DIR / "tiles.json").exists()}
    n_frames = len(list(FRAMES_DIR.glob("f*.npz"))) if FRAMES_DIR.exists() else 0
    checks["frames"] = {"dir": str(FRAMES_DIR), "n_npz": n_frames, "expected": 1503}
    checks["export_csv"] = {"path": str(EXPORT_CSV), "exists": EXPORT_CSV.exists()}
    vmask = CACHE_ROOT / "vehicle_mask.npz"
    checks["vehicle_mask"] = {"path": str(vmask), "exists": vmask.exists()}
    try:
        import torch

        checks["cuda"] = {"available": bool(torch.cuda.is_available())}
    except Exception as e:  # noqa: BLE001
        checks["cuda"] = {"available": False, "error": str(e)}
    try:
        du = shutil.disk_usage("/mnt")
        checks["disk_free_gb_mnt"] = round(du.free / 1e9, 1)
        checks["disk_ok"] = du.free / 1e9 >= 300
    except Exception as e:  # noqa: BLE001
        checks["disk_free_gb_mnt"] = None
        checks["disk_ok"] = None
        checks["disk_error"] = str(e)
    checks["git_rev"] = _git_rev()
    symlink = REPO_ROOT.parent / "Geovap_cache"
    checks["cache_symlink"] = {"path": str(symlink), "exists": symlink.exists(), "is_symlink": symlink.is_symlink()}
    return checks


def _fix_symlink() -> None:
    link = REPO_ROOT.parent / "Geovap_cache"
    target = Path("/mnt/Geovap_cache")
    if link.is_symlink() or link.exists():
        return
    if not target.exists():
        return
    link.symlink_to(target)


# ============================================================================ 0b: baseline
def _do_baseline(ctx: Ctx) -> dict:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot: dict[str, Any] = {}
    for name in ("clean_frames.json", "frame_quality.csv", "tile_summary.json"):
        src = DATASET_DIR / name
        if src.exists():
            shutil.copy2(src, BASELINE_DIR / name)
            snapshot[name] = True
        else:
            snapshot[name] = False
    seg_src = DATASET_DIR / "seg"
    if seg_src.exists():
        dst = BASELINE_DIR / "seg"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(seg_src, dst)
    counts = {}
    cf = BASELINE_DIR / "clean_frames.json"
    if cf.exists():
        d = _read_json(cf)
        for k in ("clean", "unverified", "usable", "reject"):
            counts[k] = len(d.get(k, [])) if isinstance(d.get(k), list) else d.get(k)
    bench = BASELINE_DIR / "seg" / "bench" / "results.json"
    miou = None
    if bench.exists():
        try:
            r = _read_json(bench)
            miou = r.get("eomt_city", {}).get("mIoU_core") if isinstance(r, dict) else None
        except Exception:  # noqa: BLE001
            miou = None
    return {"classes": counts, "eomt_city_mIoU_core": miou, "reference": REFERENCE["baseline_hash"]}


# ============================================================================ stage bodies
def _align_json() -> Path:
    return POSES_DIR / "align.json"


def _pass_reg_dir() -> Path:
    return OUT_DIR / "pass_reg"


def _dataset_dir(ctx: Ctx) -> Path:
    return ctx.source_dir(OUT_DIR / "dataset")


def _seg_out_dir(ctx: Ctx) -> Path:
    return ctx.source_dir(OUT_DIR / "seg_eomt")


def _segds_dir(ctx: Ctx) -> Path:
    return ctx.source_dir(CACHE_ROOT / "segds")


STAGES: list[Stage] = []


def _stage(**kw) -> Stage:
    st = Stage(**kw)
    STAGES.append(st)
    return st


_stage(name="env-check", commands=lambda ctx: [], inprocess=_do_env_check, est_min=0.5)

_stage(name="baseline", commands=lambda ctx: [], inprocess=_do_baseline, est_min=0.5)

_stage(
    name="store-columns",
    commands=lambda ctx: [
        mod_cmd("mapping.cli.store_add_columns", "--workers", "8"),
        mod_cmd("mapping.cli.pass_psid"),
    ],
    inputs=lambda ctx: {"tiles_json": STORE_DIR / "tiles.json"},
    outputs=lambda ctx: [POSES_DIR / "pass_psid.json"],
    metrics=_safe(lambda ctx: {"n_tiles": len(_read_json(STORE_DIR / "tiles.json"))}),
    est_min=1,
)

_stage(
    name="align",
    commands=lambda ctx: [mod_cmd("mapping.cli.align_frames", "--workers", "8")],
    inputs=lambda ctx: {"export_csv": EXPORT_CSV},
    outputs=lambda ctx: [_align_json()],
    metrics=_safe(lambda ctx: _align_metrics()),
    est_min=8,
)


def _align_metrics() -> dict:
    import numpy as np

    d = _read_json(_align_json())
    yaw = [r.get("yaw_offset_deg") for r in d if isinstance(r, dict) and r.get("yaw_offset_deg") is not None]
    return {"n_total": len(d), "median_yaw_offset_deg": float(np.median(yaw)) if yaw else None}


_stage(
    name="traj-rot",
    commands=lambda ctx: [mod_cmd("mapping.cli.build_trajectory", "--rot-only", "--passes", "all", "--workers", "6")],
    inputs=lambda ctx: {"export_csv": EXPORT_CSV},
    outputs=lambda ctx: [POSES_DIR / "poses_traj_rot.csv"],
    metrics=_safe(lambda ctx: {"has_trajectory_sidecar": (POSES_DIR / "poses_traj_rot.trajectory.npz").exists() or (POSES_DIR / "poses_traj_rot.csv").exists()}),
    est_min=5,
)

_stage(
    name="traj-validate",
    optional=True,
    commands=lambda ctx: [mod_cmd("mapping.cli.build_trajectory", "--validate-rot")],
    inputs=lambda ctx: {"traj_csv": POSES_DIR / "poses_traj_rot.csv"},
    outputs=lambda ctx: [],
    metrics=_safe(lambda ctx: {}),
    est_min=10,
)

_stage(
    name="refine",
    commands=lambda ctx: [
        mod_cmd(
            "mapping.cli.refine_poses",
            "--poses",
            "export",
            "--passes",
            "all",
            "--workers",
            "8",
            "--out",
            str(POSES_DIR / "poses_refined_export.csv"),
        )
    ],
    inputs=lambda ctx: {"align_json": _align_json(), "traj_csv": POSES_DIR / "poses_traj_rot.csv"},
    outputs=lambda ctx: [POSES_DIR / "poses_refined_export.csv"],
    metrics=_safe(lambda ctx: _refine_metrics()),
    est_min=52,
)


def _refine_metrics() -> dict:
    import csv

    p = POSES_DIR / "poses_refined_export.csv"
    with open(p, newline="") as f:
        n = sum(1 for _ in csv.DictReader(f))
    return {"n_rows": n}


_stage(
    name="register",
    commands=lambda ctx: [mod_cmd("mapping.cli.register_passes", "all", "--datum", "none")],
    inputs=lambda ctx: {"refined_csv": POSES_DIR / "poses_refined_export.csv"},
    outputs=lambda ctx: [_pass_reg_dir() / "pass_transforms.json", _pass_reg_dir() / "pairs.json"],
    metrics=_safe(lambda ctx: _register_metrics()),
    est_min=13,
)


def _register_metrics() -> dict:  # noqa: C901
    pairs = _read_json(_pass_reg_dir() / "pairs.json")
    rms_before = [v.get("rms_before") for v in pairs.values() if isinstance(v, dict) and v.get("rms_before") is not None]
    rms_after = [v.get("rms_after") for v in pairs.values() if isinstance(v, dict) and v.get("rms_after") is not None]
    converged = sum(1 for v in pairs.values() if isinstance(v, dict) and v.get("converged"))
    tr = _read_json(_pass_reg_dir() / "pass_transforms.json")
    return {
        "n_pairs": len(pairs),
        "n_converged": converged,
        "rms_before_median": _pctl(rms_before, 50) if rms_before else None,
        "rms_after_median": _pctl(rms_after, 50) if rms_after else None,
        "summary": tr.get("summary", {}),
        "max_t_m": max((max(abs(float(v)) for v in p_.get("t", [0, 0, 0])) for p_ in tr.get("passes", {}).values()), default=None),
        "max_yaw_deg": max((abs(float(p_.get("yaw_deg", 0.0))) for p_ in tr.get("passes", {}).values()), default=None),
    }


_stage(
    name="reg-conflict",
    optional=True,
    commands=lambda ctx: [mod_cmd("mapping.cli.validate_pass_reg", "conflict", "6")],
    inputs=lambda ctx: {"pass_transforms": _pass_reg_dir() / "pass_transforms.json"},
    outputs=lambda ctx: [],
    metrics=_safe(lambda ctx: {}),
    est_min=30,
)

_stage(
    name="assemble",
    commands=lambda ctx: [mod_cmd("mapping.cli.assemble_poses", "run")],
    inputs=lambda ctx: {
        "traj_csv": POSES_DIR / "poses_traj_rot.csv",
        "refined_csv": POSES_DIR / "poses_refined_export.csv",
        "pass_transforms": _pass_reg_dir() / "pass_transforms.json",
    },
    outputs=lambda ctx: [POSES_DIR / "poses_corrected.csv"],
    metrics=_safe(lambda ctx: {"corrected_hash6": ctx.corrected_hash()}),
    est_min=0.5,
)

_stage(
    name="products",
    commands=lambda ctx: [mod_cmd("mapping.cli.build_frames", "--workers", "16", "--poses", "corrected")],
    inputs=lambda ctx: {"poses_corrected": POSES_DIR / "poses_corrected.csv"},
    outputs=lambda ctx: [_products_dir(ctx)],
    metrics=_safe(lambda ctx: _products_metrics(ctx)),
    est_min=10,
)


def _products_dir(ctx: Ctx) -> Path:
    """Per-frame products use the SUBDIRECTORY convention (`products.frames_dir`: `frames/<hash6>`), not the
    `_<hash6>` sibling that `config.source_dir` gives the other datasets."""
    try:
        from ..poses import load_poses
        from ..products import frames_dir

        return frames_dir(load_poses(ctx.poses))
    except Exception:  # noqa: BLE001 - corrected table not assembled yet
        return FRAMES_DIR if ctx.poses == "export" else FRAMES_DIR / "pending"


def _products_metrics(ctx: Ctx) -> dict:
    d = _products_dir(ctx)
    n = len(list(d.glob("f*.npz"))) if d.exists() else 0
    return {"dir": str(d), "n_npz": n, "expected": 1503}


_stage(
    name="pose-report",
    commands=lambda ctx: [mod_cmd("mapping.cli.pose_report", "run", "--workers", "8")],
    inputs=lambda ctx: {"poses_corrected": POSES_DIR / "poses_corrected.csv"},
    outputs=lambda ctx: [POSES_DIR / "report_final.json"],
    metrics=_safe(lambda ctx: _read_json(POSES_DIR / "report_final.json")),
    est_min=26,
)

_stage(
    name="colorize",
    commands=lambda ctx: [mod_cmd("mapping.cli.colorize", "--poses", "corrected", "--tag", ctx.tag, "--workers", "5")],
    inputs=lambda ctx: {"poses_corrected": POSES_DIR / "poses_corrected.csv"},
    outputs=lambda ctx: [OUT_DIR / ctx.tag / "stats"],
    metrics=_safe(lambda ctx: {"n_meta": len(list((OUT_DIR / ctx.tag / "stats").glob("*_meta.json")))}),
    est_min=55,
)

_stage(
    name="colour-report",
    commands=lambda ctx: [mod_cmd("mapping.report", ctx.tag)],
    inputs=lambda ctx: {"stats_dir": OUT_DIR / ctx.tag / "stats"},
    outputs=lambda ctx: [OUT_DIR / ctx.tag / "report.md"],
    metrics=_safe(lambda ctx: _colour_report_metrics(ctx)),
    est_min=1,
)


def _colour_report_metrics(ctx: Ctx) -> dict:
    from .. import metrics as M
    from ..report import load_run

    metas, hists, _ = load_run(ctx.tag)
    s = M.hist_stats(hists["med"][0].total_hist())
    cov = [(m["coverage"]["med"], m.get("n", 1)) for m in metas.values() if "coverage" in m]
    w = sum(n for _, n in cov)
    return {"n_tiles": len(metas), "de00_median": s.get("median"), "coverage_mean": (sum(c * n for c, n in cov) / w) if w else None}


# ---------------------------------------------------------------------------------------- quality
def _quality_jsonl(ctx: Ctx) -> Path:
    return _dataset_dir(ctx) / "frame_quality.jsonl"


def _quality_commands_template(ctx: Ctx) -> list[list[str]]:
    d = _dataset_dir(ctx)
    return [
        mod_cmd("mapping.quality", "run", "--workers", "8", "--poses", ctx.poses, "--limit", "350"),
        mod_cmd("mapping.quality", "reclassify", "--poses", ctx.poses),
        mod_cmd("mapping.quality", "tile-summary", str(d / "frame_quality.csv"), str(d), "--poses", ctx.poses),
    ]


_stage(
    name="quality",
    commands=_quality_commands_template,
    inputs=lambda ctx: {"poses_corrected": POSES_DIR / "poses_corrected.csv", "colour_stats": OUT_DIR / ctx.tag / "stats"},
    outputs=lambda ctx: [_dataset_dir(ctx) / "clean_frames.json", _dataset_dir(ctx) / "tile_summary.json"],
    metrics=_safe(lambda ctx: _quality_metrics(ctx)),
    est_min=90,
)


def _quality_metrics(ctx: Ctx) -> dict:
    d = _dataset_dir(ctx)
    cf = _read_json(d / "clean_frames.json")
    return {k: len(cf.get(k, [])) for k in ("clean", "unverified", "usable", "reject")}


_stage(
    name="promote",
    commands=lambda ctx: [mod_cmd("mapping.quality", "promote", "--poses", "corrected")],
    inputs=lambda ctx: {"clean_frames": _dataset_dir(ctx) / "clean_frames.json"},
    outputs=lambda ctx: [CLEAN_FRAMES_JSON],
    metrics=_safe(lambda ctx: _promote_metrics(ctx)),
    est_min=0.5,
)


def _promote_metrics(ctx: Ctx) -> dict:
    prev = BASELINE_DIR / "clean_frames.json"
    new = _read_json(CLEAN_FRAMES_JSON)
    out = {"n_clean_new": len(new.get("clean", []))}
    if prev.exists():
        old = _read_json(prev)
        old_set, new_set = set(old.get("clean", [])), set(new.get("clean", []))
        out["n_clean_baseline"] = len(old_set)
        out["n_added"] = len(new_set - old_set)
        out["n_removed"] = len(old_set - new_set)
    return out


_stage(
    name="segds",
    commands=lambda ctx: [
        mod_cmd("mapping.cli.seg_build", "--poses", "corrected", "areas"),
        mod_cmd("mapping.cli.seg_build", "--poses", "corrected", "rasters"),
        mod_cmd("mapping.cli.seg_build", "--poses", "corrected", "points", "--workers", "8"),
        mod_cmd("mapping.cli.seg_build", "--poses", "corrected", "labels", "--frames", "clean", "--workers", "10"),
        mod_cmd("mapping.cli.seg_build", "--poses", "corrected", "views", "--workers", "10"),
        mod_cmd("mapping.cli.seg_build", "--poses", "corrected", "dataset"),
        mod_cmd("mapping.seg.nearfield", "--workers", "8", "--poses", "corrected"),
        mod_cmd("mapping.cli.seg_build", "--poses", "corrected", "dataset"),
    ],
    inputs=lambda ctx: {"clean_frames": CLEAN_FRAMES_JSON},
    outputs=lambda ctx: [_segds_dir(ctx) / "areas" / "report.json", _segds_dir(ctx) / "nearfield.json"],
    metrics=_safe(lambda ctx: _segds_metrics(ctx)),
    est_min=15,
)


def _segds_metrics(ctx: Ctx) -> dict:
    seg_dir = _segds_dir(ctx)
    areas = _read_json(seg_dir / "areas" / "report.json")
    nf = _read_json(seg_dir / "nearfield.json")
    return {"areas": {"n_faces": areas.get("n_faces"), "all_lines": areas.get("all_lines")}, "nearfield_summary": nf.get("summary")}


_stage(
    name="seg-eval",
    commands=lambda ctx: [
        mod_cmd("mapping.cli.seg_bench", "run", "--models", "eomt_city", "--frames", "missing:clean"),
        mod_cmd("mapping.cli.seg_bench", "run", "--models", "all", "--frames", "missing:bench"),
        # the export-era 100 bench frames (05_benchmark_segmentace.md, all six models scored there), so the
        # "same frames, new pseudo-GT" row is directly comparable with the documented 0.358
        mod_cmd("mapping.cli.seg_bench", "run", "--models", "all", "--frames", "missing:" + str(DATASET_DIR / "seg" / "export_baseline" / "bench_frames.json")),
        mod_cmd("mapping.cli.seg_bench", "evaluate", "--models", "all", "--frames", "bench"),
        mod_cmd("mapping.cli.seg_bench", "evaluate", "--models", "all", "--frames", str(DATASET_DIR / "seg" / "export_baseline" / "bench_frames.json"), "--out", str(DATASET_DIR / "seg" / "bench" / "export_frames")),
        mod_cmd("mapping.cli.seg_bench", "report"),
    ],
    inputs=lambda ctx: {"clean_frames": CLEAN_FRAMES_JSON},
    outputs=lambda ctx: [DATASET_DIR / "seg" / "bench" / "results.json", DATASET_DIR / "seg" / "bench" / "tables.md"],
    metrics=_safe(lambda ctx: _read_json(DATASET_DIR / "seg" / "bench" / "results.json")),
    est_min=17,
)

_stage(
    name="seg-project",
    commands=lambda ctx: [
        mod_cmd("mapping.cli.seg_project", "--poses", "corrected", "run", "--tag", "eomt_city", "--workers", "8"),
        mod_cmd("mapping.cli.seg_project", "--poses", "corrected", "eval"),
        mod_cmd("mapping.cli.seg_project", "--poses", "corrected", "render"),
    ],
    inputs=lambda ctx: {"clean_frames": CLEAN_FRAMES_JSON},
    outputs=lambda ctx: [_seg_out_dir(ctx) / "eval.json"],
    metrics=_safe(lambda ctx: _read_json(_seg_out_dir(ctx) / "eval.json")),
    post=lambda ctx: _promote_seg_project_eval(ctx),
    est_min=15,
)


def _promote_seg_project_eval(ctx: Ctx) -> None:
    src = _seg_out_dir(ctx) / "eval.json"
    if not src.exists():
        return
    dest_dir = DATASET_DIR / "seg"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "project_eomt_city.json"
    if dest.exists():
        shutil.copy2(dest, dest.with_suffix(".json.bak"))
    shutil.copy2(src, dest)


# ---------------------------------------------------------------------------------------- compare
def _jget(d: Any, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_fmt(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return " / ".join(_fmt(x) for x in v)
    return str(v)


def _do_compare(ctx: Ctx) -> dict:
    """out/pipeline/comparison.{md,json}: every headline metric of the chain as
    `export baseline | previous corrected run (docs, e8f3e1) | previous live repo state (baseline snapshot) | this run`.
    Reads the stage markers, the report JSONs and the repo's dataset/ tables directly, so it can be
    re-run any time with `pipeline run --only compare --force`."""
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    markers = {name: read_marker(name) for name in [s.name for s in STAGES]}

    def m(stage: str) -> dict:
        mk = markers.get(stage)
        return (mk or {}).get("metrics", {}) if isinstance(mk, dict) else {}

    def rj(path: Path) -> Any:
        try:
            return _read_json(path)
        except Exception:  # noqa: BLE001
            return None

    R = REFERENCE
    rep = rj(POSES_DIR / "report_final.json") or {}
    prev_seg_stats = rj(BASELINE_DIR / "seg" / "stats.json") or {}
    prev_bench = rj(BASELINE_DIR / "seg" / "bench" / "results.json") or {}
    prev_proj = rj(BASELINE_DIR / "seg" / "project_eomt_city.json") or {}
    new_bench = rj(DATASET_DIR / "seg" / "bench" / "results.json") or {}
    new_bench_expframes = rj(DATASET_DIR / "seg" / "bench" / "export_frames" / "results.json") or {}
    new_stats = rj(DATASET_DIR / "seg" / "stats.json") or {}
    new_proj = rj(DATASET_DIR / "seg" / "project_eomt_city.json") or {}
    exp_proj = rj(DATASET_DIR / "seg" / "export_baseline" / "project_eomt_city.json") or {}
    # the export-GT benchmark (05, 2026-09-06) lives in dataset/seg/bench/results.json until this run overwrites
    # it, i.e. in the baseline snapshot; export_baseline/ has no bench/ copy
    exp_bench = rj(DATASET_DIR / "seg" / "export_baseline" / "bench" / "results.json") or prev_bench
    new_clean = rj(CLEAN_FRAMES_JSON) or {}
    prev_clean = rj(BASELINE_DIR / "clean_frames.json") or {}
    exp_clean = rj(DATASET_DIR / "export_baseline" / "clean_frames.json") or {}
    nf_new = None
    try:
        nf_new = _jget(rj(_segds_dir(ctx) / "nearfield.json") or {}, "summary")
    except Exception:  # noqa: BLE001
        pass
    sph = rj(_VALIDATE_DIR / "screenshots" / "sphere_check.json") or {}
    checks = rj(_VALIDATE_DIR / "checks.json") or {}

    def classes(d: dict) -> dict | None:
        return {k: len(d.get(k, [])) for k in ("clean", "unverified", "usable", "reject")} if d else None

    def bench_row(res: dict, tag: str, sub: str | None = None):
        r = res.get(tag) if isinstance(res, dict) else None
        if not isinstance(r, dict):
            return None
        return _jget(r, sub, "mIoU_core") if sub else r.get("mIoU_core")

    # (label, export, docs/previous corrected run, baseline snapshot (live repo before this run), this run, note)
    rows: list[tuple] = []
    rows.append(("poses hash", "a2d74f0580", R["baseline_hash"], _jget(markers, "baseline", "metrics", "reference"), _jget(rep, "identity", "hash_b") or m("assemble").get("corrected_hash6"), "corrected table identity"))
    rows.append(("pass registration: pairs converged", "-", R["pairs_converged"]["docs"], None, f"{m('register').get('n_converged')}/{m('register').get('n_pairs')}", "register_passes --datum none"))
    rows.append(("pass registration: median RMS before -> after [m]", "-", (R["pairs_rms"]["docs_before"], R["pairs_rms"]["docs_after"]), None, (m("register").get("rms_before_median"), m("register").get("rms_after_median")), "ICP point-to-plane"))
    tr = rj(_pass_reg_dir() / "pass_transforms.json") or {}
    tr_passes = tr.get("passes", {}) if isinstance(tr, dict) else {}
    max_t = max((max(abs(float(v)) for v in p_.get("t", [0, 0, 0])) for p_ in tr_passes.values()), default=None)
    max_yaw = max((abs(float(p_.get("yaw_deg", 0.0))) for p_ in tr_passes.values()), default=None)
    rows.append(("registration: max |t| [m], max |yaw| [deg], flagged passes", "-", None, None, (max_t, max_yaw, _jget(tr, "summary", "n_flagged")), "from pass_transforms.json"))
    ib = _jget(rep, "interpolation_benefit") or {}
    rows.append(("yaw at turning frames: dense model vs neighbour-linear [deg, median]", R["interp_yaw_deg"]["export"], R["interp_yaw_deg"]["corrected_docs"], None, (_jget(ib, "dense_model_resid_deg", "median"), _jget(ib, "old_model_resid_deg", "median")), f"n={ib.get('n_turning_covered')}"))
    cg = _jget(rep, "compare_groups", "turning") or {}
    rows.append(("silhouette |du|/|dv| median px, turning frames (export -> corrected)", (_jget(cg, "abs_du_a", "median"), _jget(cg, "abs_dv_a", "median")), None, None, (_jget(cg, "abs_du_b", "median"), _jget(cg, "abs_dv_b", "median")), "quality.py residual"))
    sr = _jget(rep, "slow_regression_037") or {}
    rows.append(("tile 037 CIE76: export / corrected registered / unregistered", R["tile037_cie76"]["export"], (R["tile037_cie76"]["docs"], R["tile037_cie76"]["unregistered"]), None, (sr.get("median_de76_export"), sr.get("median_de76_corrected_registered"), sr.get("median_de76_corrected_unregistered")), f"tolerance {sr.get('tolerance')}, passes={sr.get('passes_registered')}"))
    cr = m("colour-report")
    rows.append(("colourisation tw45: dE00 median", R["colour_de00_median"]["docs"], None, None, cr.get("de00_median"), "02 §13.3 value is on export poses"))
    rows.append(("colourisation tw45: coverage", R["colour_coverage"]["docs"], None, None, cr.get("coverage_mean"), ""))
    rows.append(("clean / unverified / usable / reject", classes(exp_clean) or R["clean_classes"]["export"], R["clean_classes"]["docs"], classes(prev_clean), classes(new_clean), "quality.py on this run's poses"))
    pm = m("promote")
    rows.append(("clean set turnover vs previous", "-", "-", pm.get("n_clean_baseline"), {"added": pm.get("n_added"), "removed": pm.get("n_removed")}, ""))
    rows.append(("near-field JVF flag: n_measured / median px / flagged", tuple(R["nearfield"]["export"].values()), tuple(R["nearfield"]["docs"].values()), _jget(prev_seg_stats, "nearfield"), (nf_new or {}).get("n_measured"), ""))
    if nf_new:
        rows[-1] = (rows[-1][0], rows[-1][1], rows[-1][2], rows[-1][3], (nf_new.get("n_measured"), nf_new.get("median_of_medians_px"), nf_new.get("n_flagged")), "flag > 20 px")
    for tag in ("eomt_city", "m2f_vistas", "m2f_city", "eomt_dinov3_ade", "oneformer_city", "segformer_b5"):
        rows.append((f"bench mIoU_core {tag} (all / nf_ok)", (bench_row(exp_bench, tag), bench_row(exp_bench, tag, "nf_ok")) if exp_bench else (R["eomt_city_mIoU_core"]["docs"] if tag == "eomt_city" else None),
                     None, (bench_row(prev_bench, tag), bench_row(prev_bench, tag, "nf_ok")), (bench_row(new_bench, tag), bench_row(new_bench, tag, "nf_ok")),
                     "new bench_frames.json (this run's pseudo-GT)"))
    for tag in ("eomt_city", "m2f_vistas", "m2f_city", "eomt_dinov3_ade", "oneformer_city", "segformer_b5"):
        r_new = new_bench_expframes.get(tag) if isinstance(new_bench_expframes, dict) else None
        rows.append((f"bench mIoU_core {tag} on the EXPORT-era 100 bench frames (all / nf_ok)", (bench_row(exp_bench, tag), bench_row(exp_bench, tag, "nf_ok")),
                     None, None, (bench_row(new_bench_expframes, tag), bench_row(new_bench_expframes, tag, "nf_ok")),
                     f"same frames, new pseudo-GT; n={r_new.get('n_frames') if isinstance(r_new, dict) else None}, no GT for {r_new.get('n_no_gt') if isinstance(r_new, dict) else None}"))
    rows.append(("3D projection: coverage / pixel acc / mIoU_core", (exp_proj.get("coverage"), exp_proj.get("pixel_acc"), exp_proj.get("miou_core")), (R["seg_project_coverage"]["docs"], None, R["seg_project_mIoU_core"]["docs"]),
                 (prev_proj.get("coverage"), prev_proj.get("pixel_acc"), prev_proj.get("miou_core")), (new_proj.get("coverage"), new_proj.get("pixel_acc"), new_proj.get("miou_core")), "eval.json"))
    mg = m("merge")
    rows.append(("consolidated cloud: total points", R["expected_total_points"], None, None, mg.get("total_points") if isinstance(mg, dict) else None, f"matches={mg.get('matches_expected') if isinstance(mg, dict) else None}"))
    pn = rj(_POTREE_DIR / "cloud" / "panos" / "panos.json") or m("panos")
    rows.append(("panos: poses / az offset / n", "export, 180 (old eomt_city_seg/panos; displayed under the pre-2026-09-16 reflected camera model, not comparable to this run's convention)", None, None, (pn.get("poses_source"), pn.get("az_offset_deg"), pn.get("n_frames")), "registration " + str(pn.get("registration") is not None)))
    sv = _jget(sph, "variants") or {}
    rows.append(("sphere check NCC median: cloud/panos (corr, az0) / corr180 / exp0 / exp180", None, None, None,
                 tuple(_jget(sv, v, "ncc_median") for v in ("cloud_panos", "panos_corr180", "panos_exp0", "panos_exp180")), str(_jget(sph, "verdict", "pass"))))
    rows.append(("product checks ok / failed / skipped", None, None, None, (checks.get("n_ok"), checks.get("n_fail"), checks.get("n_skip")), "check_products.py"))
    durations = {n: round((mk or {}).get("seconds", 0) / 60, 1) for n, mk in markers.items() if isinstance(mk, dict) and mk.get("seconds")}
    if not durations:
        for n, mk in markers.items():
            if isinstance(mk, dict) and mk.get("started") and mk.get("finished"):
                try:
                    durations[n] = round((datetime.fromisoformat(mk["finished"]) - datetime.fromisoformat(mk["started"])).total_seconds() / 60, 1)
                except Exception:  # noqa: BLE001
                    pass

    md = ["# Pipeline comparison", "", f"poses: `{ctx.poses}`, tag: `{ctx.tag}`, generated {_now()}", "",
          "| metrika | export | předchozí korigovaný běh (docs, e8f3e1) | stav repa před tímto během | **tento běh** | pozn. |", "|---|---|---|---|---|---|"]
    json_rows = []
    for label, exp, docs, base, new, note in rows:
        md.append(f"| {label} | {_fmt(exp)} | {_fmt(docs)} | {_fmt(base)} | **{_fmt(new)}** | {note} |")
        json_rows.append({"metric": label, "export": exp, "docs": docs, "baseline": base, "current": new, "note": note})
    md += ["", "## Doba běhu stagí [min]", "", "| stage | min |", "|---|---|"] + [f"| {n} | {d} |" for n, d in durations.items()]
    (PIPELINE_DIR / "comparison.md").write_text("\n".join(md) + "\n")
    (PIPELINE_DIR / "comparison.json").write_text(json.dumps({"rows": json_rows, "durations_min": durations}, indent=1, default=str))
    return {"rows": len(json_rows)}


# ---------------------------------------------------------------------------------------- merge
_stage(
    name="merge",
    commands=lambda ctx: [mod_cmd("mapping.cli.merge_products", "run", "--workers", "8", "--poses", "corrected")],
    inputs=lambda ctx: {"poses_corrected": POSES_DIR / "poses_corrected.csv", "tw45_stats": OUT_DIR / ctx.tag / "stats"},
    outputs=lambda ctx: [CONSOLIDATED_DIR / "summary.json"],
    metrics=_safe(lambda ctx: _read_json(CONSOLIDATED_DIR / "summary.json")),
    est_min=25,
)

# ---------------------------------------------------------------------------------------- potree
_POTREE_DIR = POTREE_OUTPUT_DIR / "consolidated"


def _potree_commands(ctx: Ctx) -> list[list[str]]:
    return [
        sh_cmd(
            "cd pointcloud-tools && docker compose run --rm --user $(id -u):$(id -g) --entrypoint sh "
            "potreeconverter -c '/usr/local/bin/PotreeConverter $(ls /cache_out/consolidated/tiles/*.laz) -o /output/consolidated/cloud'"
        ),
        sh_cmd(
            "cd pointcloud-tools && docker compose run --rm --user $(id -u):$(id -g) --entrypoint sh "
            "potreeconverter -c '/usr/local/bin/PotreeConverter $(ls /cache_out/consolidated/objects/*.laz) -o /output/consolidated/objects'"
        ),
        sh_cmd(  # third octree: TerraScan RGB on the same registered points (page mode "RGB (original)")
            "cd pointcloud-tools && docker compose run --rm --user $(id -u):$(id -g) --entrypoint sh "
            "potreeconverter -c '/usr/local/bin/PotreeConverter $(ls /cache_out/consolidated/vendor/*.laz) -o /output/consolidated/vendor'"
        ),
        mod_cmd("mapping.cli.merge_products", "classes", "--out", str(CONSOLIDATED_DIR)),
        sh_cmd(f"cp {CONSOLIDATED_DIR}/classes.json {_POTREE_DIR}/cloud/classes.json"),
        sh_cmd(f"cp {REPO_ROOT}/pointcloud-tools/consolidated/index.html {_POTREE_DIR}/index.html"),
    ]


_stage(
    name="potree",
    commands=_potree_commands,
    inputs=lambda ctx: {"consolidated_summary": CONSOLIDATED_DIR / "summary.json"},
    outputs=lambda ctx: [_POTREE_DIR / "cloud" / "metadata.json", _POTREE_DIR / "objects" / "metadata.json", _POTREE_DIR / "vendor" / "metadata.json"],
    metrics=_safe(lambda ctx: {"cloud": _read_json(_POTREE_DIR / "cloud" / "metadata.json").get("points"), "objects": _read_json(_POTREE_DIR / "objects" / "metadata.json").get("points")}),
    est_min=15,
)

# ---------------------------------------------------------------------------------------- panos
# resized 4096-px jpgs of the pre-fix export (photos only, still valid) - archived 2026-09-16; reused via hardlinks
_OLD_PANOS_DIR = Path("/mnt/Geovap_cache/TestOutput/output/_invalid_2026-09-16/eomt_city_seg/panos")  # archive stays on /mnt


def _panos_commands(ctx: Ctx) -> list[list[str]]:
    cloud = str(_POTREE_DIR / "cloud")
    ab = {"panos_corr180": ("corrected", "180"), "panos_exp0": ("export", "0"), "panos_exp180": ("export", "180")}
    cmds = [
        # reuse the 1503 resized jpgs of the old (export) export via hardlinks in every panos dir, so
        # export_panos only writes coordinates.txt + panos.json (it skips existing images)
        sh_cmd(" && ".join(
            f"mkdir -p {d} && (cp -al {_OLD_PANOS_DIR}/f*.jpg {d}/ 2>/dev/null || true)"
            for d in [f"{_POTREE_DIR}/cloud/panos", *(f"{_POTREE_DIR}/{n}" for n in ab)])),
        # primary set: corrected poses, AZ_OFFSET_DEG default 0 (camera model fixed 2026-09-16; control sets below are A/B only)
        mod_cmd("mapping.cli.export_panos", "--cloud", cloud, "--poses", "corrected"),
    ]
    for name, (src, az) in ab.items():  # A/B sets for validate (azimuth convention, export vs corrected)
        cmds.append(mod_cmd("mapping.cli.export_panos", "--cloud", cloud, "--poses", src, "--out", str(_POTREE_DIR / name), "--az-offset", az))
    return cmds


_stage(
    name="panos",
    commands=_panos_commands,
    inputs=lambda ctx: {"cloud_metadata": _POTREE_DIR / "cloud" / "metadata.json"},
    outputs=lambda ctx: [_POTREE_DIR / "cloud" / "panos" / "panos.json"],
    metrics=_safe(lambda ctx: {k: v for k, v in _read_json(_POTREE_DIR / "cloud" / "panos" / "panos.json").items() if k in ("poses_source", "poses_hash", "registration", "n_frames", "az_offset_deg")}),
    est_min=2,
)

# --------------------------------------------------------------------------------------- validate
_VALIDATE_DIR = CONSOLIDATED_DIR / "validation"


def _validate_commands(ctx: Ctx) -> list[list[str]]:
    return [
        script_cmd("pointcloud-tools/validate/check_products.py"),
        script_cmd(
            "pointcloud-tools/validate/screenshots.py",
            "--frames",
            "12",
            "--yaws",
            "0,90,180,270",
            "--variants",
            "cloud/panos,panos_corr180,panos_exp0,panos_exp180",
            "--out-dir",
            str(_VALIDATE_DIR / "screenshots"),
        ),
        # Potree sphere vs our camera model (NCC of photo-only shots against offline renders, PLUS a
        # model-independent point-cloud pinhole silhouette reference -- see sphere_check.py docstring
        # for why NCC-vs-offline alone is circular)
        script_cmd("pointcloud-tools/validate/sphere_check.py", "--shots-dir", str(_VALIDATE_DIR / "screenshots"),
                   "--variants", "cloud_panos,panos_corr180,panos_exp0,panos_exp180", "--poses", "corrected"),
        # focus-of-expansion: model-free second opinion on the azimuth convention (no point cloud, no
        # rendering -- just frame-to-frame photo motion vs. the travel direction the model predicts)
        script_cmd("pointcloud-tools/validate/foe_check.py", "--frames", "100,150,250,300,400,498,550,650,700,750,800,850,950,1000,1100,1200",
                   "--poses", "corrected", "--out", str(_VALIDATE_DIR / "foe_check.json")),
    ]


_stage(
    name="validate",
    commands=_validate_commands,
    inputs=lambda ctx: {"cloud_metadata": _POTREE_DIR / "cloud" / "metadata.json", "panos_json": _POTREE_DIR / "cloud" / "panos" / "panos.json"},
    outputs=lambda ctx: [_VALIDATE_DIR / "checks.json"],
    metrics=_safe(lambda ctx: _read_json(_VALIDATE_DIR / "checks.json")),
    est_min=5,
)

_stage(
    name="compare",
    commands=lambda ctx: [],
    inprocess=_do_compare,
    est_min=0.5,
)

# `compare` was appended after `validate` above (it's declared mid-file, above merge/potree/panos/
# validate, to keep the source near `_do_compare`); the plan puts it right after seg-project (#15)
# and lists merge/potree/panos/validate as #16-19 afterwards. Reorder STAGES to match the plan's
# canonical order exactly (this list, not append order, is what `pipeline status`/tests iterate).
_ORDER = [
    "env-check",
    "baseline",
    "store-columns",
    "align",
    "traj-rot",
    "traj-validate",
    "refine",
    "register",
    "reg-conflict",
    "assemble",
    "products",
    "pose-report",
    "colorize",
    "colour-report",
    "quality",
    "promote",
    "segds",
    "seg-eval",
    "seg-project",
    "compare",
    "merge",
    "potree",
    "panos",
    "validate",
]
_by_name = {s.name: s for s in STAGES}
assert set(_by_name) == set(_ORDER), f"stage set mismatch: {set(_by_name) ^ set(_ORDER)}"
STAGES = [_by_name[n] for n in _ORDER]


def stage_names() -> list[str]:
    return [s.name for s in STAGES]


def get_stage(name: str) -> Stage:
    return _by_name[name]


# ------------------------------------------------------------------------------------ hashing/misc
_HASH_CONTENT_MAX_BYTES = 10 * 1024 * 1024  # small files (markers, json) hash by content; bigger ones by stat only


def _hashable(v: Any) -> Any:
    if isinstance(v, (Path,)):
        p = Path(v)
        try:
            p = p.resolve()
        except OSError:
            pass
        try:
            st = p.stat()
        except OSError:
            return ["path", str(p), None, None]
        if p.is_file() and st.st_size <= _HASH_CONTENT_MAX_BYTES:
            return ["path", str(p), hashlib.sha256(p.read_bytes()).hexdigest()]
        return ["path", str(p), st.st_mtime_ns, st.st_size]
    return v


def compute_inputs_hash(stage: Stage, ctx: Ctx) -> str:
    d = stage.inputs(ctx)
    payload = json.dumps({k: _hashable(v) for k, v in sorted(d.items())}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _poses_hash_safe(ctx: Ctx) -> str | None:
    try:
        from ..poses import load_poses

        return load_poses(ctx.poses).hash()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------- markers
def marker_path(name: str) -> Path:
    return PIPELINE_DIR / f"{name}.json"


def read_marker(name: str) -> dict | None:
    p = marker_path(name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def write_marker(name: str, marker: dict) -> None:
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    marker_path(name).write_text(json.dumps(marker, indent=1, default=str))


def is_done(stage: Stage, ctx: Ctx) -> bool:
    marker = read_marker(stage.name)
    if marker is None or marker.get("rc") != 0:
        return False
    if marker.get("inputs_hash") != compute_inputs_hash(stage, ctx):
        return False
    for p in stage.outputs(ctx):
        if not Path(p).exists():
            return False
    return True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------------------- running
# stages up to and including `assemble` build the corrected table FROM export poses; the corrected
# table does not exist yet, and several of their modules call load_poses() at import time, so they
# must see GEOVAP_POSES=export (the documented S0-S7 chain runs on export).
EXPORT_POSE_STAGES = {"env-check", "baseline", "store-columns", "align", "traj-rot", "traj-validate", "refine", "register", "reg-conflict", "assemble"}


def _env_for(ctx: Ctx, stage_name: str | None = None) -> dict:
    env = dict(os.environ)
    env["GEOVAP_CACHE"] = str(CACHE_ROOT)
    env["GEOVAP_POSES"] = "export" if stage_name in EXPORT_POSE_STAGES else ctx.poses
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _log(msg: str) -> None:
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{_now()}] {msg}\n"
    with open(PIPELINE_LOG, "a") as f:
        f.write(line)
    print(msg)


def _run_one(stage_name: str, idx: int, cmd: list[str], ctx: Ctx, cwd: Path = REPO_ROOT) -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{stage_name}_{idx}.log"
    _log(f"[{stage_name}] $ {' '.join(cmd)}")
    with open(log_path, "w") as lf, open(PIPELINE_LOG, "a") as pf:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=_env_for(ctx, stage_name), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            pf.write(line)
        rc = proc.wait()
    _log(f"[{stage_name}] rc={rc}")
    return rc


def _run_quality_stage(ctx: Ctx) -> tuple[int, list[dict]]:
    from ..poses import load_poses

    cwd = get_stage("quality").cwd
    n_total = len(load_poses(ctx.poses))
    jl = _quality_jsonl(ctx)
    cmds_run: list[dict] = []
    i = 0
    prev_have = -1
    while True:
        n_have = 0
        if jl.exists():
            n_have = sum(1 for line in jl.read_text().splitlines() if line.strip())
        if n_have >= n_total:
            break
        if n_have <= prev_have:
            # a chunk ran (rc 0) but appended nothing: don't spin forever, surface it as a failure
            _log(f"[quality] no progress ({n_have}/{n_total} frames after chunk {i - 1}); aborting the loop")
            return 1, cmds_run
        prev_have = n_have
        cmd = mod_cmd("mapping.quality", "run", "--workers", "8", "--poses", ctx.poses, "--limit", "350")
        rc = _run_one("quality", i, cmd, ctx, cwd=cwd)
        cmds_run.append({"cmd": cmd, "rc": rc})
        i += 1
        if rc != 0:
            return rc, cmds_run
    d = _dataset_dir(ctx)
    for cmd in (
        mod_cmd("mapping.quality", "reclassify", "--poses", ctx.poses),
        mod_cmd("mapping.quality", "tile-summary", str(d / "frame_quality.csv"), str(d), "--poses", ctx.poses),
    ):
        rc = _run_one("quality", i, cmd, ctx, cwd=cwd)
        cmds_run.append({"cmd": cmd, "rc": rc})
        i += 1
        if rc != 0:
            return rc, cmds_run
    return 0, cmds_run


def run_stage(stage: Stage, ctx: Ctx, force: bool = False) -> int:
    if not force and is_done(stage, ctx):
        _log(f"[{stage.name}] already done, skipping (use --force to rerun)")
        return 0
    inputs_hash = compute_inputs_hash(stage, ctx)
    started = _now()
    cmds_run: list[dict] = []
    rc = 0
    if stage.inprocess is not None:
        try:
            metrics = stage.inprocess(ctx)
        except Exception as e:  # noqa: BLE001
            metrics = {"error": str(e)}
            rc = 1
        finished = _now()
        marker = {
            "poses_hash": _poses_hash_safe(ctx),
            "git": _git_rev(),
            "started": started,
            "finished": finished,
            "cmds": [],
            "rc": rc,
            "inputs_hash": inputs_hash,
            "outputs": [str(p) for p in stage.outputs(ctx)] if rc == 0 else [],
            "metrics": metrics,
        }
        write_marker(stage.name, marker)
        return rc
    if stage.name == "quality":
        rc, cmds_run = _run_quality_stage(ctx)
    else:
        for i, cmd in enumerate(stage.commands(ctx)):
            r = _run_one(stage.name, i, cmd, ctx, cwd=stage.cwd)
            cmds_run.append({"cmd": cmd, "rc": r})
            if r != 0:
                rc = r
                break
    if rc == 0 and stage.post is not None:
        try:
            stage.post(ctx)
        except Exception as e:  # noqa: BLE001
            _log(f"[{stage.name}] post-step failed: {e}")
    finished = _now()
    metrics = stage.metrics(ctx) if rc == 0 else {"error": f"rc={rc}"}
    marker = {
        "poses_hash": _poses_hash_safe(ctx),
        "git": _git_rev(),
        "started": started,
        "finished": finished,
        "cmds": cmds_run,
        "rc": rc,
        "inputs_hash": inputs_hash,
        "outputs": [str(p) for p in stage.outputs(ctx)] if rc == 0 else [],
        "metrics": metrics,
    }
    write_marker(stage.name, marker)
    return rc


def _dry_run_print(stage: Stage, ctx: Ctx) -> None:
    print(f"=== {stage.name} ({'optional' if stage.optional else 'required'}, ~{stage.est_min:.0f} min) ===")
    if stage.inprocess is not None:
        print(f"  in-process: {stage.inprocess.__name__}")
        return
    cmds = stage.commands(ctx)
    if stage.name == "quality":
        print("  (loop until frame_quality.jsonl has len(poses) lines)")
    for cmd in cmds:
        print("  $ " + " ".join(cmd))
    if stage.post is not None:
        print(f"  post: {stage.post.__name__}")


def select_stages(args) -> list[Stage]:
    names = stage_names()
    stages = list(STAGES)
    if args.only:
        wanted = set(args.only.split(","))
        stages = [s for s in stages if s.name in wanted]
    else:
        if args.from_:
            i = names.index(args.from_)
            stages = [s for s in stages if names.index(s.name) >= i]
        if args.to:
            i = names.index(args.to)
            stages = [s for s in stages if names.index(s.name) <= i]
    if not args.with_optional:
        stages = [s for s in stages if not s.optional]
    return stages


def cmd_run(args) -> int:
    ctx = Ctx(poses=args.poses or "corrected", tag=args.tag or "tw45", with_optional=args.with_optional, force=args.force, dry_run=args.dry_run)
    stages = select_stages(args)
    if args.detach and not ctx.dry_run:
        return _detach(args)
    rc = 0
    for stage in stages:
        if ctx.dry_run:
            _dry_run_print(stage, ctx)
            continue
        r = run_stage(stage, ctx, force=args.force)
        if r != 0:
            _log(f"[pipeline] stopping: {stage.name} failed (rc={r})")
            return r
    return rc


def _running_pid() -> int | None:
    """Returns the pid recorded in PID_FILE if that process is still alive, else None (also
    cleaning up a stale PID_FILE left by a process that has since exited)."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        # process exists but owned by someone else -- treat as running
        return pid
    return pid


def _detach(args) -> int:
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    if not getattr(args, "force_detach", False):
        pid = _running_pid()
        if pid is not None:
            print(f"error: a pipeline is already running (pid {pid}, see {PID_FILE}); pass --force-detach to start another anyway", file=sys.stderr)
            return 1
    argv = [a for a in sys.argv[1:] if a not in ("--detach", "--force-detach")]
    cmd = ["setsid", "nohup", "uv", "run", "python", "-u", "-m", "mapping.cli.pipeline", *argv]
    with open(PIPELINE_LOG, "a") as lf:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    tmp = PID_FILE.with_suffix(".pid.tmp")
    tmp.write_text(str(proc.pid))
    os.replace(tmp, PID_FILE)
    print(f"detached: pid {proc.pid} (see {PIPELINE_LOG}, {PID_FILE})")
    return 0


def cmd_status(args) -> int:
    ctx = Ctx(poses=args.poses or "corrected", tag=args.tag or "tw45")
    for stage in STAGES:
        marker = read_marker(stage.name)
        if marker is None:
            state = "pending"
        elif marker.get("rc") != 0:
            state = f"FAILED (rc={marker.get('rc')})"
        elif is_done(stage, ctx):
            state = "done"
        else:
            state = "stale (inputs changed)"
        tag = " [optional]" if stage.optional else ""
        print(f"{stage.name:16s} {state}{tag}")
    return 0


def cmd_rehash(args) -> int:
    """Re-stamp `inputs_hash` of every successful marker with the hash of its CURRENT inputs. Use only after a
    change of how inputs are hashed or addressed (e.g. cache root moved/symlinked, hashing switched to resolved
    paths) when you know the recorded outputs are still the ones those inputs produce -- it silences a
    spurious "stale" without re-running anything, and would equally silence a real one."""
    ctx = Ctx(poses=args.poses or "corrected", tag=args.tag or "tw45")
    n = 0
    for stage in STAGES:
        marker = read_marker(stage.name)
        if marker is None or marker.get("rc") != 0:
            continue
        new = compute_inputs_hash(stage, ctx)
        if marker.get("inputs_hash") != new:
            marker["inputs_hash_previous"] = marker.get("inputs_hash")
            marker["inputs_hash"] = new
            marker["rehashed"] = _now()
            write_marker(stage.name, marker)
            n += 1
            print(f"{stage.name:16s} rehashed")
    print(f"{n} marker(s) re-stamped")
    return 0


def cmd_compare(args) -> int:
    ctx = Ctx(poses=args.poses or "corrected", tag=args.tag or "tw45")
    _do_compare(ctx)
    print((PIPELINE_DIR / "comparison.md").read_text())
    return 0


def cmd_env_check(args) -> int:
    if args.fix_symlink:
        _fix_symlink()
    ctx = Ctx(poses=args.poses or "corrected")
    checks = _do_env_check(ctx)
    print(json.dumps(checks, indent=1, default=str))
    stage = get_stage("env-check")
    marker = {
        "poses_hash": None,
        "git": _git_rev(),
        "started": _now(),
        "finished": _now(),
        "cmds": [],
        "rc": 0,
        "inputs_hash": compute_inputs_hash(stage, ctx),
        "outputs": [str(p) for p in stage.outputs(ctx)],
        "metrics": checks,
    }
    write_marker("env-check", marker)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--from", dest="from_", default=None, help="first stage to run (inclusive)")
    r.add_argument("--to", default=None, help="last stage to run (inclusive)")
    r.add_argument("--only", default=None, help="comma-separated stage names, overrides --from/--to")
    r.add_argument("--force", action="store_true", help="rerun a stage even if its marker says it's done")
    r.add_argument("--poses", default="corrected", help='pose table for stages that need one (default "corrected")')
    r.add_argument("--tag", default="tw45", help="colorize run tag")
    r.add_argument("--dry-run", action="store_true", help="print every command without executing")
    r.add_argument("--detach", action="store_true", help="re-exec under setsid nohup, write pipeline.pid, return immediately")
    r.add_argument("--force-detach", action="store_true", help="with --detach, start a new detached run even if pipeline.pid names a still-running process")
    r.add_argument("--with-optional", action="store_true", help="also run traj-validate/reg-conflict")

    s = sub.add_parser("status")
    s.add_argument("--poses", default="corrected")
    s.add_argument("--tag", default="tw45")

    c = sub.add_parser("compare")

    rh = sub.add_parser("rehash", help="re-stamp inputs_hash of done markers after a hashing/path change (see cmd_rehash)")

    rh.add_argument("--poses", default=None)

    rh.add_argument("--tag", default=None)
    c.add_argument("--poses", default="corrected")
    c.add_argument("--tag", default="tw45")

    e = sub.add_parser("env-check")
    e.add_argument("--fix-symlink", action="store_true", help="create the Geovap_cache symlink if missing")
    e.add_argument("--poses", default="corrected")

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "rehash":
        return cmd_rehash(args)

    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "env-check":
        return cmd_env_check(args)
    ap.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
