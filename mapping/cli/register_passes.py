"""S5 CLI: extract patches, register pass pairs, compute JVF offsets, solve the global pose graph,
write pass_transforms.json and (optionally) a registered pose table / registered LAS.

    uv run python -m mapping.cli.register_passes extract    # cache patches for all 30 passes
    uv run python -m mapping.cli.register_passes pairs      # pairwise ICP over overlapping passes
    uv run python -m mapping.cli.register_passes jvf        # own-pass curbs vs JVF road boundary (cloud; informational, see pass_reg.jvf_offset docstring)
    uv run python -m mapping.cli.register_passes jvf_photo  # per-pass (dE,dN) by photo-edge grid search -- the datum solve actually uses
    uv run python -m mapping.cli.register_passes solve [--datum none|jvf-cloud|jvf-photo]
                                                          # global pose graph -> pass_transforms.json
                                                          # default "none": pairwise-only, weak identity
                                                          # prior, no absolute JVF constraint (production;
                                                          # see pass_reg.py module docstring). jvf-cloud/
                                                          # jvf-photo kept for reference, not adopted.
    uv run python -m mapping.cli.register_passes poses      # apply_pass_transforms -> poses_export_registered.csv
    uv run python -m mapping.cli.register_passes all [--datum ...]   # extract, pairs, jvf, jvf_photo, solve, poses
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import mapping.pass_reg as pr
from geovap.domain.model.frames import FrameIndex
from mapping.poses import load_poses, write_pose_table

PASS_REG_DIR = pr.PASS_REG_DIR


def cmd_extract(workers: int = 8):
    pr.run_extract_all(workers=workers)


def cmd_pairs(workers: int = 8):
    poses = load_poses()
    pairs = pr.overlap_pairs(poses)
    cache = {}

    def get(pid):
        if pid not in cache:
            cache[pid] = pr.Patches.load(PASS_REG_DIR / f"patches_p{pid:02d}.npz")
        return cache[pid]

    out = {}
    for a, b in pairs:
        Pa, Pb = get(a), get(b)
        res = pr.register_pair(Pb, Pa)  # query=b onto ref=a
        out[f"{a}_{b}"] = res
        print(f"{a:2d}-{b:2d}: n={res['n']:5d} rms {res['rms_before']} -> {res['rms_after']}  T={np.round(res['T'], 3).tolist()} sv_min={res['sv_min']:.3g}")
    (PASS_REG_DIR / "pairs.json").write_text(json.dumps(out, indent=2))
    return out


def cmd_jvf():
    poses = load_poses()
    pass_ids = sorted(int(p) for p in np.unique(poses.pass_id))
    road_lines = pr.load_road_boundary()
    index = pr.RoadBoundaryIndex(road_lines)
    print(f"road boundary: {len(road_lines)} polylines, {len(index.samples)} resampled points")
    out = {}
    for p in pass_ids:
        pat = pr.Patches.load(PASS_REG_DIR / f"patches_p{p:02d}.npz")
        res = pr.jvf_offset(p, pat, index)
        out[p] = res
        anchor = res["rms"] is not None and max(abs(res["dE"]), abs(res["dN"])) < 0.1
        print(f"pass {p:2d}: n={res['n']:5d} dE={res['dE']:+.3f} dN={res['dN']:+.3f} dH={res['dH']:+.3f} rms={res['rms']} anchor={anchor}")
    (PASS_REG_DIR / "jvf_offsets.json").write_text(json.dumps(out, indent=2))
    return out


def cmd_jvf_photo(n_frames: int = pr.PHOTO_FRAMES_PER_PASS):
    """Per-pass (dE, dN[, dH]) by photo-edge grid search (`pr.jvf_offset_photometric`) -- this is the
    JVF datum `solve` actually consumes; see `pass_reg.py` module docstring / `jvf_offset` docstring
    for why the cloud-curb version (`cmd_jvf`) is kept only as a secondary/informational signal."""
    poses = load_poses()
    fi = FrameIndex(poses)
    pass_ids = sorted(int(p) for p in np.unique(poses.pass_id))
    road_lines = pr.load_road_boundary()
    index = pr.RoadBoundaryIndex(road_lines)
    print(f"road boundary: {len(road_lines)} polylines, {len(index.samples)} resampled points")
    out = {}
    for p in pass_ids:
        res = pr.jvf_offset_photometric(p, poses, fi, index, n_frames=n_frames)
        out[p] = res
        anchor = res["score_px"] is not None and max(abs(res["dE"]), abs(res["dN"])) < 0.1
        print(f"pass {p:2d}: n_frames={res['n_frames']:2d} dE={res['dE']:+.3f} dN={res['dN']:+.3f} dH={res['dH']:+.3f} score_px={res['score_px']} dH_n={res['dH_n']} anchor={anchor}")
    (PASS_REG_DIR / "jvf_offsets_photo.json").write_text(json.dumps(out, indent=2))
    return out


DATUM_CHOICES = ("none", "jvf-cloud", "jvf-photo")


def cmd_solve(datum: str = "none"):
    """`datum`: "none" (production default -- pairwise-only, weak identity prior, no absolute JVF
    constraint; see pass_reg.py module docstring for why), "jvf-cloud" (jvf_offsets.json, informational
    -- see jvf_offset docstring for why it is unreliable), or "jvf-photo" (jvf_offsets_photo.json,
    the best absolute-datum attempt tried -- still not adopted, see jvf_offset_photometric docstring)."""
    if datum not in DATUM_CHOICES:
        raise ValueError(f"--datum must be one of {DATUM_CHOICES}, got {datum!r}")
    poses = load_poses()
    pass_ids = sorted(int(p) for p in np.unique(poses.pass_id))
    pairs_raw = json.loads((PASS_REG_DIR / "pairs.json").read_text())
    pair_results = {}
    for k, v in pairs_raw.items():
        a, b = (int(x) for x in k.split("_"))
        pair_results[(a, b)] = v

    jvf_results: dict[int, dict] = {}
    jvf_source = "none"
    if datum == "jvf-photo":
        jvf_source = "photometric"
        jvf_raw = json.loads((PASS_REG_DIR / "jvf_offsets_photo.json").read_text())
        jvf_results = {int(k): v for k, v in jvf_raw.items()}
    elif datum == "jvf-cloud":
        jvf_source = "cloud_curb"
        jvf_raw = json.loads((PASS_REG_DIR / "jvf_offsets.json").read_text())
        jvf_results = {int(k): v for k, v in jvf_raw.items()}

    gr = pr.solve_global(pass_ids, pair_results, jvf_results)

    out = {"passes": {}, "summary": {**gr.residuals, "n_pairs": len(pair_results), "n_jvf_anchors": 0, "jvf_source": jvf_source, "datum": datum}, "poses_source": poses.source}
    n_anchor = 0
    t_all, yaw_all = [], []
    for p in pass_ids:
        x = gr.x[p]
        jr = jvf_results.get(p, {})
        is_anchor = jr.get("rms") is not None and max(abs(jr.get("dE", 0)), abs(jr.get("dN", 0))) < 0.1
        n_anchor += int(is_anchor)
        n_pairs_p = sum(1 for (a, b) in pair_results if a == p or b == p)
        cx, cy = gr.centre.get(p, [float(poses.origin[poses.pass_id == p, 0].mean()), float(poses.origin[poses.pass_id == p, 1].mean())])
        flag = bool(np.linalg.norm(x[:3]) > 0.3 or abs(x[3]) > 0.3)
        t_all.append(x[:3])
        yaw_all.append(x[3])
        out["passes"][str(p)] = {
            "yaw_deg": x[3],
            "t": [x[0], x[1], x[2]],
            "centre": [cx, cy],
            "rms": jr.get("rms"),
            "n_pairs": n_pairs_p,
            "anchor": is_anchor,
            "flag": flag,
            "jvf": jr,
        }
    out["summary"]["n_jvf_anchors"] = n_anchor
    # gauge check (datum=none): the weak identity prior alone fixes the global shift -- the mean
    # transform over all 30 nodes should sit near zero (no absolute datum is pulling it anywhere else).
    t_all = np.array(t_all)
    out["summary"]["mean_t"] = t_all.mean(axis=0).tolist()
    out["summary"]["mean_yaw_deg"] = float(np.mean(yaw_all))
    out["summary"]["n_flagged"] = sum(1 for v in out["passes"].values() if v["flag"])
    (PASS_REG_DIR / "pass_transforms.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    return out


def cmd_poses():
    poses = load_poses()
    transforms = json.loads((PASS_REG_DIR / "pass_transforms.json").read_text())
    poses2 = pr.apply_pass_transforms(poses, transforms)
    meta = {
        "status": np.array(["registered"] * len(poses2), dtype=object),
        "src": np.array(["pass_reg"] * len(poses2), dtype=object),
        "dt_s": np.zeros(len(poses2)),
    }
    out = write_pose_table(
        poses2,
        meta,
        {
            "stage": "S5_pass_reg",
            "input_poses_hash": poses.hash(),
            "pass_transforms": str(PASS_REG_DIR / "pass_transforms.json"),
            "datum": transforms.get("summary", {}).get("datum", "unknown"),
        },
        PASS_REG_DIR / "poses_export_registered.csv",
    )
    print("wrote", out)
    return out


CMDS = {"extract": cmd_extract, "pairs": cmd_pairs, "jvf": cmd_jvf, "jvf_photo": cmd_jvf_photo, "solve": cmd_solve, "poses": cmd_poses}


def _parse_datum(argv: list[str]) -> str:
    for i, a in enumerate(argv):
        if a == "--datum" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--datum="):
            return a.split("=", 1)[1]
    return "none"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    datum = _parse_datum(sys.argv[2:])
    if cmd == "all":
        cmd_extract()
        cmd_pairs()
        cmd_jvf()
        cmd_jvf_photo()
        cmd_solve(datum=datum)
        cmd_poses()
    elif cmd == "solve":
        cmd_solve(datum=datum)
    else:
        CMDS[cmd]()
