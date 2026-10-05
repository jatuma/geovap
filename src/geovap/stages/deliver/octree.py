"""`potree`: build the Potree octree the viewer streams.

ONE octree now, not three -- `merge` writes one consolidated LAZ per tile, so there is only one
cloud to convert. The previous three-octree layout (`cloud/`, `objects/`, `vendor/`) disappears with
it; `stages.deliver.merge`'s module docstring explains why the underlying LAZ set collapsed.

PotreeConverter converts every point attribute a LAZ carries -- it has no "keep these N extra
dims" flag -- so building the octree straight from `merge`'s output would put all 22 provenance
dimensions into every octree node the browser streams, paying disk and network cost for bytes the
viewer never reads (`img_grad`, `cam_dist`, `seg_src_frame`, ...). Instead this stage first writes a
THINNED copy of each tile carrying only the dimensions the viewer actually shades by --
`classification` and RGB are already standard LAS fields; `cluster_id`, `obj_class` and `dE00_med`
are the only extra dims kept -- and converts THAT. The full 22-dimension set stays in the delivered
LAZ (`merge`'s output, untouched); only the octree is thinned.

Also writes `crs.json` next to the octree (from `s.crs`), so the viewer reads the projection from
data instead of a hardcoded `proj4.defs(...)` literal, and copies `classes.json` (the common15
palette) and the consolidated viewer page.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from geovap.io.las_writer import CONSOLIDATED_DIMS
from geovap.stages.base.cli import stage_main
from geovap.stages.base.spec import StageSpec, registry

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings

#: The dimensions the viewer shades by. Standard LAS fields (xyz, rgb, classification) are always
#: carried; these are the only EXTRA dims kept in the octree -- see the module docstring.
VIEWER_EXTRA_DIMS: tuple[str, ...] = ("cluster_id", "obj_class", "dE00_med", "ref_r", "ref_g", "ref_b")

_INFRA_ROOT = Path(__file__).resolve().parents[2] / "infra"


def thin_tile(src: Path, dst: Path, extra_dims: tuple[str, ...] = VIEWER_EXTRA_DIMS) -> Path:
    """Copy `src` (a `merge` output tile) to `dst`, keeping only the standard LAS fields plus
    `extra_dims`. A straight per-point copy -- no reordering, no recomputation -- so it is cheap
    relative to `merge` itself and safe to redo on every `potree` run."""
    import laspy

    src_las = laspy.read(str(src))
    header = laspy.LasHeader(version="1.4", point_format=7)
    header.scales = src_las.header.scales
    header.offsets = src_las.header.offsets
    available = set(src_las.point_format.dimension_names)
    kept = [d for d in extra_dims if d in available]
    dims_meta = {n: (dt, desc) for n, dt, desc in CONSOLIDATED_DIMS}
    for name in kept:
        dt, desc = dims_meta.get(name, (np.float32, ""))
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dt, description=desc[:31]))
    out = laspy.LasData(header)
    for dim in ("X", "Y", "Z", "red", "green", "blue", "classification", "intensity", "gps_time",
                "point_source_id", "return_number", "number_of_returns"):
        try:
            setattr(out, dim, getattr(src_las, dim))
        except Exception:  # noqa: BLE001 - a source without this dimension simply does not carry it
            pass
    for name in kept:
        setattr(out, name, getattr(src_las, name))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(dst))
    return dst


def thin_dir(s: "Settings") -> Path:
    return s.workspace.out / "potree_stage" / "tiles"


def _potree_dir(s: "Settings") -> Path:
    return Path(s.paths.publish) / "consolidated"


def build_thin_tiles(s: "Settings", force: bool = False) -> list[Path]:
    src_dir = s.workspace.consolidated_tiles
    out_dir = thin_dir(s)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for src in sorted(src_dir.glob("*.laz")):
        dst = out_dir / src.name
        if force or not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            thin_tile(src, dst)
        written.append(dst)
    return written


def write_crs_json(s: "Settings") -> Path:
    """`crs.json` next to the octree, read by `infra/viewer/consolidated/index.html` instead of a
    hardcoded `proj4.defs(...)` literal."""
    out = _potree_dir(s) / "cloud" / "crs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"epsg": s.crs.epsg, "proj4": s.crs.proj4, "axes": s.crs.axes, "name": s.name}, indent=1))
    return out


def run_potree(s: "Settings", *, force: bool = False) -> dict:
    from geovap.runtime.procs import sh_cmd

    tiles = build_thin_tiles(s, force=force)
    if not tiles:
        raise FileNotFoundError(f"no consolidated tiles found under {s.workspace.consolidated_tiles}")

    compose = _INFRA_ROOT / "containers" / "compose.yml"
    cmd = sh_cmd(
        f"cd {compose.parent} && docker compose run --rm --user $(id -u):$(id -g) --entrypoint sh "
        "potreeconverter -c '/usr/local/bin/PotreeConverter $(ls /cache_out/potree_stage/tiles/*.laz) -o /output/consolidated/cloud'"
    )
    import os
    import subprocess

    env = dict(os.environ, **s.env())
    r = subprocess.run(cmd, env=env)
    if r.returncode:
        raise RuntimeError(f"PotreeConverter failed with rc={r.returncode}")

    write_crs_json(s)

    classes_src = s.workspace.consolidated / "classes.json"
    if classes_src.exists():
        shutil.copyfile(classes_src, _potree_dir(s) / "cloud" / "classes.json")

    index_src = _INFRA_ROOT / "viewer" / "consolidated" / "index.html"
    shutil.copyfile(index_src, _potree_dir(s) / "index.html")

    meta_path = _potree_dir(s) / "cloud" / "metadata.json"
    return {"n_tiles": len(tiles), "metadata": str(meta_path), "points": _read_points(meta_path)}


def _read_points(meta_path: Path) -> int | None:
    try:
        return json.loads(meta_path.read_text()).get("points")
    except Exception:  # noqa: BLE001
        return None


# ================================================================================================ stage
class Potree:
    spec = StageSpec(
        name="potree", after=("merge",), est_min=15,
        summary="Potree octree of the consolidated cloud (thinned to the viewer-shading dimensions)",
    )

    def available(self, s: "Settings") -> bool:
        return True

    def inputs(self, s: "Settings") -> dict[str, Path]:
        return {"consolidated_summary": s.workspace.consolidated / "summary.json"}

    def outputs(self, s: "Settings") -> list[Path]:
        return [_potree_dir(s) / "cloud" / "metadata.json", _potree_dir(s) / "cloud" / "crs.json"]

    def metrics(self, s: "Settings") -> dict[str, Any]:
        try:
            return {"points": _read_points(_potree_dir(s) / "cloud" / "metadata.json")}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    def run(self, s: "Settings", *, force: bool = False) -> None:
        run_potree(s, force=force)


STAGE = registry.add(Potree())


def _add_options(p) -> None:
    p.add_argument("--force", action="store_true", help="rethin and reconvert even if the octree already exists")


def _to_opts(args) -> dict:
    return {"force": args.force}


def main(argv=None) -> int:
    return stage_main(STAGE, argv, add_options=_add_options, to_opts=_to_opts)


if __name__ == "__main__":
    raise SystemExit(main())
