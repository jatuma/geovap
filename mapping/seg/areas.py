"""JVF areas: polygonize boundary lines and classify faces from the definition points.

The JVF export carries areas only as boundary LineStrings plus one definition Point per area, so faces
have to be rebuilt (the IS DTM does the same "plochování"). Face statuses:
    resolved      exactly one surface-tier class (small-tier points inside are ignored)
    merged        several surface classes of one visual group (road + verge) -> diagnostic class road_or_verge
    small         only small-tier points, face < SMALL_FACE_MAX_M2
    sliver        no points, tiny, inherits the dominant neighbour
    conflict      >= 2 distinct surface-tier classes -> unlabeled (a missing outline, do not guess)
    unresolved    no points (or only small-tier points in a large face)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import polygonize_full, unary_union
from shapely.strtree import STRtree

from .. import compat
from ..config import CACHE_ROOT
from . import classes as C

SEGDS_DIR = CACHE_ROOT / "segds"
AREAS_DIR = SEGDS_DIR / "areas"
PRECISION_M = 0.01


@dataclass
class Face:
    face_id: int
    polygon: Polygon
    status: str = "unresolved"
    jvf_code: str | None = None
    gt_class: str | None = None
    points: list[str] = field(default_factory=list)  # jvf codes of definition points inside

    @property
    def area(self) -> float:
        return float(self.polygon.area)

    @property
    def class_id(self) -> int:
        return C.BY_NAME[self.gt_class].id if self.gt_class else C.IGNORE


def load_objects():
    compat.ensure_experiments_on_path()
    from common import io_data

    return io_data.load_jvf_objects()


def boundary_lines(objects, all_lines: bool = False) -> list[LineString]:
    out = []
    for o in objects:
        if o.geom_type not in ("LineString", "Polygon") or len(o.coords) < 2:
            continue
        if not all_lines and o.jvfcode not in C.HARD_BOUNDARY_CODES:
            continue
        xy = o.coords[:, :2]
        if o.geom_type == "Polygon" and not np.allclose(xy[0], xy[-1]):
            xy = np.vstack([xy, xy[:1]])
        if np.isnan(xy).any():
            continue
        out.append(LineString(xy))
    return out


def definition_points(objects) -> list[tuple[str, Point]]:
    """(jvf base code, point) of area definition points: Point geometries whose code is an area code."""
    return [(o.jvfcode, Point(o.coords[0, :2])) for o in objects if o.geom_type == "Point" and o.jvfcode in C.AREA_CLASS]


def polygonize_faces(lines: list[LineString], precision_m: float = PRECISION_M) -> tuple[list[Face], dict]:
    merged = unary_union(lines)
    if precision_m > 0:
        merged = shapely.set_precision(merged, precision_m)
    polys, dangles, cuts, invalid = polygonize_full(merged)
    faces = [Face(i, p) for i, p in enumerate(polys.geoms)]
    stats = {
        "n_lines": len(lines),
        "n_faces": len(faces),
        "dangle_len_m": float(dangles.length),
        "cut_len_m": float(cuts.length),
        "invalid_len_m": float(invalid.length),
        "faces_area_m2": float(sum(f.area for f in faces)),
    }
    return faces, stats


def assign_classes(faces: list[Face], def_points: list[tuple[str, Point]]) -> None:
    tree = STRtree([f.polygon for f in faces])
    for code, pt in def_points:
        idx = tree.query(pt, predicate="within")
        for i in idx:
            faces[i].points.append(code)
    for f in faces:
        surface = sorted({c for c in f.points if c not in C.SMALL_CODES})
        surface_cls = {C.AREA_CLASS[c] for c in surface}
        small = sorted({c for c in f.points if c in C.SMALL_CODES})
        if len(surface_cls) == 1:
            f.status = "resolved"
            f.jvf_code = surface[0] if len(surface) == 1 else _nearest_code(f, def_points, surface)
            f.gt_class = surface_cls.pop()
        elif len(surface_cls) > 1:
            merged = C.CONFLICT_MERGE.get(frozenset(surface_cls))
            if merged:
                f.status, f.gt_class = "merged", merged
            else:
                f.status = "conflict"
        elif small:
            code = small[0] if len(small) == 1 else _nearest_code(f, def_points, small)
            if f.area < C.SMALL_FACE_MAX_M2.get(code, C.SMALL_FACE_MAX_M2["default"]):
                f.status, f.jvf_code, f.gt_class = "small", code, C.AREA_CLASS[code]
            else:
                f.status = "unresolved"
        else:
            f.status = "unresolved"
    _inherit_slivers(faces)


def _nearest_code(face: Face, def_points, codes: list[str]) -> str:
    c = face.polygon.centroid
    best = min(((c.distance(p), code) for code, p in def_points if code in codes and face.polygon.contains(p)), default=(0, codes[0]))
    return best[1]


def _inherit_slivers(faces: list[Face]) -> None:
    labelled = [f for f in faces if f.gt_class]
    if not labelled:
        return
    tree = STRtree([f.polygon for f in labelled])
    for f in faces:
        if f.status != "unresolved" or f.points or f.area >= C.SLIVER_MAX_M2:
            continue
        perim = f.polygon.length
        if perim <= 0:
            continue
        for j in tree.query(f.polygon, predicate="intersects"):
            g = labelled[j]
            shared = f.polygon.boundary.intersection(g.polygon.boundary).length
            if shared >= C.SLIVER_SHARED_FRAC * perim:
                f.status, f.jvf_code, f.gt_class = "sliver", g.jvf_code, g.gt_class
                break


def build(all_lines: bool = True, out_dir: Path = AREAS_DIR, objects=None) -> tuple[list[Face], dict]:
    objects = objects if objects is not None else load_objects()
    lines = boundary_lines(objects, all_lines)
    faces, stats = polygonize_faces(lines)
    pts = definition_points(objects)
    assign_classes(faces, pts)
    report = summarize(faces, pts, stats)
    report["all_lines"] = all_lines
    out_dir.mkdir(parents=True, exist_ok=True)
    write_geojson(faces, out_dir / "faces.geojson")
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    try:
        plot_map(faces, out_dir / "map.png")
    except Exception as e:  # matplotlib optional at runtime
        report["map_error"] = str(e)
    return faces, report


def summarize(faces: list[Face], pts, stats: dict) -> dict:
    by_status = defaultdict(lambda: [0, 0.0])
    by_class = defaultdict(lambda: [0, 0.0])
    for f in faces:
        by_status[f.status][0] += 1
        by_status[f.status][1] += f.area
        if f.gt_class:
            by_class[f.gt_class][0] += 1
            by_class[f.gt_class][1] += f.area
    total = stats["faces_area_m2"]
    labelled = sum(a for _, a in (by_class.values()))
    conflicts = [{"face_id": f.face_id, "centroid": list(map(float, f.polygon.centroid.coords[0])), "area_m2": round(f.area, 1), "codes": sorted(set(f.points))} for f in faces if f.status == "conflict"]
    unresolved_big = sorted(({"face_id": f.face_id, "centroid": list(map(float, f.polygon.centroid.coords[0])), "area_m2": round(f.area, 1), "codes": sorted(set(f.points))} for f in faces if f.status == "unresolved"), key=lambda d: -d["area_m2"])[:40]
    n_inside = sum(len(f.points) for f in faces)
    return {
        **stats,
        "n_def_points": len(pts),
        "def_points_inside_faces": n_inside,
        "labelled_area_m2": labelled,
        "labelled_frac": labelled / total if total else 0.0,
        "by_status": {k: {"n": v[0], "area_m2": round(v[1], 1)} for k, v in sorted(by_status.items())},
        "by_class": {k: {"n": v[0], "area_m2": round(v[1], 1)} for k, v in sorted(by_class.items(), key=lambda kv: -kv[1][1])},
        "conflicts": conflicts,
        "unresolved_largest": unresolved_big,
    }


def write_geojson(faces: list[Face], path: Path) -> None:
    feats = []
    for f in faces:
        feats.append({
            "type": "Feature",
            "geometry": mapping(f.polygon),
            "properties": {"face_id": f.face_id, "status": f.status, "jvf_code": f.jvf_code, "gt_class": f.gt_class, "class_id": f.class_id, "area_m2": round(f.area, 2), "n_points": len(f.points), "point_codes": ",".join(sorted(set(f.points)))},
        })
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def load_faces(path: Path = AREAS_DIR / "faces.geojson") -> list[Face]:
    d = json.loads(Path(path).read_text())
    out = []
    for ft in d["features"]:
        p = ft["properties"]
        f = Face(p["face_id"], shapely.geometry.shape(ft["geometry"]), p["status"], p["jvf_code"], p["gt_class"], p["point_codes"].split(",") if p["point_codes"] else [])
        out.append(f)
    return out


def plot_map(faces: list[Face], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPoly

    pal = C.palette() / 255.0
    fig, ax = plt.subplots(figsize=(16, 16))
    patches, cols = [], []
    for f in faces:
        col = pal[f.class_id] if f.gt_class else (1.0, 0.2, 0.2) if f.status == "conflict" else (0.85, 0.85, 0.85)
        patches.append(MplPoly(np.asarray(f.polygon.exterior.coords), closed=True))
        cols.append(col)
    pc = PatchCollection(patches, facecolors=cols, edgecolors="k", linewidths=0.15)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title("JVF faces: class colours, red = conflict, grey = unresolved")
    handles = [plt.Rectangle((0, 0), 1, 1, color=pal[c.id]) for c in C.CLASSES] + [plt.Rectangle((0, 0), 1, 1, color=(1, 0.2, 0.2)), plt.Rectangle((0, 0), 1, 1, color=(0.85, 0.85, 0.85))]
    ax.legend(handles, [c.name for c in C.CLASSES] + ["conflict", "unresolved"], loc="upper right", fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
