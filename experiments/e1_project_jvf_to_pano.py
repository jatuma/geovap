"""E1 - projekce JVF vektoru (linie/body) do panoramat = pseudo-ground-truth zdarma.

Pro kazdy JVF objekt najde nejblizsi panorama (prostorove, do MAX_DIST metru),
promitne jeho vrcholy pres camera.world_to_panorama_px (vektorizovane na GPU) a
vykresli jako masku + barevny overlay pro rychlou vizualni kontrolu.

Vystup:
  out/e1/coverage.csv       - pro kazdy JVF objekt: jvfcode, uid, nejblizsi snimek, vzdalenost, pokryto ano/ne
  out/e1/coverage_by_class.txt - shrnuti po tridach (kolik z n instanci padlo do dosahu snimku)
  out/e1/overlays/*.jpg     - nekolik desitek nahledovych snimku s prekreslenymi JVF liniemi
"""
from __future__ import annotations

import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch

from common import class_map, io_data
from common.camera import world_to_panorama_px

MAX_DIST_M = 20.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "e1")
OVERLAY_DIR = os.path.join(OUT_DIR, "overlays")
OVERLAY_SCALE = 4  # ulozit nahledy zmensene 8000x4000 -> 2000x1000
N_OVERLAYS_PER_CLASS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assign_objects_to_frames(objects, frames) -> np.ndarray:
    """Pro kazdy objekt vrati index nejblizsiho snimku (podle vzdalenosti stred objektu <-> origin snimku)."""
    centroids = np.stack([o.coords[:, :2].mean(axis=0) for o in objects])  # [N,2]
    centroids_t = torch.tensor(centroids, dtype=torch.float64, device=device)  # [N,2]
    origins_t = torch.tensor(frames.origin_enh[:, :2], dtype=torch.float64, device=device)  # [M,2]

    best_idx = torch.empty(len(objects), dtype=torch.long)
    best_dist = torch.empty(len(objects), dtype=torch.float64)
    chunk = 512
    for start in range(0, len(objects), chunk):
        end = min(start + chunk, len(objects))
        d = torch.cdist(centroids_t[start:end], origins_t)  # [chunk, M]
        dmin, imin = d.min(dim=1)
        best_dist[start:end] = dmin.cpu()
        best_idx[start:end] = imin.cpu()
    return best_idx.numpy(), best_dist.numpy()


def project_object(obj, frame_idx: int, frames) -> tuple[np.ndarray, np.ndarray]:
    """Vrati (u, v) pole pro vsechny vrcholy objektu v danem snimku."""
    pts = torch.tensor(obj.coords, dtype=torch.float64, device=device)  # [K,3]
    origin = torch.tensor(frames.origin_enh[frame_idx], dtype=torch.float64, device=device)
    roll = torch.tensor(frames.roll_deg[frame_idx], dtype=torch.float64, device=device)
    pitch = torch.tensor(frames.pitch_deg[frame_idx], dtype=torch.float64, device=device)
    yaw = torch.tensor(frames.yaw_deg[frame_idx], dtype=torch.float64, device=device)
    u, v, el = world_to_panorama_px(pts, origin, roll, pitch, yaw)
    return u.cpu().numpy(), v.cpu().numpy()


def draw_polyline_wrapped(canvas, u, v, color, thickness):
    """Kresli polyline v ERP prostoru, segmenty pres sev (velky skok v u) preskoci."""
    pano_w = canvas.shape[1]
    pts = np.stack([u, v], axis=1)
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        if abs(p0[0] - p1[0]) > pano_w / 2:
            continue  # sev - segment by obtekl cely obraz, radeji vynechat
        cv2.line(canvas, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), color, thickness)


def main() -> None:
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    random.seed(0)

    frames = io_data.load_frames()
    objects = io_data.load_jvf_objects()
    print(f"snimku: {len(frames)}, objektu: {len(objects)}, zarizeni: {device}")

    idx, dist = assign_objects_to_frames(objects, frames)

    covered = dist <= MAX_DIST_M
    print(f"pokryto (vzdalenost <= {MAX_DIST_M} m od nejblizsiho snimku): {covered.sum()}/{len(objects)}")

    by_class_total = defaultdict(int)
    by_class_covered = defaultdict(int)
    examples_by_class: dict[str, list[int]] = defaultdict(list)
    rows = []
    for i, obj in enumerate(objects):
        by_class_total[obj.jvfcode] += 1
        if covered[i]:
            by_class_covered[obj.jvfcode] += 1
            examples_by_class[obj.jvfcode].append(i)
        rows.append((obj.jvfcode, obj.uid, int(idx[i]), float(dist[i]), bool(covered[i])))

    csv_path = os.path.join(OUT_DIR, "coverage.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("jvfcode,uid,nearest_frame_idx,dist_m,covered\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.2f},{r[4]}\n")
    print(f"zapsano {csv_path}")

    summary_path = os.path.join(OUT_DIR, "coverage_by_class.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for code, total in sorted(by_class_total.items(), key=lambda kv: -kv[1]):
            info = class_map.lookup(code)
            name = info.name_cz if info else code
            cov = by_class_covered.get(code, 0)
            line = f"{cov:5d}/{total:<5d} ({100*cov/total:5.1f}%)  {code}  {name}"
            print(line)
            f.write(line + "\n")
    print(f"zapsano {summary_path}")

    # overlaye pro nekolik prikladu z kazde tridy (radeji vzacne tridy)
    priority_codes = ["0100000162", "0100000299", "0100000193", "0100000199", "0100000084", "0100000304"]
    codes_to_draw = priority_codes + [c for c in examples_by_class if c not in priority_codes]

    n_saved = 0
    for code in codes_to_draw:
        ex = examples_by_class.get(code, [])
        random.shuffle(ex)
        for obj_i in ex[:N_OVERLAYS_PER_CLASS]:
            obj = objects[obj_i]
            frame_i = int(idx[obj_i])
            img_path = frames.path(frame_i)
            img = cv2.imread(img_path)
            if img is None:
                continue
            u, v = project_object(obj, frame_i, frames)
            color = (0, 0, 255) if obj.geom_type == "LineString" else (0, 255, 0)
            if obj.geom_type == "LineString" and len(u) > 1:
                draw_polyline_wrapped(img, u, v, color, thickness=6)
            else:
                for uu, vv in zip(u, v):
                    cv2.circle(img, (int(round(uu)), int(round(vv))), 14, color, -1)
            small = cv2.resize(img, (img.shape[1] // OVERLAY_SCALE, img.shape[0] // OVERLAY_SCALE))
            out_name = f"{code}_{obj.uid[:8]}_frame{frame_i}.jpg"
            cv2.imwrite(os.path.join(OVERLAY_DIR, out_name), small)
            n_saved += 1
    print(f"ulozeno {n_saved} nahledovych overlay obrazku do {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
