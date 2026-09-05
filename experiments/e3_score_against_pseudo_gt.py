"""E3 - skore zero-shot baseline (E2) proti realne JVF pseudo-GT (E1).

Pro kazdy JVF objekt pokryty v E1 (coverage.csv) a segmentovany v E2 (seg_*.npz):
projektuje znovu jeho vrcholy do prirazeneho snimku, vytvori tenky "trubkovy" pas kolem linie
(sirka podle tolerance 14 cm prevedene na pixely v dane vzdalenosti, viz SS8 dokumentu) a
zmeri, jaky podil pixelu v tom pasu model oznacil verejnou tridou odpovidajici jeho JVF kodu
(class_map.py). Vysledek je "matice schopnosti po tridach" na realnych datech.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import torch

from common import class_map, io_data
from common.camera import world_to_panorama_px

E1_DIR = os.path.join(os.path.dirname(__file__), "out", "e1")
E2_DIR = os.path.join(os.path.dirname(__file__), "out", "e2")
OUT_PATH = os.path.join(os.path.dirname(__file__), "out", "e3", "per_class_score.txt")

TOL_M = 0.14  # zakonna tolerance m_xy, SS1 a SS8 03_semanticka_segmentace.md
FUSE_DOWNSCALE = 2  # musi sedet s e2_zero_shot_baseline.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_seg(frame_idx: int):
    path = os.path.join(E2_DIR, f"seg_{frame_idx}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["seg"]


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    frames = io_data.load_frames()
    objects = io_data.load_jvf_objects()
    by_uid = {o.uid: o for o in objects}

    rows = list(csv.DictReader(open(os.path.join(E1_DIR, "coverage.csv"), encoding="utf-8")))
    seg_cache: dict[int, np.ndarray | None] = {}

    manifest_path = os.path.join(E2_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        print("E2 jeste nedobehl (chybi manifest.json) - spustit e2_zero_shot_baseline.py prv.")
        return
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    id2label = manifest["id2label"]
    label2id = {v.lower(): int(k) for k, v in id2label.items()}

    scored = defaultdict(list)  # jvfcode -> list[hit_rate]
    n_no_public_class = 0
    n_no_seg = 0

    for r in rows:
        if r["covered"] != "True":
            continue
        frame_idx = int(r["nearest_frame_idx"])
        if frame_idx not in seg_cache:
            seg_cache[frame_idx] = load_seg(frame_idx)
        seg = seg_cache[frame_idx]
        if seg is None:
            n_no_seg += 1
            continue

        obj = by_uid.get(r["uid"])
        if obj is None:
            continue
        info = class_map.lookup(obj.jvfcode)
        if info is None or info.public is None:
            n_no_public_class += 1
            continue
        public_names = info.public.split("|")
        target_ids = {label2id[n] for n in public_names if n in label2id}
        if not target_ids:
            n_no_public_class += 1
            continue

        pts = torch.tensor(obj.coords, dtype=torch.float64, device=device)
        origin = torch.tensor(frames.origin_enh[frame_idx], dtype=torch.float64, device=device)
        roll = torch.tensor(frames.roll_deg[frame_idx], dtype=torch.float64, device=device)
        pitch = torch.tensor(frames.pitch_deg[frame_idx], dtype=torch.float64, device=device)
        yaw = torch.tensor(frames.yaw_deg[frame_idx], dtype=torch.float64, device=device)
        u, v, el = world_to_panorama_px(pts, origin, roll, pitch, yaw)
        u, v = u.cpu().numpy(), v.cpu().numpy()

        seg_h, seg_w = seg.shape
        scale_u, scale_v = seg_w / io_data_pano_w(), seg_h / io_data_pano_h()
        u_s, v_s = u * scale_u, v * scale_v

        dist = float(np.linalg.norm(obj.coords[:, :2].mean(axis=0) - frames.origin_enh[frame_idx, :2]))
        px_per_m = (seg_w / 360.0) / (dist * np.pi / 180.0) if dist > 0.5 else seg_w / 4  # hruby odhad uhloveho rozliseni
        radius_px = max(2, int(round(TOL_M * px_per_m)))

        mask = np.zeros((seg_h, seg_w), dtype=np.uint8)
        pts_i = np.stack([u_s, v_s], axis=1).astype(np.int32)
        if obj.geom_type == "LineString" and len(pts_i) > 1:
            for i in range(len(pts_i) - 1):
                if abs(int(pts_i[i, 0]) - int(pts_i[i + 1, 0])) > seg_w / 2:
                    continue
                cv2.line(mask, tuple(pts_i[i]), tuple(pts_i[i + 1]), 1, thickness=radius_px * 2)
        else:
            for p in pts_i:
                cv2.circle(mask, tuple(p), radius_px * 2, 1, -1)

        buffer_px = mask.sum()
        if buffer_px == 0:
            continue
        hit = np.isin(seg[mask.astype(bool)], list(target_ids)).sum()
        scored[obj.jvfcode].append(hit / buffer_px)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        header = f"{'n':>4} {'prumer skore':>13}  jvfcode  nazev -> verejna trida\n"
        print(header, end="")
        f.write(header)
        for code, scores in sorted(scored.items(), key=lambda kv: -np.mean(kv[1])):
            info = class_map.lookup(code)
            line = f"{len(scores):4d} {np.mean(scores):13.3f}  {code}  {info.name_cz} -> {info.public}\n"
            print(line, end="")
            f.write(line)
        footer = (
            f"\nobjektu bez verejneho ekvivalentu (vlastni tridy, nelze zero-shot skorovat): {n_no_public_class}\n"
            f"objektu s pridelenym snimkem, ktery E2 nezpracoval (mimo vzorek): {n_no_seg}\n"
        )
        print(footer, end="")
        f.write(footer)
    print(f"zapsano {OUT_PATH}")


def io_data_pano_w() -> int:
    return 8000


def io_data_pano_h() -> int:
    return 4000


if __name__ == "__main__":
    main()
