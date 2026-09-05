"""E2 - zero-shot baseline segmentace panoramat pomoci hotoveho Mask2Former (Mapillary Vistas
taxonomie, checkpoint facebook/mask2former-swin-large-mapillary-vistas-semantic, MIT kod /
vistas vahy - vyzkumne pouziti, viz SS5.1 03_semanticka_segmentace.md).

Recept podle SS4.3 dokumentu: prstenec 8x sklon 0 stupnu (FOV 90x90, krok 45 = prekryv 50 %)
+ 4x sklon -45 stupnu na vozovku. Cela reprojekce, model i fuze bezi na GPU.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from common import io_data
from common.reproject import extract_view, fuse_views_to_erp, make_extract_grid, pad_pano_horizontal

CKPT = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
VIEW_SIZE = 1024
FUSE_DOWNSCALE = 2  # fuze bezi na out_w/out_h = pano // FUSE_DOWNSCALE, aby se buffer [1,65,H,W] vesel do 24 GB GPU
VOID_LABEL = 255  # pixely bez zadneho pohledu (zenit/nadir) - argmax by tam byl nesmyslny
N_SAMPLE_FRAMES = int(os.environ.get("N_SAMPLE_FRAMES", "10"))  # smoke test; zvysit az po vizualni kontrole
OUT_DIR = os.path.join(os.path.dirname(__file__), "out", "e2")

RING_VIEWS = [(yaw, 0.0, 90.0) for yaw in range(0, 360, 45)]
DOWN_VIEWS = [(yaw, -45.0, 90.0) for yaw in (0, 90, 180, 270)]
ALL_VIEWS = RING_VIEWS + DOWN_VIEWS


def load_model(device: torch.device):
    proc = Mask2FormerImageProcessor.from_pretrained(CKPT)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(CKPT).to(device).eval()
    return proc, model


def colorize(seg: np.ndarray, num_labels: int) -> np.ndarray:
    rng = np.random.RandomState(0)
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[:num_labels] = rng.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    palette[VOID_LABEL] = (0, 0, 0)
    return palette[seg]


@torch.no_grad()
def segment_panorama(img_bgr: np.ndarray, proc, model, device) -> tuple[np.ndarray, np.ndarray]:
    pano_h, pano_w = img_bgr.shape[:2]
    pano = torch.from_numpy(img_bgr[..., ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)
    pad = pano_w // 4
    pano_p = pad_pano_horizontal(pano, pad)

    mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1) * 255

    view_probs = []
    for yaw, pitch, fov in ALL_VIEWS:
        grid = make_extract_grid(yaw, pitch, fov, VIEW_SIZE, pano_w, pano_h, pad, device)
        view = extract_view(pano_p, grid)  # [1,3,S,S], 0..255
        inp = (view - mean) / std
        out = model(pixel_values=inp)
        # potrebujeme pravdepodobnosti, ne jen argmax -> spocitat softmax rucne z class_queries/masks
        probs = mask2former_semantic_probs(out, VIEW_SIZE)
        view_probs.append(probs)

    out_w, out_h = pano_w // FUSE_DOWNSCALE, pano_h // FUSE_DOWNSCALE
    fused, coverage = fuse_views_to_erp(view_probs, ALL_VIEWS, pano_w, pano_h, device, out_w=out_w, out_h=out_h)
    seg = fused.argmax(dim=1)[0].byte().cpu().numpy()
    conf = fused.float().max(dim=1).values[0].cpu().numpy()
    seg[~coverage[0].cpu().numpy()] = VOID_LABEL
    return seg, conf


def mask2former_semantic_probs(outputs, out_size: int) -> torch.Tensor:
    """Rucne odvodi husty [1,num_labels,S,S] pravdepodobnostni tensor z Mask2Former vystupu
    (class_queries_logits + masks_queries_logits), misto argmax post-processingu z HF, aby slo
    fuzovat softmax pres pohledy podle SS4.3 dokumentu (nikdy nehlasovat argmaxem pred fuzi).
    """
    class_logits = outputs.class_queries_logits  # [1,Q,num_labels+1]
    mask_logits = outputs.masks_queries_logits  # [1,Q,h,w]
    mask_logits = F.interpolate(mask_logits, size=(out_size, out_size), mode="bilinear", align_corners=False)
    class_probs = class_logits.softmax(dim=-1)[..., :-1]  # zahodit "no object", [1,Q,num_labels]
    mask_probs = mask_logits.sigmoid()  # [1,Q,S,S]
    semseg = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)  # [1,C,S,S]
    return semseg


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"zarizeni: {device}")

    frames = io_data.load_frames()
    selection_path = os.path.join(os.path.dirname(__file__), "out", "e2_frame_selection.json")
    if os.path.exists(selection_path) and os.environ.get("USE_SELECTION", "1") == "1":
        with open(selection_path, encoding="utf-8") as f:
            sample_idx = json.load(f)
        print(f"pouzit vyber {len(sample_idx)} snimku z {selection_path} (pokryvaji nejvic JVF trid z E1)")
    else:
        step = max(1, len(frames) // N_SAMPLE_FRAMES)
        sample_idx = list(range(0, len(frames), step))[:N_SAMPLE_FRAMES]

    proc, model = load_model(device)
    print(f"model nacten, {model.config.num_labels} trid")
    id2label = model.config.id2label

    times = []
    manifest = []
    for i in sample_idx:
        img = cv2.imread(frames.path(i))
        if img is None:
            continue
        t0 = time.time()
        seg, conf = segment_panorama(img, proc, model, device)
        dt = time.time() - t0
        times.append(dt)
        print(f"frame {i} ({frames.filename[i]}): {dt:.2f} s, trid pritomno: {len(np.unique(seg))}")

        color = colorize(seg, model.config.num_labels)
        color_full = cv2.resize(color, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(img, 0.5, color_full[..., ::-1], 0.5, 0)
        small = cv2.resize(overlay, (img.shape[1] // 4, img.shape[0] // 4))
        out_path = os.path.join(OUT_DIR, f"seg_{i}_{frames.filename[i]}")
        cv2.imwrite(out_path, small)

        np.savez_compressed(os.path.join(OUT_DIR, f"seg_{i}.npz"), seg=seg, conf=conf)
        manifest.append({"frame_idx": int(i), "filename": str(frames.filename[i]), "seconds": dt})

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "runs": manifest}, f, ensure_ascii=False, indent=1)

    if times:
        print(f"prumerny cas na panorama: {np.mean(times):.2f} s (n={len(times)})")
        print(f"odhad pro 1503 panoramat: {np.mean(times) * 1503 / 60:.1f} min")


if __name__ == "__main__":
    main()
