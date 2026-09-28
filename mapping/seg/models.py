"""Uniform wrapper around HF segmentation checkpoints: view image -> dense class probabilities.

Mask-classification models (Mask2Former, OneFormer, EoMT) give softmax(class)[..., :-1] x sigmoid(mask) as in
experiments/e2 (`mask2former_semantic_probs`); SegFormer gives softmax of the upsampled logits. Models are
called directly with normalised tensors; the HF processors are used only for mean/std and OneFormer's task token.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import taxonomy as T


@dataclass
class ModelSpec:
    tag: str
    checkpoint: str
    family: str  # mask2former | oneformer | eomt | segformer
    taxonomy: str  # cityscapes | vistas | ade
    input_size: int = 1024
    licence: str = "MIT"
    note: str = ""
    mode: str = "resize"  # for input_size < view size: resize | tiles (overlapping crops averaged)


SPECS: dict[str, ModelSpec] = {
    "m2f_vistas": ModelSpec("m2f_vistas", "facebook/mask2former-swin-large-mapillary-vistas-semantic", "mask2former", "vistas", 1024, "MIT code, Vistas weights (research)", "E2 baseline"),
    "m2f_city": ModelSpec("m2f_city", "facebook/mask2former-swin-large-cityscapes-semantic", "mask2former", "cityscapes", 1024, "MIT code, Cityscapes weights (research)"),
    "oneformer_city": ModelSpec("oneformer_city", "shi-labs/oneformer_cityscapes_swin_large", "oneformer", "cityscapes", 1024, "MIT code, Cityscapes weights (research)"),
    "eomt_city": ModelSpec("eomt_city", "tue-mps/cityscapes_semantic_eomt_large_1024", "eomt", "cityscapes", 1024, "Apache-2.0 code, Cityscapes weights (research)", "DINOv2-L"),
    "eomt_dinov3_ade": ModelSpec("eomt_dinov3_ade", "tue-mps/eomt-dinov3-ade-semantic-large-512", "eomt", "ade", 512, "Apache-2.0 + DINOv3 licence", "DINOv3-L, 512 crops", mode="tiles"),
    "segformer_b5": ModelSpec("segformer_b5", "nvidia/segformer-b5-finetuned-cityscapes-1024-1024", "segformer", "cityscapes", 1024, "NVIDIA non-commercial (research only)"),
}


class SegModel:
    def __init__(self, spec: ModelSpec, device: torch.device, amp: bool = True):
        self.spec = spec
        self.device = device
        self.amp = amp
        self.model, self.mean, self.std, self.task_inputs = self._load()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.num_labels = len(self.id2label)
        self.to_common = torch.from_numpy(T.native_to_common(self.id2label, spec.taxonomy)).to(device)

    def _load(self):
        from transformers import AutoImageProcessor

        s = self.spec
        try:
            proc = AutoImageProcessor.from_pretrained(s.checkpoint)
            mean, std = list(proc.image_mean), list(proc.image_std)
        except Exception:
            proc, mean, std = None, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        task = None
        if s.family == "mask2former":
            from transformers import Mask2FormerForUniversalSegmentation as M
        elif s.family == "oneformer":
            from transformers import OneFormerForUniversalSegmentation as M
            from transformers import OneFormerProcessor

            p = OneFormerProcessor.from_pretrained(s.checkpoint)
            enc = p(images=np.zeros((64, 64, 3), np.uint8), task_inputs=["semantic"], return_tensors="pt")
            task = enc["task_inputs"].to(self.device)
        elif s.family == "eomt":
            from transformers import AutoModelForUniversalSegmentation as M
        elif s.family == "segformer":
            from transformers import SegformerForSemanticSegmentation as M
        else:
            raise ValueError(s.family)
        model = M.from_pretrained(s.checkpoint).to(self.device).eval()
        mean_t = torch.tensor(mean, device=self.device).view(1, 3, 1, 1) * 255
        std_t = torch.tensor(std, device=self.device).view(1, 3, 1, 1) * 255
        return model, mean_t, std_t, task

    @torch.no_grad()
    def _forward(self, inp: torch.Tensor, out_size: int) -> torch.Tensor:
        """inp [1,3,S,S] normalised -> probs [C,out,out] float32."""
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and self.device.type == "cuda"):
            if self.spec.family == "oneformer":
                out = self.model(pixel_values=inp, task_inputs=self.task_inputs)
            else:
                out = self.model(pixel_values=inp)
        if self.spec.family == "segformer":
            logits = F.interpolate(out.logits.float(), size=(out_size, out_size), mode="bilinear", align_corners=False)
            return logits.softmax(1)[0]
        cls = out.class_queries_logits.float().softmax(-1)[..., :-1]  # [1,Q,C]
        masks = F.interpolate(out.masks_queries_logits.float(), size=(out_size, out_size), mode="bilinear", align_corners=False).sigmoid()
        return torch.einsum("bqc,bqhw->bchw", cls, masks)[0]

    @torch.no_grad()
    def probs(self, view_rgb: np.ndarray) -> torch.Tensor:
        """view_rgb uint8 [S,S,3] RGB -> probs [C,S,S] float32 on device."""
        S = view_rgb.shape[0]
        x = torch.from_numpy(view_rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
        x = (x - self.mean) / self.std
        n = self.spec.input_size
        if n == S:
            return self._forward(x, S)
        if self.spec.mode == "resize":
            xr = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
            return self._forward(xr, S)
        # tiles: overlapping n x n crops with stride n/2, averaged
        stride = n // 2
        starts = list(range(0, S - n + 1, stride))
        if starts[-1] != S - n:
            starts.append(S - n)
        acc = torch.zeros(self.num_labels, S, S, device=self.device)
        wsum = torch.zeros(1, S, S, device=self.device)
        for y0 in starts:
            for x0 in starts:
                p = self._forward(x[:, :, y0 : y0 + n, x0 : x0 + n], n)
                acc[:, y0 : y0 + n, x0 : x0 + n] += p
                wsum[:, y0 : y0 + n, x0 : x0 + n] += 1
        return acc / wsum
