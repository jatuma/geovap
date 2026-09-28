"""Panorama decoding: turn a JPEG file on disk into an RGB array.

This is the one part of panorama handling that touches a file, which is why it lives in `io`
rather than alongside the pure sampling maths in `geovap.domain.math.sampling`.
"""
from __future__ import annotations

import cv2
import numpy as np

_TJ = None


def load_pano_rgb(path: str) -> np.ndarray:
    """uint8 [H,W,3] RGB. Uses TurboJPEG when installed (optional extra), else cv2."""
    global _TJ
    if _TJ is None:
        try:
            from turbojpeg import TurboJPEG  # type: ignore

            _TJ = TurboJPEG()
        except Exception:
            _TJ = False
    if _TJ:
        from turbojpeg import TJPF_RGB  # type: ignore

        with open(path, "rb") as f:
            return _TJ.decode(f.read(), pixel_format=TJPF_RGB)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
