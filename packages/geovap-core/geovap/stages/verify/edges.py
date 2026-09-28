"""Edge-agreement metric between a cloud-render screenshot and a photo screenshot.

Same idea as ``mapping.quality._residual``: Canny both images, build a distance
transform of the photo's edges, then measure how far each cloud edge pixel sits
from the nearest photo edge. Unlike ``_residual`` (which works in panorama
angular space against silhouette points reprojected from the point cloud), this
operates directly on two rendered 2D screenshots (Potree cloud-only render vs.
the blended photo sphere), so it needs no camera model - just pixels.

Used by ``visual.py`` to score AZ_OFFSET / export-vs-corrected variants
numerically instead of only "looks right" by eye.
"""
from __future__ import annotations

import numpy as np


def _load_gray(path):
    """Accepts a path (str/Path) OR an already-in-memory image (np.ndarray, BGR or grayscale) - the
    latter lets callers (e.g. sphere.py's pinhole render) score an in-memory array without a
    round-trip through disk."""
    import cv2

    if isinstance(path, np.ndarray):
        return cv2.cvtColor(path, cv2.COLOR_BGR2GRAY) if path.ndim == 3 else path
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def edge_agreement(cloud_png, photo_png, cap_px: float = 30.0, inlier_px: float = 4.0) -> dict:
    """Compare edges in ``cloud_png`` (point-cloud render) against ``photo_png``.

    Returns ``dict(median_px, mean_px, inlier4, n_edges)``:
      - ``n_edges``: number of Canny edge pixels found in the cloud render.
      - ``median_px`` / ``mean_px``: distance (px, capped at ``cap_px``) from each
        cloud edge pixel to the nearest photo edge pixel (via distanceTransform).
      - ``inlier4``: fraction of cloud edge pixels within ``inlier_px`` of a photo edge.

    NaN values (and ``n_edges == 0``) mean no edges were found in the cloud render
    (e.g. an all-black or empty screenshot) - the caller should treat that shot as
    unusable rather than as a good/bad agreement score.
    """
    import cv2

    cloud = _load_gray(cloud_png)
    photo = _load_gray(photo_png)
    if cloud.shape != photo.shape:
        # resize photo to the cloud render's resolution (screenshots should already match,
        # but tolerate a driver that produced slightly different viewport sizes)
        photo = cv2.resize(photo, (cloud.shape[1], cloud.shape[0]), interpolation=cv2.INTER_AREA)

    cloud_edges = cv2.Canny(cloud, 50, 150)
    photo_edges = cv2.Canny(photo, 50, 150)

    n_edges = int((cloud_edges > 0).sum())
    if n_edges == 0 or (photo_edges > 0).sum() == 0:
        return dict(median_px=float("nan"), mean_px=float("nan"), inlier4=float("nan"), n_edges=n_edges)

    dt = cv2.distanceTransform((photo_edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    d = dt[cloud_edges > 0].astype(np.float64)
    d = np.minimum(d, cap_px)

    return dict(
        median_px=float(np.median(d)),
        mean_px=float(np.mean(d)),
        inlier4=float((d <= inlier_px).mean()),
        n_edges=n_edges,
    )


def silhouette_agreement(cloud_png, photo_png, cap_px: float = 40.0, inlier_px: float = 6.0, min_px: int = 200) -> dict:
    """Skyline metric, the screenshot analogue of ``mapping.quality._residual``.

    The cloud-only render is taken on a black background (``?nogui=1`` sets it), so "has points" is
    simply ``gray > 8``. After closing the splat gaps, the boundary of that mask (minus the image
    border) is the cloud's silhouette: skyline, roof lines, tree crowns against the sky. Those
    silhouette pixels are compared to the nearest Canny edge of the photo-only render via a distance
    transform. Unlike a raw Canny-vs-Canny comparison this ignores the splat texture inside surfaces,
    so a half-turn (AZ_OFFSET 0 vs 180) or a metre of pose error shows up as tens of pixels, while a
    correct pose lands within a few pixels at 1600 px width.

    Returns dict(median_px, p75_px, mean_px, inlier6, n_sil): NaN + n_sil < min_px when the render has
    no usable silhouette (all sky or all points)."""
    import cv2

    cloud = _load_gray(cloud_png)
    photo = _load_gray(photo_png)
    if cloud.shape != photo.shape:
        photo = cv2.resize(photo, (cloud.shape[1], cloud.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = cloud.shape
    k = max(3, int(round(w / 200)) | 1)  # ~8 px at 1600 wide, odd
    has_pts = (cloud > 8).astype(np.uint8)
    has_pts = cv2.morphologyEx(has_pts, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    has_pts = cv2.morphologyEx(has_pts, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    eroded = cv2.erode(has_pts, np.ones((3, 3), np.uint8))
    sil = (has_pts > 0) & (eroded == 0)
    b = max(2, k)
    sil[:b, :] = sil[-b:, :] = False
    sil[:, :b] = sil[:, -b:] = False
    n_sil = int(sil.sum())
    if n_sil < min_px:
        return dict(median_px=float("nan"), p75_px=float("nan"), mean_px=float("nan"), inlier6=float("nan"), n_sil=n_sil)
    photo_blur = cv2.GaussianBlur(photo, (0, 0), 1.5)
    photo_edges = cv2.Canny(photo_blur, 40, 120)
    if (photo_edges > 0).sum() == 0:
        return dict(median_px=float("nan"), p75_px=float("nan"), mean_px=float("nan"), inlier6=float("nan"), n_sil=n_sil)
    dt = cv2.distanceTransform((photo_edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    d = np.minimum(dt[sil].astype(np.float64), cap_px)
    return dict(median_px=float(np.median(d)), p75_px=float(np.percentile(d, 75)), mean_px=float(np.mean(d)),
                inlier6=float((d <= inlier_px).mean()), n_sil=n_sil)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cloud_png")
    ap.add_argument("photo_png")
    ap.add_argument("--cap-px", type=float, default=30.0)
    ap.add_argument("--inlier-px", type=float, default=4.0)
    args = ap.parse_args()
    print(json.dumps({"canny": edge_agreement(args.cloud_png, args.photo_png, args.cap_px, args.inlier_px),
                      "silhouette": silhouette_agreement(args.cloud_png, args.photo_png)}))
