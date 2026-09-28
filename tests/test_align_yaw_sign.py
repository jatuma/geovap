"""Sign check for `Aligner.score` + `Aligner.shift_to_yaw_deg`, under the pano column convention
u = ((180 - az) mod 360) / 360 * W (az = atan2(y, x) in camera axes x-fwd/y-left; columns run
clockwise, seam at the rear; `pano_rays` inverse: az = 180 - u/W*360).

Derivation (see also the docstring on `Aligner.shift_to_yaw_deg`):
  * `score(img, valid, photo)` returns (ncc, sh) where `sh` is defined so that
    ``photo == np.roll(img, sh)`` (verified empirically below on a synthetic pair, and consistent
    with the in-code comment "photo rolled by +shift (returned) matches the render").
  * For a fixed world point, increasing the render yaw by `delta` degrees decreases its azimuth
    `az` by `delta` (yaw only rotates the world into the vehicle frame about z, before roll/pitch),
    and under the new convention u = ((180-az) mod 360)/360*W, decreasing az by `delta` increases u
    by +delta*W/360. So re-rendering at (yaw + delta) is (to the column-shift resolution of `score`)
    the same as ``np.roll(render_at_yaw, delta * W / 360)``.
  * If the photo equals the render at (yaw + delta), then ``photo == np.roll(render_at_yaw, delta*W/360)``,
    and matching that against ``photo == np.roll(render_at_yaw, sh)`` gives ``delta = sh * 360 / W``,
    i.e. `shift_to_yaw_deg` must NOT negate `sh` under this convention (unlike under the old
    u = (az mod 360)/360*W convention, where the correct formula would have been -sh*360/W).

This test does not touch real data: it builds a synthetic "render" edge image (per-column random
signal, replicated down rows, i.e. a synthetic panorama made of vertical edges) and derives a
"photo" by rolling it by the exact pixel shift that a +2 deg / -2 deg yaw change would produce
under the convention above, then checks that `score` + `shift_to_yaw_deg` recover that yaw offset.
"""
from __future__ import annotations

import numpy as np

from geovap.stages.register.align import Aligner, H, W


def _synthetic_edge_image(seed: int = 0) -> np.ndarray:
    """A (H, W) float32 image with strong vertical edges (per-column signal, replicated over rows),
    like the edge-magnitude images `Aligner.render`/`photo_gray` produce."""
    rng = np.random.default_rng(seed)
    col = rng.normal(size=W).astype(np.float32)
    img = np.repeat(col[None, :], H, axis=0)
    # a little row-wise noise so the image isn't perfectly rank-1 (more like a real edge render)
    img = img + 0.05 * rng.normal(size=(H, W)).astype(np.float32)
    return img


def test_score_shift_convention_photo_equals_roll_of_render():
    """Ground-truth check of `score`'s shift convention: photo == np.roll(render, sh)."""
    al = Aligner(store=None, poses=None)
    render = _synthetic_edge_image(seed=1)
    valid = np.ones((H, W), dtype=bool)
    for true_shift in (5, -5, 37, -123):
        photo = np.roll(render, true_shift, axis=1)
        _, sh = al.score(render, valid, photo)
        assert sh == true_shift, f"expected sh == {true_shift}, got {sh}"


def test_yaw_offset_sign_matches_new_pano_convention():
    """If the photo is what the model would render with yaw + 2 deg (resp. -2 deg), the
    score-shift -> yaw-offset conversion must yield an offset of matching sign, close to +2 (-2)
    deg up to the W/360 deg column quantisation of `score`'s integer-pixel shift."""
    al = Aligner(store=None, poses=None)
    render = _synthetic_edge_image(seed=2)
    valid = np.ones((H, W), dtype=bool)

    for true_yaw_delta in (2.0, -2.0):
        # exact pixel shift a render-time yaw change of `true_yaw_delta` would produce, under
        # u = ((180 - az) mod 360)/360*W: du/dyaw = +W/360 (see module docstring derivation)
        true_shift = int(round(true_yaw_delta * W / 360.0))
        photo = np.roll(render, true_shift, axis=1)

        _, sh = al.score(render, valid, photo)
        yaw_offset = Aligner.shift_to_yaw_deg(sh, W)

        assert sh == true_shift
        # same sign as the injected offset
        assert np.sign(yaw_offset) == np.sign(true_yaw_delta)
        # within pixel-quantisation of the true offset (quantum = 360/W deg, here 0.72 deg)
        assert abs(yaw_offset - true_yaw_delta) < 0.3, (yaw_offset, true_yaw_delta)
