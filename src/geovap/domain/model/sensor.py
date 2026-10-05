"""Sensor and tuning constants as value objects.

These used to be module-level constants in `mapping/config.py`, frozen at import time and therefore
impossible for a CLI to override after argparse. They are now plain data: `io.descriptor` builds
them from the dataset descriptor, `runtime.settings` serves them, and every `domain` function that
needs one takes it as an argument. Nothing here reads the environment or the filesystem.
"""
from __future__ import annotations

from dataclasses import dataclass

# uint32 sentinel for "no point" in point_id panoramas. A property of our own raster encoding,
# not of any dataset, so it stays a constant.
NO_POINT = 0xFFFFFFFF


@dataclass(frozen=True)
class Sensor:
    """Panorama geometry and scanner point density for one dataset."""

    pano_w: int
    pano_h: int
    zb_w: int
    zb_h: int
    r_min: float          # m, closer points are camera/scanner parallax garbage
    r_max: float          # m
    point_spacing: float  # m between neighbouring cloud points
    cell_size: float      # m, point-store cell edge
    tol_abs: float        # visibility: r <= depth + max(tol_abs, tol_rel * r)
    tol_rel: float
    splat_k: float
    splat_min_px: int
    splat_max_px: int

    @property
    def deg_per_px(self) -> float:
        return 360.0 / self.pano_w

    @property
    def zb_scale(self) -> float:
        """Full-resolution panorama pixels -> z-buffer pixels."""
        return self.zb_w / self.pano_w

    @property
    def pano(self) -> tuple[int, int]:
        """(w, h) of a full-resolution panorama, for `f(..., *sensor.pano)`."""
        return self.pano_w, self.pano_h

    @property
    def zb(self) -> tuple[int, int]:
        """(w, h) of the reduced-resolution depth panorama, for `f(..., *sensor.zb)`."""
        return self.zb_w, self.zb_h


@dataclass(frozen=True)
class Tuning:
    """Algorithm knobs that are tuned per dataset but are not physical properties of the sensor."""

    score_r0: float
    incidence_max_deg: float
    top_k: int
    mad_cutoff: float
