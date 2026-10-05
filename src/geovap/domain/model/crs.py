"""The coordinate reference system a dataset is expressed in.

Dražkov is S-JTSK / Krovak East North (EPSG:5514), which used to be hardcoded in four places --
`mapping/geometry.py`'s docstring, `mapping/poses.py`'s field comments and, as a literal proj4
string, in both viewer HTML files. It is dataset configuration, so it lives here as data and
reaches the viewer through the descriptor.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Crs:
    epsg: int
    proj4: str = ""
    # Axis order of the stored coordinates. Everything in this codebase works in a right-handed
    # (Easting, Northing, Height) frame; a dataset delivered in another order is converted by its
    # adapter, never here.
    axes: str = "ENH"

    def __str__(self) -> str:
        return f"EPSG:{self.epsg}"
