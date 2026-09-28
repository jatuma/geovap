"""Pano source for a flat directory of equirectangular panoramas, addressed by the filename a pose
row already carries (`Frames.path` in `experiments/common/io_data.py` did `os.path.join(PANO_DIR,
filename)`; this is that, with `PANO_DIR` coming from the descriptor instead of a module constant).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from geovap.io.registry import register

if TYPE_CHECKING:
    from geovap.io.descriptor import Descriptor


@register("panos", "equirect_dir")
class EquirectDirPanoSource:
    def __init__(self, descriptor: "Descriptor", table: dict):
        try:
            self._dir = descriptor.data_root / table["dir"]
        except KeyError as exc:
            raise ValueError("[panos] table is missing 'dir'") from exc
        self._width = table.get("width")
        self._height = table.get("height")

    def path(self, filename: str) -> Path:
        return self._dir / filename

    def describe(self) -> dict:
        exists = self._dir.is_dir()
        count = sum(1 for _ in self._dir.iterdir()) if exists else 0
        return {
            "dir": str(self._dir),
            "exists": exists,
            "count": count,
            "width": self._width,
            "height": self._height,
        }
