"""Resolving a pose-table frame to its panorama file on disk.

This replaces `Poses.path(i)` from the pre-refactor `mapping/poses.py`. That method cannot live on
`geovap.domain.model.poses.Poses`: `domain` holds no paths (it is pure data and maths, portable
across datasets and even across processes that never touch a filesystem), and the panorama
directory is itself a per-dataset value -- only the descriptor (via `Settings.panos`, a
`PanoSource` adapter) knows where a given dataset's panoramas live. So the lookup moved to
`runtime`, the layer that is allowed to know about both `domain` values and `io` adapters.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from geovap.domain.model.poses import Poses

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings


def pano_path(poses: Poses, k: int, *, s: "Settings | None" = None) -> str:
    """Path to the panorama file backing frame `k` of `poses`, resolved through the dataset's
    `PanoSource` adapter. `s` defaults to `geovap.runtime.settings.get()`, imported here (not at
    module scope) so this module stays importable standalone."""
    if s is None:
        from geovap.runtime import settings as _settings

        s = _settings.get()
    return str(s.panos.path(str(poses.filename[k])))
