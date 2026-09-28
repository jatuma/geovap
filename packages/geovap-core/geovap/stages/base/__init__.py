"""The Stage protocol and its registry -- the shape every pipeline step shares."""
from .spec import Stage, StageRegistry, StageSpec, registry

__all__ = ["Stage", "StageRegistry", "StageSpec", "registry"]
