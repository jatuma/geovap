"""Finding the stages that are installed.

Every group ships with the package, but `semantics` needs the `[semantics]` extra (torch,
transformers): without it its stage modules fail to import and are skipped, so the set of stages
can only be known by looking.

A stage group's `__init__.py` deliberately does NOT import its stage modules. If it did, running a
stage directly -- `python -m geovap.stages.objects.cluster`, which is the whole reason each stage
ships a CLI -- would import that module twice: once when runpy imports the parent package, and again
as `__main__`. Python warns about exactly this, and the module-level registration would run twice.
So registration is pulled from here instead, by whoever needs the whole picture: the driver and
`geovap status`.
"""
from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geovap.stages.base.spec import StageRegistry

#: Groups to look in, in the order they appear in a run.
#:
#: `semantics` is listed as its three SUB-packages rather than itself: `pkgutil.iter_modules`
#: below only walks one package's own top-level modules, and `geovap.stages.semantics` itself has
#: none -- its stage modules live one level down, in `semantics/pseudogt`, `semantics/segment` and
#: `semantics/label` (see that package's docstring for why the split). Listing each sub-package
#: here makes it walked exactly like a single-level group, with no special-casing in `discover()`.
GROUPS = (
    "prepare", "register", "colour",
    "semantics.pseudogt", "semantics.segment", "semantics.label",
    "objects", "deliver", "verify",
)

#: Modules inside a group that are libraries rather than stages. Importing them is harmless but
#: pointless, and some pull heavy dependencies (torch) that an install without `[semantics]` does not have.
_SKIP_PREFIX = "_"

#: Top-level modules the `[semantics]` extra provides; `semantics.*` stages cannot import without them.
_SEMANTICS_REQUIRES = ("torch", "transformers", "shapely", "huggingface_hub")


def discover(groups: tuple[str, ...] = GROUPS) -> "StageRegistry":
    """Import every stage module of every installed group, so `registry` holds all of them."""
    from geovap.stages.base.spec import registry

    # Without the `[semantics]` extra its groups are left out: the normal light install, not an error.
    for group in installed_groups(groups):
        package_name = f"geovap.stages.{group}"
        package = importlib.import_module(package_name)
        for info in pkgutil.iter_modules(package.__path__):
            if info.name.startswith(_SKIP_PREFIX):
                continue
            try:
                importlib.import_module(f"{package_name}.{info.name}")
            except ImportError as exc:
                # A stage whose optional dependency is missing (the `[semantics]` extra) must not
                # take down the whole listing; it simply will not be registered, and the driver
                # reports it as absent.
                print(f"geovap: skipping {package_name}.{info.name}: {exc}")
    return registry


def semantics_available() -> bool:
    """Whether the `[semantics]` extra is importable (looked up, not imported: torch is slow)."""
    return all(importlib.util.find_spec(m) is not None for m in _SEMANTICS_REQUIRES)


def installed_groups(groups: tuple[str, ...] = GROUPS) -> list[str]:
    """Groups whose stages can actually be imported here: all of them, minus the `semantics`
    sub-groups when the `[semantics]` extra is missing."""
    with_extra = semantics_available()
    return [g for g in groups if with_extra or not g.startswith("semantics.")]
