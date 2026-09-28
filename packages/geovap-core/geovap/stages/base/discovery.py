"""Finding the stages that are installed.

Which stages exist depends on which distributions are installed: `geovap-core` contributes
`prepare`, `register`, `colour`, `objects` and `verify`; `geovap-semantics` adds `semantics`;
`geovap-deliver` adds `deliver`. They all populate the same `geovap.stages` namespace package, so
the set can only be known by looking.

A stage group's `__init__.py` deliberately does NOT import its stage modules. If it did, running a
stage directly -- `python -m geovap.stages.objects.cluster`, which is the whole reason each stage
ships a CLI -- would import that module twice: once when runpy imports the parent package, and again
as `__main__`. Python warns about exactly this, and the module-level registration would run twice.
So registration is pulled from here instead, by whoever needs the whole picture: the driver and
`geovap status`.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geovap.stages.base.spec import StageRegistry

#: Groups to look in, in the order they appear in a run. A group that is not installed is skipped
#: silently -- that is the normal state of a core-only install, not an error.
GROUPS = ("prepare", "register", "colour", "semantics", "objects", "deliver", "verify")

#: Modules inside a group that are libraries rather than stages. Importing them is harmless but
#: pointless, and some pull heavy dependencies (torch) that a core-only install does not have.
_SKIP_PREFIX = "_"


def discover(groups: tuple[str, ...] = GROUPS) -> "StageRegistry":
    """Import every stage module of every installed group, so `registry` holds all of them."""
    from geovap.stages.base.spec import registry

    for group in groups:
        package_name = f"geovap.stages.{group}"
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            continue  # the distribution contributing this group is not installed
        for info in pkgutil.iter_modules(package.__path__):
            if info.name.startswith(_SKIP_PREFIX):
                continue
            try:
                importlib.import_module(f"{package_name}.{info.name}")
            except ImportError as exc:
                # A stage whose optional dependency is missing must not take down the whole
                # listing; it simply will not be registered, and the driver reports it as absent.
                print(f"geovap: skipping {package_name}.{info.name}: {exc}")
    return registry


def installed_groups(groups: tuple[str, ...] = GROUPS) -> list[str]:
    out = []
    for group in groups:
        try:
            importlib.import_module(f"geovap.stages.{group}")
        except ImportError:
            continue
        out.append(group)
    return out
