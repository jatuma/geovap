"""Named run profiles: the combinations of stages people actually ask for.

`mapping/cli/pipeline.py` had one implicit profile -- whatever `_ORDER` said, minus whatever
`--skip` removed on the day. Naming them makes the common intents explicit and gives `geovap run
--profile colour` a meaning that survives a stage being added.

A profile is only a default selection. Every flag it sets can still be overridden on the command
line, and `--profile full` is simply all of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    name: str
    summary: str
    end: str | None = None
    skip: tuple[str, ...] = ()
    with_optional: bool = False
    extra: dict = field(default_factory=dict)


PROFILES: dict[str, Profile] = {
    "full": Profile(
        name="full",
        summary="everything the installed distributions provide, including the optional branches",
        with_optional=True,
    ),
    "core": Profile(
        name="core",
        summary="the required path only: prepare, registration, colour, delivery. No semantics, no clustering.",
        skip=("cluster",),
    ),
    "prepare": Profile(
        name="prepare",
        summary="just make the dataset processable: ingest, store, per-frame products",
        end="products",
    ),
    "colour": Profile(
        name="colour",
        summary="up to and including colourisation -- the usual stopping point when tuning colour",
        end="report",
    ),
    "objects": Profile(
        name="objects",
        summary="clustering only; imports no project code and needs nothing else to have run",
        extra={"only": ("cluster",)},
    ),
}


def get(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; known: {', '.join(sorted(PROFILES))}") from None


def as_selection(profile: Profile) -> dict:
    """The `driver.select` keyword arguments this profile implies."""
    out: dict = {"skip": profile.skip, "with_optional": profile.with_optional}
    if profile.end:
        out["end"] = profile.end
    out.update(profile.extra)
    return out
