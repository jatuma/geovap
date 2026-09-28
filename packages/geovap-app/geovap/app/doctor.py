"""`geovap doctor` -- can this dataset be processed, and what would it be processed as?

It answers before anything is written: resolve the descriptor, build every adapter, and report what
they see. It is the first command anyone runs against a new dataset, and it is the one that turns
"the run failed somewhere in hour three" into "your pose table names 12 panoramas that are not on
disk".

The checks are `stages.prepare.ingest`'s, reused rather than reimplemented -- a doctor that
disagrees with the stage is worse than no doctor. The difference is that `doctor` writes nothing at
all: no manifest, no marker. It also reports what `ingest` cannot, because `ingest` is about the
data and `doctor` is about the installation: which distributions are present, which stages they
contribute, and where every path resolved from.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings


def report(s: "Settings") -> dict:
    from geovap.stages.base.discovery import discover, installed_groups
    from geovap.stages.prepare import ingest

    registry = discover()
    return {
        "dataset": {
            "name": s.name,
            "descriptor": str(s.descriptor.file),
            "data_root": str(s.paths.data_root),
            "workspace": str(s.paths.workspace),
            "publish": str(s.paths.publish),
            "baseline": str(s.workspace.baseline),
            "pose_table": s.pose_table,
        },
        "install": {
            "groups": installed_groups(),
            "stages": registry.order(),
            "optional": [n for n in registry.order() if registry[n].spec.optional],
        },
        "adapters": _adapters(s),
        "checks": ingest.checks(s),
    }


def _adapters(s: "Settings") -> dict:
    """`describe()` of each adapter, each failure isolated: an unreadable pose table must not stop
    the tile report, because the whole point is to list everything wrong in one pass."""
    out = {}
    for kind in ("poses", "tiles", "panos", "reference"):
        try:
            adapter = getattr(s, kind)
            out[kind] = {"adapter": None if adapter is None else type(adapter).__name__,
                         **({} if adapter is None else adapter.describe())}
        except Exception as e:  # noqa: BLE001
            out[kind] = {"error": f"{type(e).__name__}: {e}"}
    return out


def render(rep: dict) -> str:
    lines = []
    d = rep["dataset"]
    lines.append(f"dataset {d['name']}  ({d['descriptor']})")
    for key in ("data_root", "workspace", "publish", "baseline", "pose_table"):
        lines.append(f"  {key:<11}: {d[key]}")

    i = rep["install"]
    lines.append(f"  installed  : {', '.join(i['groups'])}")
    lines.append(f"  stages     : {', '.join(i['stages'])}")
    if i["optional"]:
        lines.append(f"  optional   : {', '.join(i['optional'])}")

    lines.append("adapters")
    for kind, info in rep["adapters"].items():
        if info.get("adapter") is None and "error" not in info:
            lines.append(f"  {kind:<11}: not configured")
            continue
        head = info.get("error") or info.get("adapter")
        lines.append(f"  {kind:<11}: {head}")
        for key, value in info.items():
            if key in ("adapter", "error"):
                continue
            lines.append(f"      {key} = {_short(value)}")

    lines.append("checks")
    failed = []
    for name, result in rep["checks"].items():
        ok = result.get("ok")
        mark = {True: "    ok", False: "FAILED", None: "  n/a "}[ok]
        lines.append(f"  [{mark}] {name}: {_short(result)}")
        if ok is False:
            failed.append(name)
    lines.append("all checks passed" if not failed else f"FAILED: {', '.join(failed)}")
    return "\n".join(lines)


def _short(value, limit: int = 160) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def ok(rep: dict) -> bool:
    return not any(r.get("ok") is False for r in rep["checks"].values())
