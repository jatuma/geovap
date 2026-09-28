"""The product must not know it was built for Dražkov.

This is the test that decides whether the restructuring actually achieved its goal. Every value
that identifies one dataset -- its paths, its tile filename prefix, its CRS, its measured point
count -- belongs in a descriptor TOML, and the only Python allowed to mention such a value is the
adapter whose job is to read that shape of data.

It runs on string CONSTANTS, not docstrings: explaining in prose which literal an adapter replaced
is exactly the documentation this refactor wants to keep, while the same text in a format string is
the bug.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: Things that identify the Dražkov dataset specifically.
DATASET_LITERAL = re.compile(
    r"ID3432"  # its tile filename prefix
    r"|/home/jatuma|/mnt/Geovap|potree_output"  # one machine's paths
    r"|Dra[zž]kov|LB5|Klad_LAZ|1_ZPS_GAD|LAZ_Dra"  # its directory and file names
    r"|584_?809_?840"  # its point count, which config.py asserted against
    r"|EPSG:5514"  # its CRS -- belongs in [crs], not in a viewer or a writer
)

#: The only places a dataset literal is legitimate: the adapter that reads that vendor format (in
#: prose and in its own regex defaults) and the descriptors themselves.
ALLOWED = (
    "packages/geovap-core/geovap/io/adapters/",
    "packages/geovap-core/geovap/io/datasets/",
)

#: Legacy modules still holding dataset literals, each deleted by the phase named. This set may only
#: SHRINK -- a new entry means a literal was reintroduced somewhere the refactor had already left.
KNOWN_LEGACY = frozenset({
    "experiments/common/class_map.py",       # Phase 2: superseded by io/adapters/reference/jvf_zps
    "experiments/common/io_data.py",         # Phase 2: superseded by io/adapters
    "mapping/cli/pipeline.py",               # Phase 1: becomes app/driver.py
    "mapping/config.py",                     # Phase 2: superseded by runtime/settings
    "mapping/panos.py",                      # Phase 1: stream E, reads [crs] from the descriptor
    "mapping/quality.py",                    # Phase 1: stream A, chart title from the dataset name
    "mapping/seg/dataset.py",                # Phase 1: stream B, chart title from the dataset name
})

SCANNED = ("packages", "mapping", "pointcloud-tools", "experiments")


def _string_constants(path: Path) -> list[tuple[int, str]]:
    """Every string literal in the file that is not a docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node, clean=False) is not None and node.body:
                docstrings.add(id(node.body[0].value))
    return [
        (n.lineno, n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
    ]


def _offenders() -> dict[str, list[tuple[int, str]]]:
    found: dict[str, list[tuple[int, str]]] = {}
    for root in SCANNED:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if any(rel.startswith(a) for a in ALLOWED):
                continue
            hits = [(ln, v) for ln, v in _string_constants(path) if DATASET_LITERAL.search(v)]
            if hits:
                found[rel] = hits
    return found


def test_no_dataset_literal_survives_outside_the_adapters():
    offenders = _offenders()
    new = {f: h for f, h in offenders.items() if f not in KNOWN_LEGACY}
    assert not new, "dataset literals outside io/adapters:\n" + "\n".join(
        f"  {f}:{ln}: {v[:70]!r}" for f, hits in new.items() for ln, v in hits
    )


def test_the_legacy_allowlist_only_shrinks():
    """A file that has been cleaned must be struck off, so the list stays an honest countdown of
    what Phase 2 has left to delete rather than quietly accumulating exemptions."""
    stale = KNOWN_LEGACY - set(_offenders())
    assert not stale, f"these no longer hold dataset literals; remove them from KNOWN_LEGACY: {sorted(stale)}"


def test_the_new_packages_are_completely_clean():
    """Stated separately because it is the actual deliverable: `packages/` is what ships."""
    offenders = {f: h for f, h in _offenders().items() if f.startswith("packages/")}
    assert not offenders, f"the shipped packages still name one dataset: {offenders}"


@pytest.mark.parametrize("bad", ["ID3432_000037.laz", "/mnt/Geovap_cache/store", "EPSG:5514"])
def test_the_detector_would_actually_catch_one(bad, tmp_path):
    """A guard test that cannot fail is worse than none."""
    f = tmp_path / "m.py"
    f.write_text(f'"""A docstring mentioning {bad} is fine."""\nx = "{bad}"\n', encoding="utf-8")
    hits = [(ln, v) for ln, v in _string_constants(f) if DATASET_LITERAL.search(v)]
    assert len(hits) == 1 and hits[0][0] == 2
