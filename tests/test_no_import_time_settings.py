"""Nothing may resolve a dataset while it is being imported.

This is the single rule the whole restructuring turns on. `mapping/config.py` froze Dražkov's paths
at import time, so by the time `argparse` had run, forty modules had already captured them and no
command-line flag could change anything. Late binding -- `settings.get()` inside the function that
needs a value -- is what makes a dataset an input.

It is also easy to undo by accident, and the failure is nasty in two different ways:

  * A module-level `SEGDS_DIR = source_dir(root, load_poses())` reads a pose table on import. On a
    machine with no dataset configured, *importing the module* raises -- which is how `pytest
    --collect-only` started failing rather than skipping.
  * A default argument `def build(out_dir: Path = raster_dir())` is evaluated once, at import. It
    silently pins the first dataset resolved in the process, so a second `configure()` in the same
    process writes to the first one's directories.

Both of those existed and were found by this rule, not by a functional test.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCANNED = ("packages", "mapping", "pointcloud-tools")

#: Calls that resolve a dataset. Matched on the attribute and the object it is called on, which is
#: enough while the project keeps to one name per module (`settings`, `pose_tables`, `store`).
RESOLVERS = {
    ("settings", "get"), ("settings", "build"), ("settings", "configure"),
    ("pose_tables", "load"), ("pose_tables", "read"),
    ("store", "open_store"),
    ("Descriptor", "load"),
}
#: Bare function names that resolve a dataset, however they were imported.
RESOLVER_NAMES = {"load_poses", "open_store", "segds_dir", "segds_root", "areas_dir"}


def _import_time_nodes(tree: ast.Module):
    """Every expression evaluated when the module is imported: module-level statements, class-body
    statements, and -- the subtle one -- function DEFAULT arguments, which are evaluated at
    definition time even though the body is not."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _defaults(node)
            continue
        if isinstance(node, ast.ClassDef):
            for inner in node.body:
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield from _defaults(inner)
                else:
                    yield inner
            continue
        yield node


def _defaults(fn: ast.FunctionDef | ast.AsyncFunctionDef):
    yield from (d for d in fn.args.defaults if d is not None)
    yield from (d for d in fn.args.kw_defaults if d is not None)


def _resolves_a_dataset(node: ast.AST) -> list[tuple[int, str]]:
    found = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Attribute):
            owner = getattr(func.value, "id", None)
            if owner and (owner, func.attr) in RESOLVERS:
                found.append((sub.lineno, f"{owner}.{func.attr}()"))
        elif isinstance(func, ast.Name) and func.id in RESOLVER_NAMES:
            found.append((sub.lineno, f"{func.id}()"))
    return found


def _offenders(repo: Path = REPO, roots: tuple[str, ...] = SCANNED) -> list[str]:
    out = []
    for root in roots:
        for path in sorted((repo / root).rglob("*.py")):
            if "node_modules" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in _import_time_nodes(tree):
                for lineno, what in _resolves_a_dataset(node):
                    out.append(f"{path.relative_to(repo)}:{lineno}: {what} at import time")
    return out


def test_nothing_resolves_a_dataset_at_import_time():
    offenders = _offenders()
    assert not offenders, (
        "these run when the module is imported, so a --dataset flag can no longer change them:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("source,why", [
    ("from geovap.runtime import settings\nX = settings.get().paths.data_root\n",
     "a module-level assignment"),
    ("from pathlib import Path\ndef f(out: Path = segds_dir()):\n    pass\n",
     "a default argument, evaluated at definition time"),
    ("class C:\n    root = load_poses()\n",
     "a class-body assignment"),
])
def test_the_detector_catches_each_shape(source, why, tmp_path):
    """A guard test that cannot fail is worse than none -- and the default-argument case in
    particular looks harmless."""
    f = tmp_path / "pkg" / "m.py"
    f.parent.mkdir()
    f.write_text(source, encoding="utf-8")
    assert _offenders(tmp_path, ("pkg",)), f"the detector missed {why}"


def test_the_detector_allows_a_call_inside_a_function_body():
    tree = ast.parse("from geovap.runtime import settings\ndef f():\n    return settings.get()\n")
    assert not [x for n in _import_time_nodes(tree) for x in _resolves_a_dataset(n)]
