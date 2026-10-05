"""Stage markers and the run manifest: what has been done, from what, and what it measured.

Two things live here, both ported out of `mapping/cli/pipeline.py`, which mixed them with the
stage declarations and the subprocess runner:

**Markers** (`<workspace>/out/pipeline/<stage>.json`) are how a 24-stage, multi-hour run resumes.
A stage is done when its marker is green, its declared inputs still hash the same, and its declared
outputs still exist. The on-disk shape is unchanged, so an existing run's markers stay valid.

**The run manifest** (`<workspace>/out/pipeline/manifest.json`) records what a run *measured* about
its dataset. It exists to replace two things a product cannot ship: `config.py:71`'s
`EXPECTED_TOTAL_POINTS = 584_809_840`, which was Dražkov's own point count used as a correctness
check, and `pipeline.py:64-79`'s `REFERENCE` dict, which froze one run's results as Python code.
Both are now measurements written here and compared against the dataset's own
`datasets/<name>/baseline/`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings
    from geovap.runtime.workspace import Workspace

#: Files at or below this size are hashed by content; larger ones by (mtime_ns, size). A 30 GB
#: memmapped store cannot be re-hashed on every `status` call, and its mtime is a good enough
#: staleness signal because nothing rewrites it in place.
HASH_CONTENT_MAX_BYTES = 64 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_rev(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return None


# ------------------------------------------------------------------------------------ input hashes
def hashable(value: Any) -> Any:
    """A JSON-serialisable identity for one declared input.

    A `Path` becomes content (small files) or (mtime_ns, size) (large ones); a missing path becomes
    a distinct marker so that a stage whose input appears later is correctly seen as stale. Anything
    else is passed through, which is how a stage declares a scalar (a tag, a worker count) as part
    of its identity.
    """
    if isinstance(value, Path):
        p = value
        try:
            p = p.resolve()
        except OSError:
            pass
        try:
            st = p.stat()
        except OSError:
            return ["path", str(p), None, None]
        if p.is_file() and st.st_size <= HASH_CONTENT_MAX_BYTES:
            return ["path", str(p), hashlib.sha256(p.read_bytes()).hexdigest()]
        return ["path", str(p), st.st_mtime_ns, st.st_size]
    return value


def inputs_hash(inputs: dict[str, Any]) -> str:
    payload = json.dumps({k: hashable(v) for k, v in sorted(inputs.items())}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------ markers
def marker_path(ws: "Workspace", name: str) -> Path:
    return ws.pipeline / f"{name}.json"


def read_marker(ws: "Workspace", name: str) -> dict | None:
    p = marker_path(ws, name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt marker means "not done", not a crash
        return None


def write_marker(ws: "Workspace", name: str, marker: dict) -> None:
    ws.pipeline.mkdir(parents=True, exist_ok=True)
    marker_path(ws, name).write_text(json.dumps(marker, indent=1, default=str), encoding="utf-8")


def is_done(stage, s: "Settings") -> bool:
    """True when `stage` need not run again: green marker, unchanged input hashes, outputs present.

    `stage` is anything satisfying `geovap.stages.base.Stage` -- this module deliberately does not
    import it, so `runtime` stays below `stages` in the layering.
    """
    ws = s.workspace
    marker = read_marker(ws, stage.spec.name)
    if marker is None or marker.get("rc") != 0:
        return False
    if marker.get("inputs_hash") != inputs_hash(stage.inputs(s)):
        return False
    return all(Path(p).exists() for p in stage.outputs(s))


def make_marker(*, name: str, rc: int, inputs: dict, outputs: list[Path], metrics: dict,
                poses_hash: str | None = None, started: str | None = None,
                repo: Path | None = None) -> dict:
    return {
        "stage": name,
        "rc": rc,
        "inputs_hash": inputs_hash(inputs),
        "inputs": {k: str(v) for k, v in inputs.items()},
        "outputs": [str(p) for p in outputs],
        "metrics": metrics,
        "poses_hash": poses_hash,
        "started": started,
        "finished": now(),
        "git_rev": git_rev(repo),
    }


# ------------------------------------------------------------------------------------- run manifest
@dataclass
class RunManifest:
    """What this run measured about its dataset. Written by `stages.prepare.ingest` and extended by
    later stages; read by `geovap status`, `geovap compare` and the verification stages."""

    dataset: str
    descriptor: str
    created: str = field(default_factory=now)
    git_rev: str | None = None
    #: Measured, never configured. `config.py` asserted against a literal 584 809 840; a second
    #: dataset has its own number and only the manifest can know it.
    total_points: int | None = None
    n_tiles: int | None = None
    n_frames: int | None = None
    poses_hash: str | None = None
    #: `describe()` of each adapter, so a run records the exact shape of the data it consumed.
    sources: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @classmethod
    def path(cls, ws: "Workspace") -> Path:
        return ws.pipeline / "manifest.json"

    @classmethod
    def load(cls, ws: "Workspace") -> "RunManifest | None":
        p = cls.path(ws)
        if not p.exists():
            return None
        return cls(**json.loads(p.read_text(encoding="utf-8")))

    def save(self, ws: "Workspace") -> Path:
        ws.pipeline.mkdir(parents=True, exist_ok=True)
        p = self.path(ws)
        p.write_text(json.dumps(asdict(self), indent=1, default=str), encoding="utf-8")
        return p
