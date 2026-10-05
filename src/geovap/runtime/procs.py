"""Running a stage in its own process, and telling that process which dataset it is working on.

Stages execute as subprocesses. This predates the restructuring and is deliberate: a stage's
memmaps and CUDA context are released when its process exits, which is what lets a 24-stage run
survive on one machine (`mapping/cli/pipeline.py:10-12`). What changes here is how the child learns
its dataset -- it used to inherit a single `GEOVAP_CACHE` and re-derive everything from
`config.py`'s guesses; now the parent exports the fully resolved roots via `Settings.env()`, so a
child can never resolve to a different dataset than its parent.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from geovap.runtime.settings import Settings

#: Stages are launched through `uv run` so each gets the workspace's resolved environment rather
#: than whatever interpreter happens to be first on PATH.
PY: list[str] = ["uv", "run", "python", "-u"]


def mod_cmd(module: str, *args: str) -> list[str]:
    return [*PY, "-m", module, *args]


def script_cmd(path: str | Path, *args: str) -> list[str]:
    return [*PY, str(path), *args]


def sh_cmd(script: str) -> list[str]:
    """A shell one-liner (docker compose, cp), run via `bash -c` so pipes/globs/subshells work."""
    return ["bash", "-c", script]


def env_for(s: "Settings", *, pose_table: str | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The child's environment: the parent's, plus this dataset's resolved roots.

    `pose_table` overrides which table the child reads. The driver needs this because the stages
    that *build* a corrected pose table must themselves run on export poses -- the corrected table
    does not exist yet, and several modules resolve it eagerly.
    """
    env = dict(os.environ)
    env.update(s.env())
    if pose_table is not None:
        from geovap.runtime.settings import ENV_POSES

        env[ENV_POSES] = pose_table
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


def run(
    cmd: Sequence[str],
    s: "Settings",
    *,
    log_path: Path | None = None,
    tee: Iterable[Path] = (),
    cwd: Path | None = None,
    pose_table: str | None = None,
    echo: bool = True,
) -> int:
    """Run one command to completion, streaming its combined output to `log_path`, to each path in
    `tee`, and (when `echo`) to this process's stdout. Returns the exit code; never raises on a
    non-zero one, because the caller has a marker to write either way."""
    sinks = []
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            sinks.append(open(log_path, "w", encoding="utf-8"))
        for p in tee:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            sinks.append(open(p, "a", encoding="utf-8"))
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            env=env_for(s, pose_table=pose_table),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            for sink in sinks:
                sink.write(line)
            if echo:
                print(line, end="")
        return proc.wait()
    finally:
        for sink in sinks:
            sink.close()


#: Start method for every worker pool in the project.
#:
#: NOT the default `fork`. A stage's parent process has, by the time it reaches its pool, imported
#: OpenCV, laspy and sometimes torch -- libraries that hold their own threads and locks. Forking
#: copies those locks in whatever state they were in, and a child that then touches one deadlocks:
#: no error, no traceback, the run simply stops. It never bit the old code because the driver ran
#: every stage as a fresh subprocess, so each pool forked from an almost-empty interpreter. As soon
#: as two stages run in one process -- which is exactly what a test suite does -- it does.
#:
#: `spawn` costs a fresh interpreter per worker (a second or so, against jobs measured in minutes)
#: and requires the worker function and its arguments to be picklable, which is why the pools here
#: pass resolved values (`Settings.env()`, a rig vector) rather than live objects.
POOL_START_METHOD = "spawn"


def pool_context():
    """The multiprocessing context every pool in the project must use. See POOL_START_METHOD."""
    import multiprocessing

    return multiprocessing.get_context(POOL_START_METHOD)
