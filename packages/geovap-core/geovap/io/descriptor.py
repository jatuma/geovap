"""Dataset descriptor: turns one TOML file into a `Descriptor`.

Everything that used to be frozen at import time in `mapping/config.py` (`DATA_ROOT`,
`PANO_W`/`PANO_H`, `SENSOR`, `TUNING`, ...) is now data that a descriptor file supplies, so a CLI
can point at "Dražkov" or at some other dataset without editing Python. Adapters (`io.adapters.*`)
still do the actual file reading; this module only resolves paths and builds the value objects
`geovap.domain` already defines.
"""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geovap.domain.model.crs import Crs
from geovap.domain.model.sensor import Sensor, Tuning

_BUILTIN_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

#: Colon-separated extra directories to search for `<name>.toml`.
ENV_SEARCH_PATH = "GEOVAP_DATASETS"

_PATH_KEYS = ("data_root", "workspace", "publish")
#: Optional: git-tracked reference results for this dataset. Defaults, in `runtime.workspace`,
#: to `<descriptor dir>/<name>/baseline`, so a descriptor need not spell it out.
_OPTIONAL_PATH_KEYS = ("baseline",)
_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class DescriptorError(ValueError):
    """A descriptor file, or the environment/overrides used to resolve it, is invalid."""


@dataclass(frozen=True)
class Descriptor:
    name: str
    file: Path  # the descriptor TOML this was built from, for adapters resolving package-relative data
    data_root: Path
    workspace: Path
    publish: Path
    crs: Crs
    sensor: Sensor
    tuning: Tuning
    poses: dict = field(default_factory=dict)
    panos: dict = field(default_factory=dict)
    tiles: dict = field(default_factory=dict)
    reference: dict | None = None
    #: Optional; `runtime.workspace.baseline_dir` supplies the default when this is None.
    baseline: Path | None = None

    @classmethod
    def load(cls, path_or_name: str | Path, *, overrides: dict[str, str] | None = None) -> Descriptor:
        overrides = overrides or {}
        file = _resolve_descriptor_file(path_or_name)
        with open(file, "rb") as f:
            data = tomllib.load(f)

        try:
            raw_paths = data["paths"]
        except KeyError as exc:
            raise DescriptorError(f"{file}: missing [paths] table") from exc
        paths = {key: _resolve_path(key, raw_paths, overrides, file) for key in _PATH_KEYS}
        paths.update(
            {
                key: (_resolve_path(key, raw_paths, overrides, file) if key in raw_paths or key in overrides else None)
                for key in _OPTIONAL_PATH_KEYS
            }
        )

        try:
            crs = Crs(**data["crs"])
            sensor = Sensor(**data["sensor"])
            tuning = Tuning(**data["tuning"])
        except KeyError as exc:
            raise DescriptorError(f"{file}: missing table {exc}") from exc
        except TypeError as exc:
            raise DescriptorError(f"{file}: {exc}") from exc

        return cls(
            name=data.get("name", Path(file).stem),
            file=file,
            data_root=paths["data_root"],
            workspace=paths["workspace"],
            publish=paths["publish"],
            baseline=paths["baseline"],
            crs=crs,
            sensor=sensor,
            tuning=tuning,
            poses=data.get("poses", {}),
            panos=data.get("panos", {}),
            tiles=data.get("tiles", {}),
            reference=data.get("reference"),
        )


def search_path() -> list[Path]:
    """Where a bare dataset name is looked up, in order.

    `$GEOVAP_DATASETS` (colon-separated) comes first so a deployment can keep its descriptors
    anywhere; then `./datasets`, which is where this repository keeps them; then the few shipped
    inside the package. Real datasets deliberately do NOT ship in the wheel -- a descriptor plus its
    tracked baseline is data about one site, and `pip install geovap-core` should not carry it.
    """
    out = [Path(p) for p in os.environ.get(ENV_SEARCH_PATH, "").split(os.pathsep) if p]
    out.append(Path("datasets"))
    out.append(_BUILTIN_DATASETS_DIR)
    return out


def _resolve_descriptor_file(path_or_name: str | Path) -> Path:
    """Explicit path -> used as-is; otherwise `<name>.toml` in each directory of `search_path()`."""
    given = Path(path_or_name)
    if given.is_file():
        return given

    name = str(path_or_name)
    candidates = [d / f"{name}.toml" for d in search_path()]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(c) for c in [given, *candidates])
    raise DescriptorError(f"no dataset descriptor found for {path_or_name!r} (looked at: {searched})")


def _resolve_path(key: str, raw_paths: dict[str, Any], overrides: dict[str, str], file: Path) -> Path:
    if key in overrides and overrides[key] is not None:
        return Path(_expand_env(str(overrides[key]), key, file))
    if key not in raw_paths:
        raise DescriptorError(f"{file}: [paths] is missing '{key}'")
    return Path(_expand_env(str(raw_paths[key]), key, file))


def _expand_env(value: str, key: str, file: Path) -> str:
    """Expand `${VAR}` references from `os.environ`; an unset variable is a hard error naming both
    the variable and the descriptor file, so a misconfigured environment fails at load time rather
    than resolving to a stray literal '${VAR}' path."""

    def replace(match: re.Match[str]) -> str:
        var = match.group(1)
        if var not in os.environ:
            raise DescriptorError(
                f"{file}: [paths].{key} references environment variable ${{{var}}}, which is not set"
            )
        return os.environ[var]

    return _VAR_PATTERN.sub(replace, value)
