"""Stage group: register -- S0-S5b, the registration pipeline.

  align                    S2: re-register frames against the cloud (image -> pose)
  traj-rot / traj-validate S3b: orientation-only dense trajectory from scanner planes, + QA
  refine                   S4: per-frame edge-ICP pose refinement
  register / reg-conflict  S5: pairwise pass-to-pass ICP + global solve, + QA
  assemble                 compose S3b/S4/S5b into poses_corrected.csv
  quality                  per-frame quality manifest (geometry / pass-conflict / motion / sharpness)

The stage modules are deliberately not imported here (see `geovap.stages.base.discovery` for why:
running a stage directly, e.g. `python -m geovap.stages.register.align`, would otherwise import that
module twice). `calib/` is a library in this group, not stages -- see its own `__init__.py`.

`geovap.runtime.pose_tables` cannot import `trajectory.Trajectory` itself (`runtime` sits BELOW
`stages` in the layering, and `trajectory.py` is ~1000 lines of scipy plane fitting that has no
business living in `runtime`), so it exposes `set_trajectory_loader` for whoever DOES import the
registration code to call. Registering it here, in this package's `__init__`, means it takes effect
as soon as the `register` group is imported AT ALL -- via `stages.base.discovery`, via any of this
group's own stage modules being imported directly, or via a plain `import
geovap.stages.register` -- not only when `trajectory.py` specifically happens to be the first of
this group's modules touched. A caller that never imports anything from `register` simply gets
`Poses.traj is None`, which is the documented best-effort behaviour.
"""
from __future__ import annotations

from geovap.runtime.pose_tables import set_trajectory_loader as _set_trajectory_loader


def _load_trajectory(path):
    from geovap.stages.register.trajectory import Trajectory

    return Trajectory.load(path)


_set_trajectory_loader(_load_trajectory)
