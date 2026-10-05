"""Pipeline stages, one subpackage per stream, in run order.

  base       stage spec, registry, discovery, shared CLI                    foundation
  prepare    ingest and dataset checks                                      foundation
  register   panorama registration                                          A registration
  semantics  pseudo-GT, zero-shot segmentation, label projection            B semantics (needs `geovap[semantics]`)
  objects    clustering                                                     C clustering
  colour     colourisation                                                  D colour
  deliver    consolidated LAZ product and Potree octree                     E delivery
  verify     visual and numeric verification                                F app

Groups never import each other; they hand off through artifacts on disk (`docs/artifacts.md`).
The `__init__` of a group does not import its stage modules, so `python -m geovap.stages.X.Y`
does not load the module twice; `base.discovery` finds the stages by walking the groups.
"""
