"""geovap: posed panoramas + tiled point clouds in, one coloured, labelled cloud per tile out.

Subpackages, by responsibility and owning stream (see `CODEOWNERS`). Imports only point down:
app -> stages -> runtime -> io -> domain (enforced by `.importlinter`).

  domain    pure geometry, math and scheme; no I/O                          foundation
  io        dataset descriptors and vendor-format adapters                  foundation
  runtime   settings, workspace layout, artifact contracts                  foundation
  stages    the pipeline steps, one subpackage per stream (see its docstring)
  infra     containers, viewer and screenshot driver, shipped as data      E delivery
  app       the `geovap` command: doctor, stages, run, status, compare      F app
"""
