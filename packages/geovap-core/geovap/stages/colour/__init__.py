"""Stage group: colour -- fusing camera colour onto the point cloud.

  colorize        pano -> cloud colour fusion, tile by tile, frame by frame
  colour-report   merge per-tile stats into report.md + PNG plots

The stage modules are deliberately not imported here -- see `geovap.stages.base.discovery`.

`accumulate` is a library in this group (the per-point accumulators `colorize` feeds), not a stage.
"""
