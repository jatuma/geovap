"""Stage group: prepare -- stream 0's foundation, the artifacts every other stream reads.

  ingest                   validate and MEASURE the dataset; writes only the run manifest
  store / store-columns    the columnar point store
  products                 per-frame depth + point-id panoramas

The stage modules are deliberately not imported here. Importing them would make `python -m
geovap.stages.prepare.products` -- the point of each stage shipping its own CLI -- import that
module twice, once via this package and again as `__main__`. Registration is pulled by
`geovap.stages.base.discovery` instead, from whoever needs the whole picture.

`render` and `masks` are libraries in this group, not stages.
"""
