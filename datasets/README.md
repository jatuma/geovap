# Datasets

One TOML per dataset, plus its git-tracked reference results under `<name>/baseline/`.

Descriptors live **here, not in the package**. `geovap` ships the adapters that can read a
posed-panorama + tiled-LAZ dataset; it does not ship any particular one. A descriptor names paths on
someone's machine and its baseline is a folder of measurements of one site — neither belongs in a
wheel. The only descriptor inside the package is `synthetic.toml`, which describes a dataset the
package can generate from nothing.

A bare name (`--dataset drazkov`) is looked up in this order:

1. each directory in `$GEOVAP_DATASETS` (colon-separated)
2. `./datasets` — this directory, when running from the repository
3. the package's built-ins — currently only `synthetic`

so a deployment keeps its descriptors wherever it likes without editing any code.

## Adding one

Copy `drazkov.toml`, then change what your data actually differs in: the `[poses].columns` indices,
the `[tiles].id_regex` and `out_name`, the `[crs]`, the `[sensor]` values. If your vendor format is
not one an adapter already reads, write the adapter — one class in
`geovap/io/adapters/<kind>/` plus a contract test — and name it in `adapter =`. Nothing in
`domain`, `runtime` or `stages` should need to change.

Check the result before processing anything:

    uv run geovap doctor --dataset mysite

## The fixture

For development and CI, generate the synthetic dataset instead of mounting a real one:

    uv run python -m geovap.io.datasets.synthetic /tmp/fixture
    export GEOVAP_SYNTHETIC_ROOT=/tmp/fixture
    export GEOVAP_SYNTHETIC_WORKSPACE=/tmp/fixture-ws GEOVAP_SYNTHETIC_PUBLISH=/tmp/fixture-pub
    uv run geovap doctor --dataset synthetic

It is a 0.65 MB street scene whose panoramas are rendered from its own point cloud, so a stage run
against it produces a checkable answer rather than noise.
