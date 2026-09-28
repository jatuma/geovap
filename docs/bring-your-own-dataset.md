# Bring your own dataset

This is a from-the-code walkthrough of how the pipeline finds and validates a dataset, and how to
point it at a new one. Every claim here is backed by a specific file — read that file if you need
the detail this page leaves out.

## 1. The three roots

`geovap.runtime.settings.Paths` (`packages/geovap-core/geovap/runtime/settings.py`) pins a dataset to
exactly three directories:

- `data_root` — read-only input: panoramas, LAZ tiles, reference vectors. Never written to.
- `workspace` — our own intermediate artifacts: the point store, per-frame products, pose tables,
  stage markers/logs. Can be large (the Dražkov workspace is tens of GB).
- `publish` — what is handed to a viewer or a customer: Potree octrees, panorama exports.

`geovap.runtime.workspace.Workspace` derives every other output path (`store/`, `frames/`,
`out/poses/`, `out/pipeline/`, `out/consolidated/tiles/`, `dataset/`, ...) from `workspace` alone —
see the module docstring for the full layout. There is deliberately no `<workspace>/<dataset>/`
level inserted: two datasets are kept apart by pointing them at two different workspaces, not by an
extra path segment.

### How a descriptor resolves these

A dataset is described by one TOML file (`geovap.io.descriptor.Descriptor.load`,
`packages/geovap-core/geovap/io/descriptor.py`). Its `[paths]` table gives `data_root`, `workspace`,
`publish` either as literal paths or as `${VAR}` references that are expanded from the process
environment at load time — an unset variable is a hard, immediate `DescriptorError` naming both the
variable and the file, never a silent fallback to a stray `${VAR}` string. `datasets/drazkov.toml`
asks for `${GEOVAP_DATA}`, `${GEOVAP_WORKSPACE}`, `${GEOVAP_PUBLISH}`; `synthetic.toml` (shipped
inside the package) asks for `${GEOVAP_SYNTHETIC_ROOT}` and friends instead.

A bare `--dataset NAME` is looked up, in order (`descriptor.search_path()`):

1. each directory in `$GEOVAP_DATASETS` (colon-separated);
2. `./datasets` — where this repository keeps its own descriptors;
3. the package's built-ins (currently only `synthetic`).

`$GEOVAP_DATASET` supplies the name when `--dataset` is not passed at all (default `"drazkov"`).
`$GEOVAP_POSES` (or `--poses`) picks which pose table a stage reads: `"export"` (the regression
anchor) or the name/path of a corrected one.

### `$GEOVAP_OVERRIDE_*` vs. a descriptor's own variables — and why both exist

There are two, deliberately different, ways an environment variable can influence a path:

- **The descriptor's own `${VAR}`** (e.g. `${GEOVAP_DATA}` in `drazkov.toml`) is *that dataset's*
  choice of variable name. Exporting `$GEOVAP_DATA` redirects Dražkov specifically, because
  Dražkov's descriptor asked for that name.
- **`$GEOVAP_OVERRIDE_DATA_ROOT` / `_WORKSPACE` / `_PUBLISH`** are blanket overrides that apply no
  matter which dataset is selected or what its descriptor asks for. They exist only to carry a CLI
  flag (`--data-root` etc.) into a stage subprocess; `geovap.runtime.procs` is their only writer.

The reason these are not the same thing: if `$GEOVAP_DATA` were *also* read as a blanket override,
exporting it to redirect Dražkov would silently redirect every other dataset too — including
`synthetic`, which has nothing to do with `$GEOVAP_DATA` and resolves `${GEOVAP_SYNTHETIC_ROOT}`
instead. That is a silent wrong-data bug: the run succeeds, against the wrong directory. So a
descriptor declares which variables *it* reads, and the three `$GEOVAP_OVERRIDE_*` variables are the
only ones with dataset-independent meaning.

## 2. Walking through `datasets/drazkov.toml`

```toml
[paths]
data_root = "${GEOVAP_DATA}"
workspace = "${GEOVAP_WORKSPACE}"
publish   = "${GEOVAP_PUBLISH}"
```
The three roots above. Any of the three can also be a literal path instead of a `${VAR}`.

```toml
[crs]
epsg = 5514
proj4 = "+proj=krovak ..."
```
Builds a `geovap.domain.model.crs.Crs`. Get the `proj4` string wrong and every coordinate
transform downstream is silently off by some constant offset or rotation — there is no runtime check
that `proj4` matches `epsg`, since either alone can already fully specify the projection some sites
use.

```toml
[poses]
adapter = "ladybug_export_csv"
file = "LB5, Camera Ladybug/export.csv"
columns = { t = 0, file = 1, e = 2, n = 3, h = 4, roll = 11, pitch = 12, yaw = 13 }
expect_columns = 17
```
`adapter` names a class registered under `("poses", "ladybug_export_csv")` in `geovap.io.registry`.
`columns` maps this vendor's column layout onto the fields `geovap.domain.model.poses.Poses` needs;
`expect_columns` is a row-width sanity check, not the number of columns actually read. Get `columns`
wrong and poses still "load" — with axes swapped or the wrong field where yaw should be — so `geovap
doctor`'s `poses_load`/`pose_time_span` checks (which sanity-check the resulting time span and frame
count) are the first thing to look at, and a visual check (any stage that renders a frame against
the cloud) is what actually catches a swapped column.

```toml
[panos]
adapter = "equirect_dir"
dir = "LB5, Camera Ladybug"
width = 8000
height = 4000
```
Resolves a pose table's `file` column (a filename) to an actual JPEG under `dir`. `width`/`height`
must match the panoramas on disk — `geovap.io.images.load_pano_rgb` decodes whatever is there, but
every equirectangular sampling call in `geovap.domain.math.sampling` assumes this exact size.

```toml
[tiles]
adapter = "laz_dir"
dir = "LAZ_Dražkov_ground"
glob = "*.laz"
id_regex = 'ID\d+_\d*(?P<id>\d{3})_JTSK\.laz'
out_name = "ID3432_000{id}{kind}.laz"
grid = "Klad_LAZ_Dražkov.geojson"
```
`id_regex` must have a named group `(?P<id>...)` and must capture the tile id **with whatever
padding you want to keep** — Dražkov keeps the last three digits including their zero-padding
(`ID3432_0000037_JTSK.laz` -> `"037"`, not `"37"`), because `out_name` reproduces exactly that
padding. Get the padding wrong and every output file this dataset writes is silently renamed
(`ID3432_0001.laz` instead of `ID3432_000001.laz`) with no error anywhere — nothing downstream
knows what the "right" filename was supposed to be, since `TileNaming.out_name` (`geovap.domain.
model.tiles`) is the only definition of "right" there is.

`out_name` is a template with `{id}` (the captured tile id) and an optional `{kind}` (a product
suffix such as `"_colored"`, empty for the consolidated product); a template without `{kind}` gets
the suffix appended before the file extension instead. `grid` is an optional GeoJSON of tile
outlines; without it, tile extents come from each LAZ file's own header bounds (see
`LazDirTileSource`'s `bbox_from_header` fallback) — cheap, since header reads don't decode points,
but a bounding box rather than the true (possibly skewed) tile outline.

**Why `tile_of` (`geovap.domain.model.tiles.tile_of`) does not parse a template backwards.**
`"ID3432_000{id}{kind}.laz"` cannot be inverted unambiguously: given `ID3432_000037_colored.laz`,
is that tile `"037"` with kind `"_colored"`, or tile `"037_colored"` with no kind at all? The old
code guessed by slicing a fixed number of characters off the front — silently assuming every tile id
is exactly three characters, dataset after dataset. `tile_of` instead generates the filename each
known `TileId` *would* produce (via `TileNaming.out_name`, the same function that wrote it) and
matches against that set — exact and dataset-independent, at the cost of needing the tile set in
hand first (never a problem in practice, since a `TileSource` enumerates its tiles up front).

```toml
[tiles.names]
cluster     = "objects_t000{id}.laz"
cluster_rgb = "rgb_t000{id}.laz"
```
Products whose filename is not `out_name` plus a suffix — the clustering stage (which imports no
project code at all, by design) writes its own prefix. Named here rather than hardcoded in the
stage, so a second dataset renames them in one place.

```toml
[reference]
adapter = "jvf_zps"
file = "1_ZPS_GAD.geojson"
codes = "jvf_zps_cz.toml"
```
Optional — see §5 below.

```toml
[sensor]
...
[tuning]
...
```
Build `geovap.domain.model.sensor.Sensor`/`Tuning` — camera intrinsics-equivalent (pano/z-buffer
resolution, splat radius bounds, tolerance) and per-dataset tuning knobs for the projection/scoring
math. These are physical properties of the actual sensor and site; copy the values, don't guess them.

## 3. The four adapter Protocols, and adding a new vendor format

`geovap.io.protocols` (`packages/geovap-core/geovap/io/protocols.py`) defines four `Protocol`s that
`domain`/`runtime`/`stages` code depends on instead of any concrete vendor format:

- `PoseSource` — `load() -> Poses`, `source_file()`, `describe()`.
- `TileSource` — `tiles() -> list[TileRef]`, `out_name(...)`, `describe()`.
- `PanoSource` — `path(filename) -> Path`, `describe()`.
- `ReferenceVectors` (optional) — `objects()`, `classes()`, `describe()`.

`geovap.io.registry` maps the `adapter = "..."` name a descriptor gives to the class that implements
it. Adding a new vendor format is:

1. One class in `geovap/io/adapters/<kind>/<name>.py`, decorated `@register("<kind>", "<name>")`
   (see `geovap/io/adapters/tiles/laz_dir.py` for the shape of one — read the whole file, it is
   short and the comments explain every design choice made against Dražkov's actual filenames).
2. Import it once, for its registration side effect, at the bottom of `geovap/io/registry.py`
   alongside the built-ins.
3. A contract test against the shared adapter test fixtures (see `tests/io/test_tiles_adapter.py`,
   `tests/io/test_poses_adapter.py` — both use a `make_descriptor_file`/`env_paths` fixture that
   builds a tiny on-disk dataset and checks the Protocol's behaviour, not any adapter's internals).

Nothing in `domain`, `runtime` or `stages` needs to change — that is the entire point of the
Protocol boundary: those packages call `settings.get().tiles`/`.poses`/`.panos`/`.reference` and
never import an adapter module directly.

## 4. `geovap doctor` — run this first

```
$ uv run geovap doctor --dataset mysite
```
resolves the descriptor and adapters, and reports everything wrong, before writing anything to disk.
Example, against the generated synthetic fixture (§5):

```
dataset synthetic  (.../geovap/io/datasets/synthetic.toml)
  data_root  : /tmp/fixture
  workspace  : /tmp/fixture-ws
  publish    : /tmp/fixture-pub
  ...
  stages     : cluster, foe, ingest, store, align, products, store-columns, traj-rot, refine, segds,
               traj-validate, register, seg-eval, assemble, pose-report, reg-conflict, seg-project,
               colorize, colour-report, merge, quality, checks, potree, visual, panos, sphere
adapters
  poses      : LadybugExportCsvPoseSource
      file = /tmp/fixture/panos/export.csv
      exists = True
      count = 12
      ...
checks
  [    ok] poses_load: {'ok': True, 'n_frames': 12}
  [    ok] tiles_list: {'ok': True, 'n_tiles': 4}
  [    ok] frames_vs_panoramas: {...'ok': True}
  [    ok] pose_time_span: {...'ok': True}
  [    ok] tiles_extent: {...'ok': True}
  [    ok] crs: {...}
  [    ok] reference_coverage: {'configured': True, 'n_objects': 4, ..., 'ok': True}
  [    ok] disk_free: {...'ok': True}
  [    ok] cuda: {...'ok': True}
  [    ok] tile_headers: {'ok': True, 'n_tiles': 4}
all checks passed
```
A broken descriptor (bad `id_regex`, missing file, wrong column count) fails one of these named
checks with the actual mismatch, not a traceback three stages later.

## 5. Trying this with no data at all

```
uv run python -m geovap.io.datasets.synthetic /tmp/fixture
export GEOVAP_SYNTHETIC_ROOT=/tmp/fixture
export GEOVAP_SYNTHETIC_WORKSPACE=/tmp/fixture-ws GEOVAP_SYNTHETIC_PUBLISH=/tmp/fixture-pub
uv run geovap doctor --dataset synthetic
```
generates a ~0.65 MB, geometrically coherent stand-in for the Dražkov shape (`geovap.io.datasets.
synthetic`): a short street, a dozen posed panoramas rendered from the scene's own point cloud (so
projecting the cloud back into a panorama lands on genuinely matching pixels, not luck), a handful
of skewed LAZ tiles, and a few JVF/ZPS reference features using real taxonomy codes. It is
deterministic given `--seed`, so it is also what CI and a new adapter's contract tests run against
instead of the real ~600 GB dataset.

## 6. `[reference]` absent

`Settings.reference` returns `None` when the descriptor has no `[reference]` table — reference
vectors are optional (`geovap.io.registry.build_reference`). A semantics stage that needs them
(pseudo-GT construction, evaluation against JVF) reports `available() is False` for that dataset via
its `StageSpec`, rather than failing — `geovap run`/`geovap stages` show it as
`[unavailable for this dataset]` and skip it; every other stage runs to completion normally.
