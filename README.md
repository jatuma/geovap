# geovap

Turns a posed-panorama + tiled-point-cloud site survey (panoramas + MLS LAZ tiles + optional
reference vectors) into one consolidated, coloured, semantically-labelled point cloud per tile,
served through Potree. Built and validated against a ~600 GB real-world dataset (Dražkov); ships a
tiny synthetic fixture so the whole pipeline runs, and every stage's contract is testable, with no
real data mounted.

## Layout

One package, `src/geovap/`; the subpackage tree says who owns what (see `CODEOWNERS`).

| subpackage | owns |
|---|---|
| `domain` | pure geometry, math and scheme, no I/O |
| `io` | dataset descriptors and vendor-format adapters |
| `runtime` | settings, workspace layout, artifact contracts |
| `stages/base`, `stages/prepare` | stage spec, registry and CLI; ingest and dataset checks |
| `stages/register` | panorama registration |
| `stages/semantics` | pseudo-GT, zero-shot segmentation, label projection (needs `geovap[semantics]`: torch, transformers) |
| `stages/objects` | clustering |
| `stages/colour` | colourisation |
| `stages/deliver`, `infra` | the consolidated LAZ product, the Potree octree, the viewer/PotreeConverter container infra |
| `app`, `stages/verify` | the `geovap` command (`doctor`/`stages`/`datasets`/`run`/`status`/`compare`); visual and numeric verification |

`pip install geovap` gives a working `doctor`, point store, frame products, colourisation and
delivery, with no torch anywhere in the tree; `pip install "geovap[semantics]"` adds the
segmentation stages, and `geovap[turbo]` faster JPEG decoding.

## Layering, and how it's enforced

Imports may only point down: `app -> stages -> runtime -> io -> domain`. `domain` must stay
importable with nothing but numpy — it is what makes the synthetic-fixture tests meaningful.
Pipeline stage groups (`register`, `colour`, `semantics`, `objects`, `deliver`, `verify`) never
import each other's Python; they hand off through artifacts on disk, whose shape is documented in
[`docs/artifacts.md`](docs/artifacts.md). None of this is just convention — it's checked in CI with
[import-linter](https://import-linter.readthedocs.io/) against the contracts in `.importlinter`:

```
uv run lint-imports
```

## Quickstart

```
uv sync
uv run python -m geovap.io.datasets.synthetic /tmp/fixture
export GEOVAP_SYNTHETIC_ROOT=/tmp/fixture GEOVAP_SYNTHETIC_WORKSPACE=/tmp/fixture-ws GEOVAP_SYNTHETIC_PUBLISH=/tmp/fixture-pub
uv run geovap doctor --dataset synthetic
```

Against a real dataset, point three environment variables (or descriptor-specific ones — see
[`docs/bring-your-own-dataset.md`](docs/bring-your-own-dataset.md)) at your data, workspace and
publish directories, then:

```
uv run geovap doctor --dataset <name>     # resolve + validate, writes nothing
uv run geovap stages  --dataset <name>    # what would run, in order
uv run geovap run     --dataset <name>    # run it
uv run geovap status  --dataset <name>    # what has been done
```

Every stage is also runnable standalone, without the `geovap` command at all:
`python -m geovap.stages.colour.colorize --dataset <name>`.

## Tests and baselines

```
uv run pytest -q
```
212 passed / 27 skipped with no dataset mounted; with `$GEOVAP_DATA`/`$GEOVAP_WORKSPACE`/
`$GEOVAP_PUBLISH` pointed at the real Dražkov dataset, 239 passed.

## Further reading

- [`docs/README.md`](docs/README.md) — index of the numbered research documents (historical, Czech)
  and the current reference docs (English).
- [`docs/bring-your-own-dataset.md`](docs/bring-your-own-dataset.md) — descriptor format, writing a
  new vendor adapter, the synthetic fixture.
- [`docs/artifacts.md`](docs/artifacts.md) — the on-disk contracts between stage groups.
- [`datasets/README.md`](datasets/README.md) — where dataset descriptors and baselines live and how
  they're looked up.
