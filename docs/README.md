# Documentation index

The eleven numbered documents below (`00`…`09`) are a **historical research record**. They were
written against the flat `mapping/` layout that existed before the `packages/` restructuring. Their
**commands and paths have been updated** in this pass so that copy-pasting one still works today
(`python -m mapping.X` invocations are followed by a `dnes:`/`today:` line giving the current
equivalent, and stale directories such as `dataset/` are rewritten to `datasets/drazkov/baseline/`).
Their **findings, measurements, tables and conclusions have not been touched** — a number in these
documents is what was measured at the time, and stays that way even where the code that produced it
has since moved or, in one case (the consolidated LAZ layout in `09`), genuinely changed shape. Where
that happened, a clearly marked note says what changed and points at where the current description
lives, instead of silently editing the historical account.

## Historical research record (Czech, findings frozen)

| doc | what it covers |
|---|---|
| [`00_prehled_dat.md`](00_prehled_dat.md) | Overview of the whole Dražkov effort, `01`–`09` summarised, "where does what live" inventory. |
| [`01_plan.md`](01_plan.md) | Original project plan. |
| [`02_obarveni_pointcloudu.md`](02_obarveni_pointcloudu.md) | Point-cloud colourisation from panoramas: camera model, z-buffer occlusion, colour fusion. |
| [`03_semanticka_segmentace.md`](03_semanticka_segmentace.md) | Semantic segmentation literature survey and approach selection. |
| [`04_cisty_dataset.md`](04_cisty_dataset.md) | Definition and construction of the "clean" (independently-verified-alignment) frame subset. |
| [`05_benchmark_segmentace.md`](05_benchmark_segmentace.md) | Zero-shot segmentation model benchmark against JVF-derived pseudo-GT. |
| [`06_umisteni_panoramat.md`](06_umisteni_panoramat.md) | Panorama placement / pose review. |
| [`07_revize_geometrie_a_data.md`](07_revize_geometrie_a_data.md) | Critical review of the geometry code and data before the pose-correction push. |
| [`08_korekce_poz_panoramat.md`](08_korekce_poz_panoramat.md) | Pose correction (S1–S7): trajectory, refinement, pass registration, validation report. |
| [`09_konsolidace.md`](09_konsolidace.md) | Consolidation run: one driver, rebuild on corrected poses, one 3D product. Contains the note on what changed since (three parallel LAZ sets -> one LAZ per tile). |
| [`09_konsolidace_comparison.md`](09_konsolidace_comparison.md) | Companion comparison tables for the consolidation run. |

## Current reference (English, kept up to date)

| doc | what it covers |
|---|---|
| [`artifacts.md`](artifacts.md) | The artifact contracts between team streams — generated from `geovap/runtime/artifacts.py`, always current. |
| [`bring-your-own-dataset.md`](bring-your-own-dataset.md) | How to point the pipeline at a new dataset: the three roots, the descriptor TOML, writing an adapter, `geovap doctor`, the synthetic fixture. |

The root [`README.md`](../README.md) covers the four `packages/*` distributions, the layering rule,
and the one-command quickstart.
