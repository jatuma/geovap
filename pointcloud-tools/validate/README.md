# pointcloud-tools/validate

Headless validation of the consolidated Potree product (`pointcloud-tools/consolidated/index.html`)
and its upstream inputs. See plan Part C6.

## Files

- `check_products.py` - no-browser cross-checks (cluster ids, classification histograms, dE00
  medians, octree metadata, verify flags from merge markers) -> `validation/checks.json`, nonzero
  exit on failure.
- `screenshots.py` - headless-browser screenshot validation: samples clean frames, drives
  `consolidated/index.html` at several yaws with photo opacity {0, 1, 0.5}, scores cloud/photo edge
  agreement, builds contact sheets, and writes `edge_metric.csv` + `summary.json` (numeric AZ_OFFSET
  / export-vs-corrected verdict).
- `edge_metric.py` - `edge_agreement()` (raw Canny-vs-Canny; saturates on splat texture, kept for
  reference) and `silhouette_agreement()` (cloud skyline on the black headless background vs photo
  edges, the analogue of `mapping.quality._residual`). Neither separates pose variants reliably on
  this rural scene (hedges, flat fields) - use them for contact sheets, not for verdicts.
- `sphere_check.py` - **the decisive convention test**: renders the same view offline through
  `mapping.geometry` (camera at the sphere centre, Potree yaw/pitch/FOV) and correlates it with the
  photo-only screenshot (NCC). **2026-09-16: no longer NCC-only.** The pre-fix "NCC 1.000 for
  AZ_OFFSET 0" verdict was circular — the test rendered through the same (reflected) camera model the
  export used, so agreement with the export proved nothing about ground truth. `sphere_check.py` now
  also renders a model-independent pinhole reference from the registered point cloud's silhouette (seen
  from the camera centre) and requires **NCC ≥ 0.95 AND** the pinhole silhouette to beat every control
  set before declaring a variant correct; `--no-pinhole` skips that half of the check. See
  `02_obarveni_pointcloudu.md §2.2` and `09_konsolidace.md §7` for the root cause (azimuth-to-column
  mapping was mirrored) and the fix. Writes `sphere_check.{csv,json}` next to the screenshots; exit 1
  when a variant fails.
- `foe_check.py` - **model-independent travel-direction check**: measures the focus of expansion (FOE)
  of frame-to-frame photo motion on straight segments and compares its column to the model's predicted
  travel-direction column. Pass if median `|Δu/W| ≤ 0.1`. Purely photometric (no point cloud, no camera
  model assumption beyond "the FOE column is where the vehicle is heading"), so it doesn't share any
  assumption with `mapping.geometry` and catches the same seam/mirror error the old `sphere_check.py`
  verdict missed.
- `driver.js` + `package.json` - Playwright-based headless Chromium driver used by `screenshots.py`
  (one page load per frame+variant, multiple shots via `window.__setCam`/`window.__setOpacity`
  without reloading the octrees). Falls back to plain `chrome --headless=new --screenshot` (one page
  load per shot) if Playwright isn't installed.

## Setup

```sh
cd pointcloud-tools/validate
npm install   # local node_modules/playwright, not global
```

Playwright's own Chromium download works out of the box, but this repo already has a
pinned Chromium at `~/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome` (used by default);
override with the `CHROME` env var if needed. Headless WebGL2 needs
`--use-angle=swiftshader --enable-unsafe-swiftshader` (already baked into `driver.js` and the chrome
fallback) - there is no GPU in this environment.

## Running

The consolidated Potree product and viewer container must exist first (Part C1-C5 of the plan);
these scripts only read from the running `viewer` container / cache, they don't build anything.

```sh
# no-browser checks (fast, always safe to run)
uv run python pointcloud-tools/validate/check_products.py

# screenshot validation (needs `docker compose up viewer` running and consolidated/ converted)
uv run python pointcloud-tools/validate/screenshots.py \
    --frames 12 --yaws 0,90,180,270 \
    --variants panos_corr0,panos_exp180,panos_exp0
```

Both scripts write into `<CONSOLIDATED_DIR>/validation/` by default
(`mapping.config.CONSOLIDATED_DIR`, i.e. `OUT_DIR/consolidated/validation`); override with
`--out-dir`. Run `--help` on either script for the full flag list, or `--dry-run` on
`screenshots.py` to see the planned job list without launching a browser.

## Reading `summary.json`

`az_offset_verdict.ranked_variants_best_first` lists each `--variants` entry by
`median_of_medians_px` (cloud/photo edge-agreement, lower is better, capped at 30 px). The correct
`AZ_OFFSET` (see `mapping/panos.py`) and pose source (export vs corrected) should show up as the
variant with the lowest median - a large gap between AZ_OFFSET 0 and 180 variants is the numeric
counterpart to the by-eye "sphere sits on the building" check.

## Potree 1.8 quirks the page works around (keep in mind when editing `consolidated/index.html`)

- `Potree.Viewer`'s constructor parses URL params itself (`opacity`, `FOV`, `pointSize`, ...) and
  crashes on `?opacity=`; the page uses `pano_opacity`.
- Point clouds are drawn in Potree's own EDL pass on top of every scene object, so a sphere with
  opacity 1 never hides the cloud: "photo only" hides the octrees, "cloud only" hides the sphere.
- There is no global `THREE`; new colours are cloned from `Potree.Gradients.*`.
- Scalar colouring computes `w = (value + offset) * scale` with `scale = |initialRange| / |range|`,
  which only normalises when the attribute's stored initialRange has length 1. `scalarRange(pc,
  name, lo, hi)` in the page passes the compensated range so uint8 0-3 / float 0-97 attributes
  display as intended.
- `?nogui=1` also sets a black background so no-point pixels are exactly black (silhouette metric).
