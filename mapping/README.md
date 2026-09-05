# `mapping` — obousměrné mapování panoramata Ladybug ↔ mračno bodů (Dražkov)

Knihovna + CLI nad daty `Geovap_data/DTM_Dražkov`. Jádro je geometrie a per-snímkové hloubkové
panorama; nad ním stojí dva klienti: **cloud → pano** (rendrování atributů a JVF vektorů do
panoramat, s okluzí) a **pano → cloud** (obarvení mračna, validace proti TerraScanu, rozhraní pro
hlasování o třídách).

## Prostředí

```
uv sync                 # .venv s Python 3.11, numpy/numba/scipy/laspy/opencv/skimage/torch
uv run pytest           # rychlé testy (~6 s); `-m slow` spustí regresi na dlaždici 037
```

Data se čtou z `Geovap_data/DTM_Dražkov`, všechny odvozené soubory jdou do `Geovap_cache/`
(přepsatelné proměnnými `GEOVAP_DATA`, `GEOVAP_CACHE`). Skripty v `experiments/` zůstávají
netknuté a používají se přes `mapping.compat` (`common.io_data`, `common.class_map`).

## Postup (jednou)

```
uv run python -m mapping.cli.build_store               # LAZ -> sloupcový memmap store (18 GB, 31 s)
uv run python -c "from mapping.vehicle_mask import *; from mapping.poses import load_poses; m,c=build(load_poses()); save(m,c)"
uv run python -c "from mapping.products import build_all_frames; build_all_frames(workers=16)"   # 1503 hloubkových panoramat (17 GB, ~10 min)
```

## Použití

```python
from mapping.poses import load_poses
from mapping.frame_select import FrameIndex
from mapping.products import FrameProducts
from mapping import geometry

poses = load_poses()                     # export.csv, seřazené, 30 průjezdů, rychlosti
fi = FrameIndex(poses)                   # R[M,3,3], C[M,3]; frames_for_points / frames_for_tile / nearest_in_time
u, v, r, el = geometry.world_to_pano(P, fi.R[k], fi.C[k])          # svět -> pixel
fp = FrameProducts.load(k)                                          # hloubka + id bodu (2000x1000)
xyz = geometry.pano_to_world(u, v, fp.range_at(u, v), fi.R[k], fi.C[k])   # pixel -> svět
pid = fp.point_at(u, v)                                             # pixel -> id bodu ve store
vis = fp.visible(r, u, v)                                           # viditelnost (z-buffer + tolerance)
```

CLI:

```
uv run python -m mapping.cli.render_frame 367 --layers rgb depth classification --jvf --overlay
uv run python -m mapping.cli.colorize --workers 8 --tag run [--no-occlusion] [--sampling nearest]
uv run python -m mapping.report run                # out/run/report.md + PNG
uv run python -m mapping.cli.calibrate             # edge-ICP kalibrace rigu, out/calib/{rig_fit,fit}.json
```

## Konvence (ověřené, neměnit)

yaw = matematický azimut CCW od +E; roll a pitch se **záporným znaménkem**; `v = 0` zenit;
šev `u = 0` v azimutu = yaw; bez zrcadlení. `geometry.world_to_pano` je maticový přepis
`experiments/common/camera.py` (test shody < 1e-6 px ve float64).

Rig model (`rig.py`): boresight (ω, φ, κ) body→kamera, lever arm v osách vozidla (x vpřed, y vlevo,
z nahoru), časový offset dt. Identita reprodukuje pilot. Produkty nesou hash rigu.

## Co se zjistilo (září 2026)

- **Regrese pilotu**: recept pilotu (nejbližší v čase, nejbližší pixel, bez okluze, **body blíž než
  3,5 m vyřazeny**) dává medián CIE76 6,08 (n 43 k) — reprodukováno na dvě desetinná místa. Bez
  vyřazení blízkých bodů je medián 7,08: body pod vozidlem se promítají na karoserii.
- **Maska vozidla** (`vehicle_mask.py`): statické hrany podle koherence gradientu přes 200 snímků;
  ~25 % obrazu (kapota, střecha, oba skenery, černá čepička) + bezpečnostní okraj 10 px (3,6°) při
  čtení, protože obrys střechy míjel zaoblená ramena (barva karoserie prosakovala na vozovku).
  S maskou medián CIE76 5,73.
- **Nejhorší snímky podle ΔE nejsou špatně zarovnané** (`align.py`, `02 §13.6`): siluety mračna sedí
  na hranách fotky ≤ 2 px; ΔE proti referenci měří volbu zdrojového snímku v TerraScanu (pomalé
  otočky obarvil z jiných snímků). Zarovnání ověřovat siluetami, ne barvou.
- **Okluze**: z-buffer 2000×1000, uzavření děr, tolerance `max(0,15 m; 0,03 r; K·lokální rozptyl hloubky)`
  (poslední člen řeší zem pod tečným úhlem). Body do hloubkového panoramatu jen z
  ±45 s okolo snímku (`TIME_WINDOW_S`): mračno je sjednocení 29 průjezdů, fotka je jeden okamžik —
  brány, zaparkovaná auta a lidé z jiných průjezdů nesmí zastiňovat.
- **Celé mračno**: běh `identity` (produkty bez časového okna) produkt medián-top-5 ΔE00 4,98,
  `%(>20)` 6,4 %, pokrytí 82 %; běh `tw45` (časové okno + tolerance) **medián 4,93, pokrytí 94,1 %**,
  nejbližší-v-čase s okluzí 60,8 % (z 48,9 %). Stratifikace podle gradientu 4,3 (hladké) → 15,3 (hrany).
  Podrobně `out/{identity,tw45}/report.md`. Paměť: 8 procesů na 27 M dlaždicích OOM na 125 GB → 5 procesů.
- **Kalibrace** (`calib/`): barevná ΔE je na Δt i boresight plochá (potvrzeno). Fungující metoda je
  edge-ICP (`calib/icp.py`): body na hloubkových nespojitostech/siluetách vůči obloze z jemného
  4000×2000 z-bufferu (výběr podle vlastní buňky, ne vítěze splatu — ten kolísá až o 1°) ↔ nejbližší
  hrana fotografie z distanční transformace, soft-L1, okna 40→6 px. Výsledek na 117 snímcích:
  boresight (0,07°, −0,01°, −0,03°), lever arm ≤ 2 cm, dt ≈ 1 ms; rozptyl mezi 5 časovými bloky
  0,05–0,17°. **Identita je v rámci šumu měření správná; pro produkci se ponechává.**
  Pokusy s renderovanou intenzitou (NGF/NMI, `calib/objective.py`) a chamfer skeny (`calib/chamfer.py`)
  byly na tomto venkovském datasetu příliš ploché — ponechány jen jako reference.
- **Zatáčky**: snímky s vysokou úhlovou rychlostí (|dyaw/dt| > 8°/s) mají ΔE 10–15 a jejich
  optimální Δt kolísá od −4 do +3,5 s (není konstantní) — chování pózy v zatáčkách/otáčkách vyžaduje
  dotaz na GEOVAP (trigger vs. expozice, definice Yaw). Na rovných úsecích jsou v pořádku.

## Čistý dataset (`quality.py`, `04_cisty_dataset.md`, `dataset/`)

Per-snímková kontrola zarovnání (siluety mračna vs. hrany fotky, konflikt průjezdů, pohyb, ostrost) →
třídy clean 825 / unverified 163 / usable 295 / reject 220. `dataset/clean_frames.json` je seznam pro
další použití; `dataset/frame_quality.csv` nese všechny veličiny.

## Výstupní LAZ (`las_out.py`)

LAS 1.4 PF7, XYZ/čas/třída/intenzita/psid bitově shodné se vstupem, `red/green/blue` = náš produkt
(8 bit × 256), extra dimenze `ref_r/g/b`, `nt_r/g/b`, `dE00_med`, `dE00_nt`, `dE00_nt_noocc`,
`src_image`, `n_views`, `col_conf`, `cam_dist`, `inc_angle`, `img_grad`, VLR `geovap_map` s provenience.
