# Přehled dat a experimentů — od průjezdu k segmentaci v mračnu

Souhrn napříč dokumenty `01`–`08`: co bylo naměřeno/dodáno, jaký experiment z toho co vyrobil, kam
výstup putuje dál. Řazeno chronologicky; `07`/`08` zpětně opravují vstupy pro `04`/`05`/`06` — tato
revize je vyznačena samostatně na konci.

## 0. Vstupní data (dodaná, nevznikla experimentem)

| data | popis | zdroj |
| --- | --- | --- |
| `LAZ_Dražkov_ground/*.laz` (38 dlaždic) | mračno bodů, ~585 M bodů, S-JTSK, `gps_time`, `class`, `intensity`, RGB od TerraScanu jako referenční obarvení | mobilní mapping (VMX-2HA), dodáno GEOVAP |
| `LB5, Camera Ladybug/export.csv` + 1 503 panoramat | pozice/orientace kamery (E, N, H, roll/pitch/yaw) a 8000×4000 ekvirektangulární JPEG pro 30 průjezdů | stejný sběr, Ladybug 5 |
| `1_ZPS_GAD.geojson` (JVF) | vektorová evidence pasportu (hraniční linie + definiční body, 4 342 objektů, žádné polygony) | GEOVAP, nezávislý na mračnu/panoramatech |
| váhy veřejných modelů (Mask2Former/Vistas, Cityscapes; EoMT; OneFormer; SegFormer) | předtrénované sémantické segmentační modely, zero-shot | veřejné checkpointy |

## 1. `01_plan.md` — Plán

Bez vlastního výstupu dat; stanovuje cíl (obarvení mračna + sémantická segmentace pro pasport) a
rozděluje práci do `02`/`03`. Uvedeno pro úplnost pořadí.

## 2. `02_obarveni_pointcloudu.md` — Obarvení mračna z panoramat

- **Vstupy**: LAZ dlaždice, `export.csv` + panoramata.
- **Metoda**: `mapping/geometry.py` (kamerový model svět↔panorama, ověřený proti `experiments/common/camera.py` na <1e-6 px), `mapping/colorize.py` (fúze medián-top-5 přes pohledy, proti pohybujícím se vozidlům), `mapping/cloud_store.py`, sférický z-buffer (2000×1000, později jemný 4000×2000) pro okluzi s časovým oknem ±45 s mezi průjezdy, `mapping/calib/icp.py` (kalibrace boresight/lever-arm metodou hrana-siluety↔hrana-fotky).
- **Výstupy**: LAS 1.4 PF7 s extra dimenzemi (`sem_class`, `src_image`, `col_conf`, `inc_angle`, `dE00`, `n_views`…) v `Geovap_cache/out/{identity,tw45}/`; `out/calib/fit.json`.
- **Metodika**: Validace probíhá proti existujícímu RGB od TerraScanu (obarvenému ze stejných panoramat) — pokud se model kamery/okluze/kalibrace shoduje s TerraScanem, je správný. Kalibrace edge-ICP vyšla prakticky identita (boresight 0,07°/−0,01°/−0,03°, lever-arm ≤2 cm).
- **Klíčový výsledek**: medián ΔE00 = 4,93 při pokrytí 94,1 % (běh `tw45`, celé mračno).
- **Navazuje**: `identity`/`tw45` obarvené mračno a `dataset/frame_quality.csv` (níže, `04`) používá právě tento ΔE jako informativní (ne rozhodující) signál.

## 3. `03_semanticka_segmentace.md` — Studie proveditelnosti segmentace (§12 experimenty na Dražkově)

- **Vstupy**: panoramata + `export.csv`, JVF `1_ZPS_GAD.geojson`, veřejné taxonomie (Mapillary Vistas, GOOSE, Cityscapes) jen pro mapování tříd.
- **Metoda**: `experiments/common/reproject.py` (gnómonická reprojekce, 8× yaw + pitch 0°/−45°, softmax fúze), zero-shot Mask2Former (váhy Vistas) na 40 panoramatech; `mapping.vectors.project_polyline` pro promítnutí JVF linií zpět do panoramat jako pseudo-GT bez ruční anotace.
- **Výstupy**: `experiments/out/e1/coverage_by_class.txt`, `experiments/out/e1/overlays/` (125 přehledů), `experiments/out/e2/`.
- **Metodika**: Znovupoužívá kamerový model z `02` k promítnutí existující vektorové evidence (JVF) do panoramat — tím vzniká „zadarmo" pseudo-GT bez nutnosti ruční anotace. Zero-shot veřejný model se porovná s tímto pseudo-GT, aby se zjistilo, zda veřejné třídy vůbec pokrývají potřebné kategorie pasportu.
- **Klíčový výsledek**: 3 161/4 342 objektů JVF (72,8 %) má panorama do 20 m; třídy pasportu jsou veřejnými modely pokryté nerovnoměrně (skóre 0,50 až 0,00), 4,7 % objektů bez veřejného ekvivalentu. Reálné roll/pitch vozidla jsou malé (medián <1,3°) — rotační citlivost z literatury (SGAT4PASS) zde nebude limitující.
- **Navazuje**: potvrzuje proveditelnost přístupu „JVF → pseudo-GT → benchmark veřejných modelů", který se plně rozpracuje v `05`.

## 4. `04_cisty_dataset.md` — Výběr čistého datasetu (klasifikace kvality snímků)

- **Vstupy**: `export.csv`, panoramata, mračno/trajektorie, per-snímkové ΔE00 z běhu `tw45` (`02`).
- **Metoda**: `mapping/quality.py` (`uv run python -m mapping.quality`) — geometrická rezidua siluet mračna vůči hraně fotky (jemný z-buffer 4000×2000, distanční transformace), konflikt sousedních průjezdů (±45 s okno), rychlost/yaw-rate z trajektorie, ostrost (rozptyl Laplaciánu).
- **Výstupy**: `dataset/frame_quality.csv` (1 503 řádků), `dataset/clean_frames.json` (indexy po třídách), `dataset/tile_summary.json`, `dataset/frame_quality_map.png`, `dataset/frame_quality_stats.png`.
- **Metodika**: Zamítá ΔE vůči TerraScanu jako kritérium (odráží volbu zdrojového snímku TerraScanem, ne geometrii). Místo toho čtyři třídy (`clean`/`unverified`/`usable`/`reject`) podle geometrického rezidua siluet, konfliktu průjezdů, pohybu a ostrosti.
- **Klíčový výsledek (na exportních pózách)**: clean 825, unverified 163, usable 295, reject 220 z 1 503.
- **Navazuje**: `clean_frames.json` je vstupní filtr pro pseudo-GT v `05` a pro render/kalibrační experimenty; později (viz `08`) je celá tato klasifikace přepočtena na korigované pózy a **promotována** jako produkční sada.

## 5. `05_benchmark_segmentace.md` — Dataset a benchmark segmentace (JVF pseudo-GT v ploše)

- **Vstupy**: `dataset/clean_frames.json` (`04`, 825 čistých snímků), JVF (`1_ZPS_GAD.geojson`), mračno, `dataset/seg/classes.json` (taxonomie).
- **Metoda**: `mapping/seg/areas.py` (polygonizace JVF linií na plochy tříd), `mapping/seg/point_labels.py` (pravidlová klasifikace bodů mračna), `mapping/seg/rasters.py`, `mapping/seg/views.py` (16 horizont-zarovnaných gnómonických dlaždic 1024² na snímek), `mapping/seg/dataset.py` (rozdělení train/val/test k-means(10) na pozicích kamery), `mapping/cli/seg_build.py` + `mapping/cli/seg_bench.py` (zero-shot inference 6 modelů: EoMT-L/Cityscapes, Mask2Former-Vistas, Mask2Former-Cityscapes, EoMT-DINOv3/ADE20K, OneFormer-Cityscapes, SegFormer-B5), `mapping/seg/fusion.py`, `mapping/seg/taxonomy.py` (společná taxonomie pro srovnání modelů).
- **Výstupy**: `Geovap_cache/segds/` (areas/rasters/points/labels/views), `dataset/seg/classes.json`, `dataset/seg/splits.json`, `dataset/seg/bench_frames.json` (100 benchmarkových snímků), `dataset/seg/bench/{results.json, summary.csv, per_class_iou.csv, boundary.csv, bands.csv, tables.md}`.
- **Metodika**: JVF nemá polygony, jen hraniční linie a jeden definiční bod na plochu — plochy se rekonstruují polygonizací (shapely) a přiřazením třídy z definičního bodu (95 % plochy vyřešeno). Každý bod mračna dostane třídu podle vzdálenosti/výšky k nejbližší linii (plot, zeď, zábradlí, půdorys budovy, vegetace nad DTM). Šest veřejných modelů se testuje zero-shot na 16 gnómonických výřezech fúzovaných zpět do ERP, hodnocení v pásu φ∈[−55°,45°] proti tomuto pseudo-GT.
- **Klíčový výsledek**: EoMT-L (DINOv2, Cityscapes) vítězí — mIoU_core 0,358 (100 snímků) / 0,330 (čistá blízká zóna), nejrychlejší (4,7 s/panorama, 3,8 GB) → doporučen jako výchozí model pro fázi 2. Zjištěna chyba: 328/755 měřitelných čistých snímků má posun JVF↔fotka > 20 px (`nearfield_bad`), přisouzeno driftu průjezdu vůči JVF ~0,3–0,6 m — **tento nález je motivací pro `08`**.
- **Navazuje**: EoMT-L a tato benchmarková infrastruktura se používají v `pointcloud-tools`/`mapping.cli.seg_build` k promítnutí segmentace zpět do mračna (commit „Project EoMT-L segmentation into the point cloud, serve in Potree").

## 6. `06_umisteni_panoramat.md` — Geometrie umístění panoramat (podrobná revize kamerového modelu)

- **Vstupy**: `export.csv`, panoramata, LAZ dlaždice.
- **Metoda**: `mapping/poses.py`, `mapping/geometry.py`, `mapping/rig.py` (kalibrace boresight/lever-arm/dt_s edge-ICP na 117 snímcích), `mapping/products.py` (`zbuffer.splat` — per-snímkové rastery `depth_mm`/`point_id`), `mapping/colorize.py`, `mapping/render.py`.
- **Výstupy**: per-snímkové produktové rastery (1 503 souborů, ~17 GB), obarvené LAS 1.4 s extra dimenzemi.
- **Metodika**: Konvence os (zenit, sešití = yaw, bez zrcadlení, roll i pitch negované) určeny ablací proti referenčnímu RGB TerraScanu — jasně oddělitelné alternativy (6,5 vs. 39/18/17 ΔE). Okluzní z-buffer s ±45 s oknem a tolerancí na šikmý dohled do země.
- **Klíčový výsledek**: kalibrace rigu prakticky identita; medián ΔE 4,93 při pokrytí 94,1 %. Otevřený problém: **rezidua yaw v zatáčkách** (ΔE 10–15, nestabilní optimální Δt) — **motivace pro `07`/`08`**.

## 7. `07_revize_geometrie_a_data.md` — Kritika geometrie a plán nových dat

- **Vstupy**: kód `mapping/` z `06` (kritický přezkum, žádný nový běh dat), nevyužité dimenze v LAZ (`user_data`, `scan_angle_rank`, `return_number`, `point_source_id`, `intensity`).
- **Metoda**: analýza + jeden diagnostický proof-of-concept fit rovin skenovací hlavy (dlaždice 011).
- **Výstupy**: tento dokument — prioritizovaný seznam slabin a konkrétní plán nových dat, realizovaný v `08`.
- **Metodika**: Diagnostikuje, že interpolace pózy mezi snímky (lineární, ~5 m/0,6 s rozestup) nemůže reprezentovat změny yaw 5–10° v zatáčkách — kořenová příčina anomálie z `06`. Navrhuje rekonstrukci hustší trajektorie přímo z mračna: body jedné skenovací hlavy VMX-2HA v okně 2 ms leží na rovině (rotující zrcadlo), normály rovin dávají orientaci ~500 Hz, pozice v rovině je afinní funkcí `scan_angle_rank`.
- **Klíčový výsledek**: proof-of-concept fit rovin RMS 3,8 mm/2,8 mm (hlavy 1/2), fitovaný počátek skenovací hlavy se liší od interpolované pozice kamery o (1,46; 0,16; 0,78) m — konzistentní s fyzickým rozměrem rigu, potvrzuje proveditelnost metody.

## 8. `08_korekce_poz_panoramat.md` — Korekce pózy panoramat, kroky S0–S7

**Klíčový bod celého přehledu**: tento dokument **přepočítává vstup pro `04`, `05`, `06`, `07`** a
vytváří `poses_corrected` — druhý, korigovaný zdroj póz vedle `export.csv`.

- **Vstupy**: `export.csv` (zachován jako regresní kotva), mračno rozšířené o `EXTRA_COLUMNS` (`user_data`, `scan_angle_rank`, `return_number`) přes `mapping/cli/store_add_columns.py`, `dataset/frame_quality.csv` (`04`), JVF hraniční linie (pokus o absolutní datum).
- **Metoda (CLI řetězec v `mapping/cli/`)**:
  1. `store_add_columns.py` + `pass_psid.py` — rozšíření store, křížová tabulka `point_source_id`.
  2. `align_frames.py` — diagnostika.
  3. `build_trajectory.py --rot-only` — hustá **pouze-rotační** trajektorie z fitů rovin skenovacích hlav (`mapping/trajectory.py`); poziční varianta (S3, plná 6DoF) se ukázala degenerovaná (RMS ~0,72 m) a byla opuštěna.
  4. `refine_poses.py` — per-snímkové 6DoF zpřesnění (`PassRefiner`/`EdgeICP`, `dt, dyaw, droll, dpitch, dlat, dh`) proti hranám fotek.
  5. `register_passes.py` — párové ICP (bod-k-rovině) registrace 30 průjezdů navzájem + pokus o absolutní JVF datum (`mapping/pass_reg.py`).
  6. `assemble_poses.py` — sestavení finálních `poses_corrected`.
  7. `pose_report.py` — validace (`compare_pose_sources`), 6 nezávislých metrik.
- **Výstupy**: `Geovap_cache/out/poses/poses_corrected.{csv,json}` (json = provenience: S3b+S4+S5b, git rev, hash `export.csv`), `out/pass_reg/{pass_transforms.json, conflict_reg.json, jvf_offsets.json}`, `out/poses/traj_diag/`, `out/poses/report_final.{md,json}`; downstream přestavěné `Geovap_cache/out/dataset_e8f3e1/` (nová `04`) a `Geovap_cache/segds_e8f3e1/` (nová `05` pseudo-GT), `dataset/seg/project_eomt_city.{json,md}` (přeprojektovaná segmentace do mračna), export-baseline zálohy v `dataset/export_baseline/` a `dataset/seg/export_baseline/`.
- **Metodika**: (a) S3b — hustá orientace ~200 Hz z rovin obou skenovacích hlav; (b) S4 — per-snímkové edge-ICP zpřesnění proti hranám fotek; (c) S5 — párové ICP mezi 30 průjezdy (řízené `point_source_id`), pokus o absolutní JVF datum **zamítnut** (detektor obrubníku na tomto venkovském datasetu příliš šumí, předpokládané rozdělení „dobré/špatné" průjezdy z `05` empiricky vyvráceno); (d) S7 — validace `export` vs. `poses_corrected` na šesti metrikách.
- **Klíčové výsledky**:
  - Párová registrace průjezdů: RMS 0,102 m → 0,035 m (66/72 párů konvergovalo); konflikty průjezdů 17→16/309.
  - Hustá trajektorie vs. naivní lineární interpolace v zatáčkách: medián chyby yaw 0,197° vs. 5,639° (~29× lepší) — jediný jednoznačný přínos.
  - JVF absolutní datum: **selhalo** (offset 0,2–0,9 m i u očekávaně dobrých průjezdů), zůstává vypnuté (`--datum none`).
  - Parallax (S6): neověřitelné — žádné edge-ICP asociace blíž než 19,5 m.
  - Nový otevřený problém: azimutálně strukturované, znaménko-měnící se reziduum v zatáčkách, nevysvětlitelné žádnou tuhou pózou jednoho snímku (možný artefakt nesimultánního sešívání Ladybug).
- **Dopad na navazující data** (přepočet `04`/`05`/`06`/`07`):
  - `dataset/clean_frames.json`: 825→**830** clean (viz `04 §5`, detailní rozpad turnoveru).
  - Segmentační pseudo-GT metrika `fence` klesla 13,4 %→11,1 % (citlivost na S5b posun).
  - Projekce segmentace do mračna (`dataset/seg/project_eomt_city.json`): pokrytí +0,8 pb, ale mIoU_core −2,4 pb (zavlečeno souběžným přestavěním pseudo-GT) — a odhalena/opravena reálná chyba, kde `BENCH_DIR` byl nesprávně citlivý na zdroj pózy (první běh měl `n_frames=0`/`coverage=0.0` na všech 38 dlaždicích).
  - **Kritické upozornění pro každého spotřebitele**: `poses_corrected` rotuje kameru kolem těžiště každého průjezdu, takže každý produkt kotvený ve světovém rámci (obarvení, export LAS, segmentační rastery) **musí** navíc použít `CloudStore(registration=pass_transforms.json)`, jinak vznikne až ~1–2 m falešný posun kamera↔mračno (např. dlaždice 037: ΔCIE76 +0,012 s registrací vs. +1,724 bez ní).

## 9. Souhrnný diagram závislostí

```text
export.csv + panoramata + LAZ ──────────────┬──────────────────────────────┐
                                             │                              │
                    02 obarvení (kalibrace, ΔE validace)                   │
                                             │                              │
                    03 feasibility (JVF→pseudo-GT, zero-shot)              │
                                             │                              │
JVF (1_ZPS_GAD.geojson) ────────────────────┤                              │
                                             ▼                              │
                    04 quality.py → clean_frames.json (825/163/295/220)    │
                                             │                              │
                                             ▼                              │
                    05 seg dataset+benchmark → EoMT-L vítěz, nearfield_bad │
                                             │                              │
                    06 detailní geometrie panoramat, anomálie v zatáčkách  │
                                             │                              │
                    07 kritika → plán husté trajektorie ze skener-hlav ────┘
                                             │
                    08 S0–S7 → poses_corrected.{csv,json}
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
   04′ dataset_e8f3e1/ (clean_frames 830/163/302/208,            05′ segds_e8f3e1/ (pseudo-GT
   promotováno jako produkční)                                   přestavěné, project_eomt_city.json
                                                                  přeprojektováno do mračna)
```

## 10. Kde co leží (verzované vs. cache)

- Verzované v `dataset/`: `frame_quality.csv`, `clean_frames.json`, `tile_summary.json`, mapy/statistiky, `seg/{classes,splits,bench_frames,stats,project_eomt_city}.json`, `seg/project_eomt_city.md`.
- Regresní baseline (export pózy, ne živá data): `dataset/export_baseline/`, `dataset/seg/export_baseline/`.
- Necachované originály / velké produkty: `Geovap_cache/out/{identity,tw45}/`, `Geovap_cache/out/poses/`, `Geovap_cache/out/pass_reg/`, `Geovap_cache/out/dataset_e8f3e1/`, `Geovap_cache/segds/`, `Geovap_cache/segds_e8f3e1/`.
- Vizualizace v Potree: `pointcloud-tools/` (viewer servíruje obarvené a segmentované mračno, `mapping/cli/seg_build.py` produkuje vstup).
