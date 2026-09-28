# Konsolidace celého procesu — jeden vstupní bod, přestavba na korigovaných pózách, jeden 3D produkt

Datum běhu: 2026-09-15/16. Navazuje na `00_prehled_dat.md` (přehled `01`–`08`) a nahrazuje ruční řetězení
CLI z `08 §5` jedním driverem. Vše níže je **naměřeno tímto během** (markery v `Geovap_cache/out/pipeline/`,
souhrn `out/pipeline/comparison.md`), reference v závorkách jsou hodnoty z předchozích dokumentů.

## 0. Proč

Stav před konsolidací (2026-09-15): cache přesunutá na `/mnt/Geovap_cache` (kód ji hledal jinde), korigované
pózy `poses_corrected.csv` + `pass_transforms.json` a běh `tw45` **na disku chyběly** (přežily jen odvozeniny
`segds_e8f3e1`, `out/dataset_e8f3e1`), store bez S1 sloupců, Potree výstupy přesunuté do
`/mnt/Geovap_cache/TestOutput/output/` a viewer kontejner držel smazaný adresář. Panoramata v Potree, která
vypadala „perfektně umístěná", byla exportována z **export.csv** (ne z korigovaných póz) nad neregistrovaným
mračnem.

## 1. Co vzniklo (kód)

| soubor | účel |
|---|---|
| `mapping/cli/pipeline.py` | **driver**: `run [--from/--to/--only] [--force] [--detach] [--with-optional]`, `status`, `compare`, `env-check`. 24 stagí, každý příkaz vlastní subprocess (`uv run python -u -m …`, env `GEOVAP_CACHE`, `GEOVAP_POSES` — do stage `assemble` včetně **export**, pak `corrected`), markery `out/pipeline/<stage>.json` (hash vstupů, výstupy, metriky), log `out/pipeline/pipeline.log` + `logs/`, `--detach` = `setsid nohup` + PID guard. `compare` → `out/pipeline/comparison.{md,json}`. |
| `mapping/config.py` | `CACHE_ROOT` = env `GEOVAP_CACHE`, jinak první existující z `../Geovap_cache`, `/mnt/Geovap_cache` (+ symlink `/home/jatuma/repos/Geovap/Geovap_cache → /mnt/Geovap_cache`); `CLEAN_FRAMES_JSON`, `QUALITY_CSV`, `POTREE_OUTPUT_DIR` (env `POTREE_OUTPUT`), `PIPELINE_DIR`, `CONSOLIDATED_DIR`. Pět natvrdo zapsaných čtení `dataset/clean_frames.json` nahrazeno konstantou. |
| `mapping/cli/colorize.py --poses`, `mapping/cli/build_frames.py`, `products.build_all_frames(skip_existing)` | obarvení a produkty pro libovolný zdroj póz, resumable |
| `mapping/seg/project.py` | `SEG_OUT_DIR`/`LAS_DIR` pose-aware (`out/seg_eomt_<hash6>`, `POTREE_OUTPUT/seg_eomt_<hash6>/tiles`), `_meta.json` nese `poses_hash`; `cli/seg_project.py --poses` |
| `mapping/quality.py reclassify | promote` | promoce `out/dataset_<hash6>/` do `dataset/` se zálohou a diffem clean množiny |
| `mapping/seg/nearfield.py main()`, `seg_bench --frames missing:<spec> | <soubor.json>`, `seg_bench evaluate --out` | doplnění chybějících predikcí, evaluace na pevné množině snímků bez přepsání hlavních výsledků |
| `mapping/las_out.py` | `write_tile(xyz=…)` (registrované souřadnice), `CONS_EXTRA_DIMS`, `OBJ_EXTRA_DIMS`, `verify(xyz_mode="registered")` |
| `mapping/merge.py`, `mapping/cli/merge_products.py` | **konsolidovaný produkt**: per dlaždice tw45 RGB + sémantická třída + clustery → `out/consolidated/tiles/*.laz` a `objects/*.laz` v registrovaném rámci, kontrola identity bodů a `poses_hash` vstupů, resumable, `summary.json` |
| `pointcloud-tools/docker-compose.yml` + `.env` | `POINTCLOUD_OUTPUT=/mnt/Geovap_cache/TestOutput/output`, converter vidí `GEOVAP_CACHE/out` jako `/cache_out` |
| `pointcloud-tools/viewer/Dockerfile` | `ARG POTREE_REF` + build-time guard na Images360 konvence |
| `pointcloud-tools/consolidated/index.html` | stránka s 6 módy (RGB tw45 / segmentace / objekty / třída objektu / ΔE00 / RGB original = TerraScan), panos panel z `view.html`, URL parametry pro headless (`mode, panos, frame, pano_opacity, yaw, pitch, fov, budget, nogui, base, objects`) |
| `pointcloud-tools/validate/{screenshots.py, driver.js, edge_metric.py, sphere_check.py, check_products.py}` | headless Chromium záběry (cloud / fotka / blend × yaw), **sphere_check** (Potree vs. náš kamerový model, NCC), kontroly produktu (cluster id, histogramy tříd, ΔE mediány, oktree metadata, verify flagy) |
| `tests/test_pipeline.py`, `tests/test_las_out_xyz.py`, `tests/test_merge.py` | 160 testů celkem, ~11 s |

## 2. Průběh běhu (`pipeline run --detach`, pózy `corrected`, tag `tw45`)

| stage | čas | výsledek (reference) |
|---|---|---|
| store-columns | 1 min | S1 sloupce + časový index, 38 dlaždic |
| align | 9 min | 310 snímků, medián yaw offset 0.0° (Aligner nikdy offset nenašel — potvrzeno) |
| traj-rot | 1 min | 19/30 průjezdů s vlastním rigem, 1373/1503 snímků rot-only; ve vlastním čase snímku \|dyaw\| 0.005° |
| refine | 95 min* | 402 refined / 105 interpolated / 996 kept (ref. 409 accepted) |
| register | 13 min | **72 párů, 66 konvergovalo, RMS 0.102 → 0.035 m** (ref. totéž), `datum=none`, mean_t ≈ 0, max \|t\| 0.46 m, max \|yaw\| 0.66°, 12 flagged |
| assemble | s | `poses_corrected.csv`, hash **`34bca9ff23`** (předchozí `e8f3e1f2b3`; liší se, protože do `align` vstoupil jiný clean list) |
| products | 9 min | 1503 hloubkových panoramat v `frames/34bca9/` |
| pose-report | 26 min | interpolace yaw v zatáčkách **0.200° vs 5.639°** (ref. 0.197/5.639); siluety turning \|du\|/\|dv\| 0.40/1.21 → 0.46/0.93 px; tile 037 CIE76 export 6.076 / corrected+registrace **6.097** / bez registrace **7.777** (ref. 6.088 / 7.80) |
| colorize + report | 60 min | **ΔE00 medián 5.03, pokrytí 93.99 %** (export běh: 4.93 / 94.1 %) — korekce póz je pro barvu neutrální, mapování kamera↔mračno drží (bez registrace by ΔE vyskočilo o +1.7, viz tile 037) |
| quality + promote | 27 min | **835 clean / 165 unverified / 304 usable / 199 reject** (e8f3e1: 830/163/302/208; export 825/163/295/220), 0 chyb; clean množina +17/−12 snímků |
| segds | 25 min | areas 1009 ploch, `all_lines`, labelled 0.950 (past se symlinkem se neopakovala); podíly tříd v pásu vs. e8f3e1 v rámci ±0.1 pb (fence 11.08 → 11.12 %); near-field **762 měřeno / medián 17.9 px / 340 flag** (e8f3e1 757/17.9/336) |
| seg-eval | ~2 h (GPU doplnění 2 + 5×76 + 5×40 snímků, 2× evaluace, report) | viz §4 |
| seg-project | 20 min | **coverage 0.808, pixel acc 0.667, mIoU_core 0.328** (e8f3e1: 0.807/0.667/0.328; export 0.799/0.683/0.352); per-class IoU v rámci ±0.003 |
| merge | 30 min | 38 dlaždic, **584 809 840 bodů**, `xyz_exact` (registered) 38/38, posun vůči vendor rámci medián 0.26 m, max 1.45 m, 100 % bodů; `poses_hash` všech vstupů = `34bca9ff23` |
| potree | 35 min | `consolidated/cloud` (32 GB, 23 atributů) a `consolidated/objects`, obě 584 809 840 b.; bbox vůči `clusters/rgb` posunut ≤ 1.5 m |
| panos | 1 min | 4 sady po 1503 koulích, jpg hardlinkované (0 nových obrázků): `cloud/panos` (corrected+registrace, az 0), `panos_corr180`, `panos_exp0`, `panos_exp180` |
| validate | ~1.7 h | `check_products`: **157 ok / 0 fail** (cluster id per dlaždice i globálně 12 276 = `global_labels.npy`, histogramy tříd = `labels/NNN.npy`, ΔE mediány = tw45 stats, oktree metadata, verify flagy); 576 záběrů (12 snímků × 4 sady × 4 yaw × 3 krytí) + kontaktní archy; **sphere_check** viz §3 |

Souhrnná tabulka „export | e8f3e1 | stav před | tento běh" je v `out/pipeline/comparison.md` (`pipeline compare`); kopie verzovaná jako `09_konsolidace_comparison.md`.

\* refine běžel souběžně s validačními screenshoty (load 60 na 32 jádrech); dokumentovaná doba 52 min.

## 3. Umístění panoramat v Potree: AZ_OFFSET 180 → **0**

`panos.AZ_OFFSET_DEG` byl „naměřený" na 180 (odvození říkalo 0). Nový test
`pointcloud-tools/validate/sphere_check.py` vykreslí týž pohled (kamera ve středu koule, známé Potree
yaw/pitch/FOV) offline přes `mapping.geometry` a koreluje ho s fotkou, jak ji vykreslí Potree:

| sada koulí | NCC vs. náš kamerový model | NCC alternativy „o půl otáčky" |
|---|---|---|
| export, az 0 | **1.000** (8/8 záběrů) | 0.71 |
| export, az 180 | 0.74 | 0.92 |

Vizuálně: na snímku 1406 staví koule s 180 osamělý strom do části oblohy, kde lidar nemá žádné body; s 0 sedí
obloha na oblohu a koruny na koruny. Dřívější potvrzení „okem" proběhlo na silnici mezi dvěma živými ploty,
která po půlotáčce vypadá stejně. Výchozí hodnota je proto **0**; sady `panos_corr180`, `panos_exp0`,
`panos_exp180` zůstávají pro A/B (`?panos=<dir>`).

Další dvě pasti Potree, které validace odhalila: konstruktor `Potree.Viewer` sám parsuje URL parametry
`opacity`/`FOV`/`pointSize` (na `?opacity=` spadne) — stránka používá `pano_opacity`; mračno se kreslí ve
vlastním EDL průchodu **přes** kouli, takže „jen fotka" musí mračno schovat, ne jen nastavit krytí 1.

> **Aktualizace 2026-09-16 (viz §7).** `sphere_check.py` NCC 1,000 výše byl spočítaný proti **starému
> (zrcadlenému) kamerovému modelu** — šlo tedy o kruhové ověření (test i model sdílely stejnou chybu),
> ne o nezávislý důkaz. `sphere_check.py` teď navíc vyžaduje NCC ≥ 0,95 **a** shodu s
> modelo-nezávislou pinhole referencí (viditelný obrys registrovaného mračna z pozice kamery). Samotné
> `AZ_OFFSET_DEG = 0` a orientace koule zůstávají správně: `mapping/panos.py` `POTREE_M` je nově
> `[[1,0,0],[0,0,1],[0,-1,0]]`, odvozené pro Potreeho výchozí `texture.repeat.x = -1` (zrcadlená
> panoramata) — stránky si tenhle `repeat.x = -1` **ponechávají** (dřív ho `view.html` vracelo zpět,
> protože panoramata sama zrcadlená nebyla; teď jsou stejná jako svět, takže se srovnává správně).
> `?flip=1` zůstává jen pro A/B kontrolu. `AZ_OFFSET_DEG = 0` platí beze změny.

## 4. Výsledky segmentace a 3D projekce

Inference se neopakovala; doplnily se jen chybějící predikce (EoMT-L 2 snímky, ostatních 5 modelů 76 + 40
snímků, GPU). Vyhodnocení proti **novému pseudo-GT** (`segds_34bca9`, registrované mračno, korigované pózy):

| model | `05` (export GT, export snímky) all / nf_ok | **stejné 100 snímky, nové GT** (`bench/export_frames/`, n=99) | **nové 100 bench snímky** (`bench/`) |
|---|---|---|---|
| eomt_city | 0.358 / 0.330 | **0.326** / 0.269 | **0.332** / 0.302 |
| m2f_vistas | 0.336 / 0.318 | **0.315** / 0.286 | **0.327** / 0.310 |
| m2f_city | 0.305 / 0.285 | **0.284** / 0.262 | **0.299** / 0.283 |
| eomt_dinov3_ade | 0.302 / 0.279 | **0.282** / 0.242 | **0.298** / 0.285 |
| oneformer_city | 0.294 / 0.280 | **0.277** / 0.252 | **0.284** / 0.276 |
| segformer_b5 | 0.250 / 0.231 | **0.242** / 0.228 | **0.247** / 0.256 |

Pořadí modelů se nemění, EoMT-L (DINOv2, Cityscapes) zůstává nejlepší. Na stejných snímcích klesá mIoU_core
všem modelům o ~3 pb — stejný jev jako v `08 §7` (0.353→0.328 ve 3D): pseudo-GT se s registrací posouvá
(fence −12 % bodů), predikce zůstávají. `nf_ok` podmnožina (48 snímků na novém benchi) je menší, protože
near-field flag zůstává na 340/762. Podrobné tabulky `dataset/seg/bench/tables.md`, per-class
`per_class_iou.csv`, pásy `bands.csv`.

**3D projekce EoMT-L** (`out/seg_eomt_34bca9/`, `dataset/seg/project_eomt_city.json`): coverage **0.808**,
pixel acc 0.667, **mIoU_core 0.328**, ground-only 0.197 (e8f3e1: 0.807/0.667/0.328; export 0.799/0.683/0.352);
per-class IoU terrain 0.571, road 0.559, vegetation 0.497, building 0.319, sidewalk 0.193, fence 0.154.
Vizuální kontrola `report/bev_*.jpg`, `report/erp_f*.jpg` (BEV labels sedí na půdorys JVF; ERP round-trip
sedí na vozovku/zeď/plot fotky; červené „střechy" nad korunami = lidar skrz bezlisté stromy, `04`).

## 5. Konsolidovaný produkt a jeho validace

**Produkt** (`out/consolidated/`, Potree `TestOutput/output/consolidated/`): jedna LAZ sada 38 dlaždic v
**registrovaném** rámci — `rgb` = tw45 (kde `n_views==0` referenční TerraScan), `classification` = common15
(255 = neoznačeno, 6.4 % na dlaždici 021), extra dimenze `src_class, seg_conf, seg_n_views, cluster_id, obj_class,
hag, ref_r/g/b, dE00_med, n_views, col_conf`; druhá sada `objects/` s paletou clusterů v RGB (Potree neumí
náhodnou barvu skaláru); třetí sada `vendor/` s původním TerraScan RGB na týchž registrovaných bodech (Potree
neumí zobrazit tři skalární extra dimenze `ref_r/g/b` jako barvu; starý oktree `clusters/rgb` je v neregistrovaném
rámci). Oktree `consolidated/{cloud,objects,vendor}`, každý 584 809 840 bodů. Provenance VLR `geovap_map` nese `poses_hash`, `registration`, cesty vstupů.

**Kontroly bez prohlížeče** (`check_products.py` → `validation/checks.json`, 157/157 ok): per dlaždice
`cluster_id` bitově shodné s `clusters/src/objects_t*.laz`, sjednocení id = 12 276 = `global_labels.npy`;
histogram `classification` = histogram `seg_eomt_34bca9/labels/NNN.npy`; medián `dE00_med` = medián tw45
histogramu (tolerance 1 bin 0.05); oba oktree 584 809 840 bodů; `verify(xyz_mode="registered")` 38/38.

**Vizuální kontroly** (`validation/screenshots/<sada>/contact_f*.png`, 12 náhodných clean snímků × 4 yaw):
fotka (krytí 1), mračno (krytí 0, černé pozadí), blend 0.5. Na registrovaném mračnu s korigovanými pózami sedí
hřebeny střech, zdi, okraje vozovky i koruny stromů (např. snímek 700). Módy stránky ověřeny headless
záběry: RGB (tw45), segmentace (`classes.json`), objekty (paleta), třída objektu (0–3), ΔE00 (0–20), RGB (original, TerraScan).

**Sphere check** (`sphere_check.py`, 48 záběrů na sadu):

| sada | pózy | az | NCC vs. kamerový model | NCC půlotáčka | verdikt |
|---|---|---|---|---|---|
| `cloud/panos` | corrected + registrace | 0 | **1.000** | 0.750 | primární — OK |
| `panos_exp0` | export | 0 | **1.000** | 0.732 | OK |
| `panos_corr180` | corrected | 180 | 0.745 | 0.892 | kontrola — chová se jako otočená (OK) |
| `panos_exp180` | export | 180 | 0.743 | 0.894 | kontrola — chová se jako otočená (OK) |

Hranová metrika (`edge_metric.csv`, silueta vs. hrany fotky) na této venkovské scéně (živé ploty, pole)
varianty **nerozliší** (medián 27–32 px pro všechny) — slouží jen ke kontaktním archům, verdikt dává sphere_check
a pro pózy `pose_report` + ΔE obarvení (§2).

**Pasti při ladění stránky** (Potree 1.8): globální `THREE` neexistuje (barvy klonovat z `Potree.Gradients`);
skalární atributy se barví `w = (v + offset)·scale` se `scale = |initialRange|/|range|` — funguje jen pro
`|initialRange| = 1`, proto `scalarRange()` předává kompenzovaný rozsah (`obj_class` 0–3 by se jinak celý
vykreslil červeně, `dE00_med` jednou barvou); `clusters/index.html` měl tytéž dvě chyby (nikdy neověřen v
prohlížeči) a je opraven.

## 6. Kde co leží

- driver a markery: `Geovap_cache/out/pipeline/` (`pipeline.log`, `<stage>.json`, `comparison.md`, `baseline/` = snapshot `dataset/` před během, `backup/`)
- pózy: `out/poses/poses_corrected.{csv,json}` (hash `34bca9ff23`), `out/pass_reg/pass_transforms.json`, `out/poses/report_final.{md,json}`
- produkty: `frames/34bca9/`, `out/tw45/{tiles,stats,report.md}`, `out/dataset_34bca9/`, `segds_34bca9/`, `out/seg_eomt_34bca9/`, `out/consolidated/{tiles,objects,summary.json,validation/}`
- Potree: `/mnt/Geovap_cache/TestOutput/output/consolidated/{cloud,objects,panos_*,index.html}` → `http://localhost:8080/pointclouds/consolidated/index.html`; staré `clusters/`, `eomt_city_seg/` zůstávají pro porovnání
- verzované tabulky: `dataset/` (promotováno z `dataset_34bca9`), `dataset/seg/{bench/,bench/baseline_frames/,project_eomt_city.json}`

## 7. Oprava kamerového modelu (2026-09-16)

Všechna čísla v tomto dokumentu (§0–6) a ve všech dokumentech `01`–`08` naměřená před 2026-09-16 byla
spočítaná se **starým, zrcadleným kamerovým modelem**. Tato sekce popisuje chybu, důkazy, opravu a
plán přeměření.

### 7.1 Kořenová příčina

Sdílený kamerový model `mapping/geometry.py::cam_to_pano`/`pano_rays` (převzatý z pilotního
`experiments/common/camera.py`) mapoval azimut na sloupec panoramatu jako
`u = (az mod 360)/360·W` — šev vpředu, sloupce proti směru hodinových ručiček. Skutečná panoramata
mají šev **vzadu** a sloupce **po směru hodinových ručiček**: `u_true = ((180 − az) mod 360)/360·W`,
tedy `u_true = W/2 − u_old (mod W)` — zrcadlení kolem příčné osy (prohodí přední/zadní, zachová
levou/pravou). Opraveno v `geometry.py`, `camera.py` a `experiments/common/reproject.py`;
`mapping/panos.py` přepočítáno na `POTREE_M = [[1,0,0],[0,0,1],[0,-1,0]]` pro Potreeho výchozí
`texture.repeat.x = -1` (správně — stránky si ho ponechávají, `?flip=1` je jen pro A/B), `AZ_OFFSET_DEG
= 0` beze změny. Nový export přesně reprodukuje uživatelem potvrzenou sadu koulí (starý model, az 180,
repeat −1) — `tests/test_panos.py::test_new_export_reproduces_user_confirmed_panos_corr180`, odchylka
0. Znaménko yaw v `align.py`: pod novou konvencí `yaw_offset = shift·360/W` je správně
(`tests/test_align_yaw_sign.py`); pod starou konvencí měl tentýž řádek opačné znaménko, tj. stage-1 NCC
návrhy yaw byly znaménkově přehozené (stage-2 barevné hledání i tak zkoušelo 0 a dolaďovalo — degradace
byla měkká, ne katastrofická).

### 7.2 Důkazy (nezávislé na modelu)

- **Ohnisko rozšíření (FOE)** pohybu mezi snímky leží na `u/W ≈ 0,41–0,60` na rovných úsecích (snímky
  100, 250, 498, 591, 900), zatímco starý model kladl směr jízdy na šev (`u/W ≈ 0/1`); nový model dává
  0,500.
- **Pinhole render** registrovaného mračna ze středu kamery odpovídá pohledu z Potree (maska NCC
  0,83–0,94 proti 0,50–0,73 zrcadleně) a odpovídá fotce jen pod novou konvencí. Nápis „VMX-2HA" je
  čitelný → fotka není zrcadlená.
- **Vendor barvy TerraScan**: CIE76 7–17 (nový model) vs. 10–22 (starý model) na 7 snímcích.
- **Pilotní hrubá síla** (`02 §2.2`) testovala offsety švu 0/90/180/270 a zrcadlení `−az` **odděleně**,
  nikdy kombinaci „zrcadlo + 180°"; kotva na dlaždici 037 je navíc na tuhle záměnu necitlivá (symetrie
  uličního kaňonu).

### 7.3 Co se změnilo

| soubor | změna |
|---|---|
| `mapping/geometry.py` | opravený `cam_to_pano`/`pano_rays` (nová konvence švu/směru) |
| `mapping/camera.py` | totéž (sdílený model) |
| `experiments/common/reproject.py` | totéž pro pilotní kód |
| `mapping/panos.py` | `POTREE_M = [[1,0,0],[0,0,1],[0,-1,0]]`, `AZ_OFFSET_DEG = 0` |
| `pointcloud-tools/validate/sphere_check.py` | přidána model-nezávislá pinhole reference (siluety registrovaného mračna), práh NCC ≥ 0,95 **a** pinhole silueta lepší než každá kontrolní sada; `--no-pinhole` pro vynechání |
| `pointcloud-tools/validate/foe_check.py` (nové) | ohnisko rozšíření z fotky vs. sloupec směru jízdy modelu; průchod při mediánu \|Δu/W\| ≤ 0,1 |
| `tests/test_panos.py::test_new_export_reproduces_user_confirmed_panos_corr180` | reprodukce potvrzené sady koulí, odchylka 0 |
| `tests/test_align_yaw_sign.py` | znaménko `yaw_offset` pod novou konvencí |

### 7.4 Čísla fáze 2 (exportní pózy, před přeběhem)

30 snímků rozprostřených po trajektorii: `Aligner.colour_de` medián ΔE 8,694 (starý) → 4,928 (nový),
nový lepší na 28/30. Kotva dlaždice 037 (pilotní recept): medián CIE76 6,076 (starý = historický
anchor) → 5,082 (nový), n = 43 058. `sphere_check` na snímku 498 (uživatelem potvrzená sada): NCC
1,000 (všechny 4 yawy) s novým offline renderem; stará primární sada 0,58–0,61 (její alternativa „o
půl otáčky" vyhrává 0,87–0,94) — signál toho, že starý test byl kruhový; pinhole silueta medián
14–39 px proti 37–40 px (strop) u starého modelu.

### 7.5 Archivovaná neplatná data

Přesunuto (`mv`) do `/mnt/Geovap_cache/_invalid_2026-09-16/`:
`out/{poses,pass_reg,calib,t037,diag,final,tw45,identity,dataset*,seg_eomt*,consolidated}`,
`frames/{34bca9,loose_export}`, `segds*` (kromě `segds/bench`), markery a logy pipeline.

A do `/mnt/Geovap_cache/TestOutput/output/_invalid_2026-09-16/`:
`consolidated`, `eomt_city_seg`, `seg_eomt`, `seg_eomt_34bca9`, `test_drazkov`,
`clusters/{colored,index.html,objects,rgb}`.

Ponecháno beze změny: `store`, `gray`, `vehicle_mask.npz`, `segds/bench`, `clusters/src`.

Archiv se smaže až po ověření nového běhu uživatelem.

### 7.6 Regenerace

```
uv run python -m mapping.cli.pipeline run --from align --force --detach
```

Spuštěno 2026-09-16 11:30 UTC, očekávaná doba ~5–6 h.

### 7.7 Výsledky nového běhu (2026-09-16, 11:30–21:04 UTC)

Běh `pipeline run --from align --force` doběhl celý (align → validate, všechny stage rc=0; stage `validate`
musela být 4× opakována kvůli chybám nástrojů, ne dat – viz níže). Nový hash korigovaných póz **c914b90bce**
(starý 34bca9ff23). Plná tabulka: `out/pipeline/comparison.md`.

| metrika | před opravou (34bca9) | po opravě (c914b9) |
|---|---|---|
| align: podezřelé snímky / rozsah yaw offsetu / časové posuny | 217 / ±180° / až ±8 s | 21 / ≤ 0,7° / žádné |
| refine: snímků s přijatou korekcí (S4) / max. dyaw | 402 / 6,6° | 715 / 1,8° |
| registrace pasů (cloud-only) | 66/72, 0,102 → 0,035 m | identická |
| interpolace yaw v zatáčkách, medián (export → korig.) | 5,64° → 0,20° | 5,64° → 0,20° |
| dlaždice 037 CIE76: export / korig.+reg. | 6,076 / 6,097 | **5,082 / 5,460** |
| ΔE zatáčkové snímky (210), medián export / korig. | 10,87 / 10,86 | 5,62 / 6,14 |
| tw45 ΔE00 medián / podíl > 20 / podíl < 5 | 5,03 / 6,6 % / 49,7 % | **3,28 / 2,1 % / 71,4 %** (cíl M1 splněn) |
| tw45 15–25 m od kamery: medián / > 20 | 6,73 / 14,4 % | 3,93 / 3,3 % |
| kvalita: clean / usable / unverified / reject | 830 / 302 / 163 / 208 | **924 / 333 / 124 / 122** |
| near-field JVF: n / medián px / flagged | 762 / 17,9 / 340 | 835 / 16,1 / 343 (beze změny – datum JVF) |
| bench mIoU_core eomt_city, stejných 100 snímků (n=96) | 0,358 | **0,495** (pixel acc 0,60 → 0,76) |
| bench mIoU_core ostatní modely, stejné snímky | 0,25–0,34 | 0,38–0,50 (+0,13 až +0,16 všechny) |
| eomt_city IoU building / vegetation (2D bench) | 0,23 / 0,44 | 0,70 / 0,70 |
| 3D projekce EoMT: pokrytí / pixel acc / mIoU_core | 0,808 / 0,667 / 0,328 | **0,862 / 0,811 / 0,498** |
| 3D projekce IoU building / vegetation | 0,32 / 0,50 | 0,78 / 0,75 |
| konsolidované mračno | 584 809 840 bodů | 584 809 840 bodů |
| sphere_check NCC: primární (korig., az 0) / corr180 / exp0 / exp180 | – (cirkulární) | **1,000 / 0,653 / 1,000 / 0,653**; půlotočka u 180-sad 0,87 |
| sphere_check pinhole silueta (px, cap 40): primární / exp0 / 180-sady | – | 13,3 / 12,1 / 40 |
| screenshots skyline (px): primární / exp0 / exp180 / corr180 | – | 18,0 / 18,1 / 28,6 / 30,7 |
| foe_check: medián \|Δu/W\| (6 použitelných rovných snímků) | – | 0,055 (model 0,500, foto 0,44–0,59; 1 outlier 0,84) |
| check_products | – | 157 ok / 0 failed / 1 skipped |

**Interpretace.** Všechny fotometrické i segmentační metriky se zlepšily řádově tak, jak předpověděla fáze 2,
a nejvíc tam, kde symetrie ulice odraz nemohla maskovat (fasády, vegetace, velké vzdálenosti). Predikce
modelů se nezměnily – zlepšení benchmarku je čistě posun pseudo-GT na správné pixely.

**Otevřený bod – korigované pózy vs. export.** Po opravě modelu jsou exportní pózy v okamžiku snímku
barevně *lepší* než korigované (dlaždice 037 5,08 vs. 5,46; zatáčkové snímky 5,62 vs. 6,14; 0 z 210
zatáčkových snímků korekce zlepšila). Výhoda korigované trajektorie zůstává jen v interpolaci mezi snímky
(0,20° vs. 5,64°). Podezřelý je S4 overlay (`refine`), který byl laděn proti odraženému modelu a nyní
přijímá 715 snímků. Doporučení: změřit `colour_de` s pózami `poses_traj_rot` (bez S4) a podle výsledku S4
vypnout nebo přeladit; pipeline test „korigované ne horší než export ±0,1" zatím nepasuje.

**Opravy nástrojů během validace** (žádná se netýká dat): `driver.js` – jednorázový reload stránky, když
Potree pod swiftshaderem nikdy nedosáhne `numNodesLoading == 0`; `sphere_check.py` – `panos_exp0` není
kontrolní sada konvence (stejná konvence, jiné pózy) a pinhole verdikt se rozhoduje mediánem + win-rate
≥ 2/3 místo „každý snímek/yaw"; `foe_check.py` – jen rovné úseky (|dyaw| ≤ 2°), minimální skóre
divergence 0,3, 16 kandidátních snímků.

**Přesun Potree dat na rychlý disk (2026-09-17).** `/mnt` je pomalý; data pro vizualizaci (`consolidated/{cloud,objects,vendor,panos*}`,
78 GB, a vstup merge `clusters/src`, 12 GB) byla zkopírována do `/home/jatuma/repos/Geovap/potree_output/` a viewer
i pipeline (`config.POTREE_OUTPUT_DIR`, `pointcloud-tools/.env POINTCLOUD_OUTPUT`) odtud čtou/zapisují. Na `/mnt`
zůstává cache (`store`, `frames`, `out`, `segds`) a archiv. Původní kopie `TestOutput/output/consolidated` na `/mnt`
je nyní redundantní.

**Archiv.** `_invalid_2026-09-16/` (135 GB + 190 GB) zůstává do vizuálního potvrzení uživatelem
(`consolidated/index.html?mode=rgb` bez parametru `panos=`), poté smazat.
