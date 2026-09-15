# `dataset/` — verzované tabulky a metadata

Těžká data (panoramata, mračno, produkty) žijí v `Geovap_cache/`; tady jsou jen malé, git-sledovatelné
tabulky a obrázky odvozené z nich, aby šly verzovat a číst bez přístupu k cache.

## Čistý dataset (`04_cisty_dataset.md`)

`clean_frames.json` — třídy `clean`/`unverified`/`usable`/`reject` pro 1503 snímků. **Od této
přestavby na korigovaných pózách** (`GEOVAP_POSES=corrected`, hash `e8f3e1f2b3`/`e8f3e1`,
registrovaná dráha — `08_korekce_poz_panoramat.md §2.4`), promotováno z
`Geovap_cache/out/dataset_e8f3e1/`: **830 clean / 163 unverified / 302 usable / 208 reject**.
`frame_quality.csv` nese všechny veličiny (kritéria `mapping/quality.py`), `frame_quality_map.png` a
`frame_quality_stats.png` mapu a rozložení, `tile_summary.json` bilanci po dlaždici LAZ (přepočteno
pro korigovaný běh — viz níže). `export_baseline/` je kopie všech čtyř souborů z **exportních póz**
(825/163/295/220), pořízená před touto přestavbou — regresní referenční bod, ne živá data.

### Přestavba na korigované pózy — souhrn běhu

`mapping.quality.run`/`reclassify` běží nad `poses_source` (`08 §2.4`) a píší do
`Geovap_cache/out/dataset_e8f3e1/` (`mapping.config.source_dir`, export beze změny —
`Geovap_cache/out/dataset/`); registraci mračna berou automaticky (`open_store(poses)`). Celý běh
1 503/1 503 snímků proti `dataset/export_baseline/frame_quality.csv`:

| třída | export | corrected (= dnešní `dataset/`) | Δ |
|---|---:|---:|---:|
| clean | 825 | **830** | +5 |
| unverified | 163 | 163 | 0 |
| usable | 295 | **302** | +7 |
| reject | 220 | **208** | −12 |

Geometrická reziduua se zlepšila napříč **všemi** čtyřmi třídami (medián \|du\|/\|dv\| clean
0,38/0,78→0,31/0,74 px, reject 0,63/6,57→0,62/6,29 px atd.) — konzistentní s hlavním přínosem
korekce (mezisnímková interpolace v zatáčkách, `08 §1`), ne jen s posunem hranice `clean`. 99 snímků
(6,6 %) změnilo třídu oběma směry (`reject→clean` 31 vs. `clean→reject` 27, ...) — jde o skutečné
překlopení blízko prahu, ne jednosměrné zlepšení. Konflikty průjezdů 26→23 snímků, částečně jiná
množina průjezdů (17, 20 nově, zbytek stejný jako export). Nejhorší průjezdy (12, 13, 14) zůstávají
nejhoršími i po korekci.

`tile_summary.json` má teď generátor (`mapping.quality.write_tile_summary`/`plot_quality`, CLI
`uv run python -m mapping.quality tile-summary <csv> <out_dir> --poses {export,corrected}`),
zrekonstruovaný z ad hoc skriptu a validovaný proti `dataset/export_baseline/tile_summary.json` na
téže (export) tabulce: **17 z 38 dlaždic přesná shoda**, zbytek se liší o pár snímků (bbox-distance
zaokrouhlení na hranicích dlaždic) — nejde o chybu generátoru, jde o hranici stejná jako u
původního ad hoc skriptu. `dataset/tile_summary.json` teď nese korigovanou bilanci (830 clean).

`dataset/clean_frames.json` (produkční množina, čtená i `mapping/seg/render_labels.py` a odvozeninami
níže) **je od této přestavby korigovaná** — už ne export. Kdo potřebuje export bilanci pro srovnání,
čte `dataset/export_baseline/`.

## Segmentační dataset (`seg/`, `03_semanticka_segmentace.md`, `05_benchmark_segmentace.md`)

Metadata k `Geovap_cache/segds*/` (JVF pseudo-GT ERP značky + gnómonické výseče): `classes.json`
(taxonomie a JVF→třída mapování), `splits.json` (prostorové train/val/test bloky), `stats.json`
(souhrn pixelů/pokrytí po snímku), `bench_frames.json` (benchmark podmnožina). `export_baseline/`
je kopie těchto čtyř souborů z běhu na **export** pózách (`Geovap_cache/segds/`), pořízená před
přestavbou na korigované pózy níže — regresní referenční bod pro porovnání.

### Přestavba na korigované pózy (`GEOVAP_POSES=corrected`, hash `e8f3e1f2b3` / `e8f3e1`)

Rebuild `mapping/seg/{areas,rasters,point_labels,render_labels,views,dataset}` na `load_poses("corrected")`
(S0–S5b, `08_korekce_poz_panoramat.md`) — registrovaná dráha + `CloudStore(registration=pass_transforms.json)`
místo prostého `CloudStore()`. Výstup v `Geovap_cache/segds_e8f3e1/` (`areas`, `rasters`, `point_labels`,
`labels_erp`, `bands_erp`, `qa`, `views`, `nearfield.json`), metadata zkopírovaná sem. **Snímková
množina beze změny v této první fázi**: `render_labels.CLEAN_JSON` je natvrdo `dataset/clean_frames.json`
(nezávisí na `--poses`), a v době tohoto běhu byl tento soubor ještě export-based, takže srovnání níže
(labels statistiky, near-field, splity) je nad **stejnými 825 `clean` snímky** — jen s korigovanou
geometrií a registrovaným mračnem. Samostatný běh, který *přehodnocuje* clean/reject přímo na
korigovaných pózách (`mapping.quality.run(poses_source="corrected")` →
`Geovap_cache/out/dataset_e8f3e1/clean_frames.json`, 830/163/302/208), byl od té doby dokončen a
**promotován do `dataset/clean_frames.json`** (viz sekce výše) — druhá fáze rebuildu níže (## Promoce
na 830 `clean`) staví `labels_erp`/`views`/`dataset`/`nearfield` znovu nad touto novou, o 5 snímků
větší množinou.

**Past se to nepovedlo napoprvé — poučení.** `seg.areas` je JVF-only (nezávisí na póze), takže návod
dovoloval symlinkovat `Geovap_cache/segds/areas` místo přestavby. Mezitím ale někdo (jiný, nesouvisející
běh v tomto stroji) přepsal `segds/areas/faces.geojson` in-place s `--hard-only` (549 ploch,
`labelled_frac` 0,75 místo 549→1009 ploch, 0,95 při `--all-lines`, výchozí) — beze změny mtime adresáře,
takže to nebylo vidět na první pohled. Symlink tak potichu natáhl degradovanou plochu do korigovaného
běhu a první průchod `rasters→points→labels` ukázal kolaps pokrytí (ERP `ignore` medián 85 %→93 %,
`terrain`/`vegetation`/`road` −75 až −80 % absolutních pixelů) — vypadalo to jako důsledek registrace,
ale bylo to jen jiné (menší) JVF pokrytí pod symlinkem. Oprava: `areas` se pro `segds_e8f3e1` přestavěl
přímo (výchozí `--all-lines`, 1009 ploch, `labelled_frac` 0,950 — shoda s exportním `03_semanticka_...`
záznamem), `rasters→points→labels→views` se přestavěly znovu nad správnou plochou. Čísla níže jsou z
**opraveného** běhu; sdílený `segds/areas/` (export) zůstal netknutý (mimo rozsah tohoto úkolu) — kdokoliv
z něj bude číst `n_faces`, ověřte si `all_lines` v `report.json`, ne jen `rasters/meta.json` (ten drží
starý `n_faces: 1009` z původní stavby, i když `areas/faces.geojson` teď nese jiná data).

**Pořadí a časy** (10 workerů kde ne jinak, stroj pod souběžnou zátěží — jiný běh `mapping.quality` na 12
workerech + tento shell): `areas` < 5 s, `rasters` 160 s, `points --workers 8` 46 s, `labels --frames clean
--workers 10` 199 s, `views --workers 10` 379 s (druhý pokus; první byl zabit po ~4 min falešného podezření
na zaseknutí — 10 workerů dlouho nepřidávalo nové soubory, ukázalo se to jako jeden pomalý úsek snímků
[1173–1192] z průjezdu 20, ~5 s/snímek i izolovaně, ne hang — druhý pokus stejným místem prošel sám),
`nearfield.run` 79 s, `dataset` 55 s. Celkem `Geovap_cache/segds_e8f3e1/` 4,8 GB (`areas` symlink na
export by ušetřil ~0 — nakonec se nepoužil, viz výše).

**Label statistiky** (`labels_erp_stats.json`, pásmo φ∈[−55°,+45°] = oficiální metrika `dataset.py`,
825 snímků oběma směry):

| třída | export % | corrected % | Δ pp |
|---|---:|---:|---:|
| road | 11,864 | 12,254 | +0,390 |
| sidewalk | 0,423 | 0,375 | −0,048 |
| building | 5,181 | 5,461 | +0,281 |
| wall | 0,556 | 0,530 | −0,026 |
| **fence** | **13,427** | **11,126** | **−2,301** |
| vegetation | 24,910 | 25,795 | +0,886 |
| terrain | 26,997 | 27,804 | +0,808 |
| water | 0,089 | 0,084 | −0,005 |
| guard_rail | 0,057 | 0,043 | −0,014 |
| stairs | 0,006 | 0,007 | +0,001 |
| pole | 0,001 | 0,001 | +0,000 |
| verge | 0,644 | 0,570 | −0,074 |
| paved_other | 1,117 | 1,143 | +0,026 |
| culvert_head | 0,041 | 0,047 | +0,006 |
| structure_other | 0,143 | 0,147 | +0,004 |
| road_or_verge | 14,547 | 14,614 | +0,067 |

Pokrytí (`labelled_frac_band_median`) beze změny: 0,2685 → 0,2655 (−0,30 pp). `frames_with_no_labels`
beze změny (42). Stejný vzorec (jediná systematická ztráta u `fence`, zbytek v šumu) se potvrzuje i na
celém mračnu bodů (`point_labels/hist.json`, 584 M bodů): fence 4,304 %→3,799 % (−0,505 pp, −12 %
relativně), `road_or_verge` −0,143 pp, vše ostatní |Δ| < 0,25 pp. Vysvětlení: `fence` je jediné pravidlo
`point_labels.py` čistě vzdálenostní k tenké linii (`fence_dist=0,35 m`, `classes.RULES`) — nejcitlivější
na posun z registrace (S5b medián posunu pasáží 0,168–0,172 m, max 1,38–1,51 m u 12/30 průjezdů, `08 §3.6`);
plošná pravidla (terrain/vegetation/road, "uvnitř plochy") jsou vůči stejnému posunu robustnější (menší
podíl okrajových pixelů). Beze změny nezůstala **jen díky opravě `areas` výše** — s degradovanou plochou
byl pokles plošný a ~4× větší u všech tříd, ne jen u fence (viz poučení výše).

**Near-field** (`mapping/seg/nearfield.py run`, JVF hranice vozovky vs. Cannyho hrana fotky,
[03_semanticka_segmentace.md](../03_semanticka_segmentace.md), 08 §3.6 motivace korekce):

| | export | corrected | Δ |
|---|---:|---:|---:|
| n_measured | 755 | 753 | −2 |
| medián mediánů [px] | 15,8 | 17,7 | +1,9 |
| p90 [px] | 43,7 | 43,4 | −0,3 |
| vlajkovaných (>20 px) | 328 | 333 | +5 |

**Nezlepšilo se — poctivě.** Toto je přesně metrika, která korekci motivovala (03: 328/755), a po
korekci zůstává **prakticky neutrální, mírně horší** medián. Souhlasí to se samostatným zjištěním S7
(`08 §3.6`, `nearfield_reg.json`, jiná metoda/vzorek 200 snímků): medián 13,3→15,3 px, vlajky 62→60 —
stejný směr (medián mírně horší, počet vlajek prakticky beze změny/mírně horší), protože S5b registrace
je jen párová mezi průjezdy (`--datum none`), **není ukotvená proti JVF** (pokus o JVF datum byl vyzkoušen
a zavržen, `08 §3.6`: medián 13,3→15,6 px, jen 1/30 průjezdů splnil kritérium kotvy). Registrace tedy
zlepšuje shodu MEZI průjezdy, ne shodu s (nehybnou) JVF vrstvou, na které stojí pseudo-GT.

Podle průjezdu (`n` = měřitelných snímků): největší zhoršení průjezd 1 (3→20 vlajek z 140), mírné
zlepšení průjezdy 0 (11→8/132), 3 (30→22/33), 9 (37→33/44); většina beze změny. 34 snímků nově
vlajkovaných, 29 nově v pořádku — čistý posun +5, ne systematický jedním směrem.

**Rozdělení splitů** (k-means(10) na kamerových pozicích, `dataset.py:make_splits`) se s korigovanými
pozicemi mírně přerozdělilo (pozice kamer se posunuly o cm–m): train 458→477, val 139→133, test
161→159, buffer 67→56. `frames_with_no_labels` beze změny (42 snímků).

### Promoce na 830 `clean` (`dataset/clean_frames.json` po přestavbě výše) — druhá fáze rebuildu

Po promoci `dataset/clean_frames.json` na korigovanou klasifikaci (830/163/302/208, viz sekce výše)
se `labels --frames clean --workers 10`, `views --workers 10`, `dataset` a `nearfield.run` v
`segds_e8f3e1` přestavěly znovu — `render_labels.CLEAN_JSON` teď čte 830 snímků místo 825 (`points`/
`areas`/`rasters` beze změny, na `clean_frames` nezávisí). Časy (10 workerů, stroj bez souběžné
zátěže): `labels` 171 s, `views` 376 s, `dataset` (1. běh) 32 s, `nearfield.run` 39 s, `dataset`
(2. běh, < 5 s) — `dataset` se spustil dvakrát, podruhé po `nearfield`, aby `stats.json`/
`bench_frames.json` nesly čerstvé `nearfield.json` (`nearfield.run` píše po prvním `dataset` kroku,
ne před ním), ne to z ještě starého souboru. `Geovap_cache/segds_e8f3e1/` 4,9 GB (+0,1 GB);
`labels_erp`/`views` obsahují navíc
32 zastaralých souborů pro snímky, které byly `clean` v 825-běhu ale ne v 830-běhu (přepisují se,
nemažou — neuklizeno, ale `dataset.build()` je do `stats.json` nezahrnuje, filtruje podle aktuálního
`clean_frames()`).

Label statistiky (pásmo φ∈[−55°,+45°], `stats.json.pixels_band_by_class`), export vs. tento
830-snímkový korigovaný běh vs. mezikrok výše (825 korigovaných, pro odlišení vlivu 5 nových snímků
od vlivu korekce samotné):

| třída | export % (825) | corrected % (825, mezikrok) | corrected % (830, finální) | Δ (exp→830) | Δ (825→830 korig.) |
|---|---:|---:|---:|---:|---:|
| road | 11,864 | 12,254 | 12,450 | +0,586 | +0,195 |
| sidewalk | 0,423 | 0,375 | 0,452 | +0,029 | +0,077 |
| building | 5,181 | 5,461 | 5,691 | +0,510 | +0,230 |
| wall | 0,556 | 0,530 | 0,526 | −0,030 | −0,004 |
| **fence** | **13,427** | **11,126** | **11,184** | **−2,243** | +0,057 |
| vegetation | 24,910 | 25,795 | 25,753 | +0,843 | −0,043 |
| terrain | 26,997 | 27,804 | 27,479 | +0,483 | −0,325 |
| water | 0,089 | 0,084 | 0,079 | −0,010 | −0,005 |
| guard_rail | 0,057 | 0,043 | 0,042 | −0,014 | −0,000 |
| stairs | 0,006 | 0,007 | 0,008 | +0,001 | +0,001 |
| pole | 0,001 | 0,001 | 0,001 | +0,000 | −0,000 |
| verge | 0,644 | 0,570 | 0,580 | −0,064 | +0,010 |
| paved_other | 1,117 | 1,143 | 1,176 | +0,060 | +0,033 |
| culvert_head | 0,041 | 0,047 | 0,047 | +0,005 | −0,000 |
| structure_other | 0,143 | 0,147 | 0,146 | +0,003 | −0,001 |
| road_or_verge | 14,547 | 14,614 | 14,388 | −0,159 | −0,226 |

**Ne jen +5 přidaných** — 825-mezikrok (export `clean` seznam) a 830-finální (korigovaný `clean`
seznam) mají jen **793 společných snímků**: 32 z původních 825 vypadlo (`reject`/`usable` po korekci),
37 nových přibylo, čistý rozdíl +5. Turnover je rozprostřený přes průjezdy (vypadlé nejvíc z 1: 7×,
přibylé nejvíc z 14: 6× a 12: 4×, ale obě skupiny zasahují většinu z 30 průjezdů — žádná jednoduchá
"jen problémové průjezdy" charakteristika). Přesto rozdíl v rozložení tříd zůstává v mezích šumu
(\|Δ\| ≤ 0,33 pp, `terrain` nejvíc) — `fence` pokles je **výhradně efektem registrace/opravy `areas`**
z první fáze (viz poučení výše), ne téhle výměny snímků. `labelled_frac_band_median`: export 0,2685 →
830-korigovaný 0,2657 (−0,28 pp, prakticky stejné jako 825-mezikrok 0,2655). `frames_with_no_labels`:
export 42, 825-mezikrok 42, 830-finální **41**.

**Near-field** (`mapping/seg/nearfield.py run` nad 830 snímky):

| | export (825) | corrected (825, mezikrok) | corrected (830, finální) |
|---|---:|---:|---:|
| n_measured | 755 | 753 | 757 |
| medián mediánů [px] | 15,8 | 17,7 | 17,9 |
| p90 [px] | 43,7 | 43,4 | 43,3 |
| vlajkovaných (>20 px) | 328 | 333 | **336** |

Stejný závěr jako v mezikroku, teď na plné 830-snímkové množině: near-field zarovnání (hlavní motivace
korekce, `08 §3.6`) se nezlepšilo, počet vlajek dál mírně roste (328→336, +2,4 %). Konzistentní s
turnoverem výše (37 nových `clean` snímků různě po průjezdech) — silueto-kritérium `quality.py` (10–40 m,
necitlivé na chyby v blízkém poli) propouští snímky, které near-field metrika (3–10 m) dál vidí jako
posunuté; obě metriky měří jinou vzdálenost od kamery, proto se neshodnou 1:1.

**Splity a benchmark**: `split_counts` train 490 / val 129 / test 153 / buffer 58 (825-mezikrok:
477/133/159/56; export: 458/139/161/67) — k-means na 830 kamerových pozicích dá jiné shluky než na
825, posun je řádu jednotek procent, ne strukturální. `bench_frames.json` (100 snímků,
`dataset.py:select_bench_frames`, seed 0) **není fixní podmnožina** — je deterministicky odvozená ze
`clean_frames()` + `stats`/`tiles` toho běhu, takže se s každou změnou vstupní množiny přepočítá:
830-finální vs. 825-mezikrok sdílí jen 27/100 snímků, 830-finální vs. export-baseline 30/100. Kdo
potřebuje stabilní benchmark napříč běhy, musí si množinu snímků zafixovat sám (`bench_frames.json`
dnešního běhu jako whitelist) — dosavadní chování ji přepočítává znovu při každém `dataset` kroku.

**Shrnutí**: přestavba na korigované pózy + registrované mračno mění pseudo-GT jen mírně a v jednom
konkrétním, vysvětlitelném místě (fence, −2,2 pp v pásmu, efekt opravy `areas`/registrace, ne výběru
snímků); near-field zarovnání (hlavní motivace korekce) se nezlepšilo ani po promoci na plnou
830-snímkovou `clean` množinu — vlajek mírně přibylo (328→336). Hlavní přínos `poses_corrected`
zůstává, jak shrnuje `08 §1`, oprava mezisnímkové interpolace v zatáčkách — ne zlepšení segmentačního
pseudo-GT jako takového. `dataset/clean_frames.json` a `Geovap_cache/segds_e8f3e1/` jsou od této
přestavby produkční (korigovaná) množina; `dataset/export_baseline/` a `dataset/seg/export_baseline/`
zůstávají jako regresní kotva na starých pózách.
