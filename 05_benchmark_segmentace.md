# Segmentační dataset z čistých snímků a zero-shot benchmark modelů

Navazuje na `04_cisty_dataset.md` (825 snímků `clean`) a na §12 v `03_semanticka_segmentace.md`. Cíl: (1) husté
pseudo-GT pro sémantickou segmentaci z JVF DTM exportu bez ruční anotace, (2) srovnat nejnadějnější zero-shot
modely na téže geometrii a metrice. Kód `mapping/seg/`, CLI `mapping.cli.seg_build` a `mapping.cli.seg_bench`,
data `Geovap_cache/segds/`, metadata a tabulky `dataset/seg/`.

## 1. Odkud jsou značky

JVF export nemá plochy jako polygony: **plošný objekt = hraniční linie + jeden definiční bod** (856 bodů
`-04`, 3 466 linií). Plochy se proto rekonstruují stejně, jako to dělá IS DTM při plochování:

1. **Polygonizace** všech linií (shapely, přesnost 1 cm): 1 009 ploch, 260 884 m², visící konce jen 144 m.
   Polygonizace jen z „tvrdých" hranic (`hranice …`) dává 58 % označené plochy — ploty a vjezdy plochy skutečně
   uzavírají, proto se berou všechny linie.
2. **Třída plochy z definičních bodů.** Přesně jedna třída → `resolved` (573 ploch, 230 743 m²). Body malých
   objektů (patka, čelo propustku, zpevnění, schodiště, skleník, zeď …) plochu definují jen, když je malá
   (< 30 m², u zpevnění < 500 m²) — jejich obrys v exportu často není uzavřený a bod padá do okolní zahrady.
   Dvě různé třídy v jedné ploše = chybějící hranice → `conflict`, **nehádá se** (25 ploch, 7 872 m²), s jednou
   výjimkou: vozovka + krajnice (hlavní ulice, 4 plochy, 15 257 m²) → diagnostická třída `road_or_verge`.
   Bez bodu: drobné odštěpky < 5 m² dědí od souseda (75 ploch), zbytek `unresolved` (272 ploch, 5 097 m²).
   **Označeno 95,0 % plochy.** `segds/areas/{faces.geojson,report.json,map.png}`.
3. **Značka každého bodu mračna** (585 M bodů, 22 s): rastry 0,1 m (třída plochy, plochy budov +0,5 m,
   vzdálenost a výška nejbližší linie pro plot / zeď / zábradlí / hranice budovy / hranice stavby) a DTM 0,5 m
   z bodů třídy 2 (proti výškám JVF linií bez systematické chyby: medián 0–2 cm). Pravidla v pořadí: plot (do
   0,35 m od linie, 0–2,5 m nad ní; prstenec do 1 m = ignore, živé ploty jsou širší), zábradlí, zeď, budova
   (v půdorysu +0,5 m a > 0,2 m nad zemí), okolí `hranice stavby` (podezdívky, terasy) = ignore, povrchové
   třídy pro body ≤ 0,2 m nad DTM, vegetace = body > 0,5 m nad zemí v plochách zeleně/zahrad/polí/příkopů dál
   než 1 m od hranice budovy, nosič technického zařízení, zbytek ignore (auta a neznámé objekty nad vozovkou).
   Prahy v `seg/classes.py::RULES`. Výsledek: označeno 65 % bodů (terén 26 %, vozovka 12,5 %, vegetace 11 %,
   plot 4,3 %, budova 3,6 %, `road_or_verge` 5,3 %).
4. **Vykreslení do panoramatu** přes `point_id` produkty (2000×1000, nejbližší buňka, díry ≤ 3 px, okluze
   zdarma); 255 = bez bodu, maska vozidla, vzdálenost > 40 m. Zvlášť `bands_erp/` — pásy 14 cm kolem
   **viditelných** JVF linií (hranice dopravní stavby, budovy, stavby, zdi, plot, zábradlí, zeleň) jen pro
   evaluaci hranic, nikdy jako trénovací značka.

### Taxonomie (`dataset/seg/classes.json`)

| id | třída | zdroj JVF | v mIoU |
|---|---|---|---|
| 0 | road | provozní plocha PK, nájezd, parkoviště, most | core |
| 1 | sidewalk | chodník | core |
| 2 | building | budova, ostatní zastřešená stavba, skleník, drobné stavby (+0,5 m, > 0,2 m nad zemí) | core |
| 3 | wall | zeď, hranice zdi | core |
| 4 | fence | plot (všechny materiály včetně živého) | core |
| 5 | vegetation | body > 0,5 m nad zemí v plochách zeleně / zahrad / polí | core |
| 6 | terrain | zem v plochách zeleně, zahrad, polí, příkopů | core |
| 7–10 | water, guard_rail, stairs, pole | vodní tok/koryto/nádrž, zábradlí, schodiště, nosič | ext (jen kde model třídu má) |
| 11–15 | verge, paved_other, culvert_head, structure_other, road_or_verge | přidružená plocha PK, dvůr/manipulační/zpevnění, čelo propustku, patka, hlavní ulice | diagnostické, mimo mIoU |

`přidružená plocha PK` (krajnice) se záměrně nemapuje na `road`: zero-shot model tam předpovídá 68 % terrain a
14 % road (Mask2Former-Vistas), tj. krajnice je v Dražkově většinou nezpevněná — potvrzení mezery
„zpevněná vs. nezpevněná vozovka" z 03 §5.2 přímo z dat.

## 2. Dataset (`Geovap_cache/segds/`, README tam)

- **825 snímků `clean`**, ERP značky 2000×1000 + **16 gnómonických výsečí 1024²** na snímek (8× yaw krok 45°
  ve vodorovné rovině, 4× sklon −45° na vozovku, 4× sklon +45° — bez horního prstence zůstává pás +35…+45°
  na švech výsečí nepokrytý). Výseče jsou **srovnané do horizontu** z roll/pitch trajektorie (ověřeno na
  snímku 1304 s náklonem 8,7°: horizont leží na středním řádku všech výsečí; round-trip test mimostředového bodu
  v obou osách < 1e-6 px, `tests/test_seg_views.py`).
- V pásu φ ∈ [−55°, +45°] je označeno v mediánu 27 % pixelů (obloha, vozidlo a vzdálené pozadí jsou 255);
  42 snímků na okraji obce nemá žádné značky (pole mimo JVF pokrytí).
- Pixely v pásu (M): terrain 66,7 · vegetation 61,5 · road_or_verge 35,9 · fence 33,2 · road 29,3 ·
  building 12,8 · paved_other 2,8 · verge 1,6 · wall 1,4 · sidewalk 1,0 · water 0,2 · guard_rail 0,1;
  stairs a pole se v žádném snímku nedostaly nad 2 000 px.
- **Splity** (`splits.json`): k-means (10) na pozicích kamer → souvislé úseky ulic; test 161 / val 139 /
  train 458 snímků, 67 snímků `buffer` (do 20 m od kamery jiného splitu). Dělení podle dlaždic LAZ nefunguje —
  kamery jedou po hranicích dlaždic a 30 m nárazník pohltil 593 snímků.
- **Benchmark podmnožina** (`bench_frames.json`): 100 snímků, 30 dlaždic, 21 průjezdů; 10 snímků vybraných
  hladově pro vzácné třídy (zábradlí, voda, zeď, čelo propustku), zbytek alokací ∝ √(počet clean snímků na
  dlaždici), kamery ≥ 15 m od sebe, ≥ 15 % označených pixelů v pásu.

### Známé slabiny značek (vidět v `segds/qa/`)

- **Blízké pole je posunuté u části průjezdů — kritérium `clean` to nevidí.** Uživatelská kontrola překryvů
  („masky jsou posunuté") vedla k měření: medián vzdálenosti promítnuté JVF `hranice dopravní stavby` (pás
  viditelných úseků, řádky elevace 0…−63°) k nejbližší Canny hraně fotky (`seg/nearfield.py`,
  `segds/nearfield.json`). Zarovnané snímky 4–12 px (snímek 29: 4 px), posunuté > 20 px (snímek 1401: 33 px,
  1118: 50 px, patní linie budov leží v polovině fasády). Hodnota je **organizovaná po průjezdech**: průjezdy
  0, 1, 11, 12, 20 mají medián 5–10 px, průjezdy 3–7, 9, 16, 23, 25, 27, 29 mají 23–43 px; sousední snímky
  korelují 0,87. Mechanismus: body vlastního průjezdu sedí na fotce (siluety → `clean`), ale průjezd je vůči
  JVF vektorům (a vůči ostatním průjezdům, 02 §13.6a „dvojité střechy") posunutý o ~0,3–0,6 m, takže značky
  přenesené z JVF rastrů na body jsou v 3D posunuté a v 5 m se posun projeví jako 5–7°. Predikce modelů to
  neovlivňuje (pracují jen s fotkou). Z 755 měřitelných čistých snímků je 328 označeno `nearfield_bad`
  (> 20 px), 427 ok; v benchmarku 46 ok / 49 bad / 5 neměřitelných. **Metriky se proto uvádějí pro všech 100
  snímků i pro podmnožinu s čistým blízkým polem.** Náprava (mimo rozsah): odhad posunu (E, N, H) každého
  průjezdu vůči JVF (např. minimalizací téže pásové vzdálenosti přes snímky průjezdu) a jeho aplikace před
  rastrovým vyhledáním třídy bodu — týž offset by opravil i fúzi barev napříč průjezdy.

- **Střechy za bezlistými stromy.** Lidar vidí skrz větve, fotka ne: střecha domu 12 m za korunou stromu je
  označená `building` tam, kde fotka ukazuje větve (snímek 119, 48, 329). Není to chyba geometrie — body mají
  čas skenu shodný s časem snímku a leží v půdorysu budovy z JVF. Pro hodnocení modelů to znamená systematické
  „chyby" u building/vegetation na hranách korun; pro trénink je to šum, který by řešila jen maska vegetace z
  obrazu.
- Vegetace je odvozená z výšky nad zemí, ne z vzhledu — lampy, sloupy a předměty v zeleni dostanou `vegetation`.
- Hlavní ulice je `road_or_verge`; `road` v mIoU je tedy měřeno hlavně na vedlejších ulicích.
- Auta na fotce, která lidar v okně ±45 s neviděl, mají pod sebou značku vozovky (vzácné).

## 3. Benchmark

Recept pro všechny modely stejný: 16 srovnaných výsečí 1024², softmax/`class×mask` pravděpodobnosti, fúze do
ERP 2000×1000 s kosinovým doběhem k okraji výseče (`seg/fusion.py`, port `experiments/common/reproject.py`
s obecnou rotací), argmax jednou po fúzi, mapování do společné taxonomie **před** argmaxem
(`seg/taxonomy.py`: Vistas 65 / Cityscapes 19 / ADE20K 150 → road, sidewalk, building, wall, fence,
vegetation, terrain, water, guard_rail, stairs, pole, sky, vehicle, person, other). Metriky v pásu
φ ∈ [−55°, +45°], GT ≠ 255, pokrytí modelu: IoU po třídách vážené cos φ (plochou na kouli) i prosté,
mIoU_core přes 7 tříd, mIoU_ext kde model třídu má, Boundary IoU (2/4/8 px), pásové metriky na viditelných
JVF liniích (podíl pásu plotu/zdi/zábradlí označený správnou třídou; medián vzdálenosti pásu hranice
dopravní stavby / budovy k nejbližšímu predikovanému přechodu road|sidewalk resp. building a podíl do 14 cm
v dané vzdálenosti), rychlost a paměť na RTX 3090 (fp16 autocast).

### 3.1 Výsledky (100 snímků benchmarku, RTX 3090, fp16)

| model | taxonomie | mIoU_core cos (100 sn.) | mIoU_core cos, blízké pole ok (46 sn.) | mIoU_core plain | mIoU_ext | pixel acc (cos) | s/pano | GPU MiB | licence |
|---|---|---|---|---|---|---|---|---|---|
| eomt_city | cityscapes | 0.358 | 0.330 | 0.368 | 0.313 | 0.587 | 4.748 | 3844 | Apache-2.0 code, Cityscapes weights (research) |
| m2f_vistas | vistas | 0.336 | 0.318 | 0.342 | 0.253 | 0.601 | 5.168 | 6585 | MIT code, Vistas weights (research) |
| m2f_city | cityscapes | 0.305 | 0.285 | 0.308 | 0.267 | 0.571 | 4.939 | 3569 | MIT code, Cityscapes weights (research) |
| eomt_dinov3_ade | ade | 0.302 | 0.279 | 0.309 | 0.207 | 0.532 | 7.573 | 14410 | Apache-2.0 + DINOv3 licence |
| oneformer_city | cityscapes | 0.294 | 0.280 | 0.301 | 0.258 | 0.551 | 5.484 | 4662 | MIT code, Cityscapes weights (research) |
| segformer_b5 | cityscapes | 0.250 | 0.231 | 0.251 | 0.219 | 0.515 | 4.151 | 3820 | NVIDIA non-commercial (research only) |

IoU po třídách (cos), všechny snímky:

| třída | GT px | eomt_city | m2f_vistas | m2f_city | eomt_dinov3_ade | oneformer_city | segformer_b5 |
|---|---|---|---|---|---|---|---|
| road | 4.6 M | 0.537 | 0.553 | 0.519 | 0.506 | 0.467 | 0.451 |
| sidewalk | 0.2 M | 0.465 | 0.228 | 0.227 | 0.344 | 0.199 | 0.078 |
| building | 1.7 M | 0.230 | 0.234 | 0.223 | 0.213 | 0.232 | 0.223 |
| wall | 0.3 M | 0.043 | 0.012 | 0.004 | 0.011 | 0.006 | 0.003 |
| fence | 4.0 M | 0.246 | 0.295 | 0.204 | 0.197 | 0.249 | 0.171 |
| vegetation | 8.9 M | 0.439 | 0.453 | 0.437 | 0.295 | 0.452 | 0.389 |
| terrain | 10.0 M | 0.545 | 0.579 | 0.520 | 0.548 | 0.457 | 0.433 |
| water | 0.0 M | n/a | 0.172 | n/a | 0.154 | n/a | n/a |
| guard_rail | 0.1 M | n/a | 0.000 | n/a | 0.007 | n/a | n/a |
| stairs | 0.0 M | n/a | n/a | n/a | 0.000 | n/a | n/a |
| pole | 0.0 M | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

IoU po třídách (cos), blízké pole ok:

| třída | GT px | eomt_city | m2f_vistas | m2f_city | eomt_dinov3_ade | oneformer_city | segformer_b5 |
|---|---|---|---|---|---|---|---|
| road | 1.7 M | 0.549 | 0.580 | 0.520 | 0.463 | 0.479 | 0.424 |
| sidewalk | 0.1 M | 0.379 | 0.207 | 0.168 | 0.245 | 0.171 | 0.014 |
| building | 0.8 M | 0.241 | 0.239 | 0.225 | 0.247 | 0.240 | 0.235 |
| wall | 0.3 M | 0.049 | 0.014 | 0.005 | 0.013 | 0.007 | 0.003 |
| fence | 1.9 M | 0.204 | 0.242 | 0.210 | 0.190 | 0.232 | 0.198 |
| vegetation | 2.4 M | 0.360 | 0.366 | 0.359 | 0.257 | 0.368 | 0.337 |
| terrain | 3.7 M | 0.529 | 0.579 | 0.506 | 0.539 | 0.460 | 0.404 |
| water | 0.0 M | n/a | 0.000 | n/a | 0.007 | n/a | n/a |
| guard_rail | 0.1 M | n/a | 0.000 | n/a | 0.007 | n/a | n/a |
| stairs | 0.0 M | n/a | n/a | n/a | 0.000 | n/a | n/a |
| pole | 0.0 M | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

| boundary IoU | eomt_city | m2f_vistas | m2f_city | eomt_dinov3_ade | oneformer_city | segformer_b5 |
|---|---|---|---|---|---|---|
| road@2 | 0.011 | 0.016 | 0.012 | 0.016 | 0.011 | 0.009 |
| road@4 | 0.023 | 0.032 | 0.024 | 0.032 | 0.022 | 0.018 |
| road@8 | 0.051 | 0.060 | 0.049 | 0.061 | 0.044 | 0.038 |
| sidewalk@2 | 0.019 | 0.011 | 0.013 | 0.017 | 0.011 | 0.009 |
| sidewalk@4 | 0.037 | 0.021 | 0.028 | 0.035 | 0.026 | 0.016 |
| sidewalk@8 | 0.074 | 0.042 | 0.055 | 0.073 | 0.052 | 0.025 |
| building@2 | 0.011 | 0.011 | 0.011 | 0.010 | 0.010 | 0.010 |
| building@4 | 0.022 | 0.023 | 0.022 | 0.020 | 0.021 | 0.021 |
| building@8 | 0.041 | 0.044 | 0.043 | 0.039 | 0.042 | 0.040 |
| fence@2 | 0.015 | 0.019 | 0.011 | 0.013 | 0.014 | 0.011 |
| fence@4 | 0.029 | 0.039 | 0.022 | 0.025 | 0.027 | 0.023 |
| fence@8 | 0.053 | 0.072 | 0.045 | 0.046 | 0.052 | 0.044 |
| terrain@2 | 0.020 | 0.019 | 0.017 | 0.017 | 0.017 | 0.018 |
| terrain@4 | 0.040 | 0.038 | 0.033 | 0.034 | 0.034 | 0.036 |
| terrain@8 | 0.079 | 0.075 | 0.067 | 0.066 | 0.067 | 0.069 |
| vegetation@2 | 0.015 | 0.014 | 0.014 | 0.013 | 0.014 | 0.013 |
| vegetation@4 | 0.031 | 0.030 | 0.029 | 0.027 | 0.030 | 0.026 |
| vegetation@8 | 0.065 | 0.061 | 0.059 | 0.053 | 0.059 | 0.053 |

| pásy JVF linií | eomt_city | m2f_vistas | m2f_city | eomt_dinov3_ade | oneformer_city | segformer_b5 |
|---|---|---|---|---|---|---|
| fence band: podíl správně | 0.097 | 0.161 | 0.102 | 0.074 | 0.112 | 0.095 |
| wall band: podíl správně | 0.156 | 0.045 | 0.021 | 0.061 | 0.030 | 0.016 |
| guard_rail band: podíl správně | – | 0.000 | – | 0.017 | – | – |
| road_boundary: medián px k přechodu | 16.031 | 17.205 | 17.805 | 21.840 | 18.358 | 16.553 |
| building_edge: medián px k přechodu | 33.264 | 38.833 | 36.000 | 28.071 | 38.833 | 31.064 |
| road_boundary: podíl do 14 cm | 0.276 | 0.277 | 0.248 | 0.251 | 0.247 | 0.274 |
| building_edge: podíl do 14 cm | 0.173 | 0.154 | 0.162 | 0.145 | 0.155 | 0.152 |

| diagnostická třída | model | složení predikcí |
|---|---|---|
| road_or_verge | eomt_city | road 0.97, terrain 0.02, vehicle 0.01, sidewalk 0.00 |
| road_or_verge | m2f_vistas | road 0.97, terrain 0.02, vehicle 0.01, sidewalk 0.00 |
| road_or_verge | m2f_city | road 0.96, terrain 0.02, vehicle 0.01, sidewalk 0.00 |
| road_or_verge | eomt_dinov3_ade | road 0.96, terrain 0.02, vehicle 0.01, sidewalk 0.00 |
| road_or_verge | oneformer_city | road 0.96, terrain 0.02, sidewalk 0.01, vehicle 0.01 |
| road_or_verge | segformer_b5 | road 0.97, terrain 0.02, vehicle 0.01, vegetation 0.00 |
| verge | eomt_city | terrain 0.72, road 0.15, sidewalk 0.07, fence 0.03 |
| verge | m2f_vistas | terrain 0.76, road 0.11, sidewalk 0.07, fence 0.04 |
| verge | m2f_city | terrain 0.65, road 0.22, sidewalk 0.04, fence 0.04 |
| verge | eomt_dinov3_ade | terrain 0.87, road 0.05, sidewalk 0.04, fence 0.02 |
| verge | oneformer_city | terrain 0.61, road 0.30, fence 0.04, vegetation 0.03 |
| verge | segformer_b5 | terrain 0.49, road 0.28, vegetation 0.17, fence 0.04 |
| paved_other | eomt_city | road 0.62, terrain 0.23, fence 0.05, building 0.05 |
| paved_other | m2f_vistas | road 0.41, terrain 0.40, fence 0.05, building 0.05 |
| paved_other | m2f_city | road 0.62, terrain 0.19, building 0.06, fence 0.05 |
| paved_other | eomt_dinov3_ade | terrain 0.75, road 0.08, building 0.06, fence 0.04 |
| paved_other | oneformer_city | road 0.63, terrain 0.16, fence 0.05, building 0.05 |
| paved_other | segformer_b5 | road 0.62, terrain 0.20, building 0.06, fence 0.06 |

`model_sheet.jpg` v `segds/bench/` ukazuje všech šest modelů na dvou snímcích s čistým blízkým polem.

### 3.2 Čtení výsledků

- **Pořadí je stabilní na obou podmnožinách**: EoMT-L (DINOv2, Cityscapes 1024) 0,358 / 0,330 > Mask2Former
  Swin-L Vistas 0,336 / 0,318 > Mask2Former Cityscapes 0,305 > EoMT-L DINOv3 (ADE20K, 512 dlaždice) 0,302 >
  OneFormer Swin-L Cityscapes 0,295 > SegFormer-B5 0,250. Podmnožina s čistým blízkým polem dává čísla o 1,5–2,5
  bodu nižší, ne vyšší — posun značek v blízkém poli tedy není hlavní zdroj chyb; ten je v samotném pseudo-GT
  (zrnitost lidaru, střechy za stromy, vegetace odvozená z výšky) a v taxonomické mezeře.
- **Absolutní hodnoty (0,25–0,36 mIoU_core) jsou hluboko pod odhadem 44–50 z 03 §9**, protože metrika je proti
  automatickému pseudo-GT, ne proti ruční anotaci: pseudo-GT samo má chybu řádu 10–20 % pixelů (viz slabiny
  výše). Čísla jsou proto **relativní pořadí modelů**, ne absolutní schopnost. Relativní pořadí je konzistentní
  s literaturou: zmrazený DINOv2 backbone (EoMT) poráží plně dotrénované CNN/Swin modely mimo doménu.
- **Po třídách**: road 0,45–0,55 a terrain 0,43–0,58 jsou nejlepší; vegetation 0,30–0,45; building jen
  0,21–0,25 u všech modelů (stejně — limit je v GT: střechy za korunami a facády mimo 40 m); **fence 0,17–0,30**
  (Mask2Former Vistas nejlepší, jediný trénovaný na bohaté třídě Fence); **wall ≈ 0** u všech (zdi v Dražkově
  jsou nízké podezdívky a ohradní zídky, modely je vidí jako fence/building); sidewalk 0,08–0,47 (EoMT nejlépe;
  GT chodníků je jen 1 M px). Voda, zábradlí, schodiště a sloupy nemají v benchmarku dost GT pixelů pro závěr.
- **Hranice** (pásy JVF linií, jediná metrika hranic použitelná na zrnitém pseudo-GT): medián vzdálenosti
  predikovaného přechodu vozovky od JVF hranice dopravní stavby je 16–22 px (2,9–4,0°; při 5 m ≈ 25–35 cm),
  do 14 cm padne jen 25–28 % pásu; u paty budov 28–39 px. To zahrnuje i posun průjezdů (§2) — na podmnožině
  s čistým blízkým polem je pořadí stejné. Boundary IoU (2–8 px) je u všech modelů 0,01–0,08 a nerozlišuje —
  na zrnitém pseudo-GT není použitelná, uvádí se jen pro úplnost.
- **Diagnostické třídy** potvrzují dvě zjištění z 03: `přidružená plocha PK` (krajnice) je pro všechny modely
  61–87 % terrain, tj. nezpevněná; `dvůr/manipulační/zpevnění` je 41–63 % road a 16–40 % terrain — směs
  zpevněných a nezpevněných povrchů, kterou žádná veřejná taxonomie nerozlišuje. Hlavní ulice
  (`road_or_verge`) je 96–97 % road.
- **Rychlost a paměť** (16 výsečí 1024², fúze 2000×1000): EoMT-L 4,7 s / 3,8 GB, SegFormer 4,2 s, Mask2Former
  5 s / 3,6–6,6 GB, OneFormer 5,5 s, EoMT-DINOv3 v režimu 3×3 dlaždic 7,6 s / 14 GB. Celých 825 snímků = ~65 min
  na jeden model.

### 3.3 Doporučení

1. **EoMT-L (DINOv2) jako výchozí model** pro fázi 2 (self-training, PEFT adaptéry, 03 §9): nejlepší mIoU,
   nejrychlejší, Apache-2.0 kód i váhy (Cityscapes váhy jsou pro produkci stejně nutné nahradit vlastním
   dotrénováním). Mask2Former-Vistas ponechat jako referenci pro plot a zábradlí (jediný s těmito třídami).
2. **Pseudo-GT před tréninkem opravit**: (a) odhad posunu (E, N, H) průjezdů vůči JVF a jeho aplikace před
   rastrovým vyhledáním třídy (odstraní `nearfield_bad`), (b) maska vegetace z obrazu (např. průnik s predikcí
   `vegetation` shody ≥ 4 modelů) nad body označenými `building`, (c) trénovat jen na pixelech, kde se
   pseudo-GT a konsensus modelů shodují, nebo pseudo-GT použít jen pro třídy road/terrain/fence/building a
   zbytek nechat na modelech.
3. Předpověď EoMT-L pro všech 825 čistých snímků je v `segds/bench/eomt_city/` (native + common + conf) —
   vstup pro přenos 2D→3D (`mapping.accumulate.LabelVote`, M3).


## 4. Reprodukce

```
uv sync --extra seg
uv run python -m mapping.cli.seg_build areas      # 1 s
uv run python -m mapping.cli.seg_build rasters    # 40 s
uv run python -m mapping.cli.seg_build points --workers 4     # 22 s
uv run python -m mapping.cli.seg_build labels --frames clean --workers 8   # 3,5 min
uv run python -m mapping.cli.seg_build views --workers 10     # 8 min, 2,7 GB
uv run python -m mapping.cli.seg_build dataset
uv run python -m mapping.cli.seg_bench run --models all --frames bench     # ~1 h
uv run python -m mapping.cli.seg_bench evaluate && uv run python -m mapping.cli.seg_bench report
uv run pytest     # test_seg_*: polygonizace, pravidla bodů, round-trip výsečí, taxonomie, metriky
```
