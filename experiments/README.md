# Experimenty nad daty Dražkov

Doprovodné skripty k `03_semanticka_segmentace.md` §12. Ověřují tvrzení dokumentu přímo na
reálném vzorku od GEOVAP (`/home/jatuma/repos/Geovap/Geovap_data/DTM_Dražkov`), ne jen na
proxy veřejných datasetech z rešerše.

## Prostředí

```
pip install laspy[lazrs]          # chybelo, LAZ cteni
pip install -U scikit-learn       # puvodni verze v prostredi mela numpy2/pyarrow ABI konflikt,
                                   # ktery shazoval `import transformers`
```

`pandas` je v tomto prostředí rozbité (numpy2 vs. pyarrow ABI) — záměrně se nepoužívá, `io_data.py`
čte `export.csv` čistým `csv` modulem.

GPU (RTX 3090) se využívá pro: dávkovou projekci JVF objektů v E1 (`torch.cdist`, `camera.py`),
celou gnomonickou reprojekci a fúzi softmaxů v E2 (`torch.nn.functional.grid_sample`), a inferenci
modelu. CPU zůstává jen na JPEG dekódování, `cv2` rasterizaci a I/O.

## Spuštění

```
cd geovap/experiments
python3 e0_data_sanity.py
python3 e1_project_jvf_to_pano.py
python3 e2_zero_shot_baseline.py          # pouzije out/e2_frame_selection.json (40 snimku, viz nize)
python3 e3_score_against_pseudo_gt.py
```

## E0 — sanity check dat

Vše sedí: 1503/1503 JPG ↔ `export.csv`, 4342 JVF objektů v `1_ZPS_GAD.geojson` (98.9 % → 100 % po
doplnění `class_map.py`), souřadnice geojsonu/export.csv/LAZ se překrývají ve stejném S-JTSK CRS.
Rozložení tříd viz `out/e1/coverage_by_class.txt` (sloupec vlevo od `%` je celkový počet v Dražkově).

## E1 — projekce JVF vektorů do panoramat (pseudo-GT zdarma)

**Jádro celého plánu, funguje.** 3161/4342 objektů (72.8 %) má panorama do 20 m. Vizuální kontrola
(`out/e1/overlays/`, 125 náhledů) potvrzuje, že projekce sedí i pro vzácné třídy:

- `hranice budovy` (0100000299) — červená linie leží přesně na patě fasády
- `plot` (0100000162) — leží na hraně keře/oplocení pozemku
- `čelo propustku` (0100000193, jen 28 instancí v obci) — padá na okraj vozovky/příkopu
- `zábradlí` (0100000199, jen 8 instancí) — sedí přesně na červeném kovovém zábradlí mostku
- `studna` (0100000084) — padá do zahrady za plotem

Použitý kamerový model a znaménkové konvence jsou identické s `02_obarveni_pointcloudu.md` §2.1
(ověřeno tam nezávisle na TerraScan RGB, ΔE medián 6,08). Shoda vizuální kontroly tady je druhý,
nezávislý důkaz, že model je správně implementovaný — tentokrát na sémantických objektech, ne na barvě.

Výstupy: `out/e1/coverage.csv` (per objekt), `out/e1/coverage_by_class.txt` (souhrn), `out/e1/overlays/*.jpg`.

## E2 — zero-shot baseline (Mask2Former, Mapillary Vistas taxonomie)

Checkpoint `facebook/mask2former-swin-large-mapillary-vistas-semantic` (MIT kód, Vistas váhy —
**výzkumné použití**, licence komerčního nasazení viz SS5.1 `03_semanticka_segmentace.md`, neřešeno
zde). 65 tříd Vistas taxonomie.

Recept podle SS4.3: prstenec 8× yaw (0°,45°,…,315°) × pitch 0°, FOV 90°×90° + 4× pitch −45°
(yaw 0°,90°,180°,270°), rozlišení vysece 1024². Fúze softmaxů (ne argmax hlasování) s kosinovým
dobehem, `torch.nn.functional.grid_sample` na GPU. Fúzní buffer běží na out_w/out_h = pano/2 a
float16 (jinak OOM na 24 GB — `[1, 65, 4000, 8000]` float32 samo o sobě je 8,3 GB).

**Dva bugy nalezené a opravené za běhu** v `common/reproject.py`, `view_sample_grid`:

1. Špatné pořadí/znaménko inverzní rotace (pitch) — coverage mapa ukazovala tvrdé mezery mezi
   sousedními výsečmi (`out/e2/coverage_bug_before_fix.jpg`, ~11 % nepokryto v pásu φ∈[−55°,+45°])
   a segmentace obsahovala 25 % pixelů třídy „Bird" (argmax defaultu na nulách). Po opravě
   nepokryto v operačním pásu jen 0,76 % (numerický okraj), viz `out/e2/coverage_ring_down_fixed.jpg`.
   Nepokryté pixely (zenit/nadir) se teď explicitně značí `VOID_LABEL=255`, ne argmaxem nesmyslné nuly.
2. **Vertikální souřadnice byla zrcadlená** — `y = up/fwd*f` mělo být `y = -up/fwd*f`, protože
   `up` (směr nahoru ve světě) a `gy` (řádek obrazu vysece, roste opačně) jsou navzájem záporné.
   Efekt byl vidět jako lokálně vzhůru nohama segmentace (typicky u střech, kde se sbíhaly dvě
   výseče s různou elevací) — **odhaleno až vizuální kontrolou uživatele** na frame 119 (bílá
   dodávka v příjezdové cestě vypadala „jinde a zrcadlená"). Ani test středu výseče, ani test
   čistě horizontálního posunu tohle nezachytí — je potřeba round-trip test mimostředového bodu
   zvlášť ve vertikální ose. Srovnání před/po: `out/e2/seg_119_260408_115046334.jpg` (aktuální,
   opravená verze) — před opravou měly střechy viditelně obrácený (vzhůru nohama) tvar a dodávka
   byla posunutá/rozpitá mimo své skutečné obrysy.

**Rozpočet:** 3,43 s/panorama na RTX 3090 → **86 minut na celých 1503 panoramat** (§4.4 dokumentu
odhadoval "jednotky GPU-hodin" pro EoMT-L/SegFormer-B2 — Mask2Former Swin-L je pomalejší backbone,
ale pořád pod 1,5 h pro celý vzorek na jedné kartě).

Spuštěno na 40 panoramatech vybraných hladově tak, aby pokryla co nejvíc JVF tříd z E1 (36/46
tříd, `out/e2_frame_selection.json`). Vizuální kontrola (`out/e2/seg_*.jpg`) — budovy, vegetace,
plot, silnice na rozumných místech.

**Neproběhlo (rozpočet této iterace):** plný běh přes všech 1503 panoramat. Příkaz je stejný,
jen smazat/ignorovat `out/e2_frame_selection.json` a nastavit `N_SAMPLE_FRAMES` nebo upravit výběr.

## E3 — skóre proti pseudo-GT

Metoda: kolem projektované JVF linie/bodu se vykreslí pás o šířce podle tolerance 14 cm (§1, §8
dokumentu) převedené na pixely v dané vzdálenosti; měří se podíl pixelů v pásu, které model označil
odpovídající veřejnou třídou (`class_map.py`). Výsledek v `out/e3/per_class_score.txt`:

| n | skóre | JVF třída → veřejná |
|---|---|---|
| 2 | 0.50 | přidružená plocha pozemní komunikace → road |
| 9 | 0.41 | hranice přírodního a polopřírodního objektu → terrain\|vegetation |
| 3 | 0.33 | ostatní zastřešená stavba → building |
| 3 | 0.31 | chodník → sidewalk |
| 120 | 0.27 | hranice dopravní stavby nebo plochy → road\|sidewalk\|curb |
| 21 | 0.27 | hranice udržované zeleně → vegetation |
| 50 | 0.22 | hranice budovy → building |
| 9 | 0.22 | budova → building |
| 23 | 0.14 | udržovaná plocha zeleně → vegetation |
| 12 | 0.11 | hranice zdi → wall |
| 176 | 0.04 | hranice stavby → building |
| 20 | 0.00 | hranice vodního díla → water |

(číslo výše je z běhu PO opravě obou bugů z `common/reproject.py`; první, chybný běh dával mírně
jiné pořadí, ale stejný celkový obrázek — viz `03_semanticka_segmentace.md` §12.4.)

**Toto je záměrně přísná metrika** (úzký pás v toleranci 14 cm, ne plošné mIoU) — nízká čísla
nejsou v rozporu s odhadem 44–50 mIoU v §9 dokumentu (to je jiná, mnohem shovívavější metrika).
Hlavní hodnota E3 není absolutní číslo, ale **relativní pořadí tříd** a fakt, že jde měřit vůbec —
poprvé na skutečné geometrii Dražkova, ne na DensePASS/GOOSE proxy. `hranice stavby` skóruje výrazně
hůř než `hranice budovy`, ačkoli obojí mapuje na `building` — to je přesně ten druh zjištění, který
proxy datasety nemohly odhalit (JVF rozlišuje budovu/ostatní stavbu jemněji, než veřejné taxonomie).

202 objektů nemá žádný veřejný ekvivalent (vlastní třídy, viz `class_map.py` `public=None`) — pro
ně zero-shot skóre nedává smysl, to je materiál pro syntetickou data / few-shot podle §11 fáze 3
dokumentu.

## E4 — rotační citlivost (částečně, z rozpočtových důvodů)

Reálné rozdělení roll/pitch pro všech 1503 snímků (z `export.csv`, viz `e0_data_sanity.py`):

| | medián \|x\| | p90 | p99 | max | podíl \|x\|>5° |
|---|---|---|---|---|---|
| roll | 1,07° | 2,07° | 3,95° | 8,66° | 0,33 % |
| pitch | 1,23° | 2,11° | 3,52° | 5,99° | 0,13 % |

**Riziko z §3.3 dokumentu (SGAT4PASS: 5° perturbace → rozptyl mIoU ~100×) je pro tohle vozidlo v
praxi vzácné** — jen 0,1–0,3 % snímků má vůbec |roll| nebo |pitch| > 5°. Medián je ~1°. To nesnižuje
hodnotu srovnání do horizontu (pořád "zadarmo", jeden převzorkovací průchod), ale znamená to, že
**naměřený dopad bude pravděpodobně mnohem menší** než umělá ±5° ablace v citované práci naznačuje —
protože reálná data leží skoro celá uvnitř ±3°, ne na hranici ±5°.

Neproběhla už A/B srovnávací část (E2 se zapnutým/vypnutým srovnáním do horizontu na podvzorku s
nejvyšším |roll|/|pitch|) — vyžaduje druhý běh E2, přeskočeno v této iteraci podle plánu (E4 měla
nejnižší prioritu). Kandidátní snímky pro takový test: filtrovat `frames.roll_deg`/`pitch_deg` z
`common/io_data.py` na |x|>3° (~30-50 snímků).
