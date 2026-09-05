# Sémantická segmentace panoramat

Jaký model zvolit, jak ho pustit na equirektangulární snímky, kde vzít trénovací data a co to bude stát.

---

## 1. Sedm zjištění, která mění zadání

1. **Cílem není „model natrénovaný na Cityscapes".** Vítězný recept 2024–2026 pro segmentaci mimo doménu je **zmrazený self-supervised ViT (DINOv2/DINOv3) plus ~1 % trénovatelných adaptérů**, ne plně dotrénovaný supervised backbone. Plné dotrénování ničí naučený prior.
2. **Model se nesmí pouštět na syrové equirektangulární panorama.** Reprojekce na perspektivní výseče dá **+5,6 mIoU** proti přímému běhu na ERP.
3. **Rozlišení 8000×4000 dělá reprojekci zadarmo.** 8000 px / 360° = **22,2 px/°**, takže výseč 90° se vyrenderuje jako **2000×2000 nativně bez jakéhokoli převzorkování**. Reprojekce *je* zároveň strategie dlaždicování.
4. **Mapillary Vistas je taxonomicky nejblíž JVF DTM — a jeho licence zakazuje přesně to, co chceme dělat.** Tohle je největší nákladová položka celého programu a musí se vyřešit jako první.
5. **Seznam tříd je na třech místech špatně, protože JVF DTM ty objekty nemodeluje tak, jak jsme předpokládali.** Viz §2.
6. **Nesmíme produkovat polygony.** IS DTM dělá plochování sám. Naším výstupem jsou **topologicky uzavřené 3D konstrukční linie správného typu plus jeden definiční bod na plochu**.
7. **Cílová přesnost je právně závazná: m_xy ≤ 0,14 m, m_H ≤ 0,12 m** (třída přesnosti 3), ověřovaná dle ČSN 01 3410. Mobilní laserové skenování je přitom metodikou ČÚZK **výslovně doporučeno** právě pro ZPS silnic II. a III. třídy — náš případ užití je to, co regulátor zamýšlel.

---

## 2. Co JVF DTM opravdu žádá

Tohle je nutné opravit dřív, než se začne cokoli anotovat.

### 2.1 Čtyři opravy seznamu tříd

| Náš předpoklad | Realita v JVF DTM |
|---|---|
| **`obrubník`** jako objekt | **Neexistuje.** Obrubník je linie `hranice dopravní stavby nebo plochy` (`0100000304`) s atributem `typ dopravní stavby nebo plochy`. Jde tedy o úlohu **sémantické hranice** („co tahle linie ohraničuje?"), ne o detekci objektu |
| **`dopravní značka`** | **V DTM vůbec není.** Značky jsou samostatný produkt (pasport DZ). Nejbližší objekt je `nosič technického zařízení` (bod) |
| **`brána`** | **Neexistuje.** Nejbližší je `stavebně upravený vjezd na pozemek` (`0100000165`), linie s atributy průjezdná šířka a výška |
| **`plot`** jako plocha | `plot` (`0100000162`) je **linie**, ne plocha, s právě pěti hodnotami materiálu: dřevěný, drátěný, kovový, zděný, **živý** |

**Živý plot je právně PLOT, ne vegetace.** Všechny veřejné datasety označují živé ploty jako `Vegetation`. Tohle je povinná vlastní třída, kterou nikde nekoupíme.

### 2.2 Klíčové konstrukční linie

Všechny mají geometrii `linie`:

| Kód | Název | Doména atributu |
|---|---|---|
| `0100000299` | hranice budovy | způsob pořízení |
| `0100000300` | hranice stavby | podezdívka, rampa, terasa, komín, skleník, bazén, patka/deska/monolit/pilíř, stavba pro zpevnění povrchu, **čelo propustku**, drobná sakrální/kulturní stavba… |
| `0100000302` | hranice zdi | zeď volně stojící / **opěrná** / zárubní / městské hradby / nezjištěno |
| **`0100000304`** | **hranice dopravní stavby nebo plochy** | pozemní komunikace, chodník, cyklostezka, parkoviště, dopravní ostrůvek, nájezd/sjezd/vjezd, přidružená plocha, **příkop/násyp/zářez**, plocha mostní konstrukce… |
| `0100000305` | hranice přírodního objektu | vodní tok, jezero, zemědělská plocha, zahrada, les, hospodářsky nevyužívaná plocha |
| `0100000308` | hranice udržované zeleně | — |

Dále: `zábradlí` `0100000199` (linie), `svodidlo` `0100000318`, `terénní hrana` `0100000217` (hrana / pata), `nosič technického zařízení` `0100000201` (bod), `studna` `0100000084` (bod), `neidentifikovaný objekt` `0100000202` — únikový poklop, kterým lze ohraničit skoro cokoli.

Atribut **`převažující povrch`** u komunikací má hodnoty asfalt, beton, dlažba, R-materiál, písek/štěrkopísek, **šotolina**, **nezpevněno**, nezjištěno. Žádný veřejný dataset nerozlišuje zpevněnou a nezpevněnou vozovku — je to samostatná atributová hlava.

Dobrá zpráva: `příkop, násyp, zářez dopravní stavby` je **jeden sloučený objekt** (`0100000051`). Nemusíme sémanticky rozlišovat příkop od náspu od zářezu.

---

## 3. Modely

### 3.1 Srovnání

| Model | Cityscapes mIoU | ADE20K | Vistas | Rychlost | Licence |
|---|---|---|---|---|---|
| SegFormer-B5 | 84,0 | 51,0 | — | — | NVIDIA nekomerční |
| SegNeXt-S | 81,3 | — | — | 124 GFLOPs | Apache-2.0 |
| **Mask2Former** Swin-L | 83,3 | 56,1 | — | — | **MIT** |
| **OneFormer** ConvNeXt-L | 83,0; PQ 68,5 | — | **63,2** | — | **MIT** |
| **OneFormer** DiNAT-L | 83,1 | — | **64,0** | — | MIT |
| ViT-Adapter + BEiT-3 | **85,2** | **62,8** | — | pomalé | Apache-2.0 |
| InternImage-H | ~86 | 62,5 | — | vyžaduje DCNv3 CUDA | MIT |
| **EoMT-L (DINOv2)** | **84,2** @1024² | 58,4 | — | **25 FPS na H100, 4× rychlejší než M2F** | Apache-2.0 |
| **EoMT-L (DINOv3)** | — | **59,5** | — | 137–200 FPS | + licence DINOv3 |

Cityscapes je nasycený na 84–87 mIoU a rozdíly na špičce jsou ~2 body — pro nás irelevantní. Rozhoduje chování mimo doménu, licence, cena dotrénování a kvalita hranic.

### 3.2 Rozhodující důkaz: zmrazený foundation backbone

Měření z Rein (CVPR 2024), přenos GTAV → reálná data, průměr přes Cityscapes / BDD100K / Mapillary:

| Backbone | Režim | Trénovatelné parametry | Průměr mIoU |
|---|---|---|---|
| MAE-L | plné dotrénování | 331 M | 54,2 |
| **MAE-L** | **zmrazený** | 0 | 43,0 |
| SAM-H | plné dotrénování | 632 M | 56,9 |
| **SAM-H** | **zmrazený** | 0 | 54,2 |
| DINOv2-L | plné dotrénování | 304 M | 61,7 |
| **DINOv2-L** | **zmrazený** | **0** | **61,1** |
| **DINOv2-L + adaptéry** | **PEFT** | **2,99 M** | **64,3** |

Tři závěry s čísly za sebou:

- **Zmrazený DINOv2-L s nulou trénovatelných parametrů poráží plně dotrénovaný MAE-L o 18 bodů.** Self-supervised rysy jsou největší jednotlivá páka pro přenos mimo doménu.
- **SAM je špatný sémantický backbone** (zmrazený 54,2). Je to návrhář masek, ne klasifikátor. Použít na zpřesnění hranic a na anotaci, ne na klasifikaci.
- **PEFT poráží plné dotrénování.** To přímo určuje tréninkový recept při malé české anotační sadě.

**DINOv3** posouvá lineární sondu na zmrazených rysech na Cityscapes **81,1** (+5,5 proti DINOv2), ADE20K 55,9 (+6,4). Pozor ale na licenci — je vlastní, ne Apache-2.0. Komerční užití povoluje, ale vyžaduje právní posouzení. DINOv2 (Apache-2.0) je záloha za cenu ~5 mIoU.

### 3.3 Rotační citlivost — přímo akční pro měřický vůz

SGAT4PASS změřil, co udělá perturbace panoramat o **5° v pitch a roll**: rozptyl mIoU vyskočil z **0,056 na 5,147**, tedy zhruba 100×, střední pokles ≈ 4 mIoU.

**Máme trajektorii s roll a pitch pro každý snímek. Panoramata je proto nutné před segmentací srovnat do vodorovné roviny.** Jeden převzorkovací průchod, několik mIoU zadarmo.

---

## 4. Jak pustit model na panorama

### 4.1 Rozhodující ablace

360SFUDA (CVPR 2024) měnil zorný úhel projekce při pevném modelu natrénovaném na perspektivních snímcích, DensePASS, 19 tříd:

| Zorný úhel projekce | mIoU | Δ proti ERP |
|---|---|---|
| bez projekce (syrové ERP) | 38,65 | — |
| 60° | 44,03 | +5,38 |
| 72° | 44,16 | +5,51 |
| **90°** | **44,28** | **+5,63** |
| 120° | 44,02 | +5,37 |
| 180° | 41,65 | +3,00 |
| 360° (celé ERP) | 40,31 | +1,66 |

Optimum je široké a ploché mezi 60° a 120°.

### 4.2 Proč to u nás funguje ještě líp

Zkreslení ERP roste jako **1/cos φ** a diverguje u pólů. Ale:

- Horní čepička je **obloha** — beztexturní, triviálně oddělitelná, zkreslení 5,76× nestojí nic.
- Dolní čepička je **střecha vozidla** — musí se natvrdo maskovat, ne segmentovat. Kdyby tam zůstala, otrávila by jakoukoli self-training smyčku.
- **Omezením na φ ∈ [−55°, +45°]** je maximální horizontální roztažení **1/cos 55° = 1,74×**, a v pásmu ±45° jen 1,41× — srovnatelné s běžnou augmentací měřítka.

Argument pro panorama-nativní architekturu je tedy u našich dat **slabý**. Propad 38,65 vs 80,9 není hlavně zkreslením, ale **nesouladem zorného úhlu a měřítka** — a to reprojekce na 90° řeší.

> **Past, kterou nelze přehlédnout.** Vozovka, obrubníky a vodorovné značení leží na elevaci **−40° až −80°**. Vozidlo zabírá spodních možná 15–25°, ne spodních 60°. **Je nutné přidat spodní prstenec výsečí se sklonem ≈ −45°**, jinak přijdeme přesně o ty třídy, o které v DTM jde.

### 4.3 Konkrétní recept

- **Výseče:** prstenec **8 pohledů, sklon 0°, zorný úhel 90°×90°, krok 45°** (překryv 50 %) + **4–6 pohledů se sklonem −45°** na vozovku + volitelně 4 se sklonem +45°.
- **Velikost renderu:** 1600–2048² — nativní na rovníku, žádné převzorkování.
- **Slučování: průměrovat logity nebo softmax, nikdy ne hlasovat argmaxem.** Akumulovat `Σ w·softmax` do bufferu tvaru `[C, H, W]` ve float16 s **kosinovým doběhem váhy** k okrajům výseče, a argmax udělat jednou globálně na konci. Pro 8000×4000 a ~25 tříd má buffer ~1,5 GB. Hlasování argmaxem zahazuje přesně tu informaci o konfidenci, kterou překryvy poskytují, a dělá skvrny.
- **Šev:** při reprojekci **žádný nevzniká** — gnómonické výseče nikdy nepřekračují λ = ±π. Pro jakékoli dodatečné zpracování v ERP (CRF, morfologie) horizontálně **zabalit okraje** o šířku receptivního pole a pak oříznout.

### 4.4 Výpočetní rozpočet

~16–24 výsečí × 1 503 panoramat = **24 000–36 000 průchodů** při 1024–2048². EoMT-L běží 25 FPS na H100, RTX 4090 je zhruba 3–4× pomalejší. To je **jednotky GPU-hodin s EoMT-L nebo SegFormer-B2, pod den s Mask2Former Swin-L**. Zvládne to jedna pracovní stanice.

---

## 5. Datasety a licence

### 5.1 Licenční tabulka — číst před jakýmkoli závazkem

| Aktivum | Licence | Komerčně? |
|---|---|---|
| Mask2Former, OneFormer, InternImage (kód i váhy) | **MIT** | ✅ |
| EoMT (závislosti Apache-2.0) | Apache-2.0 | ✅ |
| DINOv2 | Apache-2.0 | ✅ |
| DINOv3 | vlastní licence | ⚠️ ano, ale posoudit |
| Rein (adaptéry) | **GPL-3.0** | ⚠️ virální — adaptér reimplementovat, nelinkovat |
| SAM / SAM 2 | Apache-2.0 | ✅ |
| SAM 3 / 3.1 | vlastní SAM License | ⚠️ posoudit |
| **Cityscapes (data)** | nekomerční | ❌ |
| **Mapillary Vistas (data)** | Research Use License 2019 — bez derivátů, bez produktů | ❌ existuje komerční licence, nutno vyjednat |
| **GOOSE / GOOSE-Ex** | **CC BY-SA 4.0** | ✅ (share-alike) |
| **BDD100K** | **BSD-3-Clause** | ✅ |
| **SensatUrban** | **MIT** | ✅ |
| KITTI-360, SemanticKITTI, nuScenes, Waymo, ACDC, Toronto-3D | nekomerční | ❌ |

**Komerčně čistý trénovací korpus je tenký: GOOSE + BDD100K + SensatUrban + vlastní česká data.** Počítat se dvěma modely — výzkumný na prototypování a benchmarky, produkční jen z čistého korpusu.

### 5.2 Mezera mezi JVF a veřejnými datasety

| JVF cíl | Nejlepší veřejný zdroj | Verdikt |
|---|---|---|
| pozemní komunikace | Road, Service Lane, Road Shoulder (Vistas) | ✅ |
| *obrubník* → `hranice dopravní stavby` | **Curb** (Vistas, IDD, GOOSE, Waymo) | ✅ 4 zdroje |
| chodník | Sidewalk, Pedestrian Area | ✅ |
| **nájezd/sjezd/vjezd** | **Driveway + Curb Cut** — jen Vistas | ⚠️ licenčně blokované |
| **příkop / násyp / zářez** | nic; proxy `soil`+`gravel`+`low_grass` (GOOSE) | ❌ **vlastní** — ale je to geometrický jev, detekce zlomu sklonu v LiDARu porazí segmentaci obrazu |
| budova | Building, Garage | ✅ |
| **plot drátěný** | **`wire`** — jen GOOSE | ⚠️ jediný zdroj |
| **plot živý** | **`hedge`** — jen GOOSE | ⚠️ jediný zdroj |
| **plot dřevěný / zděný / kovový** | jen obecný `Fence` | ❌ **vlastní** |
| zeď | Wall | ✅ |
| **opěrná zeď** | sloučeno do Wall | ❌ vlastní, nebo odvodit z výškového rozdílu v LiDARu |
| zábradlí | Guard Rail | ⚠️ veřejný `guard_rail` = svodidlo, ne zábradlí |
| **čelo propustku** | nic; nejblíž `pipe` (GOOSE) | ❌ **vlastní** |
| zeleň | 12třídní rozklad vegetace (GOOSE) | ✅ nejlepší dostupné |
| **zahrada** | nic | ❌ **je to využití území, ne vzhled — fúzovat s KN/LPIS, neanotovat** |
| vodní tok | Water | ✅ |
| **šachta / vpusť** | **Manhole, Catch Basin** — jen Vistas | ⚠️ licenčně blokované |
| **studna** | nic | ❌ vlastní |
| sloup | Utility Pole, Pole Group (Vistas) | ✅ |
| **převažující povrch** | nic | ❌ **žádný dataset nerozlišuje zpevněnou a nezpevněnou vozovku** — samostatná atributová hlava |

**Povinné vlastní třídy: ~8.** Bez Mapillary **+6 dalších.**

---

## 6. Anotace

### 6.1 Základní cena

| Zdroj | Číslo |
|---|---|
| Cityscapes | „více než **1,5 h na snímek**"; hrubá anotace „méně než 7 min" |
| ACDC | **3,3 h na snímek** |
| Plná pixelová anotace (ECCV'16) | **239,7 s/snímek**; 1 bod na třídu **22,1 s**; klikyhák **34,9 s** |
| SAM | **34 s → 14 s na masku**, 6,5× rychlejší než COCO |
| SAM 2 | **37,8 → 4,5 s na snímek** (≈8,4×) |

Při sazbách Cityscapes by 5 000 českých panoramat v granularitě ZPS znamenalo **~7 500 h ≈ 4,5 člověkoroku**. Rozlišení 8000×4000 s ~25 jemnými třídami bude *horší* než Cityscapes, ne lepší.

### 6.2 Páky, s čísly

**(1) Semi-supervised učení — největší jednotlivá páka.** UniMatch na Cityscapes, ResNet-101:

| Podíl anotací | Snímků | Supervised | UniMatch | Δ |
|---|---|---|---|---|
| 1/16 | 186 | 66,3 | **76,6** | +10,3 |
| 1/8 | 372 | 72,8 | **77,9** | +5,1 |
| 1/4 | 744 | 75,0 | 79,2 | +4,2 |
| 1/2 | 1 488 | 78,0 | 79,5 | +1,5 |

**~186 hustě anotovaných snímků plus neanotovaný zbytek dá 76–78 mIoU — tedy 95–98 % plné suvervize za ~6 % anotačních nákladů.** Tohle je číslo do obchodního případu.

**(2) Aktivní učení.** PixelPick dosáhl „**~96 % plně supervizované baseline s 0,06 % anotací**". Akvizice musí být na úrovni regionů/superpixelů, ne celých snímků. Nejlepší akviziční funkce byla **margin sampling**.

**(3) Slabá supervize.** Body jsou **10,8× levnější** než plné masky a při stejném rozpočtu *přesnější*. Pro tenké liniové prvky ZPS (obrubník, zábradlí, plot, čelo propustku) je správná kombinace **klikyháky + rozšíření přes SAM** — klikyhák je přirozený prompt pro SAM a objekt je topologicky jednoduchý.

**(4) Temporální propagace přes SAM 2.** Po sobě jdoucí panoramata podél trajektorie jsou video. Anotovat plot nebo obrubník jednou a propagovat podél jízdy — **8,4× naměřeno**. Tohle je největší nástrojová výhra specifická pro mobilní mapování.

**(5) Syntetická data pro geometrické třídy bez pokrytí.** Geometrii ovládáme. Vyrenderovat **příkop/násep/zářez, opěrnou zeď, čelo propustku** a materiálově odlišené ploty v Blenderu nebo CARLA **rovnou v equirektangulární projekci**. Pro tyhle tři třídy je syntetika s následným few-shot dotrénováním skoro jistě levnější než ruční anotace.

**(6) Zahradu a zemědělskou plochu se nepokoušet segmentovat z obrazu.** Je to využití území, ne vzhled. Fúzovat s katastrem a LPIS.

**(7) Anotační nástroje — praktické zjištění.** CVAT, Label Studio, Segments.ai, Encord ani Supervisely **nedokumentují nativní equirektangulární segmentaci**, a 8000×4000 přesahuje pohodlné pracovní rozlišení všech testovaných editorů. **Anotovat na gnómonických dlaždicích, ne na syrovém ERP** — projekce → anotace při 1024–2048 px v běžném nástroji → zpětná projekce masek. Tím je anotační reprezentace totožná s inferenční.

---

## 7. Přenos na české vesnické scény

**Taxonomická mezera je větší než vzhledová.** Cityscapes je pro DTM skoro nejhorší možná volba pro předtrénování — chybí mu *curb*, *manhole*, *driveway*, *road shoulder* i *utility pole*, tedy pět nejcennějších tříd.

| Selhání | Pokryto čím | Naše mezera |
|---|---|---|
| Nezpevněná / šotolinová vozovka | GOOSE terén; Vistas Sand/Terrain | **Žádný městský dataset neoznačuje zpevněnou vs. nezpevněnou** |
| Komunikace bez obrubníku, okraj přecházející do trávy | Vistas Road Shoulder / Road Side | **Nejslabší jednotlivý případ.** Modely z Cityscapes kladou hranici silnice a chodníku na obrubník, který tam není |
| Živé ploty | nic — všechny datasety říkají `Vegetation` | **Vlastní třída, nepřenositelná** |
| Drátěné a dřevěné ploty | obecný `Fence` | Tenké struktury; **empirický strop IoU 42,8–49,5** na Toronto-3D |
| Příkopy | GOOSE / off-road terén | Geometrický příznak ≫ fotometrický → argument pro fúzi 2D+3D |
| Sezónnost | GOOSE, WildDash 2 | Český listnatý výkyv je zásadní — **plánovat vícesezónní sběr** |

**Vícedatasetové trénování je druhá spolehlivá páka.** MSeg ukázal, že **naivní míchání datasetů nefunguje — kupuje se až sladěním taxonomií**. Pro nás je relevantní technika překrývajících se návěští (Bevandić & Šegvić), která řeší přesně náš problém: JVF `chodník` vs. Vistas Sidewalk + Curb + Curb Cut.

### 7.1 Český stav praxe — otevřená nika

- Metodika ČÚZK **vyžaduje a doporučuje přesně naše vstupy**. Kapitola 9: mobilní laserové skenování je „určena zejména pro mapování dat ZPS silnic II. a III. třídy". Kapitola 8.2 předepisuje odevzdat **LAS v S-JTSK/Bpv s intenzitou nebo RGB, panoramatické JPG plus XYZ a úhly vnější orientace středů snímků v ASCII, obličeje a SPZ rozmazané.** To *je* náš dataset.
- Atribut `způsob pořízení` u ZPS připouští `geodeticky – fotogrammetricky` a `geodeticky – pozemním laserovým skenováním`. Hodnota „AI" neexistuje — automatická extrakce se deklaruje pod senzorem, který data pořídil.
- Od pilotu 2024 **nelze dokumentaci nahrávat přes webové rozhraní, jen přes API**. Pipeline musí obsahovat JVF writer a API klienta.
- **Nenašel se jediný publikovaný český ani slovenský produkt, článek, práce nebo tendr, který by automaticky odvozoval objekty ZPS z mobilního mapování pomocí ML.** Je to příležitost, ale zároveň to znamená žádnou referenční implementaci a žádné publikované baseline.

---

## 8. Evaluace

Výstupem jsou **3D konstrukční linie na 0,14 m**. Plošné mIoU je pro to slabý zástupce. V tomto pořadí:

1. **Boundary IoU jako hlavní metrika.** Je výrazně citlivější na chyby hranic než plošné IoU a nepřepenalizuje malé objekty. Vykazovat při několika šířkách dilatace, které obklopí naši toleranci 14 cm převedenou na pixely.
2. **IoU po třídách, nikdy jen průměr.** Program stojí a padá s `plot` (strop 42,8–49,5 na nejbližší obdobě), `hranice dopravní stavby`, `zábradlí` a `čelo propustku` — vzácnými tenkými třídami, které průměr rozmělní.
3. **Panoptic Quality** pro počitatelné objekty (sloup, nosič, šachta, studna).
4. **Metrika, která rozhoduje o převzetí: 3D polohové RMSE extrahované linie proti nezávisle zaměřeným kontrolním bodům dle ČSN 01 3410**, vykázané jako m_xy a m_H proti limitům 0,14 / 0,12 m. Všechno výše je jen mezidiagnostika.
5. **Podíl topologicky platných ploch.** Model s 85 mIoU, který produkuje neuzavřené prstence, má nulovou hodnotu.

### 8.1 Férová evaluace na panoramatech

Prosté ERP mIoU **převáží póly**, což u nás znamená převážit oblohu a střechu vozidla — čísla to nafoukne. Vykazovat **mIoU vážené kosinem zeměpisné šířky** (plochou na kouli) vedle prostého. Dál:

- Maskovanou nadirovou čepičku **vyloučit ze všech metrik** a napsat to.
- Vykazovat metriky **omezené na φ ∈ [−55°, +45°]** jako provozně smysluplné číslo, celou kouli jako druhotné.
- **Testovat rotační stabilitu** perturbací ±5° v pitch a roll a vykázat rozptyl.
- Realistický cíl je **venkovní režim 19 tříd: 50–60 mIoU bez anotací nebo s málo anotacemi**, víc po dotrénování na vlastních datech. Neporovnávat s vnitřními datasety (60–79 mIoU na 8–13 třídách).

---

## 9. Doporučený postup

### Fáze 0 — rozhodnutí před psaním kódu (1–2 týdny)

- **Vyřešit komerční licenci Mapillary Vistas.** Tahle jediná odpověď mění ~6 tříd z „dostupné" na „vlastní" a zhruba zdvojnásobuje rozsah anotací.
- Právní posouzení **licence DINOv3** a **SAM License**. Rozhodnout DINOv3 vs. DINOv2 (Apache-2.0, −5 mIoU).
- **Přemapovat seznam tříd na skutečný katalog JVF** (§2). Zahodit `obrubník` a `dopravní značka` jako třídy, přidat `hranice dopravní stavby` + `typ` jako úlohu hranic, přidat atributovou hlavu `převažující povrch` a příznak `hranice jiného objektu`.
- Zmrazit ontologii sladěnou přes Vistas/WildDash/GOOSE technikou překrývajících se návěští.

### Fáze 1 — zero-shot baseline (1–2 týdny)

- Srovnat panoramata do vodorovné roviny z trajektorie, zamaskovat nadirovou čepičku.
- Postavit reprojekci a slučování logitů s kosinovými vahami (§4.3). Ověřit, že to projde tam a zpět.
- Pustit **OneFormer-ConvNeXt-L (váhy Mapillary Vistas)** a **EoMT-L** přes všech 1 503 panoramat.
- Ručně označit **10–20 panoramat jen jako validační sadu.** Vykázat kosinem vážené mIoU, IoU po třídách a Boundary IoU. Očekávat zhruba **44–50 mIoU** na přenositelné podmnožině a téměř nulu na vlastních třídách.
- **Výstup: poctivá matice schopností po třídách.** Ta řekne, kam přesně má jít anotační rozpočet.

### Fáze 2 — self-training a malá anotovaná sada (4–8 týdnů)

- Source-free doménová adaptace na všech 1 503 panoramatech. Očekávaný režim analogicky: **38,65 → ~50–55** zadarmo.
- **Ručně opravit 30–60 pseudo-anotovaných panoramat** (rovníkový pás, dlaždice ne syrové ERP), vybraných **margin samplingem na úrovni regionů**. Použít **propagaci přes SAM 2 podél trajektorie** (8,4×) a **klikyháky** pro tenké liniové prvky (10,8×).
- Dotrénovat se **zmrazeným DINOv2/v3 a ~1 % adaptérů**. **Neprovádět plné dotrénování.**
- Pustit semi-supervised trénink typu UniMatch přes anotovaný i neanotovaný díl.

### Fáze 3 — vlastní třídy a 3D (8–16 týdnů)

- **Syntetický ERP rendering** pro příkop/násep/zářez, opěrnou zeď a čelo propustku, pak few-shot dotrénování na reálných datech.
- Ruční anotace jen pro: materiály plotů, studnu, zábradlí vs. svodidlo, drobné stavby.
- **Fúze s katastrem a LPIS** pro zahradu a zemědělskou plochu.
- Postavit přenos 2D→3D: viditelnost přes z-buffer, vážená akumulace pravděpodobností, pak **naučená společná síť nad 2D evidencí a 3D geometrií**.
- Volitelně přepnout 2D model na **Trans4PASS+ / DATR / Deformable Mamba**, jakmile bude ≥100 anotovaných panoramat — sníží to inferenci z ~20 průchodů na panorama na jeden.

---

## 10. Proč segmentovat obrazy a promítat, a ne segmentovat mračno

**Obojí, fúzovaně — obraz jako nositel sémantiky, geometrie jako nositel přesnosti a topologie.**

1. **Obraz vyhrává v sémantice** ve scénách bohatých na vegetaci a atributy: GOOSE **2D 46,53 vs. 3D 34,32** mIoU. Drátěný plot je pro LiDAR skoro průhledný, živý plot je geometricky nerozlišitelný od keře, a `převažující povrch` (asfalt vs. dlažba vs. šotolina) **nemá žádný geometrický podpis**.
2. **Geometrie vyhrává v metrické přesnosti a topologii.** JVF žádá 3D uzavřené linie na 14 cm. Obrazy dávají značky, ne souřadnice.
3. **Naivní projekce prokazatelně nestačí.** Peters, Brenner a Schindler (ISPRS J. 2023) na 88 M anotovaných bodech a 2 205 orientovaných snímcích — tedy přesně na formátu, který ČÚZK předepisuje — píší doslova, že *„naivní segmentace v obrazovém prostoru a mapování výsledných značek na mračno nestačí, protože výsledek ovlivňují vizuální nejednoznačnosti, zbytkové kalibrační chyby atd."*, a že naučený vícepohledový přenos značek *„významně zlepšuje výkon různých segmentačních backbonů"*.
4. **Reprezentace jsou prokazatelně komplementární.** Sonata (CVPR 2025): ScanNet lineární sonda **Sonata 72,5, DINOv2 promítnutý 63,1, kombinace 76,4**.
5. **Destilace odstraní závislost na obrazech v inferenci.** 2DPASS destiluje 2D priory do 3D sítě při tréninku; při inferenci nejsou obrazy potřeba — cenné, protože rozmazání obličejů a SPZ je povinné.

> **Klíčové varování z benchmarků.** Na Toronto-3D dosahuje nejlepší metoda 82,9 mIoU, ale **Fence jen 49,5**, zatímco Road 96,1. Tenké, propustné, materiálově proměnlivé struktury — přesně ty objekty JVF s nejbohatší doménou atributů (`druh plotu`, 5 hodnot) — jsou úzkým hrdlem celého programu a **geometrie sama je nevyřeší.** To je nejsilnější argument pro zvolenou cestu přes obrazy.

Peters et al. zároveň jmenují **zbytkové kalibrační chyby** jako primární zdroj chyb přenosu značek. **Kalibrace boresightu kamery a LiDARu tedy sedí na rozpočtu přesnosti, nejen na rozpočtu sémantiky** — viz `02_obarveni_pointcloudu.md`.

---

## 11. Rizika, seřazená

1. **Licence Mapillary Vistas** — blokuje taxonomicky nejlepší zdroj. Řešit první.
2. **Vektorizace a topologie**, ne percepce, je místo, kde projekty tohoto tvaru nejčastěji selhávají. IS DTM dělá plochování; neuzavřené prstence nebo špatná hierarchie znamenají odmítnutou dodávku bez ohledu na mIoU.
3. **Tenké a propustné třídy** (`plot` ve všech pěti materiálech, `zábradlí`, `čelo propustku`). Empirický strop ~45–50 IoU.
4. **Rozpočet 14 cm** spotřebuje trajektorie + kalibrace boresightu + prokládání linií + generalizace.
5. **Sezónní generalizace** — 1 503 panoramat z jedné kampaně v jedné vesnici negeneralizuje přes roční období ani na jiné obce. Před slibem produktu naplánovat vícesezónní a vícelokalitní sběr.
6. **Žádný předchůdce v ČR/SR** — příležitost, ale zároveň žádná referenční implementace a žádné publikované baseline.

---

## 12. Experimenty na datech Dražkov

Sekce 1–11 jsou rešerše — čísla z jiných datasetů (Cityscapes, GOOSE, DensePASS, Toronto-3D…).
Tahle sekce testuje stejná tvrzení přímo na vzorku, který GEOVAP předal k nulté fázi (menší obec
Dražkov: 1503 panoramat Ladybug 8000×4000, `export.csv` s roll/pitch/yaw, 38 dlaždic LAZ terén/ostatní,
a hlavně kompletní atributovaný JVF export `1_ZPS_GAD.geojson` — 4342 objektů s reálnými 3D
souřadnicemi). Kód a podrobné výstupy: `geovap/experiments/` (`README.md` tam popisuje spuštění).

### 12.1 Skutečné rozložení JVF tříd v Dražkově

Na rozdíl od §5.2 (proxy odhad pokrytí veřejnými datasety) je tohle **přímé měření** na reálné obci:

| Instancí | JVF kód | Název |
|---|---|---|
| 1239 | `0100000304` | hranice dopravní stavby nebo plochy |
| 630 | `0100000162` | **plot** |
| 512 | `0100000300` | hranice stavby |
| 437 | `0100000299` | hranice budovy |
| 194 | `0100000051` | příkop, násep, zářez dopravní stavby |
| 166 | `0100000165` | stavebně upravený vjezd na pozemek |
| 56 | `0100000193` | **čelo propustku** |
| 17 | `0100000084` | studna na veřejném prostranství |
| 8 | `0100000199` | **zábradlí** |

Přesně ty vzácné/tenké třídy, o kterých §5.2 a §11 říkají, že je žádný veřejný dataset nepokrývá,
tady existují v reálné, geodeticky přesné 3D podobě — plná tabulka v `experiments/out/e1/coverage_by_class.txt`.

### 12.2 Klíčové zjištění: JVF vektory se dají promítnout zpět do panoramat — pseudo-GT zdarma

Tentýž kamerový model, ověřený v `02_obarveni_pointcloudu.md` §2.1 proti TerraScan RGB (ΔE medián
6,08), promítá i JVF linie/body zpět do panoramat. **3161/4342 objektů (72,8 %) má panorama do 20 m**
a vizuální kontrola (125 náhledů, `experiments/out/e1/overlays/`) potvrzuje, že projekce sedí přesně —
`hranice budovy` leží na patě fasády, `plot` na hraně oplocení, `čelo propustku` u kraje vozovky,
**`zábradlí` (jen 8 instancí v celé obci) sedí přesně na červeném kovovém zábradlí mostku**, `studna`
padá do zahrady za plotem. Tohle je druhý, nezávislý důkaz správnosti projekčního modelu — tentokrát
na sémantice, ne na barvě — a zároveň způsob, jak získat 2D pseudo-ground-truth bez ruční anotace
(vstup pro M2/M3/M4 v `01_plan.md`).

### 12.3 Zero-shot baseline (Mask2Former, Mapillary Vistas) na reálných panoramatech

Recept §4.3 (prstenec 8× yaw × pitch 0° + 4× pitch −45°, fúze softmaxů s kosinovým dobehem,
`grid_sample` na GPU) implementován a otestován na 40 panoramatech vybraných tak, aby pokryla
nejvíc JVF tříd z §12.2 (36/46 tříd). **3,43 s/panorama na RTX 3090 → odhad 86 minut pro celých
1503 panoramat** (§4.4 odhadoval "jednotky GPU-hodin" pro rychlejší backbony — potvrzeno, jen s
pomalejším Mask2Former Swin-L je to blíž horní hranici toho odhadu).

**Dvě nalezené a opravené chyby při implementaci** jsou samy o sobě poučení pro kohokoli, kdo bude
recept z §4.3 reprodukovat:

1. Špatné pořadí inverzní rotace v reprojekci nechalo ~11 % pixelů v operačně důležitém pásu
   φ∈[−55°,+45°] nepokrytých žádnou výsečí (viditelné jako tvrdé mezery mezi sousedními pohledy a
   25 % pixelů spadlých do náhodné výchozí třídy při fúzi). Po opravě nepokryto jen 0,76 %
   (numerický okraj).
2. **Vertikální (elevační) souřadnice se při zpětné fúzi výseče do ERP počítala se špatným
   znaménkem** — směr "nahoru ve světě" se zaměnil za řádek obrazu vysece (ten roste opačně, řádek 0
   = nahoře). Efekt: segmentace byla lokálně vzhůru nohama přesně v místech, kde do fúze přispívala
   nenulová elevace (tedy skoro všude mimo horizont) — na overlayi to vypadalo jako "obrys auta je
   jinde a zrcadlený", protože se do stejného pixelu skládaly dvě verze pravděpodobnosti z různých
   výsečí, jedna správně orientovaná, druhá obrácená. **Tuhle chybu odhalil až vizuální review
   uživatelem** na konkrétním snímku (bílá dodávka v příjezdové cestě, viz `experiments/out/e2/`
   frame 119 před/po) — ani test středu výseče, ani test čistě horizontálního posunu ji neodhalí,
   protože se projeví jen na bodech s nenulovým vertikálním posunem od středu pohledu. Test, který ji
   chytí: ověřit round-trip **mimostředového** bodu v obou osách zvlášť, ne jen support bod ve středu.

Obě chyby jsou teď opravené v `common/reproject.py`. **Kosinový dobeh k okraji výseče (§4.3) bez
explicitního testu pokrytí celé koule a bez mimostředového round-trip testu neodhalí ani jednu z
nich — potvrzující test typu "vyrenderuj mapu vah, zkontroluj mezery, a ověř mimostředový bod v obou
osách" patří do checklistu, ne jen vizuální kontrola jedné segmentované dlaždice.**

### 12.4 Matice schopností po třídách — poprvé na reálné geometrii, ne na proxy datasetu

Metoda: pás kolem projektované JVF linie/bodu o šířce podle tolerance 14 cm (§1, §8) v pixelech
dané vzdálenosti; podíl pixelů v pásu s odpovídající veřejnou třídou Vistas taxonomie
(`experiments/common/class_map.py`, rozšíření §5.2 o kódy skutečně přítomné v Dražkově):

| n | skóre | JVF třída → veřejná |
|---|---|---|
| 2 | 0,50 | přidružená plocha pozemní komunikace → road |
| 9 | 0,41 | hranice přírodního a polopřírodního objektu → terrain\|vegetation |
| 3 | 0,33 | ostatní zastřešená stavba → building |
| 3 | 0,31 | chodník → sidewalk |
| 120 | 0,27 | hranice dopravní stavby nebo plochy → road\|sidewalk\|curb |
| 21 | 0,27 | hranice udržované zeleně → vegetation |
| 50 | 0,22 | hranice budovy → building |
| 9 | 0,22 | budova → building |
| 23 | 0,14 | udržovaná plocha zeleně → vegetation |
| 12 | 0,11 | hranice zdi → wall |
| 176 | 0,04 | hranice stavby → building |
| 20 | 0,00 | hranice vodního díla → water |

(čísla po opravě obou chyb z §12.3 — proti první, chybné verzi se pořadí tříd mírně mění, ale
celkový obrázek je stejný: silnice/chodník/vegetace nejlépe, `hranice stavby` výrazně hůř než
`hranice budovy`, voda nulová.)

Tohle je **záměrně přísná metrika** (úzký pás na 14 cm, ne plošné mIoU) — nízká absolutní čísla
nejsou v rozporu s odhadem 44–50 mIoU v §9 (jiná, shovívavější metrika). Hodnota je v **relativním
pořadí a v tom, že jde vůbec měřit na pravé geometrii**: `hranice stavby` skóruje výrazně hůř než
`hranice budovy`, přestože obě mapujeme na `building` — JVF rozlišuje budovu od "ostatní stavby"
jemněji, než to umí veřejná taxonomie, a tohle rozlišení proxy datasety nemohly odhalit.
**202 ze 4342 objektů (~4,7 %) nemá žádný veřejný ekvivalent** — to je přímé, měřené číslo pro
rozsah vlastních tříd z §11, ne odhad.

### 12.5 Rotační citlivost na reálných datech (§3.3 revidováno)

Skutečné rozdělení roll/pitch pro všech 1503 snímků z `export.csv`:

| | medián \|x\| | p90 | p99 | max | podíl \|x\|>5° |
|---|---|---|---|---|---|
| roll | 1,07° | 2,07° | 3,95° | 8,66° | 0,33 % |
| pitch | 1,23° | 2,11° | 3,52° | 5,99° | 0,13 % |

SGAT4PASS (§3.3) měřilo dopad **umělé** perturbace ±5°. Na reálných datech tohohle vozu leží skoro
všechny snímky uvnitř ±3° a jen 0,1–0,3 % překračuje 5°. **Srovnání do vodorovné roviny zůstává
správný krok** (pořád "jeden převzorkovací průchod zadarmo"), ale očekávaný přínos bude pravděpodobně
menší, než ±5° ablace naznačuje, protože reálná data se ke krajní hodnotě z té ablace skoro nikdy
nepřiblíží. Přímé A/B srovnání (se/bez srovnání do horizontu, na podvzorku s nejvyšším náklonem)
neproběhlo v této iteraci — je to nejlevnější zbývající krok k dokončení, viz `experiments/README.md`.

### 12.6 Co z toho plyne pro plán

- **§12.2 (projekce JVF→panorama) je hotová infrastruktura pro M3** (`01_plan.md`) — funguje už teď,
  ne až po M2. Stojí za zvážení přesunout ji před M2 v pořadí prací, protože je levnější a dává
  okamžitě pseudo-GT pro cokoliv, co M2/M4 potřebují validovat.
- **Matice schopností po třídách (§12.4) je reprodukovatelná na celém vzorku** — chybí jen spustit
  E2/E3 přes všech 1503 panoramat (~86 min GPU čas) a zpřesnit odhad šířky pásu (aktuální je hrubý
  odhad úhlového rozlišení, ne kalibrovaný na skutečnou vzdálenost jako v `02_obarveni_pointcloudu.md`).
- **Riziko rotační citlivosti (§3.3) je pravděpodobně přeceněné** pro tenhle konkrétní vůz/kalibraci.

### 12.7 Dodatek (září 2026): okluze v projekci JVF a inverzní mapování

Balíček `mapping/` (viz `02_obarveni_pointcloudu.md` §13 a `mapping/README.md`) nahrazuje kreslení z E1/E3:

- **E1 a E3 kreslily bez okluze.** S hloubkovým panoramatem snímku (`mapping.vectors.project_polyline`) se ukazuje, že nemalá část JVF linií v „pokrytých" snímcích je z daného snímku zastíněná (plot za zemědělskými stroji, hranice budovy za živým plotem, paty zábradlí za obrubou). Skóre E3 v §12.4 je tedy kontaminované — pás okolo zastíněné linie model správně neoznačí a metrika to počítá jako chybu. E3 je třeba přepočítat s maskou viditelných úseků (`mapping.cli.render_frame --jvf` zapisuje masku i zastíněné části zvlášť).
- Dále opraveno proti E1: 3D segmenty se dělí po ≤ 0,25° (přímky jsou v ERP křivky), šev se rozděluje místo vynechání segmentu, šířka pásu se počítá z 0,14 m ve vzdálenosti každého vzorku (E3 měla jednu šířku na objekt podle těžiště).
- **Inverzní směr existuje**: `FrameProducts.point_at(u, v)` vrací id bodu ve store a `geometry.pano_to_world` bod na paprsku, takže 2D značky (masky, pravděpodobnosti) lze přenášet na body přímo, bez opakované projekce; `mapping.accumulate.LabelVote` je připravené rozhraní pro vážené hlasování podle §9 fáze 3 (tytéž váhy `V·cos ι·1/(1+(r/8)²)` jako u barvy).
- Snímky v zatáčkách (|dyaw/dt| > 8°/s) mají nevyřešený časový/orientační problém (02 §13.6) — pro pseudo-GT je zatím bezpečnější je vynechat.

---

## Zdroje

**Modely a backbony**
- [EoMT — Your ViT is Secretly an Image Segmentation Model, CVPR 2025](https://arxiv.org/abs/2503.19108) · [repo](https://github.com/tue-mps/eomt)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) · [OneFormer](https://github.com/SHI-Labs/OneFormer)
- [SegFormer](https://arxiv.org/abs/2105.15203) · [SegNeXt](https://arxiv.org/pdf/2209.08575) · [InternImage](https://github.com/OpenGVLab/InternImage)
- [DINOv3](https://arxiv.org/html/2508.10104v1) · [repo](https://github.com/facebookresearch/dinov3) · [licence](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
- [Rein — Stronger, Fewer & Superior, CVPR 2024](https://arxiv.org/abs/2312.04265) · [repo (GPL-3.0)](https://github.com/w1oves/Rein)
- [SAM](https://ar5iv.labs.arxiv.org/html/2304.02643) · [SAM 2](https://ar5iv.labs.arxiv.org/html/2408.00714) · [SAM 3](https://arxiv.org/pdf/2511.16719)
- [Boundary IoU, CVPR 2021](https://arxiv.org/abs/2103.16562)

**Panoramatická segmentace**
- [Panoramic Scene Analysis: A Survey, 2026](https://arxiv.org/abs/2606.27745)
- [360SFUDA, CVPR 2024 — ablace zorného úhlu](https://arxiv.org/html/2403.12505v2) · [360SFUDA++](https://arxiv.org/html/2404.16501v1)
- [Trans4PASS, CVPR 2022](https://arxiv.org/abs/2203.01452) · [Trans4PASS+](https://arxiv.org/abs/2207.11860) · [repo](https://github.com/jamycheung/Trans4PASS)
- [SGAT4PASS, IJCAI 2023 — rotační citlivost](https://arxiv.org/abs/2306.03403)
- [DATR, ICCV 2023](https://arxiv.org/abs/2308.05493) · [Deformable Mamba](https://arxiv.org/html/2411.16481v2)
- [OmniSAM, ICCV 2025](https://arxiv.org/abs/2503.07098) · [Open Panoramic Segmentation, ECCV 2024](https://arxiv.org/html/2407.02685v2)
- [Tangent Images, CVPR 2020](https://arxiv.org/abs/1912.09390) · [360MonoDepth](https://ar5iv.labs.arxiv.org/html/2111.15669)
- [Sekkat et al., Sci. Reports 2022 — planární vs. sférické konvoluce](https://pmc.ncbi.nlm.nih.gov/articles/PMC8942985/)
- [DensePASS](https://arxiv.org/abs/2108.06383) · [WildPASS](https://github.com/elnino9ykl/WildPASS)

**Datasety**
- [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) · [Research Use License 2019](https://www.mapillary.com/dataset/assets/mapillary-object-dataset-research-use-license-2019.pdf)
- [GOOSE — definice tříd](https://goose-dataset.de/docs/class-definitions/) · [GOOSE paper](https://arxiv.org/html/2310.16788)
- [Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/) · [BDD100K](https://github.com/bdd100k/bdd100k) · [IDD](https://idd.insaan.iiit.ac.in/dataset/details/) · [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/documentation.php)
- [Toronto-3D](https://arxiv.org/abs/2003.08284) · [SensatUrban](https://arxiv.org/abs/2201.04494)

**Anotační efektivita**
- [UniMatch](https://ar5iv.labs.arxiv.org/html/2208.09910) · [AllSpark](https://ar5iv.labs.arxiv.org/html/2403.01818) · [PixelPick](https://ar5iv.labs.arxiv.org/html/2104.06394) · [What's the Point](https://ar5iv.labs.arxiv.org/html/1506.02106)
- [MSeg](https://arxiv.org/abs/2112.13762) · [Překrývající se návěští](https://arxiv.org/abs/2108.11224)

**Přenos 2D→3D a 3D segmentace**
- [Peters, Brenner & Schindler — Multi-view label transfer for MLS, ISPRS J. 2023](https://doi.org/10.1016/j.isprsjprs.2023.05.018)
- [2DPASS](https://arxiv.org/abs/2207.04397) · [Point Transformer V3](https://arxiv.org/abs/2312.10035) · [Sonata, CVPR 2025](https://arxiv.org/html/2503.16429v1) · [Pointcept](https://github.com/Pointcept/Pointcept)

**České DTM**
- [Vyhláška 393/2020 Sb. — konsolidované znění](https://www.cuzk.gov.cz/DMVS/JVF-DTM/platne_zneni_se_zmenami_navrhy.aspx)
- [Metodika pořizování dat DTM](https://cuzk.gov.cz/getattachment/DMVS/Metodika/Metodika_porizovani_dat_DTM_final_signed.pdf.aspx?lang=cs-CZ) · [Metodika pro geodety v2.2](https://cuzk.gov.cz/DMVS/Metodika/Metodika_pro_geodety_k_aktualizaci_DTM_v2-2_final.aspx)
- [JVF DTM 1.4.3](https://cuzk.gov.cz/DMVS/JVF-DTM/JVF_DTM_1423_StrukturaFormatu.aspx) · [JVF DTM 1.5.0](https://cuzk.gov.cz/DMVS/JVF-DTM/Budouci-verze/JVF_DTM_150_StrukturaFormatu.aspx)
- [DTMwiki — přesnost](https://dtmwiki.cuzk.gov.cz/01_pravidla/01_zaklad/02_presnost) · [DTMwiki — plošné objekty ZPS](https://dtmwiki.cuzk.gov.cz/01_pravidla/01_zaklad/04_plosne_objekty_zps)

---

## Co ověřit před závazkem

1. **Komerční licence Mapillary Vistas** — kontaktovat Mapillary/Meta.
2. **Příloha č. 2 vyhlášky 393/2020 Sb.** — potvrdit čísla třídy přesnosti 3 z primárního textu (automatizované stažení bylo blokované).
3. **Kódy objektů JVF** byly přepsány z konsolidovaného znění *s vyznačenými návrhy změn*. Před psaním schématu validovat proti autoritativnímu XSD JVF 1.5.0.
4. **Peters et al. 2023 a Cao et al. 2025** (ISPRS J.) — paywall, čísla po třídách neověřena. Získat institucionální přístup.
