# Automatizace DTM z mračna bodů — celkový plán

Projekt MULTIMA × GEOVAP. Cíl: nahradit ruční vektorizaci digitální technické mapy automatickým zpracováním dat z mobilního mapování.

| Dokument | Obsah |
|---|---|
| **01_plan.md** | tenhle — celkový plán, milníky, rizika |
| **02_obarveni_pointcloudu.md** | projekce obrazů na body, kalibrace, validace |
| **03_semanticka_segmentace.md** | modely, datasety, anotace, licence |

---

## 1. Zadání jednou větou

Pořizovatel dnes v mračnu bodů ručně vyhledává objekty a obtahuje jejich hrany liniemi s atributem z číselníku DTM. Zaučení trvá půl roku až rok. Chceme, aby co největší část téhle práce převzal stroj — a to tak, aby **oprava automatického výstupu nedala pořizovateli víc práce, než kdyby to nakreslil sám**.

## 2. Zvolená cesta

```
panoramata ──► sémantická segmentace ──┐
                                        ├──► fúze ──► body se značkami ──► vektorizace ──► JVF DTM
mračno bodů ──► geometrie, viditelnost ─┘
```

Segmentovat **obrazy** a promítnout výsledek na body, ne segmentovat mračno přímo. Důvod je měřitelný: na datasetu GOOSE dává 2D segmentace 46,53 mIoU proti 34,32 pro 3D. Drátěný plot je pro LiDAR skoro průhledný, živý plot je geometricky nerozlišitelný od keře, a povrch vozovky (asfalt / dlažba / šotolina) nemá žádný geometrický podpis. Naopak geometrie nese přesnost a topologii, které obraz dát nemůže.

Postup má proto dvě větve, které běží paralelně a potkají se ve fúzi.

---

## 3. Větev A — projekce a obarvování

**Proč první.** U sémantických značek nemáme s čím výsledek porovnat — když projekce sedne o metr vedle, plot se označí jako budova a nikdo si toho hned nevšimne. RGB je jiný případ: TerraScan mračno obarvil ze stejných panoramat, takže uložené `red/green/blue` je **referenční odpověď na přesně tu úlohu, kterou řešíme**.

RGB tedy není cíl. Je to testovací sonda se známou správnou odpovědí, která licencuje použití téhož kódu pro značky.

### Stav: projekční model ověřen ✅

Změřeno na dlaždici 037, 49 897 bodech, 44 panoramatech:

```
u = ((az − yaw) mod 360) / 360 × 8000     # az = atan2(dN, dE), šev v azimutu = yaw
v = (90 − el) / 180 × 4000                 # v = 0 v zenitu
roll a pitch se aplikují se ZÁPORNÝM znaménkem
```

| Metrika | Hodnota |
|---|---|
| Medián ΔE (CIE76) proti uloženému RGB | **6,08** |
| Podíl pod ΔE 10 | 70,9 % |
| Špatné varianty pro srovnání | zrcadlení 17,05 · nadir místo zenitu 39–42 |

A to **bez jakéhokoli řešení okluzí**. Rozestupy proti špatným variantám jsou tak velké, že o správnosti modelu není pochyb.

### Co z pilotu vyplynulo dál

- **Chyba se rozkládá na dvě složky.** V nejhladším decilu gradientu obrazu je medián ΔE 3,59, v nejhranatějším 9,34. Prvních ~3,5 je radiometrická podlaha (tónové mapování TerraScanu, JPEG), zbytek je geometrie. **Poměr 2,64 je referenční hodnota, proti které se měří každé zlepšení.**
- **Boresight se barevnou metrikou doladit nedá.** Mřížka ±3° je plochá — 6,28 v nule proti 6,08 v minimu. Na kalibraci je potřeba hranová metoda nebo vzájemná informace s intenzitou laseru.
- **Globální barevná korekce nepomáhá** (6,28 → 6,17) a v hladkých oblastech dokonce škodí. Radiometrický rozdíl není lineární transformace.
- **Parallax ze sešívání se nepotvrdil** jako periodicita 60° (R² = 0,008), zato je silný jednocyklický vzor v azimutu (R² = 0,587). Buď chyba lever armu, nebo osvětlení — rozliší to stratifikace podle gradientu.
- **Body nad horizontem jsou 2,6× horší** (ΔE 15–16 proti 5,8). Koruny stromů a dráty budou nejproblematičtější i pro značky.

### Zbývá

1. Okluze — sférický z-buffer 2000×1000. Největší nevyužitá rezerva.
2. Kalibrace boresightu přes vzájemnou informaci s intenzitou.
3. Multi-view fúze přes 29 průjezdů — mediánová, což zároveň zdarma filtruje zaparkovaná auta.
4. Rozhodnout, jestli je jednocyklická chyba lever arm nebo osvětlení.

Detaily v `02_obarveni_pointcloudu.md`.

---

## 4. Větev B — sémantická segmentace

### Zásadní korekce zadání

Rešerše odhalila, že seznam tříd, ze kterého jsme vycházeli, **neodpovídá tomu, co JVF DTM skutečně modeluje**:

- **`obrubník` jako objekt neexistuje.** Je to linie `hranice dopravní stavby nebo plochy` (`0100000304`) s atributem typu. Jde tedy o úlohu sémantické hranice, ne detekce objektu.
- **`dopravní značka` v DTM vůbec není** — je to samostatný produkt (pasport DZ).
- **`brána` neexistuje**, nejbližší je `stavebně upravený vjezd na pozemek`.
- **`plot` je linie**, ne plocha, s pěti materiály. A **živý plot je právně PLOT, ne vegetace** — přitom všechny veřejné datasety ho označují jako `Vegetation`.

Tohle je nutné vyřešit **před** jakoukoli anotací, jinak se anotuje špatná ontologie.

### Tři technická rozhodnutí, která rešerše rozhodla

1. **Zmrazený DINOv2/DINOv3 plus ~1 % adaptérů, ne plné dotrénování.** Měření: zmrazený DINOv2-L s nulou trénovatelných parametrů poráží plně dotrénovaný MAE-L o **18 mIoU**. S adaptéry 64,3 proti 61,7 pro plné dotrénování.
2. **Nepouštět model na syrové ERP.** Reprojekce na výseče 90° dává **+5,6 mIoU**. Naše rozlišení 8000×4000 = 22,2 px/°, takže výseč 90° se vyrenderuje jako 2000×2000 **nativně, bez převzorkování**. Reprojekce je zároveň strategie dlaždicování.
3. **Panoramata srovnat do vodorovné roviny z trajektorie.** Perturbace 5° v pitch/roll zvyšuje rozptyl mIoU zhruba 100×. Máme roll i pitch pro každý snímek — jeden převzorkovací průchod za několik mIoU.

### Nejvýznamnější riziko: licence

**Mapillary Vistas je taxonomicky nejblíž JVF DTM a jeho licence zakazuje komerční užití i deriváty.** Bez něj se ~6 tříd mění z „dostupné" na „nutno anotovat vlastní". Komerčně čistý korpus je tenký: GOOSE (CC BY-SA), BDD100K (BSD-3), SensatUrban (MIT) a vlastní data.

Detaily v `03_semanticka_segmentace.md`.

---

## 5. Milníky

| # | Milník | Trvá | Závisí na | Výstup |
|---|---|---|---|---|
| **M0** | Licenční a ontologická rozhodnutí | 1–2 týdny | — | Vyřešená licence Mapillary; přemapovaný seznam tříd na katalog JVF; zmrazená ontologie |
| **M1** | Projekce hotová a validovaná | 2–3 týdny | — | Z-buffer, kalibrovaný boresight, multi-view fúze; ablační tabulka; **cílově medián ΔE < 5 a podíl ΔE > 20 pod 5 %** |
| **M2** | Zero-shot baseline segmentace | 1–2 týdny | M0 | OneFormer a EoMT přes všech 1 503 panoramat; 10–20 ručně označených validačních panoramat; **matice schopností po třídách** |
| **M3** | Značky na bodech | 2–3 týdny | M1 + M2 | Táž geometrie jako M1, ale místo RGB se odečítá vektor pravděpodobností; vážené hlasování; argmax až na konci |
| **M4** | Dotrénovaný model | 4–8 týdnů | M2 | 30–60 opravených panoramat, aktivní učení, semi-supervised trénink; cíl 50–60 mIoU na venkovním režimu |
| **M5** | Vlastní třídy | 8–16 týdnů | M4 | Syntetický ERP rendering pro příkop/opěrnou zeď/čelo propustku; ruční anotace materiálů plotů; fúze s katastrem a LPIS pro zahrady |
| **M6** | Vektorizace a topologie | ? | M5 | **Největší riziko celého projektu** — viz §6 |

**M1 a M2 běží paralelně** a jsou na sobě nezávislé. M3 je spojuje.

---

## 6. Rizika, seřazená podle závažnosti

### 1. Vektorizace na 14 cm není vyřešená úloha

Cílová přesnost je právně závazná: **m_xy ≤ 0,14 m, m_H ≤ 0,12 m** (třída přesnosti 3), ověřovaná dle ČSN 01 3410.

Celá oblast automatické vektorizace HD map (MapTR, MapTRv2, PivotNet) se vyhodnocuje na Chamferových prazích **0,5 / 1,0 / 1,5 m** — tedy 3,5 až 10× hruběji, než potřebujeme. Hluboké polygonové metody vykazují IoU nebo Chamfer normalizovaný na úhlopříčku, ne metry. **Žádná automatická metoda v letech 2015–2026 nevykazuje 0,14 m ověřených proti nezávislému geodetickému měření.**

Nejlepší nalezené číslo — RMSE obrubníků 0,060–0,142 m — téměř jistě měří vnitřní konzistenci proti referenci digitalizované ze **stejného** mračna, ne absolutní přesnost.

Navíc samotná tolerance vektorizace spotřebuje **0,05–0,15 m** z rozpočtu, tedy skoro celý.

**Důsledek pro plánování: neslibovat plnou automatizaci.** Realistický cíl 2026 je systém „návrh a kontrola" — automatická klasifikace a návrh vektorů pokrývající možná 60–80 % délky linií za příznivých podmínek, s povinnou kontrolou operátora.

### 2. Tenké a propustné třídy

Na Toronto-3D dosahuje nejlepší metoda 82,9 mIoU celkem, ale **Fence jen 49,5**, zatímco Road 96,1. Ploty jsou přitom objekt JVF s nejbohatší doménou atributů (pět materiálů). Empirický strop ~45–50 IoU platí i pro zábradlí a čela propustků.

### 3. Dvě mezery bez jakékoli literatury

- **Automatická extrakce hran příkopů a terénních zlomů** — nenašla se jediná metoda s publikovanými čísly. Ne mezera v nástrojích, mezera ve výzkumu.
- **Ploty v geodetické přesnosti.**

Obojí rozpočtovat jako ruční práci.

### 4. Licence Mapillary Vistas

Viz §4.

### 5. Sezónní a lokalitní generalizace

1 503 panoramat z jedné kampaně v jedné vesnici negeneralizuje. Před slibem produktu naplánovat vícesezónní a vícelokalitní sběr.

### 6. Kalibrace sedí na rozpočtu přesnosti

Peters et al. (ISPRS J. 2023) jmenují zbytkové kalibrační chyby jako primární zdroj chyb přenosu značek. Kalibrace boresightu tedy není jen otázka sémantiky, ale přímo se odečítá ze 14 cm.

---

## 7. Co hraje pro nás

**Český právní rámec je nezvykle příznivý.** Metodika ČÚZK:

- **výslovně doporučuje mobilní laserové skenování** právě pro ZPS silnic II. a III. třídy — náš případ užití je to, co regulátor zamýšlel;
- připouští **jakýkoli postup**, který zajistí požadovanou kvalitu — přejímka je vázaná na QA (kontroly topologie, nezávislé kontrolní body, podpis ÚOZI), ne na to, jak geometrie vznikla;
- předepisuje odevzdat **LAS v S-JTSK/Bpv s RGB plus panoramata s vnější orientací v ASCII** — což *je* přesně náš dataset;
- zavádí pojem **„oblast se souvislou plošnou geometrií"**: úplná plošná topologie se validuje jen uvnitř vyznačené oblasti, mimo ni stačí validita jednotlivých polygonů. **Tím se z problému „všechno nebo nic" stává inkrementální rollout.**

Dále: **IS DTM dělá plochování sám.** Nemusíme produkovat polygony, jen topologicky uzavřené 3D konstrukční linie správného typu plus jeden definiční bod na plochu.

A konečně — **nenašel se jediný publikovaný český ani slovenský produkt, který by tohle dělal.** Je to otevřená nika, byť za cenu žádné referenční implementace.

---

## 8. Čtyři otázky na GEOVAP

Levné dotazy, které smrsknou většinu zbývající nejistoty:

1. **Jaký poloměr sešívací koule byl použit u Ladybugu?** Výchozí je 20 m, pro uliční scénu by mělo být ~5 m. Jednořádková odpověď, mění model parallaxu.
2. **Jeden řádek `export.csv` s vyplněnými sloupci Direction/Up nebo Omega/Phi/Kappa.** Odstranilo by veškerou zbylou nejednoznačnost v konvenci rotací. Pro ně pět minut.
3. **Nastavení TerraScanu „Extract color from images"** — která metoda výběru snímku, jaký footprint, běželo „Balance using intensity", byly použité color points? Změní to reverzní inženýrství na kontrolovaný experiment.
4. **Existují surové snímky ze šesti objektivů** místo sešitých panoramat? To je jediná skutečná cesta, jak se zbavit parallaxu na krátkou vzdálenost.

---

## 9. Co je realisticky dosažitelné

**Zralé, jde dnes:**
- Klasifikace bodů na 80–87 % mIoU u strukturálních tříd.
- Pseudo-anotace z panoramat jako akcelerátor anotace — řádové snížení ruční práce.
- Automatický návrh obrysů budov a městských obrubníků.
- Plně automatická konstrukce topologie, plochování a definiční body — tahle část je skutečně vyřešená (PostGIS, GRASS, CGAL, JTS).
- Export do JVF DTM.

**Výzkum, nelze slibovat:**
- Jakákoli automatická extrakce linií certifikovaná na 0,14 m proti nezávislému měření.
- Hrany příkopů a terénní zlomy.
- Ploty v geodetické přesnosti.
- End-to-end naučená vektorizace.

**Obchodní přínos není plná automatizace.** Je to změna geodetova dne z *kreslení všeho* na *kontrolu a opravu návrhu*. Pozicování produktů Trimble a Terrasolid implicitně potvrzuje, že právě tam jsou dnes peníze — obě firmy své nástroje pro extrakci prvků popisují výhradně jako poloautomatické a **žádná nepublikuje čísla přesnosti**.
