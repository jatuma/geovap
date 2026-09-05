# Promítání panoramat na mračno bodů

Jak z panoramatických snímků obarvit body mračna, jak ověřit, že projekce je správná, a jak tu samou cestu později použít pro sémantické značky.

Stav: **projekční model je ověřený na reálných datech** (viz §2). Zbytek dokumentu je návrh postupu opřený o rešerši a o pilotní měření.

---

## 1. Proč tohle děláme první

Cílem projektu není obarvit mračno — to už je obarvené. Cílem je promítnout na body **sémantické značky** z obrázků. Jenže u značek nemáme s čím výsledek porovnat: když projekce sedne o metr vedle, plot bude označený jako budova a nikdo si toho hned nevšimne.

RGB je jiný případ. TerraScan mračno obarvil **ze stejných panoramat**, takže uložené `red/green/blue` je referenční odpověď na přesně tu úlohu, kterou řešíme. Když naše projekce zreprodukuje TerraScanovo RGB, máme dokázáno, že geometrie sedí — a **tentýž kód se stejnou geometrií pak můžeme použít pro značky s důvěrou**.

RGB tedy není cíl. Je to testovací sonda, u které známe správnou odpověď.

> **Zásadní omezení, které je nutné pojmenovat hned.** Shoda s TerraScanem ověřuje náš *kód*, ne *absolutní geometrickou přesnost*. Chyba ze sešívání panoramat (§3.3) je společná oběma pipeline a ve srovnání se vyruší. Pro absolutní přesnost je potřeba nezávislá kontrola — vlícovací body nebo metoda přes vzájemnou informaci s intenzitou (§7.2).

---

## 2. Ověřený projekční model

Tohle už není návrh. Změřeno na dlaždici 037, 49 897 bodech a 44 panoramatech.

### 2.1 Výsledný vzorec

```python
dE = P.E - F.E          # P = bod z LAZ, F = řádek z export.csv
dN = P.N - F.N
dH = P.Z - F.H

# 1) srovnání do směru jízdy (yaw je matematický azimut, CCW od osy +Easting)
x =  dE*cos(yaw) + dN*sin(yaw)      # dopředu
y = -dE*sin(yaw) + dN*cos(yaw)      # doleva
z =  dH                             # nahoru

# 2) roll a pitch, OBĚ SE ZÁPORNÝM ZNAMÉNKEM
y, z = y*cos(-roll) - z*sin(-roll),  y*sin(-roll) + z*cos(-roll)
x, z = x*cos(-pitch) + z*sin(-pitch), -x*sin(-pitch) + z*cos(-pitch)

# 3) na equirektangulární pixel
az = degrees(atan2(y, x))
el = degrees(atan2(z, hypot(x, y)))
u  = (az % 360) / 360 * 8000        # šev je v azimutu = yaw
v  = (90 - el) / 180 * 4000         # v = 0 je ZENIT
```

### 2.2 Jak se každá konvence rozhodla

Konvence v datech nikde popsané nejsou (sloupce `Direction`, `Up` a `Omega/Phi/Kappa` jsou prázdné ve všech 1 503 řádcích), takže se určily hrubou silou proti uloženému RGB.

| Konvence | Varianty | Výsledek (medián ΔE) | Závěr |
|---|---|---|---|
| Svislá osa | v=0 zenit vs. nadir | **6,53** vs. 39–42 | Zenit, jednoznačné |
| Šev / offset azimutu | 0°, 90°, 180°, 270° | **6,53** / 18,66 / 18,32 / 13,81 | Offset 0, šev v azimutu = yaw |
| Zrcadlení | `az−yaw` vs. `−(az−yaw)` | **6,53** vs. 17,05 | Bez zrcadlení |
| Znaménka roll/pitch | 8 kombinací znamének a pořadí | **6,28** (obě záporná) vs. 6,53 (bez) vs. 7,29 (obě kladná) | Obě záporná |
| Pořadí rotací | roll→pitch vs. pitch→roll | 6,276 vs. 6,277 | Nerozlišitelné, úhly jsou malé |

Rozestupy jsou obrovské tam, kde na tom záleží (6,5 vs. 39 u svislé osy, 6,5 vs. 17 u zrcadlení), takže tyhle tři konvence jsou určené s jistotou. Znaménka roll/pitch mají slabší signál, protože |roll| ≤ 9° a |pitch| ≤ 6°, ale směr zlepšení je konzistentní.

### 2.3 Naměřená shoda

Ve výsledné konfiguraci, **bez jakéhokoli řešení okluzí**:

| Metrika | Hodnota |
|---|---|
| Medián ΔE (CIE76) | **6,08** |
| p25 / p75 | 3,46 / 11,27 |
| Podíl pod ΔE 5 | 41,0 % |
| Podíl pod ΔE 10 | 70,9 % |

Rozklad podle vzdálenosti bodu od kamery:

| Vzdálenost | n | Medián ΔE | Pod ΔE 10 |
|---|---|---|---|
| 3–4 m | 7 315 | **8,90** | 53,4 % |
| 4–6 m | 24 125 | **4,70** | 77,9 % |
| 6–10 m | 12 601 | 7,56 | 64,7 % |
| 10–15 m | 3 554 | 6,68 | 71,4 % |
| 15–25 m | 2 272 | 5,06 | 85,4 % |

Rozklad podle klasifikace:

| Třída | n | Medián ΔE |
|---|---|---|
| 1 (ostatní) | 42 237 | 6,13 |
| 2 (terén) | 7 660 | 5,91 |

> **Ověřená negativní hypotéza.** Standardní workflow Terrasolid obarvuje body terénu z **ortofota** a body nad terénem z panoramat — což by znamenalo, že třída 2 je nesrovnatelná. Test to vyvrací: třída 2 vychází dokonce mírně lépe. Obě třídy pocházejí ze stejných panoramat a referenční RGB je homogenní.

### 2.4 Rozklad chyby na geometrickou a radiometrickou složku

Klíčové měření: stratifikace ΔE podle **lokálního gradientu obrazu** v okolí 5×5 px kolem odečteného pixelu. V hladkých oblastech malé posunutí barvu nezmění, takže tam ΔE měří čistě radiometrii. Na hranách naopak měří čistě geometrii.

| Decil gradientu | Rozsah | n | Medián ΔE | Pod ΔE 5 |
|---|---|---|---|---|
| 1 (nejhladší) | 0,0–1,5 | 4 715 | **3,59** | 65,4 % |
| 2 | 1,5–2,4 | 5 239 | 3,69 | 65,2 % |
| 3 | 2,4–3,2 | 4 757 | 4,37 | 56,9 % |
| 4 | 3,2–4,0 | 5 073 | 5,55 | 44,2 % |
| 5 | 4,0–5,0 | 5 027 | 6,47 | 36,0 % |
| 6 | 5,0–6,0 | 5 123 | 6,98 | 31,3 % |
| 7 | 6,0–7,4 | 4 991 | 7,45 | 27,7 % |
| 8 | 7,4–9,2 | 4 940 | 7,62 | 25,2 % |
| 9 | 9,2–12,5 | 5 042 | 7,95 | 24,2 % |
| 10 (nejhranatější) | 12,5–56,3 | 4 990 | **9,34** | 19,5 % |

**Poměr nejvyšší/nejnižší decil = 2,64.** Monotónní růst je přesně podpis geometricky správné, ale ne dokonale seřízené projekce. Rozklad:

- **ΔE ≈ 3,5 je radiometrická podlaha** — TerraScanovo tónové mapování, průměrování přes footprint, JPEG artefakty. Bez znalosti jeho nastavení nesnížitelná.
- **Nárůst na 9,3 na hranách je geometrický** — a je to přesně ta složka, na kterou má zabírat řešení okluzí a doladění boresightu.

Tohle je referenční hodnota, proti které se měří každé zlepšení pipeline.

### 2.5 Globální barevná korekce nepomáhá

Proložení 3×3 matice + offsetu z našeho RGB na referenční (nejmenší čtverce, 49 897 bodů) posune medián ΔE jen z 6,28 na 6,17 — a v nejhladším decilu ho dokonce **zhorší** z 3,53 na 4,07.

Závěr: radiometrický rozdíl **není globální lineární transformace**. To odpovídá tomu, že TerraScan používá triangulované *color points* (.CPT) s prostorově proměnnou korekcí jasu a barevného vyvážení, případně měl zapnuté „Balance using intensity". Globální korekci tedy do pipeline nezařazovat — buď se nastavení TerraScanu dozvíme (§12), nebo se radiometrický rozdíl musí přijmout jako daný a měřit se jen relativní zlepšení.

### 2.6 Prostorové rozložení chyby — azimut a elevace

Rešerše předpovídala, že parallax ze sešívání (§3.3) se projeví jako **periodicita 60°** v relativním azimutu, protože Ladybug má šest objektivů. Test to **nepotvrdil**:

| Perioda | Amplituda | R² | Interpretace |
|---|---|---|---|
| 60° (švy objektivů) | 0,28 | **0,008** | Žádný signál — parallax ze švů se neprojevuje |
| 180° | 0,92 | 0,083 | Slabý |
| **360°** | **2,24** | **0,587** | **Dominantní** |

Naměřený profil je zhruba půl na půl: azimuty 60–195° mají medián ΔE 3,2–5,3, azimuty 195–315° mají 6,4–11,1.

Jednocyklická závislost na azimutu má dvě možná vysvětlení a je nutné je rozlišit:

- **Chyba lever armu** — příčné posunutí mezi počátkem skeneru a optickým středem kamery dává úhlovou chybu úměrnou sin(azimut). Tohle by šlo dofitovat.
- **Osvětlení** — přisvětlená versus protisvětlová strana ulice. Nedá se odstranit, jen popsat.

Rozliší je stratifikace podle gradientu z §2.4: **chyba lever armu poroste s gradientem obrazu, efekt osvětlení ne.** Tenhle test je potřeba udělat před jakýmkoli fitováním lever armu.

Rozklad podle elevace:

| Elevace | n | Medián ΔE |
|---|---|---|
| −90° až −30° | 29 253 | 5,79 |
| −30° až −15° | 13 879 | 6,15 |
| −15° až −5° | 5 289 | 6,13 |
| **−5° až +5°** | 860 | **16,15** |
| **+5° až +15°** | 367 | **15,90** |
| **+15° až +30°** | 222 | **15,30** |

Body **nad horizontem jsou 2,6× horší.** Je jich málo (1 449 z 49 897), ale jde přesně o koruny stromů, dráty a horní hrany fasád — tenké struktury, kde okluze a parallax bolí nejvíc a kde je zároveň největší šance, že bod „prosvítá" na oblohu. Pro sémantické značky je to varování: **tyhle třídy budou nejhorší** a je vhodné je vyhodnocovat zvlášť.

### 2.7 Co ještě zbývá vyřešit

**Boresight se barevnou metrikou doladit nedá.** Mřížka ±3° v azimutu × 0–6° v elevaci je extrémně plochá: medián ΔE jde z 6,28 v bodě (0°, 0°) na 6,08 v minimu, a to minimum utíká na okraj mřížky. Rozdíl 0,2 ΔE napříč šesti stupni znamená, že objektivní funkce nemá použitelné minimum.

Důvod je fyzikální: barva je lokálně hladká, takže malé úhlové posunutí barvu skoro nezmění všude kromě hran. Barevná odchylka spolehlivě odliší **správný model od špatného** (6 vs. 39), ale ne **dobře seřízený od mírně rozhozeného**.

Na boresight je proto potřeba jiná úloha — hranová nebo přes vzájemnou informaci, viz §7.2. Alternativně počítat posunovou křivku jen na bodech z **nejvyššího decilu gradientu** (§2.4), kde je citlivost 2,6× vyšší.

---

## 3. Rozpočet přesnosti

Bez těchhle čísel se nedá rozhodovat, co je ještě chyba a co už šum.

| Veličina | Hodnota | Odvození |
|---|---|---|
| Úhlové rozlišení panoramatu | **0,045 °/px** | 360° / 8000 px |
| Velikost pixelu na objektu ve 4,4 m | **3,5 mm** | 4,4 m × 7,854e-4 rad |
| Střední rozestup bodů při 382 b/m² | **51 mm** | 1/√382 |
| Bodů na jeden pixel ve 4,4 m | ~0,01 | mračno je ~15× hrubší než obraz |
| Chyba 1 cm v poloze ve 4,4 m | 0,13° = **2,9 px** | atan(0,01/4,4) |
| Boresight 0,1° ve 4,4 m | 7,7 mm = **2,2 px** | |
| Chyba časové synchronizace 10 ms při 30 km/h | 83 mm = **24 px** ve 4,4 m | |
| Parallax ze sešívání ve 4,4 m | ~0,5–0,6° = **11–13 px** | b·(1/r − 1/R), b≈50–60 mm, R=20 m |

Dva závěry, které z toho plynou přímo:

1. **Mračno je vůči obrazu silně podvzorkované.** Odečítat jeden pixel na bod zahazuje informaci — lepší je průměr přes malý footprint, což mimochodem dělá i TerraScan.
2. **Dominantní zdroje chyby ve 4,4 m jsou v pořadí:** časová synchronizace a boresight (desítky pixelů), parallax ze sešívání (~12 px), lever arm (neznámý). Interpolace barvy je proti tomu bezvýznamná.

### 3.3 Parallax ze sešívání — nejzávažnější nemodelovatelná chyba

Panorama z Ladybugu **není projekce z jednoho středu**. Teledyne to říká přímo: stitching algoritmus předpokládá, že všechny body scény leží ve stejné vzdálenosti od kamery, a **výchozí kalibrační poloměr je 20 metrů**. Pro objekty od 10 m do nekonečna je chyba pod 1 px. Náš medián je 4,4 m — hluboko v pásmu, kde ta záruka neplatí.

Úhlový posun objektu ve vzdálenosti *r*, když stitcher předpokládá *R*:

```
Δθ ≈ b · (1/r − 1/R)      b = odsazení objektivu od středu rigu ≈ 50–60 mm
```

| Vzdálenost | Δθ | px | na objektu |
|---|---|---|---|
| 2 m | 1,29° | 29 px | 4,5 cm |
| **4,4 m** | **0,51°** | **11 px** | 3,9 cm |
| 10 m | 0,14° | 3,2 px | 2,5 cm |
| 20 m | 0 | 0 | 0 |

**Tohle přesně vysvětluje naměřený profil v §2.3** — pásmo 3–4 m je nejhorší (ΔE 8,90), 15–25 m nejlepší (5,06). Rigidní boresight to neodstraní, protože chyba závisí na vzdálenosti i na azimutu vůči švům.

Ověřitelný podpis: v mapě reziduí v souřadnicích (azimut, elevace) se má objevit **šest svislých pruhů** v místech švů mezi objektivy. Pokud tam jsou, je to parallax, ne chyba projekce.

Jediné skutečné řešení je pracovat ze **šesti surových snímků** s vlastními extrinsiky místo ze sešitého panoramatu, případně přesešít s poloměrem koule ~5 m. Obojí vyžaduje data a součinnost, které zatím nemáme.

---

## 4. Co víme o referenčním RGB

Uložené RGB **není surový pixel z projekce**. Prošlo TerraScanovým zpracováním a to má několik nastavení, která výsledek mění:

| Nastavení TerraScanu | Dopad na referenci |
|---|---|
| **Footprint** | Barva se převzorkovává z kruhové oblasti kolem bodu, ne z jednoho pixelu (u mobilního mapování v pixelech) |
| **Use image** | Šest metod výběru snímku. Náš profil vzdáleností (medián 4,4 m) odpovídá „Mobile — closest in time" |
| **Balance using intensity** | Pokud bylo zapnuté, je uložená barva funkcí intenzity laseru a nikdy ji nezreprodukujeme |
| **Color points (.CPT)** | Triangulovaný model korekce jasu a barevného vyvážení — prostorově proměnná, neparametrická transformace |
| **Depth maps** | TerraScanův test okluze, s parametry Radius a Tolerance |
| **Use normal vectors** | Snímky, které vidí odvrácenou stranu plochy, se ignorují |

Zjištěné z dat: uložené RGB je **8bitové × 256** (max 51 456 = 201 × 256, `red % 256` má jedinou hodnotu). Maximum 8bitové složky je ~201, ne 255 — což naznačuje nějaké tónové mapování a je dalším důvodem, proč nečekat ΔE blízko nuly.

Praktický důsledek: **dokonalá geometrická reimplementace se s referencí stejně neshodne přesně.** Validace proto musí oddělit *geometrickou* shodu (kam paprsek dopadl) od *radiometrické* (jaká transformace se na barvu použila). Viz §7.3.

---

## 5. Okluze

Zatím neřešeno — a je to největší zbývající rezerva. Bod za zdí se momentálně obarví barvou zdi.

### 5.1 Doporučení: sférický z-buffer

Pro equirektangulární kameru je přirozenou strukturou **hloubkové panorama ve sníženém rozlišení**:

1. `depth = np.full((1000, 2000), inf, np.float32)` — 0,18°/px, tedy ~4 buňky na plný pixel
2. Scatter-min vzdáleností všech kandidátních bodů, každý bod splatnutý přes jádro o poloměru `k · rozestup / vzdálenost` (bod ve 2 m pokryje víc buněk než ve 30 m)
3. Bod je viditelný, pokud `r <= depth[buňka] · (1 + tol) + eps`, s `tol ≈ 0,02` a `eps ≈ 0,05 m`
4. Volitelně morfologické uzavření hloubkové mapy, aby pozadí neprosakovalo dírami

Dimenzování z našich dat: rozestup 51 mm ve 4,4 m odpovídá 0,66° = **14,8 px v plném rozlišení**. Mračno je tedy v plném rozlišení řidší než obraz a naivní z-buffer by byl děravý — proto podvzorkovaný buffer 2000×1000.

Parametry pro tento dataset:
- rozlišení bufferu **2000 × 1000**
- poloměr splatu `1,2 · 0,051 / r` rad, ořezaný na [1, 8] px
- tolerance `max(0,15 m; 0,03 · r)`
- maximální dosah **40–60 m**
- doplňkově zamítnutí odvrácených ploch přes normály (`n · (C − p) ≤ 0`), normály jednou na dlaždici z Open3D s k=20

### 5.2 Proč ne Hidden Point Removal

Katzův operátor (spherical flipping + konvexní obal) je klasika a Open3D ho má hotový jako `hidden_point_removal()`. Pro nás se ale nehodí:

- konvexní obal nad 6 M bodů × 1 503 pohledů je neúnosný
- Mehra et al. ukázali, že u zašuměných a nerovnoměrně vzorkovaných mračen optimální poloměr konverguje k mnohem větší hodnotě s **velkým počtem falešně pozitivních** — a MLS mračno je přesně ten případ
- jediný neprůhledný parametr `radius` proti z-bufferu, kde se dá ladit tolerance zvlášť

Ponechat jako kontrolní rameno ablace na malém vzorku, ne jako produkční řešení.

---

## 6. Multi-view fúze

Máme **29 jízdních průjezdů**, takže typický bod vidí 5–20 snímků. To je velká rezerva, kterou zatím nevyužíváme.

### 6.1 Výběr snímku

| Metoda | Kdy použít |
|---|---|
| **Nejbližší v čase** | Pro validaci proti TerraScanu — téměř jistě to, co použil on. Nejlevnější (binární vyhledávání v čase) |
| **Vážený výběr podle skóre** | Pro produkt |
| Nejbližší v 3D | **Nepoužívat naivně** — nejbližší kamera může být na fasádu pod 80°, tedy zkrácená a s mizerným rozlišením |

Skórovací funkce pro snímek *k* a bod *p*:

```
w_k = V_k · cos^a(ι) · 1/(1 + (r/r0)^b) · g(Δt)

V_k  ∈ {0,1}  viditelnost ze z-bufferu
ι             úhel dopadu mezi normálou a paprskem
r0 ≈ 8 m, a ≈ 1, b ≈ 2
```

### 6.2 Skládání barev

- **Pracovat v lineárním světle.** Průměrování v gama prostoru ztmavuje a posouvá odstín. Odgamovat, zprůměrovat, zagamovat zpět.
- **Medián místo průměru** přes top 3–5 snímků. Medián je zdarma filtr pohyblivých objektů: auto zaparkované v jednom průjezdu je přehlasováno zbylými 28.
- Zahazovat vzorky dál než ~2,5 MAD od mediánu.

### 6.3 Pohyblivé objekty

Auta a chodci jsou v obrazech, ale ne v mračnu — jejich barva se rozmaže na geometrii za nimi.

- Terrasolid to řeší **ručně** kreslenými výběrovými tvary. Neškáluje.
- **Mediánová fúze přes průjezdy** to řeší z velké části zdarma.
- **Sémantické maskování** — segmentační síť označí `car / person / bicycle`, ty pixely dostanou nulovou váhu. Tohle je zároveň infrastruktura pro fázi 2, takže se staví jednou.

---

## 7. Validační metodika

### 7.1 Dvě nezávislé evaluace

**E1 — shoda s TerraScanem.** Odpovídá na „dělám totéž co TerraScan?". Snadné, máme hotové.

**E2 — absolutní kontrola bez reference.** Odpovídá na „je moje projekce správně?". Tohle se běžně vynechává a je to přitom to důležitější, protože parallax ze sešívání se v E1 vyruší.

### 7.2 E2 — kontroly nezávislé na TerraScanu

1. **Vzájemná informace s intenzitou laseru.** Vyrenderovat z mračna panorama intenzity a maximalizovat normalizovanou vzájemnou informaci proti fotografii. Kanonická metoda (Pandey et al., AAAI 2012), zcela nezávislá na TerraScanově barvě — a na rozdíl od ΔE dává **ostré minimum**, takže z ní vypadne boresight. **Tohle je doporučená cesta pro §2.4.**
2. **Ruční vlícovací body.** 30–50 ostrých prvků (paty sloupů, rohy značek, obruby šachet) rozprostřených v azimutu, elevaci i vzdálenosti. Reziduum hlásit v pixelech i stupních. Tohle je číslo do zprávy pro zákazníka.
3. **Konzistence mezi průjezdy.** Obarvit tytéž body ze dvou různých průjezdů a porovnat. Dává dosažitelnou přesnost bez jakékoli externí reference — pokud se dva průjezdy shodnou jen na ΔE 8, nemá smysl čekat lepší shodu s TerraScanem.

### 7.3 E1 — oddělení geometrie od radiometrie

Tři nástroje, které je nutné použít, jinak čísla nic neznamenají:

1. **Nejdřív odhadnout a odstranit globální barevnou transformaci.** Robustní regrese (Huber, RANSAC) 3×3 matice + offset z našeho RGB na jejich. Reportovat ΔE **před i po**. Rozdíl mezi těmi dvěma čísly *je* velikost radiometrického zkreslení.
2. **Stratifikovat podle lokálního gradientu obrazu.** V nejnižším decilu gradientu je geometrický příspěvek téměř nulový, takže tam ΔE měří **čistě radiometrii**. V nejvyšším decilu měří **čistě geometrii**. Poměr těch dvou je nejčistší jednotlivá diagnostika, kterou lze postavit.
3. **Pořadová (Spearmanova) korelace** — imunní vůči jakékoli monotónní tónové křivce.

Vždy dále rozdělovat podle: klasifikace, vzdálenostního decilu, úhlu dopadu, azimutu (hledat periodicitu 60° = švy), `point_source_id` a indexu průjezdu.

### 7.4 Posunová křivka

Nejcennější jednotlivý experiment. Aplikovat umělý posun (du, dv) na odečítání a vykreslit medián ΔE jako funkci toho posunu, du, dv ∈ [−60, +60] px:

- Plocha musí mít **jasné minimum** a to minimum musí být v (0, 0)
- **Poloha minima přímo měří zbytkový boresight**: du px × 0,045° = chyba v yaw, dv px × 0,045° = chyba v pitch
- Počítat **zvlášť po vzdálenostních pásmech**: pokud du systematicky driftuje s 1/r, jde o **lever arm**; pokud je konstantní, jde o **boresight**; pokud se mění s azimutem s periodou 60°, je to **parallax**

Pozor: naše zkušenost z §2.4 říká, že u téhle úlohy je ta jáma velmi mělká. Křivku je proto lepší počítat na podmnožině bodů s **vysokým gradientem obrazu**, kde je citlivější.

### 7.5 Rychlé kontroly, které odhalí špatnou konvenci za minuty

První dvě nepotřebují mračno vůbec:

1. **Promítnout trajektorii samu do sebe.** Do snímku *i* promítnout středy kamer *i±1..i±20*. Musí ležet **blízko horizontu** ve směru dopředu a dozadu. Ověří znaménko yaw, offset švu i orientaci bez jakýchkoli externích dat.
2. **Promítnout nadir.** Bod 2 m přímo pod kamerou musí padnout na `v ≈ H`. Spolu s (1) určí svislou osu.
3. **Obarvit jednu dlaždici z jednoho snímku a podívat se.** Barva oblohy na fasádách → svislé překlopení. Zrcadlené nápisy → vodorovné překlopení. Scéna otočená o 90° → záměna azimutu a azimutu od severu.
4. **Test znaménka roll/pitch.** Omezit se na snímky s |roll| > 4°. Při obráceném znaménku tam ΔE výrazně vyskočí proti snímkům s |roll| < 1°.
5. **Sken časového offsetu.** Odečítat barvu s časy posunutými o Δt ∈ [−1, +1] s. Medián ΔE musí mít minimum v Δt = 0. Nenulové minimum odhalí **chybu časové synchronizace** — což při 30 km/h stojí 8 m na sekundu a nic jiného ze seznamu to nezachytí.

### 7.6 Metriky a jejich vykazování

- **CIEDE2000 (ΔE00)** jako hlavní metrika. *Pozor: naše pilotní čísla jsou CIE76* — jednodušší, perceptuálně nerovnoměrné, hodí se na třídění, ne do zprávy.
- Rozklad na **ΔL\*, ΔC\*ab, ΔH\*ab zvlášť.** Rozdíl jen v ΔL znamená expozici nebo gamu. Rozdíl v ΔH znamená vyvážení bílé nebo jiný snímek. Velké ΔE při malém ΔL a ΔH znamená, že jsme trefili **jiný objekt** — tedy geometrickou chybu.
- **Podíl hrubých chyb** `%(ΔE00 > 20)` — míra geometrických přešlapů, mnohem diagnostičtější než průměr.
- **Pokrytí** — jaký podíl bodů jsme obarvili proti tomu, kolik jich obarvil TerraScan.

**Robustní statistika, ne průměry.** Rozdělení reziduí je bimodální: úzké jádro (stejná plocha, malý radiometrický posun) plus těžký chvost (úplně jiná plocha — selhání okluze, pohyblivý objekt). Průměr je vážená kombinace „jak dobrý jsem obvykle" a „jak často katastrofálně selžu" a neodpovídá ani na jedno. Vykazovat **medián, MAD, percentily P25/50/75/90/95/99** a **histogram s logaritmickou osou**, aby byla bimodalita vidět.

### 7.7 Ablace

Zafixovat evaluační sadu a měnit jeden faktor:

| ID | Faktor | Ramena |
|---|---|---|
| A1 | Okluze | žádná / z-buffer (tol 0,01; 0,02; 0,05) / HPR |
| A2 | Výběr snímku | nejbližší v čase / v 3D / v XY / min. úhel dopadu |
| A3 | Fúze | jeden snímek / průměr k=3 / **medián k=3–5** / vážený průměr |
| A4 | Odečítání | nejbližší pixel / bilineárně / průměr přes footprint |
| A5 | Boresight | nula / dofitovaný (MI) |
| A6 | Práh úhlu dopadu | vypnuto / 85° / 80° / 70° |
| A7 | Dosah | 15 / 25 / 40 / ∞ m |

Očekávané chování jako kontrola vlastní instrumentace: **A1 má dramaticky srazit podíl ΔE > 20, ale mediánem skoro nehnout** (okluze je jev chvostu). Pokud A1 pohne mediánem hodně, je něco jiného špatně.

---

## 8. Známé pasti

1. **Rolling shutter — nehrozí.** Ladybug5+ má šest globálních závěrek Sony Pregius, Ladybug5 CCD. Zkreslení pohybem řádků odpadá.
2. **Časová synchronizace.** Největší riziko: 10 ms = 83 mm = 24 px ve 4,4 m při 30 km/h. Testovat sken z §7.5.5 a zkontrolovat, že obě strany jsou ve stejném GPS týdnu bez posunu o přestupné sekundy.
3. **Parallax ze sešívání** — §3.3.
4. **Rozmazání pohybem.** Při 30 km/h a expozici 5 ms se bod ve 4,4 m posune o 42 mm = 12 px. Počítat per-snímek ostrost (rozptyl Laplaciánu) a nejhorší decil vyloučit z fitování.
5. **Body oblohy a ptáci** — promítnou se na oblohu a dostanou sebejistě špatnou modrou. Filtrovat přes výšku nad terénem a hustotu.
6. **Přesvětlené pixely a retroreflexní plochy.** Značky a vodorovné značení se v obraze vysytí a v intenzitě laseru vyskočí. Vyloučit `(r==255)|(g==255)|(b==255)` a `r<=2`, podíl hlásit.
7. **Tečné dopady.** Při 80° se footprint protáhne 5,8×, při 85° 11,5×. Práh 75–80°, úhel ukládat, aby se dal přebírat dodatečně.
8. **JPEG artefakty.** Podvzorkování barvy 4:2:0 způsobí ΔE 2–5 na barevných hranách čistě z komprese. Neinterpretovat ΔE 2–3 u hran jako chybu projekce.
9. **TerraScan použil jinou politiku výběru snímku.** Pokud on „nejbližší v čase" a my „nejbližší v 3D", neshodneme se na 30–50 % bodů *z principu*, bez jakékoli geometrické chyby.

---

## 9. Nástroje

| Nástroj | Na co | Poznámka |
|---|---|---|
| **laspy + lazrs** | Čtení/zápis LAZ, extra dimenze | ~23 M bodů/s. Používat `chunk_iterator()`, ne `read()` |
| **PDAL** | Dlaždicování, reprojekce, COPC | **`filters.colorization` je pouze rastrový a pro tuhle úlohu nepoužitelný** — nemá kameru ani okluzi |
| **Open3D** | Normály, KD-stromy, HPR jako kontrola | `KDTreeFlann` je pomalejší než `scipy.cKDTree` |
| **OpenCV / PyTurboJPEG** | Dekódování JPEG, gradienty | 2–4× rychlejší než Pillow |
| **scikit-image** | `rgb2lab`, `deltaE_ciede2000`, `phase_cross_correlation` | Nepsat převod do Lab ručně |
| **Numba** | Scatter-min z-bufferu | `np.minimum.at` je notoricky pomalé |
| **CloudCompare** | Vizuální kontrola, obarvení podle ΔE | Obarvování z 360° snímků **nemá** implementované |

### Výkon

**Iterovat přes snímky, ne přes body.** Dekódované panorama má 96 MB; všech 1 503 je 144 GB, takže je nelze držet. Per-image gather dekóduje každý snímek právě jednou:

```
předdlaždicovat mračno podle trajektorie
pro každý snímek i:
    načíst panorama i
    vytáhnout body do R_max od kamery i
    transformovat, z-buffer, odečíst, akumulovat
    uvolnit panorama i
```

Akumulátory indexované globálním ID bodu: `best_score[N] float32`, `best_rgb[N,3] uint8`, `best_img[N] uint16` — pro 150 M bodů ~1,35 GB, vejde se.

Rozpočet: ~0,2 s na snímek (100 ms dekódování + 50 ms matematika + 30 ms z-buffer + 20 ms odečítání) × 1 503 ≈ **5 minut jednovláknově**, 1–2 minuty na 4–8 procesech. Dominovat bude I/O mračna, 10–30 minut. **Compute není potřeba přeoptimalizovávat.**

Praktické detaily, které pomůžou: odečítat rozdíl v float64 (souřadnice mají 7 číslic), pak lokální vektor přetypovat na float32; panoramata držet v `uint8`; načítání překrýt s výpočtem prefetch vláknem.

---

## 10. Ukládání výsledku

**Past v LAS 1.2:** pole `classification` má v LAS 1.1–1.3 jen **bity 0–4 pro třídu (0–31)**, bity 5–7 jsou příznaky Synthetic / Key-point / Withheld. Zápis hodnoty 64 do klasifikace v našem souboru tiše vyrobí třídu 0 se dvěma nastavenými příznaky. Plných 256 tříd má až **LAS 1.4 s formáty 6–10**, kde je uživatelský rozsah 64–255.

Doporučení: **migrovat na LAS 1.4 PF7** a dát hlavní značku do rozšířené klasifikace 64+, aby ji každý GIS ukázal ve standardním rozhraní, **a zároveň** nést experimentální značky a diagnostiku jako extra dimenze:

```python
las.add_extra_dims([
    laspy.ExtraBytesParams("sem_class", np.uint8,   description="sémantická třída"),
    laspy.ExtraBytesParams("src_image", np.uint16,  description="index zdrojového snímku"),
    laspy.ExtraBytesParams("col_conf",  np.uint8,   description="konfidence 0-255"),
    laspy.ExtraBytesParams("inc_angle", np.uint8,   description="úhel dopadu ve stupních"),
    laspy.ExtraBytesParams("dE00",      np.float32, description="ΔE2000 vs TerraScan"),
])
```

Pro web 3D klienta: **COPC** (jeden LAZ soubor s vestavěným oktree a HTTP range reads) místo EPT nebo Potree 1.x, které se rozpadnou na miliony souborů.

Do souboru přidat vlastní VLR s provenience — dofitovaný boresight, lever arm, časový offset, verze software, konvence škálování RGB.

---

## 11. Doporučený postup

**Fáze 0 — forenzní ohledání reference** (hotovo z velké části)
Ověřit škálování RGB (hotovo: 8 bit × 256), spočítat neobarvené body, zkontrolovat korelaci jasu s intenzitou (odhalí „Balance using intensity"), ověřit překryv GPS času.

**Fáze 1 — základní projekce** ✅ hotovo
Konvence určeny, medián ΔE 6,08 bez okluzí.

**Fáze 2 — diagnóza systematického posunu**
Vyrenderovat panoramata intenzity, fázová korelace proti fotografiím, sesbírat `(du, dv)` per snímek. Vynést `dv` proti azimutu — **hledat roll sinusoidu**. Vynést velikost rezidua proti vzdálenosti (ploché = boresight, 1/r = lever arm) a proti rychlosti vozidla (= časová synchronizace).

**Fáze 3 — fit**
Stratifikovaný podvzorek 200 k bodů vyvážený přes azimut, elevaci, vzdálenost a bloky snímků. Hrubá mřížka ±2° po 0,1° na MI objektivu, pak Powell. **Křížová validace na zadržených snímcích** a fit na 5 časových blocích zvlášť — pravý boresight je konstantní, driftující znamená problém s trajektorií.

**Fáze 4 — plná pipeline**
Z-buffer, footprint odečítání, práh dopadu 80°, R_max 40 m, mediánová fúze k=3–5, dofitovaný boresight. Vydat **dvě** mračna: (a) „nejbližší v čase, jeden snímek" pro validaci proti TerraScanu, (b) „medián top-5" jako produkt.

**Fáze 5 — ablace a zpráva** podle §7.7.

**Fáze 6 — sémantické značky**
Stejná geometrie, stejná viditelnost, stejné váhy. Změny jen tři:
1. **Nikdy neinterpolovat značky.** Buď nejbližší soused, nebo — lépe — odečítat celý vektor pravděpodobností tříd, fúzovat pravděpodobnosti a argmax udělat až na konci.
2. **Fúze = vážené hlasování** s vahami podle vzdálenosti, úhlu dopadu a viditelnosti.
3. **Použít segmentační model uzpůsobený panoramatům**, ne perspektivní model naslepo. Viz `03_semanticka_segmentace.md`.

---

## 12. Co si vyžádat od GEOVAPu

Tři levné dotazy, které smrsknou většinu zbývající nejistoty:

1. **Jaký poloměr sešívací koule byl použit?** Výchozí je 20 m; pro uliční scénu by mělo být ~5 m. Mění to model parallaxu a je to jednořádková odpověď.
2. **Jeden řádek `export.csv` s vyplněnými sloupci Direction/Up nebo Omega/Phi/Kappa.** Odstranilo by to veškerou zbylou nejednoznačnost v konvenci rotací. Pro ně pět minut práce.
3. **Screenshot nastavení TerraScanu „Extract color from images".** Řekne nám, která metoda výběru snímku, jaký footprint, jestli běželo „Balance using intensity" a jestli byly použité color points. Tím se z reverzního inženýrství stane kontrolovaný experiment.

Dále, pokud existuje: **surových šest snímků z Ladybugu** místo sešitých panoramat. To je jediná skutečná cesta, jak se zbavit parallaxu na krátkou vzdálenost.

---

## Zdroje

**Projekční model a Ladybug**
- [Overview of the Ladybug Image Stitching Process — Teledyne](https://www.teledynevisionsolutions.com/support/support-center/application-note/iis/overview-of-the-ladybug-image-stitching-process/)
- [Geometric Vision using Ladybug Cameras — Teledyne](https://www.teledynevisionsolutions.com/en-hk/support/support-center/application-note/iis/geometric-vision-using-ladybug-cameras/)
- [3D Geometry for Panoramic Images — John Lambert](https://johnwlambert.github.io/panos/)
- [Yaw, Pitch, Roll a Omega, Phi, Kappa — Pix4D](https://support.pix4d.com/hc/en-us/articles/202558969)
- [EPSG:5514 S-JTSK / Krovak East North](https://epsg.io/5514)

**TerraScan / TerraPhoto**
- [TerraScan: Extract color from images](https://terrasolid.com/guides/tscan/prjextractcolorfromimages.html)
- [TerraPhoto: Compute depth maps](https://terrasolid.com/guides/tphoto/mwcomputedepthmaps.html)
- [TerraPhoto: Process panoramic images](https://terrasolid.com/guides/tphoto/process-panoramic-images.html)
- [TerraPhoto: Color Points and Selection Shapes](https://terrasolid.com/guides/tphoto/cp.html)
- [TerraPhoto: Creating orthophotos and colored point clouds](https://terrasolid.com/guides/tphoto/prjwcreatingorthophotoscoloredpointclouds.html)

**Okluze**
- [Katz, Tal, Basri — Direct visibility of point sets, SIGGRAPH 2007](https://www.weizmann.ac.il/math/ronen/sites/math.ronen/files/uploads/katz_tal_basri_-_direct_visibility_of_point_sets.pdf)
- [Katz & Tal — On the Visibility of Point Clouds, ICCV 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Katz_On_the_Visibility_ICCV_2015_paper.pdf)
- [Mehra et al. — Visibility of Noisy Point Cloud Data, C&G 2010](http://vecg.cs.ucl.ac.uk/Projects/SmartGeometry/robustPointVisibility/paper_docs/VisibilityOfNoisyPointCloud_small.pdf)
- [Open3D PointCloud API](https://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html)

**Kalibrace**
- [Pandey et al. — Automatic Targetless Extrinsic Calibration by Maximizing Mutual Information, AAAI 2012](http://robots.engin.umich.edu/publications/gpandey-2012a.pdf)
- [Koide et al. — General, Single-shot, Target-less LiDAR-Camera Calibration, ICRA 2023](https://arxiv.org/abs/2302.05094) · [kód](https://github.com/koide3/direct_visual_lidar_calibration)
- [Reflectance Intensity Assisted Calibration of 3D LiDAR and Panoramic Camera](https://arxiv.org/pdf/1708.05514)
- [Li et al. — Registration of panoramic image sequence and MLS data, ISPRS J. 2018](https://www.sciencedirect.com/science/article/abs/pii/S0924271617303829)

**Obarvování mračen — prior art s čísly**
- [OmniColor, ICRA 2024](https://arxiv.org/html/2404.04693v1) · [kód](https://github.com/liubonan123/OmniColor/) — 0,475° / 3,06 cm
- [Registration of Vehicle-Borne Point Clouds and Panoramic Images, Sensors 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5422198/) — 0,10–0,20 m horizontálně
- [Cost Effective MMS for Color Point Cloud Reconstruction, Sensors 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7696296/) — barevná registrace 4,6 cm
- [LiDAR Point Cloud Colourisation Using Multi-Camera Fusion, Sensors 2025](https://arxiv.org/abs/2509.25859)
- [Evaluating the Quality of TLS Point Cloud Colorization, Remote Sensing 2020](https://www.mdpi.com/2072-4292/12/17/2748)

**Metriky**
- [skimage.color — rgb2lab, deltaE_ciede2000](https://scikit-image.org/docs/stable/api/skimage.color.html)
- [Delta E 101 — prahy](http://zschuessler.github.io/DeltaE/learn/)
- [Assessing objective quality metrics for point cloud coding](https://arxiv.org/pdf/2403.00410)

**Nástroje a formáty**
- [ASPRS LAS 1.4 R12](https://www.asprs.org/a/society/committees/standards/LAS_1_4_r12.pdf)
- [laspy — extra dimenze a VLR](https://laspy.readthedocs.io/en/latest/lessbasic.html)
- [PDAL — writers.copc](https://pdal.io/en/2.9.0/stages/writers.copc.html)
- [Colouring point clouds with PDAL — proč filters.colorization nestačí](https://www.spatialised.net/colouring-point-clouds-with-pdal/)
- [COPC — formát](https://www.cadinterop.com/en/formats/cloud-point/copc.html)
