# Revize přístupu k umístění panoramat a mapování fotka ↔ body + nevyužitá data

Kritické čtení `06_umisteni_panoramat.md` / `mapping/`. První část říká, co je solidní a kde jsou
skutečné slabiny; druhá vyjmenovává data, která máme a nepoužíváme, a data, která by šla získat.

---

## Část I — Revize přístupu

### 1. Co je udělané dobře a nemá smysl to měnit

* **Obousměrnost přes jeden pár produktů.** Hloubka + `point_id` na snímek řeší okluzi, pano→svět
  i svět→pano jednou datovou strukturou. To je správná abstrakce a je levná.
* **Určení konvencí ablací.** Svislá osa, šev a zrcadlení mají rozestupy 6,5 vs. 39 / 17 — to není
  fit šumu, to je důkaz.
* **Oddělení geometrie od radiometrie** stratifikací podle gradientu obrazu. Bez toho by se
  optimalizovalo na TerraScanovo tónové mapování.
* **Regrese pilotu na dvě desetinná místa** a hash rigu v produktech. Reprodukovatelnost je zajištěná.
* **Závěr „identita je v rámci šumu správná"** je podepřený tím, že se nezlepšilo ani reziduum,
  ani hold-out. To je poctivé negativní zjištění, ne rezignace.

### 2. Slabiny, seřazené podle dopadu

#### 2.1 Póza se interpoluje mezi snímky vzdálenými ~5 m — a to je hlavní příčina problému v zatáčkách

`Poses.interp` je po částech lineární **mezi snímky**, ne po skutečné trajektorii. Při 30 km/h je
krok 0,6 s; v otočce na návsi se za 0,6 s změní yaw o 5–10°, tedy o **110–220 px**. Každý test
časového posunu (`dt_s`), každý sken Δt v `13.7` a každá interpolace pózy tak pracuje s
trajektorií, která je v zatáčkách hrubě aproximovaná. Není divu, že optimální Δt „kolísá od −4 do
+3,5 s" — hledá se korekce, kterou lineární interpolace nemůže vyjádřit.

**To se dá vyřešit bez součinnosti GEOVAPu** — viz §5.1: trajektorii lze rekonstruovat přímo
z mračna při ~500 Hz.

#### 2.2 Parallax ze sešívání se neřeší vůbec, přitom dominuje reziduu

11 px ve 4,4 m (§3.3 v `02_…`) je víc než dosažitelná přesnost edge-ICP (2–4 px). Model
`Δθ ≈ b·(1/r − 1/R)` je znám, ale nikde se neaplikuje ani nekalibruje. Přitom má **měřitelný
podpis**: šest svislých pruhů v mapě reziduí (azimut × elevace) — a ten se dosud nikdy nevykreslil.
Pokud tam jsou, dá se z reziduí edge-ICP nafitovat tabulka korekcí `Δu(azimut, 1/r)` se šesti
sedlovými body a odečítat ji při projekci. To je několik hodin práce s potenciálem srazit
geometrickou složku chyby o polovinu — víc, než přinesla celá kalibrace rigu.

**Zásadní důsledek pro §2.2 v `06`:** dokud parallax není modelovaný, je „boresight v rámci šumu"
tvrzení o šumu, který má systematickou strukturu. Edge-ICP průměruje přes všechny azimuty a
vzdálenosti a tím parallax rozmaže do rezidua — proto vyšla plochá cílová funkce.

#### 2.3 Časové okno ±45 s je náhražka za informaci, kterou v datech máme

Okno je heuristika, která má oddělit „body z tohoto průjezdu" od „bodů z jiných průjezdů".
Jenže `point_source_id` v LAZ **identifikuje průjezd přímo** (v dlaždici 011: 8 hodnot, 113–128,
dvě dominantní). Filtr `psid == psid(snímku)` je přesnější (žádné okno neselže na dvou průjezdech
30 s po sobě), levnější a nemá volný parametr. Okno navíc vytváří díry v renderech tam, kde daný
úsek nebyl v okně naskenován (13.6c) — s psid díry nevzniknou, protože se použije celý průjezd.

#### 2.4 Průjezdy nejsou vzájemně registrované a nic se s tím nedělá

13.6a to konstatuje („dvojité střechy, siluety se rozcházejí o stupně") a pipeline to jen obchází
časovým oknem. Přitom to kontaminuje **všechno navazující**: fúze barev napříč průjezdy,
značky segmentačního datasetu na hranách objektů, i budoucí vektorizaci hran (rozdvojená hrana
střechy je pro vektorizátor horší než chybějící hrana). Registrace průjezd-na-průjezd (ICP nad
rovinnými primitivy, po psid) je standardní úloha a mělo by být krokem 0 celé pipeline, ne
poznámkou. Bez ní se přesnost mračna sama drží na úrovni desítek centimetrů a všechny naše
2–4px ambice v obraze jsou akademické.

#### 2.5 Rastr `point_id` v 2000×1000 kvantuje zpětné mapování na 0,18°

Pro obarvování to nevadí. Pro **segmentační dataset a vektorizaci vadí**: hrana objektu se v ERP
masce určí s přesností jedné buňky z-bufferu, tedy 4 px plného rozlišení. Kalibrace už si kvůli
tomu staví vlastní 4000×2000 buffer (13.2 bod 3). Buď zvýšit rozlišení produktů, nebo ke každé
buňce ukládat i sub-buňkový offset vítězného bodu.

#### 2.6 Splat používá konstantní rozestup bodů 0,051 m

Skutečná hustota kolísá s dálkou, úhlem dopadu a počtem průjezdů přes daný povrch klidně o řád.
Konstanta znamená, že blízké husté povrchy se splatují zbytečně velkými disky (rozmazaná okluzní
hrana) a vzdálené řídké mají díry (falešné „viditelné" body za povrchem). Per-bodový rozestup
z k-NN je jednorázový předvýpočet nad store.

#### 2.7 Vegetace se v okluzi chová jako pevná stěna

Lidar prochází korunou (`number_of_returns` až 6), kamera ne. Bod za listím se dnes buď obarví
listem, nebo projde jako viditelný — podle náhody. `return_number` / `number_of_returns` v datech
jsou a nepoužívají se ani v okluzi, ani jako feature.

#### 2.8 Radiometrie mezi snímky se nikde nesjednocuje

Per-snímkový medián ΔE kolísá 3–13 podle osvětlení a expozice (13.3). Body viditelné ze dvou a
více snímků dávají **přeurčenou soustavu na per-snímkový zisk a bílý bod** — klasické radiometrické
vyrovnání, řádově tisíce neznámých, řeší se v minutách. Medián top-5 nesrovnalost jen maskuje,
neodstraní ji, a na švech mezi průjezdy je vidět.

#### 2.9 Validace stojí na referenci, o které víme, že je vadná

TerraScanovo RGB je v pomalých úsecích obarvené z jiných snímků (13.6). ΔE proti němu je tedy
metrika s vestavěnou podlahou i vestavěnými odlehlými hodnotami a v zatáčkách měří něco jiného
než v přímém úseku. Primární metrikou by měla být **hranová**: medián odchylky siluet mračna od
hran fotografie (`align.py` to už umí), případně skóre na ručně označených kontrolních bodech.
ΔE ponechat jako sekundární.

### 3. Co z revize plyne pro cíl projektu

Cílem není hezké obarvení, ale **anotace mračna z existující DTM a následná vektorizace**. Pro ten
cíl je pořadí důležitosti jiné, než jaké má dnešní pipeline:

1. registrace průjezdů (§2.4) — bez ní jsou hrany rozdvojené,
2. skutečná trajektorie (§2.1) — bez ní se nedá věřit projekci v zatáčkách, tj. v křižovatkách,
   což je přesně tam, kde je uliční čára nejsložitější,
3. parallax (§2.2) — určuje, jestli má smysl se v obraze bavit o jednotkách pixelů,
4. teprve pak barva a její fúze.

---

## Část II — Jaká další data použít

### 4. Co už máme v souborech a nepoužíváme

Vstupní LAZ je LAS 1.2 PF3 a nese víc, než store čte (`COLUMNS` = xyz, intensity, classification,
rgb, gps_time, psid, orig_index):

| Dimenze | Co v ní reálně je (ověřeno na dlaždici 011, 18,5 M bodů) | K čemu |
|---|---|---|
| `user_data` | **1 / 2, přesně 50 : 50** → identifikátor skenovací hlavy VMX-2HA | rekonstrukce trajektorie (§5.1), oddělení parallaxu obou hlav, kontrola vzájemné kalibrace hlav |
| `scan_angle_rank` | plný rozsah −128…127, ~1,02°/jednotka → úhel zrcadla | §5.1; úhel dopadu paprsku bez odhadu normály; detekce hran na hranici pásma |
| `number_of_returns` | 1 (92,5 %), 2 (7,2 %), 3–6 (0,3 %) | vegetace vs. pevný povrch — okluze (§2.7), feature pro segmentaci, „průhledné" body |
| `return_number` | 1–6 | totéž; poslední odraz = zem pod vegetací |
| `point_source_id` | 8 hodnot 113–128 = průjezd/mise | náhrada za časové okno (§2.3), klíč pro registraci průjezdů (§2.4) |
| `intensity` | 16bit s offsetem (min 10 922) | radiometrická kontrola nezávislá na kameře; TerraScan ji možná použil na „balance using intensity" |
| `classification` | jen 1 a 2 (ostatní/zem) | slabá, ale zdarma dostupná prior pro zem |

Dále v datovém adresáři leží **`vydej_zps_ref_0.jvf.xml` (21 MB)**, ze kterého se dnes používá jen
odvozený `1_ZPS_GAD.geojson`. Plný JVF nese atributy prvků, hierarchii, definiční body a metadata
o původu/přesnosti — tedy přesně to, co `seg.areas` dnes rekonstruuje heuristikou „třída podle
definičních bodů v 95 % plochy".

### 5. Data, která si můžeme vyrobit z toho, co máme

#### 5.1 Trajektorie a orientace přímo z mračna — ověřeno, funguje

Body jedné skenovací hlavy v okně 2 ms leží na rovině (rotující zrcadlo). Změřeno:

```
dlaždice 011, t = 301 680,908 s, okno ±2 ms
hlava 1: n = 991,  RMS od roviny 3,8 mm, normála (−0,821, 0,030, 0,571)
hlava 2: n = 1280, RMS od roviny 2,8 mm, normála (−0,455, −0,704, 0,546)
```

Dvě různé, vzájemně skloněné roviny = „butterfly" konfigurace VMX-2HA. Z toho plyne:

* **Orientace**: dvě normály v každém okamžiku určují úplnou 3-DoF orientaci senzorové hlavy
  (dvě nezávislé roviny), a to při ~500 Hz — proti 1 503 vzorkům z export.csv.
* **Poloha**: uvnitř roviny je směr na bod afinní funkcí `scan_angle_rank`, takže střed skenování
  se dá dopočítat proložením. Zkušební fit (soft-L1, hlava 2) dal počátek
  `(−642 438,29; −1 055 307,44; 224,78)` proti interpolované poloze kamery
  `(−642 436,83; −1 055 307,28; 225,56)` — **rozdíl 1,46 / 0,16 / 0,78 m**, což je přesně tvar
  ramene mezi hlavou skeneru a kamerou na stožáru (1,37 m dopředu, 0,78 m nahoru vůči yaw 26,8°).
  Reziduum naivního fitu bylo 0,62°, tj. metoda potřebuje dopilovat (robustní inicializace,
  společné řešení obou hlav, regularizace v čase), ale princip je potvrzený a nezávislý na export.csv.

Co tím získáme: hustou trajektorii pro interpolaci pózy (§2.1), nezávislou verifikaci konvence
roll/pitch/yaw (dnes odvozené jen z barvy!), přímé měření ramene kamera↔skener, a odpověď na
otázku „co je Timestamp a Yaw v zatáčkách", aniž bychom čekali na GEOVAP.

#### 5.2 Sférické stereo ze sousedních panoramat

Snímky jsou spouštěné po ~5 m, tedy s **5m bází**. Při 0,045°/px je to na 20 m paralaxa ~14°,
tedy stovky pixelů — pro ERP stereo velmi příznivý poměr. Hloubka z obrazu dává:
* nezávislou kontrolu registrace obraz↔mračno (bez TerraScanovy reference),
* přímé měření parallaxu ze sešívání (jeho podpis se v obrazové hloubce projeví jinak než v mračnu),
* hustou hloubku tam, kde má mračno díry (sklo, tmavé povrchy, zákryty),
* kandidáta na foto-konzistenční zpřesnění boresightu, které není ploché tak jako ΔE.

#### 5.3 Odvozené geometrické featury mračna

Normály, křivosti, planarita/linearita/sféricita v několika měřítkách, výška nad DTM (už je),
lokální hustota — standardní vstup pro klasifikaci bodů a jednorázový předvýpočet nad store.

### 6. Co si vyžádat od GEOVAPu (aktualizace §12 v `02_…`)

Beze změny platí: poloměr sešívací koule, jeden řádek export.csv s vyplněnými Omega/Phi/Kappa,
screenshot nastavení TerraScanu, surových šest snímků z Ladybugu. K tomu přibývá, seřazeno podle
poměru přínos/náklad na jejich straně:

1. **SBET / soubor trajektorie z POSPacu** (nebo jakýkoli export IMU+GNSS, ideálně 100–200 Hz).
   Odstraní §2.1 jedním souborem a je to standardní artefakt každé MMS zakázky — s velkou
   pravděpodobností existuje. Toto je dnes nejlevnější velká výhra.
2. **Kalibrační protokol RIEGL VMX-2HA** — vzájemná kalibrace obou hlav, ramena a boresight kamery
   vůči IMU. Ověří §5.1 a udělá z odhadů měření.
3. **Protokol kontroly kvality / hlášení o přesnosti** — RMS na kontrolních bodech, výsledek
   vyrovnání průjezdů (jestli vůbec proběhlo). Rozhodne, jestli §2.4 je náš problém, nebo jejich.
4. **Původní LAS před řezáním na dlaždice** — s plnými dimenzemi a možná vyšším point formatem
   (dnešní PF3 nemá `scanner_channel` ani přesný `scan_angle`, PF6+ ano).
5. **Páry mračno + hotová DTM z vln DTM1/DTM2** (už je v zadání) — to je hlavní zdroj anotací a
   ostatní body jsou proti němu detail.

### 7. Externí veřejná data

| Zdroj | Co dá | Poznámka |
|---|---|---|
| **ČÚZK ortofoto** (WMS/WMTS, 0,2 m) | barva zemi a střech tam, kde ji panorama vidí pod tečným úhlem; nezávislá kontrola polohy v půdorysu | jiné datum pořízení než mračno |
| **ČÚZK DMR 5G / DMP 1G** | nezávislá výšková reference — odhalí výškový posun jednotlivých průjezdů | přesnost ~0,18 m v terénu |
| **RÚIAN / katastr — obvody budov** | třída „budova" zdarma a absolutní kontrola polohy v E/N | obvod katastru ≠ obvod střechy |
| **ZABAGED, DTM krajů** | komunikace, vodní prvky, ploty jako slabá anotace | různé stáří a přesnost |
| **OSM** | osy komunikací, chodníky | jen pro hrubý kontext |

Používat je jako **slabou supervizi a kontrolu**, ne jako pravdu — každý má jiné stáří a jiný
model reality (13.6 ukazuje, co udělá jedna nekonzistentní reference s celou metrikou).

### 8. Externí datasety pro předtrénování a benchmark

* **Mračna z mobilního mapování s anotací**: Toronto-3D, Paris-Lille-3D, KITTI-360, SemanticKITTI,
  nuScenes-lidarseg — pro předtrénování backbonu pro klasifikaci bodů; třídy jsou hrubší než DTM,
  ale geometrie nosiče (auto, ulice) sedí.
* **Letecká / UAV mračna**: DALES, SensatUrban, Hessigheim — pro budoucí větev „dron" ze zadání.
* **Equirektangulární segmentace**: Stanford2D3D-pano, WildPASS/DensePASS, Matterport — pro
  doladění ERP modelů z `05_benchmark_segmentace.md`, které dnes běží zero-shot na perspektivních
  vahách; jsou to jediné veřejné sady se stejnou deformací obrazu, jakou máme my.
* **Street-level RGB**: Mapillary Vistas / Cityscapes — bohatá taxonomie pro pseudo-značky
  v panoramatech, ale perspektivní; nutná gnómonická projekce (`seg.views` už ji dělá).

---

## 9. Doporučené pořadí kroků

| # | Krok | Náklad | Přínos |
|---|---|---|---|
| 1 | Vyžádat SBET/trajektorii a kalibrační protokol | e-mail | odstraní §2.1 a §2.2 nejistotu u ramen |
| 2 | Nahradit časové okno filtrem `point_source_id` | ~půl dne | přesnější okluze, bez děr, bez parametru |
| 3 | Vykreslit mapu reziduí edge-ICP v (azimut × elevace) | ~půl dne | rozhodne, jestli je parallax ze švů reálný |
| 4 | Registrace průjezdů přes psid | dny | podmínka pro vektorizaci hran |
| 5 | Rekonstrukce trajektorie z rovin skenu (§5.1) | dny | nezávislá kontrola i náhrada, když SBET nepřijde |
| 6 | Radiometrické vyrovnání snímků (§2.8) | dny | odstraní švy mezi průjezdy v produktu |
| 7 | Doplnit `user_data` / návraty / scan angle do store a featur | ~den | okluze ve vegetaci, vstupy pro klasifikaci |
| 8 | Sférické stereo (§5.2) | týden+ | až po 1–5, jako nezávislá verifikace |
