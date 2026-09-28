# Umístění a orientace panoramat ve 3D a mapování fotka ↔ body

Popis toho, co pipeline `mapping/` skutečně dělá: odkud se bere poloha a natočení každého
panoramatu ve světových souřadnicích a jak se obousměrně převádí mezi pixelem panoramatu
a bodem mračna. Čísla a konvence jsou ověřené (viz `02_obarveni_pointcloudu.md` §2).

---

## 1. Vstupní data

| Zdroj | Obsah |
|---|---|
| `LB5, Camera Ladybug/export.csv` | 1 503 řádků = 1 503 panoramat: `Timestamp`, `Filename`, `Origin(E,N,H)`, `Roll(X)`, `Pitch(Y)`, `Yaw(Z)` ve stupních |
| `*.jpg` | equirektangulární panorama 8000 × 4000 px (celá koule, 0,045°/px) |
| `LAZ_Dražkov_ground/*.laz` | mračno, 584,8 M bodů, S-JTSK (E, N, H) + `gps_time` + třída + intenzita |

Sloupce `Direction`, `Up` a `Omega/Phi/Kappa` jsou v export.csv **prázdné ve všech řádcích** —
orientace je tedy dána výhradně trojicí roll/pitch/yaw a jejich konvence se musela určit
hrubou silou proti uloženému RGB (§3).

Časová báze je společná: `Timestamp` je sekunda GPS týdne, stejně jako `gps_time` v LAZ.
To je jediná vazba, která říká, *kdy* byl bod naskenován vzhledem k expozici snímku, a používá
se dvakrát — pro výběr „nejbližšího snímku v čase" a pro časové okno okluzí.

---

## 2. Umístění a orientace snímku (`poses.py`, `geometry.py`, `rig.py`)

### 2.1 Střed kamery

`C = Origin(E, N, H)` z export.csv, doplněný o **lever arm** rigu (offset středu kamery vůči
referenčnímu bodu trajektorie, v osách vozidla):

```
C = Origin + R_v^T · lever_arm      (lever_arm = (0,0,0) pro identitu)
```

### 2.2 Natočení

Rotace svět → vozidlo, se znaménkovou konvencí ověřenou měřením:

```
y = yaw, r = −roll, p = −pitch                       (POZOR na záporná znaménka)
Ry = [[ cy, sy, 0], [−sy, cy, 0], [0, 0, 1]]         yaw = matematický azimut, CCW od +Easting
Rr = Rx(r)                                            roll kolem osy x (dopředu)
Rp = Ry(p)                                            pitch kolem osy y (doleva)
R_v = Rp · Rr · Ry                                    svět → osy vozidla (x vpřed, y vlevo, z nahoru)
```

Nad tím je **boresight** rigu (konstantní rotace vozidlo → kamera):

```
R_b = Rz(κ) · Ry(φ) · Rx(ω)
R   = R_b · R_v                                       svět → osy kamery
```

Rig (`rig.py`) je tedy trojice: boresight (ω, φ, κ), lever arm (x, y, z) a časový posun `dt_s`
(snímek byl exponován v `t_csv + dt_s`; pro `dt ≠ 0` se póza lineárně interpoluje po trajektorii
uvnitř téhož průjezdu). **Provozně je rig identita** — edge-ICP kalibrace na 117 snímcích dala
boresight (0,07°, −0,01°, −0,03°), lever arm ≤ 2 cm, dt ≈ 1 ms, což je v rámci šumu měření nula.

Trajektorie je rozsekaná na **30 průjezdů** (mezera v čase > 5 s nebo skok > 15 m); interpolace
pózy nikdy nepřechází přes hranici průjezdu.

`FrameIndex` (`frame_select.py`) předpočítá pro všech M snímků `R[M,3,3]` a `C[M,3]` a drží
2D KD-strom středů kamer — z toho plyne „které snímky vidí tento bod / tuto dlaždici"
(poloměr `R_MAX` = 40 m) a `nearest_in_time(gps_time)`.

---

## 3. Mapování bod → pixel (dopředná projekce)

`geometry.world_to_pano(P, R, C)`:

```
x_cam = R · (P − C)                       odečtení v float64, pak float32 (chyba ~1e−3 px na 40 m)
az    = atan2(y, x)                       azimut v osách kamery
el    = atan2(z, hypot(x, y))             elevace
u     = (az mod 360) / 360 · 8000         šev u = 0 leží v azimutu = yaw
v     = (90 − el) / 180 · 4000            v = 0 je ZENIT
r     = |x_cam|                            euklidovská vzdálenost (dálka podle paprsku)
```

Konvence byly určeny ablací proti referenčnímu RGB z TerraScanu (medián ΔE CIE76):

| Konvence | Varianty | Výsledek | Závěr |
|---|---|---|---|
| svislá osa | zenit vs. nadir | **6,53** vs. 39–42 | zenit |
| offset azimutu | 0/90/180/270° | **6,53** / 18,7 / 18,3 / 13,8 | 0 (šev = yaw) |
| zrcadlení | ano/ne | 17,05 vs. **6,53** | bez zrcadlení |
| znaménka roll/pitch | 8 kombinací | **6,28** (obě −) | obě záporná |
| pořadí rotací | roll→pitch vs. opačně | 6,276 vs. 6,277 | nerozlišitelné |

Rozestupy jsou tak velké, že první tři konvence jsou jisté. Výsledná shoda s referencí bez
řešení okluzí: medián ΔE 6,08; stratifikace podle gradientu obrazu 3,59 (hladké) → 9,34 (hrany),
tj. ~3,5 je radiometrická podlaha a zbytek je geometrie.

Filtry, které se na projekci vždy váží: `R_MIN = 1 m` (blíž je parallax mezi skenerem a kamerou),
`R_MAX = 40 m`, maska vozidla (~25 % obrazu: kapota, střecha, skenery, černá čepička, + okraj 10 px).

---

## 4. Mapování pixel → bod (zpětná projekce)

Zpětný převod potřebuje hloubku, protože pixel je jen paprsek. Proto se pro **každý snímek**
předpočítá dvojice rastrů (`products.py`, 1 503 souborů, ~17 GB):

* `depth_mm` — uint16 [1000 × 2000], euklidovská vzdálenost v mm, 0 = prázdno
* `point_id` — uint32 [1000 × 2000], globální index bodu ve store, `0xFFFFFFFF` = prázdno

Stavba (`zbuffer.splat`): body do 40 m od kamery se promítnou dopřednou projekcí a scatter-min
splatují do sférického z-bufferu 2000 × 1000. Poloměr splatu = `1,2 · 0,051 m / r` v radiánech
(clamp 1–8 px), s roztažením v `u` o `1/cos(el)` u pólů. Vyhrává nejbližší dálka; do buňky se
zapíše i **id vítězného bodu** — to je právě ta tabulka pixel → bod.

Do z-bufferu vstupují jen body naskenované v okně **±45 s** kolem času snímku: mračno je
sjednocení 29 průjezdů, fotka je jeden okamžik, takže brány, zaparkovaná auta a lidé z jiných
průjezdů nesmí zastiňovat.

Použití:

```python
fp  = FrameProducts.load(k)
xyz = geometry.pano_to_world(u, v, fp.range_at(u, v), R, C)   # pixel -> souřadnice ve světě
pid = fp.point_at(u, v)                                        # pixel -> globální id bodu
ti, row = store.locate(pid)                                    # id -> (dlaždice, řádek v LAZ)
```

Paprsek samotný: `d_cam = (cos el·cos az, cos el·sin az, sin el)` z pixelu, `d_world = R^T · d_cam`,
`P = C + r · d_world`. Vzorkování rastru je v souřadnicích plného rozlišení, škáluje se
faktorem 2000/8000 = 0,25 (`FrameProducts.cell`); `u` se wrapuje, `v` clampuje.

Globální id bodu je prostě `row_offset(dlaždice) + lokální řádek`, takže je stabilní přes celé
mračno a `store.locate()` ho jednoznačně vrátí zpět na (dlaždici, řádek).

---

## 5. Viditelnost (okluze)

Bod promítnutý do pixelu je viditelný, když jeho dálka nepřesahuje hloubku v buňce o toleranci:

```
viditelný  ⟺  r ≤ depth_closed[cell] + tol
tol = max( 0,15 m ; 0,03·r ; min(1,0·lokální rozptyl hloubky, 2 m) ; min(0,12/ sin|el| , 2 m) )
```

* `depth_closed` = z-buffer po uzavření malých děr (grey opening dálky).
* Člen s lokálním rozptylem 3×3 a člen s `1/sin|el|` řeší **zem pod tečným úhlem** — tam se
  hloubka mění o ~0,5 m na buňku a bez toho by se země „zastiňovala sama".
* Prázdná buňka = neviditelný.

Efekt na celém mračnu: běh bez časového okna (`identity`) — medián ΔE00 4,98, pokrytí 82 %;
s časovým oknem a tolerancemi (`tw45`) — **medián 4,93, pokrytí 94,1 %**.

---

## 6. Dva klienti nad stejnou geometrií

**cloud → pano** (`render.py`, `vectors.py`, `seg/`): atributy bodů (RGB, třída, intenzita, výška)
a JVF vektory se rendrují do panoramatu; okluze je zadarmo, protože `point_id` rastr už drží
vítěze. Tímhle vznikají i ERP masky segmentačního datasetu a 16 gnómonických výsečí 1024².

**pano → cloud** (`colorize.py`): po dlaždicích se pro každý kandidátní snímek promítnou body
dlaždice, otestuje viditelnost, navzorkuje obraz (nearest / bilineárně / 3×3 footprint) a plní
tři akumulátory:

* **produkt** — medián top-5 vzorků, skóre `1/(1+(r/8)²)` × |cos dopadu|, bez saturovaných pixelů;
* **nejbližší v čase s okluzí** — validace proti TerraScanu;
* **nejbližší v čase bez okluze** — reprodukce pilotu (medián CIE76 6,08 na dvě desetinná místa).

Výstup je LAS 1.4 PF7 s původními XYZ/časem/třídou bitově shodnými, `red/green/blue` = náš produkt
a extra dimenzemi `ref_*`, `nt_*`, `dE00_*`, `src_image`, `n_views`, `col_conf`, `cam_dist`,
`inc_angle`, `img_grad` + VLR `geovap_map` s proveniencí (mj. hash rigu).

---

## 7. Co je otevřené

* **Zatáčky**: snímky s |dyaw/dt| > 8°/s mají ΔE 10–15 a jejich optimální Δt kolísá od −4 do
  +3,5 s (není konstantní). Chování pózy při otáčení vyžaduje dotaz na GEOVAP — co přesně
  spouští expozici vůči zápisu pózy a jak je definován Yaw. Na rovných úsecích je vše v pořádku.
* **Parallax ze sešívání** Ladybugu je nemodelovatelná složka chyby (§3.3 v `02_…`).
* Nejhorší snímky podle ΔE **nejsou špatně zarovnané** — siluety mračna sedí na hranách fotky
  ≤ 2 px; ΔE tam měří volbu zdrojového snímku v TerraScanu. Zarovnání se ověřuje siluetami,
  ne barvou.
