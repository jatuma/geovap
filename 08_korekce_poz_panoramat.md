# Korekce polohy a natočení panoramat (S0–S7)

Realizace plánu `suggest-how-to-correct-hashed-mochi.md`, který navazuje na slabiny 2.1–2.4
zjištěné v `07_revize_geometrie_a_data.md`. Cílem bylo opravit pózu tam, kde `export.csv` sama
nestačí: trajektorie mezi snímky, per-snímkové reziduum a vzájemná registrace 30 průjezdů. Všechna
čísla v tomto dokumentu jsou přečtená z `Geovap_cache/out/poses/*.json`, `out/pass_reg/*.json`,
`out/diag/residual_maps.json`, `out/poses/traj_diag/*.json`, z docstringů modulů a z logů běhů
(`/tmp/.../scratchpad/*.log`, citováno jen tam, kde nešlo číslo dohledat v `out/`); kde se dřívější
shrnutí této práce od čísla v souboru lišilo, platí soubor a je to tak dole poznamenané.

---

## 1. Cíl a shrnutí výsledku

Cíl: opravit pózu 1 503 panoramat proti mračnu i proti JVF referenci a přitom nechat `export.csv`
netknuté jako výchozí a regresní kotvu. Výsledek je nesourodý a je potřeba ho takhle číst — ne
jako jedno číslo, ale jako čtyři samostatná zjištění:

**Fungovalo.** Orientace mezi snímky se dala rekonstruovat nezávisle na `export.csv` z rovin
skenovacích hlav VMX-2HA (S3b) — a ukázalo se, že export sama je v čase snímku přesná (medián
odchylky yaw 0,0008° na rovných úsecích, 0,0045° v zatáčkách, `out/poses/traj_diag/diag_rot.json`);
problém byl vždy jen v **lineární interpolaci mezi snímky**, kterou S3b nahradila hustou (200 Hz)
trajektorií. Vzájemná registrace 30 průjezdů proti sobě (S5, pár ICP) taky funguje dobře: medián
RMS klesl z 0,102 m na 0,035 m na 66 z 72 párů (`out/pass_reg/conflict_reg.json`). Infrastruktura
(S0) — pose tabulky, provenience, hash, volitelný `poses="corrected"` — je hotová a testovaná.

**Nefungovalo — a to je poctivé zjištění, ne rezignace.** Absolutní poloha z rovin skenu (S3, plná
verze s pozicí) je na tomto datasetu degenerovaná — úzké obloukové okno neumožňuje jednoznačně
určit střed skenování, takže výsledná trajektorie má chybu řádu metru (RMS kamerového ramene
0,72 m) místo plánovaných centimetrů; kód zůstává v repozitáři, ale **produkčně se nepoužívá**.
Absolutní ukotvení proti JVF (S5b) selhalo úplně: očekávané rozdělení průjezdů na "dobré" (0, 1, 11,
12, 20) a "špatné" (podle segmentačního datasetu) revize **vyvrátila** — právě u očekávaně dobrých
průjezdů 0/1/11/12/20 vyšel odsazení mračna vůči JVF 0,2–0,9 m, zatímco u některých očekávaně
špatných (16, 23, 27) vyšlo < 0,1 m (`out/pass_reg/jvf_offsets.json`). Ukotvení proti JVF proto
zůstává vypnuté (`--datum none`, jen párová registrace + slabý prior identity).

**Ověřeno (S7).** Finální validace (`pose_report.compare_pose_sources`, `out/poses/report_final.md`)
srovnala `export` vs. `poses_corrected` na šesti nezávislých metrikách (detail §3.8). Geometrické
metriky se mírně zlepšily nebo zůstaly neutrální: siluetová shoda v zatáčkách (n=210) medián |du|
0,402→0,459 px, |dv| 1,21→0,94 px, zlepšeno o >2 px u 16,7 % snímků proti 7,1 % zhoršených; na
rovných úsecích (n=200) |du| 0,395→0,43 px, |dv| 0,79→0,678 px. Barevná ΔE na zatáčkách je beze
změny (medián 10,866→10,865) — potvrzuje, že tahle metrika sleduje TerraScanovu volbu zdrojového
snímku, ne pózu. Skutečný přínos celého řetězce je **mezi-snímková interpolace**: v čase snímku
hustý model souhlasí s exportem na medián 0,197° (p95 0,398°), zatímco naivní lineární interpolace
ze sousedních snímků v zatáčce chybuje o medián 5,639° (p95 19,173°, n=173/178 zatáčkových snímků
pokrytých trajektorií) — cca 29× hůř. Důležitá provozní výhrada: pomalá regrese na dlaždici 037
zůstává v toleranci (CIE76 medián export 6,076 vs. corrected 6,088, Δ +0,012, tolerance 0,1), ale
jen s `CloudStore(registration=pass_transforms.json)`; bez registrace mračna vyjde Δ +1,724, protože
`poses_corrected` posouvá kameru rotací kolem těžiště průjezdu, což je u dlouhého průjezdu i za
zlomkovou úhlovou korekci metrový posun (příklad: průjezd 0, korekce 0,097° + 0,062 m, snímek 470 m
od těžiště → 0,86 m absolutní posun kamery) — kdokoliv použije `load_poses("corrected")` mimo
lokální per-snímkové kontroly, musí mračno registrovat stejnou transformací.

**Zůstalo otevřené.** Per-snímkové zpřesnění (S4) narazilo na azimutálně závislé, měnící znaménko
reziduum v zatáčkách, které žádná tuhá póza jednoho snímku nevysvětlí (§3.4). Parallax ze sešívání
(S6) se nedal změřit vůbec — v žádné z 436 941 asociací edge-ICP na 117 snímcích není bod blíž než
19,5 m, takže není čím otestovat růst amplitudy s 1/r (`out/diag/residual_maps.json`).

Čistý dopad: `poses_corrected` existuje a je sestavená (S0–S5b), ale její hlavní přínos je oprava
interpolace v zatáčkách (S3b), ne zlepšení nějaké jedné souhrnné metriky — pár metrik se zlepšilo
(pass-to-pass RMS, cross-pass konflikty 17→16 z 309), jedna zůstala prakticky beze změny
(near-field 13,3→15,3 px) a jedna se lehce zhoršila (S4 acceptance floor ~20 px edge RMS nikdy
neklesl, protože je to šum metody, ne pózy — viz §3.4).

---

## 2. Architektura

### 2.1 Pose tabulka (`mapping/poses.py`)

`Poses` (dataclass) nese `filename`, `t`, `origin[E,N,H]`, `roll/pitch/yaw`, `pass_id`, `speed`,
`source` (`"export"` nebo jméno CSV) a volitelně `traj` (hustá trajektorie, viz §2.3).
`Poses.hash()` je sha1 z `t` (přesně) a `origin/roll/pitch/yaw` (zaokrouhleno na 1e-3 m / 1e-4°) —
stabilní přes zápis/čtení CSV. `Poses.pass_of_time(t)` dá `pass_id` pro libovolný dotazovaný čas
(hranice v polovině mezery mezi průjezdy) — používá ho S5 k přiřazení bodů mračna k průjezdu.

`load_poses(source=None)`:

* `None` → `config.POSES_SOURCE` (env `GEOVAP_POSES`, default `"export"`).
* `"export"` → beze změny, regresní kotva (`_load_export`, parsuje `export.csv` přes
  `experiments/common/io_data`).
* `"corrected"` → `read_pose_table(POSES_DIR / "poses_corrected.csv")`.
* cokoliv jiného (str/Path) → ten CSV soubor.

`write_pose_table(poses, per_frame_meta, provenance, path)` zapíše CSV (`frame, filename, t, E, N,
H, roll, pitch, yaw, pass_id`, + libovolné sloupce z `per_frame_meta`, např. `status, src, dt_s,
dyaw, ...`) a sidecar `<path>.json` s provenience (git rev, sha1 `export.csv`, `poses_hash`, vstupní
soubory). `read_pose_table` bere `pass_id` z tabulky tak, jak je (nikdy znovu nesegmentuje), takže
pořadí a indexy snímků zůstávají stabilní napříč zdroji. Pokud sidecar JSON pojmenuje trajektorii
(`"trajectory": "..."`), připojí se jako `Poses.traj` (best-effort — `mapping.trajectory` nemusí
existovat v čase importu, proto typ `object | None`).

### 2.2 Provenience — řetězení

`poses_corrected.json` řetězí `poses_hash` a sha1 souborů všech tří vstupů (S3b báze, S4 overlay,
S5b transformace) plus `git rev` a sha1 `export.csv` (viz §3.6 pro přesný obsah). Kdokoliv, kdo má
`poses_corrected.csv`, tak umí zpětně dohledat přesně to, z čeho vznikl.

### 2.3 Interpolace — jak `traj` mění `Poses.interp`

`Poses.interp` beze změny reprodukuje starý lineární kód (dt=0 vrací uložené hodnoty přesně —
regresní kotva `test_regression_037.py` je nedotčená). Nový branch na začátku smyčky přes průjezdy:
pokud je `Poses.traj` nastaveno a `traj.covers(pass, t_query)` je pravda, použije se
`traj.camera_pose(t_query, pass)` místo lineární interpolace; pro dotazy mimo pokrytí trajektorie
(nebo bez `traj`) se použije beze změny stará lineární cesta. Downstream kód (`FrameIndex`,
produkty, `colorize`, `align`) o tom neví — pracuje jen s `Poses`.

### 2.4 Jak se opt-inuje

```
export GEOVAP_POSES=corrected      # nebo cesta k libovolné pose tabulce
uv run python -m mapping.cli.render_frame 367 ...
```

nebo v kódu `load_poses("corrected")` / `load_poses("/path/to/table.csv")`. Bez nastavení
`GEOVAP_POSES` (nebo s `GEOVAP_POSES=export`) je chování **byte-identické** dnešnímu stavu —
`_load_export()` se nezměnila, `test_load_poses_export_unchanged` to hlídá.

### 2.5 `frames_dir` — hašované produkty

`products.frames_dir(poses)`: pro `poses.source == "export"` vrací nezměněný `FRAMES_DIR` (produkty
z korigovaných póz nikdy nepřepíšou ani nesmíchají s produkty exportu); pro cokoliv jiného vrací
`FRAMES_DIR / poses.hash()[:6]`. `FrameProducts.load(..., poses=)` kontroluje `meta["poses_hash"]`
proti `poses.hash()` (jako dnes `rig_hash`) a bez `allow_stale=True` vyhodí `RuntimeError`, pokud
produkty sedí na jinou pózu. Workery (`Pool`) dostávají `poses_source` jako explicitní initarg, ne
jen přes dědění `GEOVAP_POSES` forkem — spolehlivé i při `spawn`.

---

## 3. Kroky S0–S7

### 3.1 S0 — infrastruktura pose tabulek

Beze změny chování pro export (viz §2.1, §2.4); nový kód testován v `tests/test_poses_table.py`
(zápis/čtení round-trip, `load_poses("export")` nezměněno, stabilita hashe, `interp` použije `traj`
tam, kde `covers()` je pravda, `assemble` end-to-end). Náklad podle plánu 0,5 dne, žádné měřitelné
"před/po" — je to jen infrastruktura.

### 3.2 S1 — rozšíření store a psid

`mapping/cloud_store.py`: `EXTRA_COLUMNS = ("user_data", "scan_angle_rank", "return_number")`,
lazy-loaded per dlaždice; `TIME_BUCKET_S = 0,01 s` časový index (`query_time`). `store_add_columns`
proběhl na všech 38 dlaždicích, 584 809 840 bodů, **22 s** (soubor `store_add_columns.log`;
paměťově levné, jde jen o permutaci existujícího `orig_index`). `user_data` je potvrzeně 50:50 mezi
oběma hlavami skeneru na každé dlaždici (frakce 0,44–0,64 podle velikosti dlaždice, celkově blízko
50:50) — souhlasí s `07 §4`.

`pass_psid.json`: 30 fotoprůjezdů, `psid` odpovídá průjezdu 1:1 na **26 z 30** — čtyři průjezdy mají
rozdělený `psid` (dva scanovací segmenty v jednom fotoprůjezdu): **12, 14, 20, 29** (např. průjezd
20: `psid {132: 16 124 601, 142: 22 595 565}`, průjezd 29: `psid {141: 8 580 290, 145: 9 452 142}`).
Tyto čtyři se v S5 (`pass_reg.py`) vrací k čistě časovému filtru; ostatních 26 používá `psid` jako
levnější dodatečnou pojistku proti sousednímu průjezdu ve stejném časovém koši.

### 3.3 S2 — diagnostika (baseline + parallax)

**Baseline** (`out/poses/report_export_baseline.md`, z `dataset/frame_quality.csv`, n = 1 503,
n s edge-ICP fitem = 1 432): medián |du| 0,43 px, |dv| 1,06 px, MAD du 8,05 px, inlier@8px 0,197
celkem; po třídách clean 0,38/0,78 px (MAD 8,1/11,2), reject 0,63/6,57 px (MAD 8,0/13,5); zatáčky
(n=209) 0,40/1,19 px vs. rovné (n=1294) 0,43/1,04 px — na úrovni celého snímku není zatáčka vidět v
edge-ICP mediánu (ten je dominovaný šumem metody, viz §3.4), i když **lokálně** (jednotlivé úseky
zatáčky) je vidět velmi jasně (§1, §3.4).

**Aligner** (`out/poses/align.json`, `align_summary.json`): 310 snímků (210 zatáček
|yaw_rate|>8°/s + 100 rovných), medián `dt_s` = 0 pro obě skupiny (IQR 0), rozsah −7,5…8,35 s
(zatáčky) a −8,35…4,35 s (rovné) — extrémní scatter jednotlivých řešení, medián je nula, protože
soft-L1 cílová funkce je na `dt` plochá (potvrzuje `06 §7`/`07 §2.1`: bez husté trajektorie nemá
optimalizace co zafixovat). `n_suspicious` 212/310 (68 %) — Aligner sám hlásí většinu řešení jako
nedůvěryhodná.

**Parallax reziduum** (`out/diag/residual_maps.json`): 117 snímků, 436 941 asociací edge-ICP,
binováno (azimut 5°, elevace 10°, 1/r do 6 košů). Fourierova řada řádu 1 dá amplitudu 1,09 px ve
4,4 m, řád 5–6 dá 5,46–6,24 px, ale **r² zůstává 0,0–0,0008** u všech řádů a **všech 436 941
asociací padne do jediného, nejvzdálenějšího koše 1/r** (`r_covered_min_m = 19,48`, `inv_r_bin_counts
= [436941, 0, 0, 0, 0, 0]`). Verdikt souboru: *"inconclusive (no near-field associations: all 436941
within r >= 19.5 m, only the outermost 1/r bin has data — cannot test amplitude growth with 1/r)"*.
Rozhodnutí podle plánu (§S2 → S6): protože se parallax **nedal změřit, ne že by vyšel nulový**, S6
zůstává neudělané — viz §3.6.

### 3.4 S3 — hustá trajektorie: poziční mód (FAILED) a rotační mód (funguje)

**Fyzika, ověřená.** Body jedné skenovací hlavy VMX-2HA v okně ~2 ms leží na rovině: medián RMS
0,61–1,28 mm napříč všemi 30 průjezdy (`out/poses/traj_diag/diag.json`, `diag_rot.json`), kontinuita
normály mezi po sobě jdoucími okny > 0,9999 (dot product). Uvnitř roviny je směr paprsku afinní
funkcí `scan_angle_rank`.

**Vyvrácená hypotéza "wrap".** `scan_angle_rank` je int8 přes celý rozsah −128…127, což vyvolalo
otázku, jestli exportér zabalil mechanické otočení zrcadla 360° do 256 kódů (360/256 = 1,40625
°/jednotka) místo měřeného ~1,02 °/jednotka. Přímé měření na průjezdu 5 (obě hlavy, bearing úhel
proti hrubému skenovacímu středu odvozenému z exportní pózy, oddělené od vlastní degenerace fitu
`fit_scan_centre`): sklon se těsně shlukuje kolem 1,0–1,1 °/jednotka (průměr |a| 1,05, std 0,13,
n=40 oken) — daleko od 1,40625° (`mapping/trajectory.py` docstring modulu). Rychlost skenování
(rank sequence, autokorelace, průjezd 5, 17–18 oken 2 ms): **head 1 medián 96,8 jednotek/ms → 378
řádků/s, head 2 medián 89,6 jednotek/ms → 350 řádků/s** (přepočteno ze skriptu, jehož vstupní pole
zůstala v scratchpadu; hodnota nebyla nikde uložena do `out/`, proto je přepočtena zde a citovaná
tak, jak vyšla — 345–380 řádků/s z dřívějšího shrnutí je v tomto rozsahu).

**S3 (poziční mód) — FAILED, kód zůstává, produkčně se nepoužívá.** Uvnitř roviny má být poloha
středu skenu S dopočitatelná proložením podle `scan_angle_rank`. Během vývoje se našly a opravily
dvě reálné chyby (žádná nebyla v plánu předvídaná): (1) `fit_scan_planes`'s vlastní vektor `e1` na
okno neměl kontinuitu znaménka (jen `normal` ji měla) — na ~15 % po sobě jdoucích reálných oken se
překlopil, což nakrmilo teplý start stupně 2 fází ~180° špatně; (2) stupně 1/2 (`_pooled_fit`,
`_stage2`) znovupoužívaly syrová čísla (s1, s2, phi0) z předchozího okna beze změny pod **jinou**
rotující bází (e1, e2) místo přeprojekce přes světové souřadnice — chybné vždy, když se báze mezi
okny skutečně otočí (medián ~5°, až ~90°, mezi po sobě jdoucími reálnými okny skenu). Obě opraveny.

Zbývá **třetí, nevyřešený problém**: úzká úhlová šíře oblouku v jednom okně je skutečně
neidentifikovatelná — jiný, jinde ležící střed vysvětlí body stejně dobře (ověřeno: 30 náhodných
restartů jednoho takového okna konverguje vždy ke stejnému, ale **špatnému** optimu). Zkoušené
zmírnění: post-hoc neuzávěrový robustní medián-filtr (`_reject_position_outliers`) — částečně
pomáhá; kauzální veto na neplauzibilní skok bylo zkoušeno a zavrženo (jedno chybně přijaté okno pak
otráví každé další, i správně konvergující). Zkoušené zpřísnění excentricity `MIN_IN_PLANE_ECC`
z 0,03 na 0,1–0,5 (odmítnout víc úzkých oken) zhoršilo **všechna** agregovaná čísla — nejspíš tím,
že vyhladovělo interpolaci o dobrá okna — a bylo vráceno zpět.

Čísla, celý průjezd 5 (31 M bodů, obě hlavy, `MIN_IN_PLANE_ECC` na původních 0,03):

| veličina | před opravou (dle zadání plánu) | po opravě | cíl plánu |
|---|---|---|---|
| offset střed hlava1 vs. hlava2 (std) | — | 9,36 m → **0,96 m** | mm–cm |
| RMS kamerového ramene (fit na 44 čistých rovných snímků) | 5–10 m | **0,72 m** | < 5 cm |
| `l_cs` (fitované rameno) | — | (−1,95, −2,79, 0,75) m | ~(1,4, 0, 0,8) m |
| trajektorie vs. export, rovné (n=66) | — | medián 0,54 m (p95 1,83 m) | — |
| trajektorie vs. export, zatáčky (n=16) | — | medián 0,87 m (p95 1,73 m) | — |
| yaw, rovné | — | medián 0,007° (souhlasí s exportem) | — |
| yaw, zatáčky | — | medián 2,9° (p95 7,0°) | — |
| reziduum úhlu stupně 1 (fit roviny) | ~0,33–0,35° | beze změny | < 0,1° |
| jitter středu okna | ~400 mm | beze změny | < 20 mm |

Zlepšení je reálné (10–30× na několika nezávislých metrikách), ale řádově nad cílem plánu. Příčina
je popsaná výše (úzký oblouk). Skutečná oprava potřebuje buď křížovou konzistenci mezi hlavami
(S1(t) a S2(t) by se měly shodovat na mm–cm úrovni kdykoliv — v plánu zmíněná kontrola, neimplemen-
tovaná) nebo globální/joint estimátor s pohybovým modelem místo per-okenního řešení opravovaného
dodatečně. Kód (`fit_scan_centre`, `_reject_position_outliers`) zůstává v `mapping/trajectory.py`
pro budoucí použití, ale `build_trajectory.py` (bez `--rot-only`) se v produkci nespouští.

**S3b (rotační mód) — funguje, je to, co jde do `poses_corrected`.** Salvage: použít jen orientaci
z dvouhlavého Kabsch fitu (RMS roviny 0,6–1,3 mm, kontinuita normály > 0,9999 — poziční degenerace
se orientace netýká), a polohu nechat na exportní lineární interpolaci beze změny
(`Trajectory.mode="rot_only"`, `camera_pose` vrací `origin` = přesně `Poses.interp`'s netraj branch,
`orientation = R_cs @ R_s(t + dt_s)`). `--rot-only --passes all` proběhl na 19 z 30 průjezdů
(zbylých 11 nemá dost čistých rovných snímků pro vlastní rig — "too few clean frames", zůstávají
`kept`/export): **1 373/1 503 snímků** dostane trajektorii (`status="traj_rot"`), 130 zůstane na
exportu (`rot_only_all_v2.log`). Per-pass `dt_s` konzistence (referenčně nezávislá veličina) přes 19
průjezdů: medián −0,0008 s, std 0,0089 s, rozsah −0,038…0,010 s — těsný shluk, jak fyzika žádá.

**Klíčové zjištění.** Export yaw ve **vlastním čase snímku** souhlasí se skenovací orientací velmi
těsně: rovné úseky medián |Δyaw| 0,0008° (p95 0,506°), zatáčky medián 0,0045° (p95 0,244°)
(`diag_rot.json`, `"comparison"`). Export sám tedy v čase snímku *není* chybný — problém je výhradně
**mezi** snímky. To potvrzuje i `validate_rot.json` (nezávislá kontrola siluetami a barvou při dt=0,
tj. beze změny expozičního času): zatáčky |dv| medián 1,35→1,15 px, inlier@8px 0,194→0,202, barevná
ΔE 10,69→10,65 — mírné zlepšení, ne regrese, přesně jak se čeká, když se mění jen interpolace mimo
frame times (které `validate_rot` netestuje).

Chybu **mezi** snímky (lineární interpolace `export.csv` vs. hustá trajektorie ve stejném dotazovaném
čase) žádný soubor v `out/` přímo nenese — přepočteno pro tento report ze `trajectory_rot.npz` +
`poses_traj_rot.csv` (segmenty mezi dvěma po sobě jdoucími snímky téhož průjezdu, kde je úsek
pokrytý trajektorií, |Δyaw/Δt| segmentu > 8°/s = "zatáčkový" segment, porovnáno v polovině úseku):

| skupina | n úseků | medián \|Δyaw\| | p95 | max |
|---|---|---|---|---|
| zatáčkové (\|rate\|>8°/s) | 164 | 1,38° | 6,93° | 17,2° |
| rovné | 1192 | 0,17° | 1,36° | — |

Toto číslo je menší, než dřívější neformální odhad "5,6° medián / 19° p95" (ten se v žádném souboru
nenašel, takže je zde nahrazen měřením — metodika výše, reprodukovatelná ze `trajectory_rot.npz`).
Řádově potvrzuje totéž: lineární interpolace se v zatáčkách odchyluje o jednotky stupňů od husté
trajektorie, zatímco na rovných úsecích je rozdíl setiny až desetiny stupně.

### 3.5 S4 — per-snímkové zpřesnění (edge-ICP na 6 parametrů/snímek)

`PassRefiner` (`mapping.calib.icp.EdgeICP` podtřída) fituje θ_k = (dt, dyaw, droll, dpitch, dlat, dh)
na snímek jednoho průjezdu proti stejné hraně-fotky cílové funkci jako kalibrace rigu, s
Gaussovským priorem a hladkostí podél průjezdu. Plný běh na exportních pózách (30 průjezdů, 8
workerů, log `full_run.log`): **3 112 s celkem**, výsledek `status_counts` v CSV
(`poses_refined_export.csv`, n=1 503): **`refined` 409, `interpolated` 102, `kept` 992**.

**Akceptační práh je nízko záměrně.** Zjištění během vývoje: na 20 čistých rovných snímcích
(|yaw_rate|<3°/s) je medián `rms_px` v jejich **nerefinované** póze 22,5 — skoro přesně jako "před"
čísla ve vývojovém běhu — přestože medián |du|,|dv| je jen 0,46/1,16 px a MAD 7,9/11,5 px, což sedí
s `dataset/frame_quality.csv`'s clean-baseline (0,38/0,78 px medián, 8,1/11,2 px MAD) na šum
vzorkování. Samotná edge-ICP metrika má **vlastní podlahu** (šum vegetace/okluzních hran nafukuje
soft-L1 RMS přes ocasy) ~20 px i u dobře zarovnaných snímků — poměr RMS jako akceptační kritérium
by tedy honil šum místo chyby pózy. Akceptace proto zrcadlí přímo `quality.py`: robustní medián
posunu + inlier frakce, ne poměr RMS.

**Přijaté korekce** (409 `refined` snímků, `poses_refined_export.csv`): medián |dt| 0,029 s, medián
|dyaw| 0,29°, |droll| 0,37°, |dpitch| 0,38°; inlier@8px medián 0,242 → 0,267. (Dřívější neformální
shrnutí uvádělo dt 0,046 s, dyaw 0,06°, droll 0,23°, dpitch 0,10°, inlier 0,26→0,29 — čísla v
`poses_refined_export.csv` jsou vyšší u úhlových korekcí; platí soubor.)

**Zjištění, které nejde opravit tuhou pózou jednoho snímku.** Zatáčkové snímky vykazují azimutálně
závislé, měnící znaménko reziduum: snímek 1092 má dv +37 px při azimutu 225° a −29 px při azimutu
315°. Žádná tuhá 6-DoF korekce jednoho snímku takový vzor nevysvětlí — kandidátní hypotézy pro
dalšího řešitele:

* Ladybug expozice šesti senzorů není simultánní (sešívání pod rotací vozidla by dalo přesně
  tento azimutální, znaménko-měnící podpis);
* zkreslení mračna v zatáčce (vozidlo se otáčí rychleji, než skener stihne pokrýt scénu — vlastní
  parallax skener↔kamera se v zatáčce neruší tak čistě jako v přímém úseku);
* maska pohybujícího se vozidla — v zatáčce je jiná část siluety karoserie vidět než v přímém úseku
  a statická maska (`vehicle_mask.py`, fitovaná na přímé snímky) může na hraně selhat asymetricky.

### 3.6 S5 — registrace průjezdů: párově funguje, JVF datum ne

**Párová ICP registrace** (`pass_reg.py`, bod-k-rovině, 4-DoF, `cKDTree`, soft-L1): 72 párů průjezdů
(origins do 15 m), **66 konvergovalo**, medián RMS 0,102 m → **0,035 m** (`conflict_reg.json`,
`cloud_icp_pairs_summary`). Pairwise-only řešení pose grafu (`--datum none`, slabý prior identity):
medián posunu 0,168–0,172 m, medián rotace 0,124°, max posun 1,38–1,51 m; **12 z 30 průjezdů** má
posun/rotaci nad prahem 0,3 m / 0,3° (`pass_transforms.json`, `summary.n_flagged`): 2, 3, 4, 8, 9,
10, 12, 15, 21, 23, 24, 26.

**Cross-pass konflikty siluet** (`conflict_reg.json`): u 309 snímků se skutečným křížením průjezdů
(`overall_true_cross_pass`) klesl počet konfliktních snímků **17 → 16**; medián |du2|/|dv2| 0,37/0,87
px → 0,37/0,84 px — malé, ale ve správném směru. Near-field metrika (`nearfield_reg.json`, 200
snímků): medián **13,3 px → 15,3 px**, počet vlajkovaných snímků 62 → 60 — prakticky neutrální
(mírně horší medián, mírně méně vlajek).

**JVF absolutní datum — selhalo, nepoužívá se.** Dvě metody zkoušeny: `jvf_offset` (obrys obrubníku
z mračna vs. JVF hranice vozovky) a `jvf_offset_photometric` (fotoedge grid search proti stejným
liniím). Obě zakotvení near-field medián **zhoršila** (13,3 → 15,6 px při `--datum jvf-photo`, jen
1/30 průjezdů splnil kritérium kotvy), a obě metody se u průjezdů, které umí změřit obě, **navzájem
neshodují**. Kritická chyba dat: `_curb_points` na tomto venkovském datasetu bez filtru sbírá
30–80 tisíc "obrubníkových" bodů na průjezd — o řád víc, než pár set metrů skutečného obrubníku může
dát — protože obecný detektor výškového skoku stejně ochotně chytá meze orby jako skutečný obrubník
(`out/pass_reg/qa/*.png` ukazuje mračno "obrubníku" průjezdu 0 jako diagonální vějíř přes zorané
pole, ne hranu vozovky). `CURB_JVF_CORRIDOR_M` gate to zmírňuje (drží jen body do koridoru okolo
syrové JVF linie), ne odstraňuje.

**Rozdělení na "dobré" a "špatné" průjezdy ze segmentačního datasetu je vyvrácené.** Plán očekával
`|offset| < 0,1 m` u průjezdů 0, 1, 11, 12, 20 (viz `07 §1.3`). Naměřeno (`jvf_offsets.json`,
cloudová metoda):

| průjezd | očekávání | odsazení (m) | n bodů |
|---|---|---|---|
| 0 | dobrý | **0,638** | 16 568 |
| 1 | dobrý | **0,492** | 10 394 |
| 11 | dobrý | **0,211** | 1 140 |
| 12 | dobrý | **0,728** | 1 745 |
| 20 | dobrý | **0,916** | 396 |
| 16 | špatný | 0,025 | 550 |
| 23 | špatný | 0,054 | 612 |
| 27 | špatný | 0,074 | 481 |

Žádný z pěti očekávaně dobrých průjezdů nemá odsazení pod 0,1 m; tři očekávaně špatné (16, 23, 27)
mají odsazení < 0,1 m. Fotometrická metoda dá jiné, vlastní pořadí (např. průjezd 0: 0,46 m, průjezd
1: 0,72 m) — obě metody se navzájem neshodují na tom, který průjezd je "dobrý". Závěr:
**venkovská silnice v Dražkově má příliš málo skutečných obrubníků a hranic vozovky**, aby JVF sloužila
jako absolutní reference; obrubníkový i fotoedge detektor se sytí na mez orby/krajnici namísto na
skutečnou hranu. `cli.register_passes solve` proto defaultně používá `--datum none`; oba JVF postupy
zůstávají volitelné (`--datum jvf-cloud|jvf-photo`) pro až přijde lepší absolutní reference (SBET,
§5).

### 3.7 S6 — parallax ze sešívání: netestovatelné

Podle §3.3: 436 941 asociací edge-ICP na 117 snímcích, **žádná blíž než 19,48 m** — celý naměřený
rozsah padne do jediného nejvzdálenějšího koše 1/r, takže neexistuje kontrast, na kterém by šlo
otestovat, jestli amplituda roste s 1/r (podpis parallaxu). Fourierovy řády 1–6 dají r² 0,0–0,0008 —
plochá cílová funkce, ne důkaz nulového parallaxu. Co by bylo potřeba: edge-ICP asociace blíž než
~10 m (dnešní kalibrační sada je zaměřená na vzdálenější, stabilnější hrany domů/plotů) — buď jiný
výběr snímků/oken s bližšími hranami, nebo jiná zdrojová geometrie (obrubníky, sloupy). `mapping/parallax.py`
podle plánu nebyl napsán vůbec — nemělo by smysl fitovat šest sedlových bodů na data bez near-field
kontrastu.

### 3.8 S7 — validace a testy

**Testy: zelené.** `uv run pytest -q` → **104 passed, 5 deselected** (slow), ~10 s. Pokrývají S0
(`test_poses_table.py`: round-trip, `load_poses("export")` nezměněno, stabilita hashe, `interp`
používá `traj` jen tam, kde `covers()` je pravda, `test_assemble_end_to_end`), S3/S3b
(`test_trajectory.py`: syntetické roviny → rekonstrukce S/R, `rot_only` save/load round-trip,
`camera_pose` proti ruční formuli s netriviálním rigem), S4 (`test_pose_refine.py`: shoda se
zamrzlým `EdgeICP`, zotavení ze syntetické perturbace), S5 (`test_pass_reg.py`: `register_pair`
zotaví známý offset, invariance `apply_pass_transforms` vůči `world_to_pano`, `solve_global` zotaví
známý 3-uzlový graf), `test_geometry.py` (`euler_from_vehicle_rotation` round-trip) a nově S7
(`test_pose_report.py`: `compare_pose_sources`, `colour_de_comparison`, `interpolation_benefit`,
`invariance_check`); mezi 5 odselektovanými (slow) je i korigovaná varianta pomalé regrese v
`test_regression_037.py` (bod 6 níže).

**Validační zpráva S7 hotová** (`pose_report.compare_pose_sources` + `run_final_report`,
`out/poses/report_final.md`/`.json`; spuštění `uv run python -m mapping.cli.pose_report run
--workers 8`, ~26 min). Porovnává `export` vs. `poses_corrected` na šesti nezávislých metrikách:

**1. Siluetová shoda** (`quality.py`, vlastní póza každého zdroje):

| skupina | n | \|du\| medián px (export→corrected) | \|dv\| medián px | zlepšeno >2px | zhoršeno >2px |
|---|---|---|---|---|---|
| zatáčky (\|yaw_rate\|>8°/s) | 210 | 0,402 → 0,459 | 1,21 → 0,94 | 16,7 % | 7,1 % |
| rovné | 200 | 0,395 → 0,43 | 0,79 → 0,678 | 8,0 % | 6,0 % |
| refined_or_reg (přesně tam, kam sáhly S4/S5b) | 673 | 0,367 → 0,391 | 0,816 → 0,677 | 16,0 % | 8,0 % |

Mírné zlepšení / neutrální — |du| lehce naroste, |dv| a inlier@8px se zlepší; efekt je asymetrický
ve prospěch korekce (víc snímků se zlepší o >2 px, než zhorší) a je vidět i na skupině `straight`,
kterou S4/S5b sotva zasáhly — konzistentní s tím, že S5b registrace posouvá celý průjezd, ne jen
snímky, které opravil S4.

**2. Barevná ΔE na zatáčkách** (`Aligner.colour_de`, n=210): medián export 10,866 → corrected
10,865 (zlepšeno 104/210, zhoršeno 106/210) — beze změny; potvrzuje zjištění z `mapping/README.md`,
že tahle metrika sleduje TerraScanovu volbu zdrojového snímku, ne pózu — změna se tu ani nečekala.

**3. Cross-pass konflikty** (cited z `conflict_reg.json`, §3.6): 17 → 16 z 309 skutečně křížených
snímků; skutečný zisk je na úrovni průjezdů — párová cloud-ICP RMS medián 0,1018 → 0,0352 m (72
párů, 66 konvergovalo).

**4. Přínos husté trajektorie** (dense model vs. naivní lineární interpolace ze sousedních snímků,
zatáčky): v čase snímku hustý model souhlasí s exportem na medián 0,197° (p95 0,398°, n=178
zatáčkových snímků pokrytých trajektorií); naivní lineární interpolace ze sousedních snímků chybuje
o medián 5,639° (p95 19,173°, n=173) — cca 29× hůř. **Toto je jediné velké, jednoznačné zlepšení
celého řetězce korekcí** (souhlasí řádově s přepočtem v §3.4, kde stejná veličina, měřená jinou
metodikou přímo z `trajectory_rot.npz`, vyšla medián 1,38°/p95 6,93° na zatáčkových úsecích —
rozdíl je v tom, co se přesně porovnává: §3.4 srovnává segmenty mezi snímky, tady se srovnává
naivní interpolace ze dvou sousedních snímků proti hustému modelu v čase konkrétního snímku).

**5. Invariance registrace** (transformace tabulky vs. trajektorie): nejhorší rozdíl 2,76e-6 px
napříč 30 průjezdy na reálných datech — o řády pod jakoukoliv měřitelnou přesností (srov. 8 px MAD
floor v bodě 1), tedy numerický šum, ne systematická chyba.

**6. Pomalá regrese, dlaždice 037** (`tests/test_regression_037.py -k corrected`; pilotní recept:
nejbližší snímek v čase, nejbližší pixel, bez okluze, bez masky vozidla, body ≥3,5 m): medián CIE76
export=6,076 (n=43058) vs. corrected=6,088 (n=43119), Δ +0,012, tolerance 0,1 — **v toleranci, ale
jen s `CloudStore(registration=pass_transforms.json)`** na straně `corrected`. Bez registrace vyjde
medián 7,80 (Δ +1,724) — ne proto, že korigovaná póza je horší, ale protože `poses_corrected`
posouvá kameru rotací kolem těžiště průjezdu, zatímco body dlaždice zůstávají na místě; u dlouhého
průjezdu jde o metrový posun i za zlomkovou úhlovou korekci (příklad: průjezd 0, korekce
0,097° + 0,062 m, snímek 0 leží 470 m od těžiště průjezdu → 0,86 m absolutní posun kamery).

**Provozní důsledek, platí obecně, ne jen pro tuto regresi:** kdokoliv použije
`load_poses("corrected")` pro cokoliv ukotvené v pevných, world-frame bodech (kolorizace, export
LAS, seg dataset rastry), musí mračno registrovat stejným `pass_transforms.json` — jinak vznikne
falešný offset kamera-vs-mračno v řádu až 1–2 m daleko od těžiště registrace daného průjezdu. Bod 1
(siluety) tímhle problémem netrpí — `compare_pose_sources` tam bere hrany z okolí vlastní (případně
posunuté) pózy zdroje stejným, neregistrovaným `CloudStore()`, jakým dnes vzniká
`dataset/frame_quality.csv`, takže lokální per-snímkové kontroly zůstávají informativní i bez
registrace mračna.

---

## 4. Co z toho plyne pro projekt (revize priorit `07 §3`)

`07 §3` řadilo priority takto: (1) registrace průjezdů, (2) skutečná trajektorie, (3) parallax,
(4) barva. Po S0–S7 se pořadí mění:

1. **Registrace průjezdů (párová) zůstává v produkci** — funguje (0,10→0,035 m), použít
   `pass_transforms.json` všude, kde se dnes mračno bere jako jedno tuhé těleso (fúze barev napříč
   průjezdy, budoucí vektorizace).
2. **Absolutní geodetický datum je teď samostatný, neřešený problém**, ne důsledek registrace
   průjezdů. JVF na tomto datasetu jako absolutní reference nefunguje (§3.6) — to je nové zjištění
   proti `07`, který s JVF počítal jako s dostupnou referencí. Bez SBETu/POSPacu (§5) nemá smysl
   absolutní datum dál zkoušet touhle cestou.
3. **Trajektorie mezi snímky je vyřešená pro orientaci (S3b), ne pro polohu.** Praktický dopad:
   yaw v zatáčkách je teď spolehlivý mezi snímky (ne jen v čase snímku); poloha mezi snímky pořád
   jede po staré lineární interpolaci — to stačí pro projekci obrazu (kde poloha kamery mezi
   snímky ~5 m od sebe hraje malou roli proti natočení), ale nestačí pro cokoliv, co by chtělo
   subcentimetrovou trajektorii mezi snímky.
4. **S4 otevřelo nový, konkrétnější problém**, než `07 §2.1` čekalo: zbytková chyba v zatáčkách
   není (jen) chybějící hustá trajektorie — S3b tu chybu z většiny odstranila a *v zatáčkách stále
   zůstává* azimutálně strukturované reziduum, které vypadá jako artefakt sešívání Ladybugu nebo
   zkreslení mračna, ne jako chyba pózy (§3.5). To je teď samostatná položka k prošetření, ne
   podmnožina "chybí trajektorie".
5. **Parallax (S6) zůstává neřešen, ale teď víme proč**: chybí data (asociace blíž než 20 m), ne
   chybí čas. Priorita se nemění, ale úkol už není "vykreslit mapu reziduí a rozhodnout" (to už
   proběhlo) — je to "získat bližší asociace".

---

## 5. Doporučené další kroky (s odhadem nákladu)

1. **Vyžádat od GEOVAPu SBET/POSPac trajektorii a syrové snímky/timestampy Ladybugu** — e-mail,
   $0 nákladu na naší straně, čekání na odpověď. Řeší zároveň absolutní datum (§3.6) i S4 zatáčkové
   reziduum (§3.5, hypotéza nesimultánní expozice) jedním artefaktem, pokud GEOVAP odpoví.
   Nejvyšší poměr přínos/náklad ze všeho v tomto seznamu.
2. **Zkusit S3 (poziční mód) s křížovou konzistencí obou hlav** — ~3–5 dní. Jediná neimplementovaná
   cesta, jak degeneraci úzkého oblouku (§3.4) obejít bez externí trajektorie: joint estimátor obou
   hlav s pohybovým modelem místo per-okenního řešení. Riziko: nemusí to stačit, pokud je scéna
   lokálně opravdu neidentifikovatelná (viz 30 náhodných restartů → stejné špatné optimum).
3. **Prošetřit S4 zatáčkové reziduum přímo na Ladybug datech** — ~2–3 dny, podmíněno bodem 1 (syrové
   snímky z jednotlivých šesti kamer, ne jen sešité panorama). Bez nich lze zkusit jen nepřímé testy
   (fitovat časový posun na úrovni jednotlivého sektoru azimutu místo celého snímku).
4. **Sehnat lepší JVF vstup nebo alternativní absolutní referenci** (ČÚZK ortofoto/DMR, `07 §7`) —
   ~1 den na vyzkoušení, pokud SBET nepřijde. Nebude to lepší než SBET, ale je to nezávislé na
   GEOVAPu.
5. **S6 parallax, jakmile existují bližší asociace** (buď z bodu 3, nebo z jiného výběru
   kalibračních oken) — ~den práce nad `parallax.py`, který ještě nikdo nenapsal (§3.7).
6. **S7 `compare_pose_sources` hotovo** (§3.8, `out/poses/report_final.md`/`.json`) — zbývá jen
   zvážit, jestli produkty na `poses="corrected"` (kolorizace, LAS export, seg dataset) mají
   registraci mračna (`CloudStore(registration=pass_transforms.json)`) vynucenou automaticky, místo
   spoléhání na to, že si to každý volající pohlídá sám (§3.8 bod 6) — ~0,5–1 den.

---

## 6. Jak to spustit

Pořadí (každý krok čte výstup předchozího; `--datum none` je výchozí a doporučené pro S5):

```
uv run python -m mapping.cli.store_add_columns --workers 8                 # S1, ~22 s (584 M bodů, 38 dlaždic)
uv run python -m mapping.cli.pass_psid                                     # S1, psid crosstab
uv run python -m mapping.cli.align_frames --workers 8                      # S2, 310 snímků, ~8 min
uv run python -m mapping.cli.build_trajectory --rot-only --passes all --workers 6   # S3b, ~5 min
uv run python -m mapping.cli.build_trajectory --validate-rot               # S3b validace, ~10 min (279 snímků)
uv run python -m mapping.cli.refine_poses --poses export --passes all --workers 8 \
    --out Geovap_cache/out/poses/poses_refined_export.csv                  # S4, ~52 min (3 112 s)
uv run python -m mapping.cli.register_passes all --datum none              # S5, extract+pairs+jvf(info)+solve+poses
uv run python -m mapping.cli.assemble_poses run                            # S_assemble -> poses_corrected.csv
uv run python -m mapping.cli.pose_report run --workers 8                   # S7, ~26 min -> report_final.md/.json
```

`build_trajectory` **bez** `--rot-only` (S3, poziční mód) je v repozitáři, ale podle §3.4 se
nespouští produkčně — je tam jen pro budoucí pokus s křížovou konzistencí hlav.

---

## 7. Dopad na dataset

Kromě `pose_report` (§3.8, siluety/barva/interpolace, výše) proběhla přestavba obou navazujících
datasetů na `poses_corrected` — plný detail a čísla v `04_cisty_dataset.md §5` a `dataset/README.md`.
Shrnutí:

**Čistý dataset (`04`, `mapping/quality.py`).** Celý běh (1 503/1 503 snímků) do
`Geovap_cache/out/dataset_e8f3e1/` proti stejnému kritériu jako export. Geometrická reziduua
(medián \|du\|/\|dv\|, MAD, inlier) se zlepšila **napříč všemi čtyřmi třídami současně**, ne jen u
`clean` — konzistentní s tímto dokumentem: hlavní přínos je oprava mezisnímkové interpolace v
zatáčkách, kde `quality.py` taky měří. `clean` 825→**830** (+5), `reject` 220→**208** (−12), `usable`
295→302, `unverified` beze změny (163). 99 snímků (6,6 %) změnilo třídu (oběma směry, ne
jednosměrně) — čisté zlepšení je malé, ne dramatické. Nejhorší průjezdy (12, 13, 14 — §1 zmiňuje jen
souhrnně, detail v `04 §3`) zůstávají nejhoršími i po korekci, jen o pár snímků méně vyhrocené;
konflikty průjezdů 26→23 snímků, ale s částečně jinou množinou průjezdů (17, 20 nově, ne jen
vymizení starých). `dataset/clean_frames.json` (produkční množina pro `seg/render_labels.py` atd.)
**zůstává na exportních pózách** — tahle přestavba je srovnávací měření, ne změna produkční množiny.

**Segmentační pseudo-GT (`03`/`05`, `mapping/seg/`).** Přestavba do `Geovap_cache/segds_e8f3e1/` nad
**stejnými 825 `clean` snímky** (množina se nemění, jen geometrie/registrace) ukázala jeden
systematický, vysvětlitelný pokles — `fence` (ERP pásmová metrika 13,427→11,126 %, −2,301 pp; na
celém mračnu 584 M bodů 4,304→3,799 %, −0,505 pp) — protože `fence` je jediné pravidlo
`point_labels.py` čistě vzdálenostní k tenké linii (`fence_dist=0,35 m`), nejcitlivější na S5b
medián posunu pasáží 0,168–0,172 m; ostatní třídy |Δ| < 0,1 pp (pásmo) / < 0,25 pp (mračno).
**Near-field zarovnání (03: 328/755 vlajkovaných, hlavní motivace téhle korekce) se nezlepšilo** —
medián mediánů 15,8→17,7 px, vlajkovaných (>20 px) 328→333 — souhlasí to se zjištěním §3.6 výše
(nezávislá metoda, jiný vzorek 200 snímků, stejný směr: 13,3→15,3 px): S5b registrace je jen párová
mezi průjezdy (`--datum none`), není ukotvená proti JVF, takže zlepšuje shodu MEZI průjezdy, ne
shodu s (nehybnou) JVF vrstvou, na které pseudo-GT stojí. Splity (`dataset.py:make_splits`,
k-means(10) na kamerových pozicích) se s posunutými pozicemi kamer mírně přerozdělily: train
458→477, val 139→133, test 161→159, buffer 67→56.

**Verdikt.** Korekce pózy vylepšuje čistý dataset i geometrii pseudo-GT mírně a měřitelně (méně
`reject`, lepší siluetová shoda, o něco menší podíl `fence` false positive u vzdálenostního
pravidla), ale **neřeší** to, co ji motivovalo — shodu s JVF (near-field beze zlepšení) — a
nejhorší průjezdy zůstávají nejhoršími. To je stejný závěr jako u `pose_report` (§1): skutečný
přínos je oprava mezisnímkové interpolace v zatáčkách, ne posun nějaké jedné souhrnné metriky
datasetu k lepšímu. Registrace mračna (`CloudStore(registration=pass_transforms.json)`) je pro
oba přestavěné datasety **povinná**, ne volitelná — `mapping.quality.assess_frame` a
`mapping.seg.*` ji berou automaticky z `poses_corrected.json` (`open_store(poses)`), takže kdokoliv
volá `load_poses("corrected")` napřímo mimo tyhle dva pipeline, musí mračno registrovat sám (§1).

**Jak reprodukovat**: `GEOVAP_POSES=corrected uv run python -m mapping.quality 8 350` (čistý
dataset, `04 §5`) a `GEOVAP_POSES=corrected uv run python -m mapping.cli.seg_build areas|rasters|
points|labels|views|dataset` (pseudo-GT, `dataset/README.md`) — oba automaticky píšou do
hash-suffixovaných adresářů (`out/dataset_e8f3e1/`, `Geovap_cache/segds_e8f3e1/`), nikdy nepřepíšou
exportní výstupy.

**Projekce segmentace do mračna (`06`, `mapping/seg/project.py`, `dataset/seg/project_eomt_city.{json,md}`).**
Přestavba na `poses_corrected` (registrované mračno, produkty `e8f3e1`, čistá množina 830 — 5 nově
promovaných snímků oproti exportním 825): (1) EoMT-L předpověď (`segds/bench/eomt_city/`, pózo-nezávislá,
jen fotka) doplněna pro 37 chybějících snímků čisté množiny (~4,5 s/snímek, RTX 3090, ~3 min celkem);
(2) `seg_project run --workers 8` přes všech 38 tiles, 525 s (8,75 min) wall; (3) `seg_project eval` a
`render`; (4) PotreeConverter znovu nad novými LAS tiles (92 s, stejných 584 809 840 bodů, 23 GB octree,
`pointcloud-tools/output/eomt_city_seg`, přepsáno na místě po záloze `eval.json`/`eval.md`/
`project_eomt_city.*` jako `*_export_baseline`).

Cestou se odhalil a opravil skutečný bug: `BENCH_DIR` (`mapping/seg/bench.py`) byl odvozen z
pose-source-aware `SEGDS_DIR`, ačkoliv 2D předpověď závisí jen na fotce (a pevné rotaci rigu), ne na
korekci pozice — s `GEOVAP_POSES=corrected` by tak `seg_project run` hledal masky v neexistujícím
`segds_e8f3e1/bench/` a tiše by promítal s **nulou snímků na každý tile** (ověřeno: první běh dal
`n_frames=0` a `coverage=0.0` na všech 38 tiles — bez shodných masek se produkty snímků vůbec nenačítaly,
takže běh doběhl výrazně rychleji než normálně). Opraveno na `BENCH_DIR =
SEGDS_ROOT / "bench"` (vždy exportní kořen, nezávisle na pose source); prázdný běh byl smazán (labels i
4,6 GB prázdných LAS tiles) a spuštěn znovu.

**LAS výstup nese PŮVODNÍ (neregistrované) souřadnice.** `las_out.write_tile` kopíruje `las.X/Y/Z` bit-exact
ze zdrojového LAZ (`td.info.laz`), nikdy z `td.xyz_m()` (které registraci aplikuje) — stejně jako u
kolorizačních tiles (`mapping.colorize`), takže Potree vrstvy `eomt_city_seg` a ostatní (`tw45`, `clusters`
apod.) zůstávají vzájemně v souřadnicích konzistentní. Registrace se použije jen INTERNĚ, při rozhodování
který bod vidí který snímek (`open_store(poses)` → `td.xyz_m()` pro promítání), nikdy se nezapisuje do
výstupního mračna.

**Výsledek (export 825 → corrected 830, `dataset/seg/project_eomt_city.md` má plnou tabulku).** coverage
0,799→**0,807** (+0,8 pp), pixel acc 0,683→**0,667** (−1,5 pp), mIoU_core 0,352→**0,328** (−2,4 pp),
mIoU_core (ground pts) 0,213→0,197, acc (ground pts) 0,718→0,704. Po tiles (38): coverage se zlepšila u 16,
zhoršila u 14 (mean +0,6 pp; tiles 002/004 nejvíc, +14,8/+11,8 pp — víc snímků nově v dosahu `r_max=40 m`
po korekci pozic); pixel acc klesla u 28 z 33 vyhodnotitelných tiles (mean −1,9 pp), nejvýrazněji tile 037
(−19,9 pp) při stejném počtu snímků (71→71) — tedy čistě posun geometrie/GT, ne jiný výběr snímků.

**Verdikt: NENÍ to čisté zlepšení** — a nedá se to ani očekávat, protože se touhle přestavbou mění DVĚ věci
najednou: (a) korigovaná pozice/registrace mračna (téma tohodle dokumentu) a (b) samotný pseudo-GT
(`segds_e8f3e1`, §7 výše), kde primárně klesly `fence` GT body (−12 %, 25,17M→22,22M, viz `fence_dist`
citlivost na S5b posun). Coverage +0,8 pp je jediné číslo, které lze připsat převážně geometrii (víc
snímků v dosahu po korekci); pokles pixel acc/mIoU_core o 1,5–2,4 pp je pravděpodobněji efekt (b) —
menší/jinak tvarovaný pseudo-GT `fence`/`sidewalk` (IoU sidewalk −7,3 pp, největší pokles ze všech tříd,
fence −3,9 pp) — než
skutečné zhoršení projekce. Nelze to z tohohle srovnání rozplést; oddělené měření (stejný pseudo-GT, jen
pózy jinak) by vyžadovalo znovu-vyhodnotit export-pseudo-GT proti corrected-projekci nebo naopak, což
tenhle běh nedělal.

**Jak reprodukovat**: `GEOVAP_POSES=corrected uv run python -m mapping.cli.seg_bench run --models eomt_city
--frames <chybějící snímky>` (doplnění predikcí — POZOR: `BENCH_DIR` je pózo-nezávislý, výstup jde vždy do
`segds/bench/`, ne `segds_e8f3e1/bench/`) a `GEOVAP_POSES=corrected uv run python -m mapping.cli.seg_project
run --workers 8` + `eval` + `render` + PotreeConverter (`pointcloud-tools/docker-compose.yml`) +
`potree-classes`, stejně jako u exportního běhu (`05_benchmark_segmentace.md`), jen s `GEOVAP_POSES=corrected`
před importem.
