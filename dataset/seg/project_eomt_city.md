# Projekce EoMT-L segmentace do mračna — korigované pózy (`poses_corrected`, registrované mračno, produkty `e8f3e1`, čistá množina 830)

Srovnání s exportní verzí (`export_baseline/project_eomt_city.{json,md}`, 825 čistých snímků, exportní pózy,
neregistrované mračno): body i GT se v mezidobí posunuly OBOJÍ (korekce pozic + korigovaný pseudo-GT ze
`segds_e8f3e1`), takže rozdíly IoU nejsou čistě geometrický efekt — viz `08_korekce_poz_panoramat.md §7`.

| metrika | export | corrected | Δ |
|---|---|---|---|
| coverage | 0.799 | 0.807 | +0.008 |
| pixel acc | 0.683 | 0.667 | −0.015 |
| mIoU_core | 0.352 | 0.328 | −0.024 |
| mIoU_ext | 0.000 | 0.000 | 0.000 |
| mIoU_core (ground pts) | 0.213 | 0.197 | −0.016 |
| acc (ground pts) | 0.718 | 0.704 | −0.015 |
| evaluated points | 273,127,198 | 273,355,072 | +227,874 |

Per-tile (38 tiles): coverage se zlepšila u 16, zhoršila u 14 tiles (mean Δ +0.6 pp, rozptyl od −3.8 pp do +14.8 pp —
tiles 002/004 získaly nejvíc pokrytí, +14.8/+11.8 pp, protože se u nich s korekcí přiřadilo víc snímků do
dosahu `r_max=40 m`). Pixel acc klesla u 28/33 vyhodnotitelných tiles (mean Δ −1.9 pp); nejvýraznější propad
tile 037 (−19.9 pp, 0.601→0.402) při prakticky stejném počtu snímků (71→71) — jde tedy o posun geometrie/GT,
ne o jiný výběr snímků.

| class | IoU (all) | IoU (ground pts) | pred points | GT points |
|---|---|---|---|---|
| road | 0.561 | 0.523 | 135,884,312 | 73,140,273 |
| sidewalk | 0.194 | 0.160 | 4,745,326 | 774,349 |
| building | 0.317 | 0.000 | 21,353,487 | 21,340,470 |
| wall | 0.003 | 0.003 | 2,246,201 | 659,006 |
| fence | 0.155 | 0.038 | 11,120,165 | 22,215,742 |
| vegetation | 0.495 | 0.001 | 87,964,972 | 64,276,832 |
| terrain | 0.572 | 0.656 | 206,705,229 | 154,623,114 |
| water | 0.000 | 0.000 | 0 | 320,415 |
| guard_rail | 0.000 | – | 0 | 75,317 |
| stairs | 0.000 | 0.000 | 0 | 18,256 |
| pole | 0.000 | 0.000 | 258,650 | 324 |
| sky | – | – | 0 | 0 |
| vehicle | 0.000 | 0.000 | 1,776,648 | 0 |
| person | 0.000 | 0.000 | 4,887 | 0 |
| other | 0.000 | 0.000 | 24,773 | 0 |

coverage 0.807, pixel acc 0.667, mIoU_core 0.328, mIoU_ext 0.000, evaluated points 273,355,072