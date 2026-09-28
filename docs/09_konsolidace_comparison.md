# Pipeline comparison

poses: `corrected`, tag: `tw45`, generated 2026-09-16T06:10:57.419151+00:00

| metrika | export | předchozí korigovaný běh (docs, e8f3e1) | stav repa před tímto během | **tento běh** | pozn. |
|---|---|---|---|---|---|
| poses hash | a2d74f0580 | e8f3e1f2b3 | e8f3e1f2b3 | **34bca9ff23** | corrected table identity |
| pass registration: pairs converged | - | 66/72 | - | **66/72** | register_passes --datum none |
| pass registration: median RMS before -> after [m] | - | 0.102 / 0.035 | - | **0.102 / 0.035** | ICP point-to-plane |
| registration: max |t| [m], max |yaw| [deg], flagged passes | - | - | - | **0.461 / 0.661 / 12** | from pass_transforms.json |
| yaw at turning frames: dense model vs neighbour-linear [deg, median] | 5.639 | 0.197 | - | **0.200 / 5.639** | n=178 |
| silhouette |du|/|dv| median px, turning frames (export -> corrected) | 0.402 / 1.210 | - | - | **0.461 / 0.931** | quality.py residual |
| tile 037 CIE76: export / corrected registered / unregistered | 6.076 | 6.088 / 7.800 | - | **6.076 / 6.097 / 7.777** | tolerance 0.1, passes=True |
| colourisation tw45: dE00 median | 4.930 | - | - | **5.025** | 02 §13.3 value is on export poses |
| colourisation tw45: coverage | 0.941 | - | - | **0.940** |  |
| clean / unverified / usable / reject | clean 825, unverified 163, usable 295, reject 220 | clean 830, unverified 163, usable 302, reject 208 | clean 830, unverified 163, usable 302, reject 208 | **clean 835, unverified 165, usable 304, reject 199** | quality.py on this run's poses |
| clean set turnover vs previous | - | - | 830 | **added 17, removed 12** |  |
| near-field JVF flag: n_measured / median px / flagged | 755 / 15.800 / 328 | 757 / 17.900 / 336 | flag_px 20.000, bad 336, ok 421, unmeasured 73 | **762 / 17.900 / 340** | flag > 20 px |
| bench mIoU_core eomt_city (all / nf_ok) | 0.358 / 0.330 | - | 0.358 / 0.330 | **0.332 / 0.302** | new bench_frames.json (this run's pseudo-GT) |
| bench mIoU_core m2f_vistas (all / nf_ok) | 0.336 / 0.318 | - | 0.336 / 0.318 | **0.327 / 0.310** | new bench_frames.json (this run's pseudo-GT) |
| bench mIoU_core m2f_city (all / nf_ok) | 0.305 / 0.285 | - | 0.305 / 0.285 | **0.299 / 0.283** | new bench_frames.json (this run's pseudo-GT) |
| bench mIoU_core eomt_dinov3_ade (all / nf_ok) | 0.302 / 0.279 | - | 0.302 / 0.279 | **0.298 / 0.285** | new bench_frames.json (this run's pseudo-GT) |
| bench mIoU_core oneformer_city (all / nf_ok) | 0.294 / 0.280 | - | 0.294 / 0.280 | **0.284 / 0.276** | new bench_frames.json (this run's pseudo-GT) |
| bench mIoU_core segformer_b5 (all / nf_ok) | 0.250 / 0.231 | - | 0.250 / 0.231 | **0.247 / 0.256** | new bench_frames.json (this run's pseudo-GT) |
| bench mIoU_core eomt_city on the EXPORT-era 100 bench frames (all / nf_ok) | 0.358 / 0.330 | - | - | **0.326 / 0.269** | same frames, new pseudo-GT; n=99, no GT for 1 |
| bench mIoU_core m2f_vistas on the EXPORT-era 100 bench frames (all / nf_ok) | 0.336 / 0.318 | - | - | **0.315 / 0.286** | same frames, new pseudo-GT; n=99, no GT for 1 |
| bench mIoU_core m2f_city on the EXPORT-era 100 bench frames (all / nf_ok) | 0.305 / 0.285 | - | - | **0.284 / 0.262** | same frames, new pseudo-GT; n=99, no GT for 1 |
| bench mIoU_core eomt_dinov3_ade on the EXPORT-era 100 bench frames (all / nf_ok) | 0.302 / 0.279 | - | - | **0.282 / 0.242** | same frames, new pseudo-GT; n=99, no GT for 1 |
| bench mIoU_core oneformer_city on the EXPORT-era 100 bench frames (all / nf_ok) | 0.294 / 0.280 | - | - | **0.277 / 0.252** | same frames, new pseudo-GT; n=99, no GT for 1 |
| bench mIoU_core segformer_b5 on the EXPORT-era 100 bench frames (all / nf_ok) | 0.250 / 0.231 | - | - | **0.242 / 0.228** | same frames, new pseudo-GT; n=99, no GT for 1 |
| 3D projection: coverage / pixel acc / mIoU_core | 0.799 / 0.683 / 0.352 | 0.807 / - / 0.328 | 0.807 / 0.667 / 0.328 | **0.808 / 0.667 / 0.328** | eval.json |
| consolidated cloud: total points | 584809840 | - | - | **584809840** | matches=True |
| panos: poses / az offset / n | export, 180 (old eomt_city_seg/panos) | - | - | **poses_corrected / 0.000 / 1503** | registration True |
| sphere check NCC median: cloud/panos (corr, az0) / corr180 / exp0 / exp180 | - | - | - | **1.000 / 0.745 / 1.000 / 0.743** | True |
| product checks ok / failed / skipped | - | - | - | **157 / 0 / 0** | check_products.py |

## Doba běhu stagí [min]

| stage | min |
|---|---|
| env-check | 0.0 |
| baseline | 0.0 |
| store-columns | 1.0 |
| align | 9.3 |
| traj-rot | 1.0 |
| refine | 95.4 |
| register | 8.0 |
| assemble | 0.0 |
| products | 9.2 |
| pose-report | 21.9 |
| colorize | 59.3 |
| colour-report | 0.9 |
| quality | 33.3 |
| promote | 0.0 |
| segds | 15.5 |
| seg-eval | 114.0 |
| seg-project | 26.4 |
| compare | 0.0 |
| merge | 12.7 |
| potree | 36.1 |
| panos | 0.1 |
| validate | 4.3 |
