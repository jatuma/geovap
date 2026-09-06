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