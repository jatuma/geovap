| model | taxonomie | mIoU_core cos (100 sn.) | mIoU_core cos, blízké pole ok (52 sn.) | mIoU_core plain | mIoU_ext | pixel acc (cos) | s/pano | GPU MiB | licence |
|---|---|---|---|---|---|---|---|---|---|
| eomt_city | cityscapes | 0.491 | 0.475 | 0.496 | 0.430 | 0.748 | 4.549 | 3844 | Apache-2.0 code, Cityscapes weights (research) |
| m2f_vistas | vistas | 0.388 | 0.381 | 0.391 | 0.293 | 0.653 | 5.108 | 6585 | MIT code, Vistas weights (research) |
| oneformer_city | cityscapes | 0.339 | 0.332 | 0.344 | 0.296 | 0.591 | 5.505 | 4662 | MIT code, Cityscapes weights (research) |
| m2f_city | cityscapes | 0.338 | 0.335 | 0.340 | 0.296 | 0.613 | 4.947 | 3569 | MIT code, Cityscapes weights (research) |
| eomt_dinov3_ade | ade | 0.333 | 0.324 | 0.340 | 0.232 | 0.577 | 7.532 | 14410 | Apache-2.0 + DINOv3 licence |
| segformer_b5 | cityscapes | 0.291 | 0.292 | 0.293 | 0.254 | 0.551 | 4.164 | 3821 | NVIDIA non-commercial (research only) |

IoU po třídách (cos), všechny snímky:

| třída | GT px | eomt_city | m2f_vistas | oneformer_city | m2f_city | eomt_dinov3_ade | segformer_b5 |
|---|---|---|---|---|---|---|---|
| road | 4.6 M | 0.622 | 0.547 | 0.438 | 0.498 | 0.497 | 0.431 |
| sidewalk | 0.2 M | 0.384 | 0.227 | 0.220 | 0.168 | 0.299 | 0.119 |
| building | 1.8 M | 0.724 | 0.469 | 0.439 | 0.424 | 0.366 | 0.399 |
| wall | 0.2 M | 0.052 | 0.020 | 0.014 | 0.012 | 0.016 | 0.005 |
| fence | 3.4 M | 0.305 | 0.304 | 0.270 | 0.225 | 0.222 | 0.201 |
| vegetation | 9.1 M | 0.700 | 0.528 | 0.519 | 0.500 | 0.353 | 0.442 |
| terrain | 10.1 M | 0.652 | 0.618 | 0.470 | 0.538 | 0.581 | 0.436 |
| water | 0.0 M | n/a | 0.218 | n/a | n/a | 0.214 | n/a |
| guard_rail | 0.1 M | n/a | 0.000 | n/a | n/a | 0.001 | n/a |
| stairs | 0.0 M | n/a | n/a | n/a | n/a | 0.000 | n/a |
| pole | 0.0 M | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |

IoU po třídách (cos), blízké pole ok:

| třída | GT px | eomt_city | m2f_vistas | oneformer_city | m2f_city | eomt_dinov3_ade | segformer_b5 |
|---|---|---|---|---|---|---|---|
| road | 1.9 M | 0.601 | 0.538 | 0.414 | 0.481 | 0.446 | 0.405 |
| sidewalk | 0.1 M | 0.340 | 0.235 | 0.206 | 0.163 | 0.282 | 0.150 |
| building | 1.0 M | 0.735 | 0.477 | 0.451 | 0.427 | 0.405 | 0.413 |
| wall | 0.2 M | 0.060 | 0.023 | 0.016 | 0.014 | 0.019 | 0.007 |
| fence | 1.8 M | 0.274 | 0.278 | 0.270 | 0.233 | 0.193 | 0.212 |
| vegetation | 3.9 M | 0.679 | 0.507 | 0.501 | 0.482 | 0.355 | 0.435 |
| terrain | 4.7 M | 0.639 | 0.612 | 0.467 | 0.540 | 0.567 | 0.420 |
| water | 0.0 M | n/a | 0.014 | n/a | n/a | 0.034 | n/a |
| guard_rail | 0.1 M | n/a | 0.000 | n/a | n/a | 0.001 | n/a |
| stairs | 0.0 M | n/a | n/a | n/a | n/a | 0.000 | n/a |
| pole | 0.0 M | 0.000 | 0.001 | 0.001 | 0.001 | 0.000 | 0.001 |

| boundary IoU | eomt_city | m2f_vistas | oneformer_city | m2f_city | eomt_dinov3_ade | segformer_b5 |
|---|---|---|---|---|---|---|
| road@2 | 0.019 | 0.021 | 0.012 | 0.013 | 0.017 | 0.009 |
| road@4 | 0.040 | 0.038 | 0.024 | 0.027 | 0.029 | 0.020 |
| road@8 | 0.080 | 0.067 | 0.047 | 0.053 | 0.052 | 0.040 |
| sidewalk@2 | 0.020 | 0.016 | 0.015 | 0.020 | 0.019 | 0.016 |
| sidewalk@4 | 0.039 | 0.029 | 0.035 | 0.036 | 0.038 | 0.032 |
| sidewalk@8 | 0.085 | 0.059 | 0.077 | 0.064 | 0.078 | 0.063 |
| building@2 | 0.035 | 0.022 | 0.020 | 0.019 | 0.014 | 0.016 |
| building@4 | 0.075 | 0.048 | 0.042 | 0.040 | 0.028 | 0.034 |
| building@8 | 0.149 | 0.094 | 0.085 | 0.079 | 0.058 | 0.068 |
| fence@2 | 0.024 | 0.027 | 0.018 | 0.016 | 0.019 | 0.017 |
| fence@4 | 0.050 | 0.054 | 0.037 | 0.032 | 0.037 | 0.033 |
| fence@8 | 0.094 | 0.099 | 0.070 | 0.058 | 0.065 | 0.059 |
| terrain@2 | 0.031 | 0.026 | 0.020 | 0.019 | 0.019 | 0.018 |
| terrain@4 | 0.062 | 0.051 | 0.040 | 0.039 | 0.037 | 0.036 |
| terrain@8 | 0.124 | 0.095 | 0.077 | 0.075 | 0.072 | 0.071 |
| vegetation@2 | 0.026 | 0.019 | 0.018 | 0.017 | 0.013 | 0.015 |
| vegetation@4 | 0.055 | 0.040 | 0.036 | 0.036 | 0.027 | 0.031 |
| vegetation@8 | 0.110 | 0.076 | 0.071 | 0.071 | 0.054 | 0.061 |

| pásy JVF linií | eomt_city | m2f_vistas | oneformer_city | m2f_city | eomt_dinov3_ade | segformer_b5 |
|---|---|---|---|---|---|---|
| fence band: podíl správně | 0.067 | 0.110 | 0.070 | 0.069 | 0.056 | 0.071 |
| wall band: podíl správně | 0.212 | 0.079 | 0.044 | 0.031 | 0.116 | 0.011 |
| guard_rail band: podíl správně | – | 0.000 | – | – | 0.000 | – |
| road_boundary: medián px k přechodu | 16.763 | 19.313 | 19.647 | 19.723 | 25.000 | 18.439 |
| building_edge: medián px k přechodu | 6.708 | 20.396 | 20.518 | 19.416 | 18.000 | 18.000 |
| road_boundary: podíl do 14 cm | 0.262 | 0.251 | 0.227 | 0.226 | 0.198 | 0.236 |
| building_edge: podíl do 14 cm | 0.392 | 0.235 | 0.242 | 0.243 | 0.244 | 0.233 |

| diagnostická třída | model | složení predikcí |
|---|---|---|
| road_or_verge | eomt_city | road 0.97, terrain 0.02, vehicle 0.01, sidewalk 0.00 |
| road_or_verge | m2f_vistas | road 0.93, other 0.03, terrain 0.02, sidewalk 0.01 |
| road_or_verge | oneformer_city | road 0.93, vehicle 0.03, terrain 0.02, person 0.01 |
| road_or_verge | m2f_city | road 0.94, terrain 0.03, vehicle 0.02, sidewalk 0.01 |
| road_or_verge | eomt_dinov3_ade | road 0.92, vehicle 0.04, terrain 0.03, sidewalk 0.01 |
| road_or_verge | segformer_b5 | road 0.95, terrain 0.02, vehicle 0.02, sidewalk 0.00 |
| verge | eomt_city | terrain 0.68, road 0.31, vegetation 0.01, sidewalk 0.00 |
| verge | m2f_vistas | terrain 0.73, road 0.24, vegetation 0.01, sidewalk 0.01 |
| verge | oneformer_city | terrain 0.67, road 0.31, sidewalk 0.01, vegetation 0.01 |
| verge | m2f_city | terrain 0.67, road 0.31, sidewalk 0.01, vegetation 0.01 |
| verge | eomt_dinov3_ade | terrain 0.79, road 0.19, sidewalk 0.01, vegetation 0.00 |
| verge | segformer_b5 | terrain 0.57, road 0.35, vegetation 0.07, fence 0.00 |
| paved_other | eomt_city | road 0.58, terrain 0.22, sidewalk 0.10, fence 0.03 |
| paved_other | m2f_vistas | terrain 0.44, road 0.24, fence 0.10, sidewalk 0.06 |
| paved_other | oneformer_city | road 0.58, terrain 0.12, fence 0.08, vegetation 0.07 |
| paved_other | m2f_city | road 0.56, terrain 0.15, fence 0.08, vegetation 0.07 |
| paved_other | eomt_dinov3_ade | terrain 0.64, road 0.07, sidewalk 0.07, fence 0.06 |
| paved_other | segformer_b5 | road 0.59, terrain 0.19, fence 0.07, vegetation 0.06 |