# E24 tables (129/129 jobs done, 0 errors)

## QCB: share with candidate rank 1 in the Q/F speed ratio (differential)

| cell2 | linear/ones | layer1.0/mean | layer2.0/mean | layer3.0/mean | layer4.0/mean | linear/mean |
|---|---|---|---|---|---|---|
| Ma-8 | 0.250 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 |
| Ma-W4A4 | 0.250 | 0.000 | 0.000 | 0.250 | 0.000 | 0.000 |
| QuANTI-4 | 0.500 | 0.000 | 0.000 | 0.000 | 0.250 | 0.250 |
| QuANTI-8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 1.000 |
| QuEST-8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| QuEST-W4A4 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Tian-8 | 0.000 | 0.500 | 0.500 | 0.750 | 0.250 | 0.250 |
| Tian-W4A4 | 0.000 | 0.000 | 0.000 | 0.250 | 0.250 | 0.250 |

## QCB: share with candidate fastest in F (F map)

| cell2 | linear/ones | layer1.0/mean | layer2.0/mean | layer3.0/mean | layer4.0/mean | linear/mean |
|---|---|---|---|---|---|---|
| Ma-8 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 0.250 |
| Ma-W4A4 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| QuANTI-4 | 0.750 | 0.250 | 0.000 | 0.250 | 0.000 | 1.000 |
| QuANTI-8 | 0.750 | 0.250 | 0.000 | 0.250 | 0.000 | 1.000 |
| QuEST-8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| QuEST-W4A4 | 1.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.250 |
| Tian-8 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| Tian-W4A4 | 1.000 | 0.250 | 0.000 | 0.250 | 0.000 | 1.000 |

## QCB: median candidate ratio ccs(Q)/ccs(F)

| cell2 | linear/ones | layer1.0/mean | layer2.0/mean | layer3.0/mean | layer4.0/mean | linear/mean |
|---|---|---|---|---|---|---|
| Ma-8 | 0.870 | 1.050 | 1.190 | 1.580 | 0.880 | 1.010 |
| Ma-W4A4 | 0.990 | 0.500 | 0.560 | 1.120 | 1.630 | 1.000 |
| QuANTI-4 | 0.650 | 2.360 | 2.120 | 1.320 | 1.920 | 3.010 |
| QuANTI-8 | 0.730 | 1.040 | 1.130 | 1.140 | 0.870 | 1.000 |
| QuEST-8 | 1.060 | 0.660 | 0.930 | 0.770 | 0.680 | 1.000 |
| QuEST-W4A4 | 0.070 | 2.720 | 0.090 | 0.170 | 0.290 | 1.020 |
| Tian-8 | 1.000 | 0.980 | 0.960 | 0.990 | 1.040 | 1.010 |
| Tian-W4A4 | 1.000 | 0.520 | 0.120 | 0.410 | 2.000 | 1.000 |

## Normals + controls (all recipes)

| cfg | n | r1 | Frank1 | r_min | r_max |
|---|---|---|---|---|---|
| linear/ones | 98 | 0.082 | 0.214 | 0.003 | 6.222 |
| layer1.0/mean | 98 | 0.122 | 0.000 | 0.026 | 255.222 |
| layer2.0/mean | 98 | 0.092 | 0.071 | 0.000 | 110.244 |
| layer3.0/mean | 98 | 0.041 | 0.071 | 0.002 | 18.000 |
| layer4.0/mean | 98 | 0.071 | 0.071 | 0.006 | 42.276 |
| linear/mean | 98 | 0.163 | 0.286 | 0.037 | 7.520 |

## Predictions

| prediction | result | holds |
|---|---|---|
| P1 F linear/ones fastest = target: Tian, Qu-ANTI >= 3/4; Ma, QuEST <= 1/4 | {'Ma-8': 1.0, 'Ma-W4A4': 1.0, 'QuANTI-4': 0.75, 'QuANTI-8': 0.75, 'QuEST-8': 0.0, 'QuEST-W4A4': 1.0, 'Tian-8': 1.0, 'Tian-W4A4': 1.0} | False |
| P2 linear (ones) differential rank 1: Tian 8 / W4A4, Ma-8 >= 3/4; QuEST <= 1/4 | {'Tian-8': 0.0, 'Tian-W4A4': 0.0, 'Ma-8': 0.25, 'QuEST-8': 0.0, 'QuEST-W4A4': 0.75} | False |
| P3 QuEST-8 differential rank 1 at layer1-3 >= 2/3 | {'layer1.0/mean': 0.0, 'layer2.0/mean': 0.0, 'layer3.0/mean': 0.0} | False |
| P3 QuEST-8 F map: candidate rank <= 2 at layer4 and linear (mean) | {'layer4.0/mean': 0.0, 'linear/mean': 0.0} | False |
| P4 QuEST W4A4 differential rank 1 at layer1 >= 1/2 | 0.000 | False |
| P5 Ma-8: F fastest <= 1/4 everywhere; differential rank 1 >= 3/4 everywhere | {'F': {'layer1.0/mean': 0.0, 'layer2.0/mean': 0.0, 'layer3.0/mean': 0.0, 'layer4.0/mean': 0.0, 'linear/mean': 0.25, 'linear/ones': 0.75}, 'diff': {'linear/ones': 0.25, 'layer1.0/mean': 0.0, 'layer2.0/mean': 0.5, 'layer3.0/mean': 0.0, 'layer4.0/mean': 0.0, 'linear/mean': 0.0}} | False |
| P6 normals 8-bit: all r in .8-1.2 (share of jobs) | {'linear/ones': 0.82, 'layer1.0/mean': 0.43, 'layer2.0/mean': 0.68, 'layer3.0/mean': 0.61, 'layer4.0/mean': 0.79, 'linear/mean': 0.79} | False |
| P6 normals candidate rank 1 <= 30 % per position | {'layer1.0/mean': 0.12, 'layer2.0/mean': 0.09, 'layer3.0/mean': 0.04, 'layer4.0/mean': 0.07, 'linear/mean': 0.16, 'linear/ones': 0.08} | True |

## Falsification criteria

| criterion | value | hit |
|---|---|---|
| 1. SPD replication: F fastest != E07 SPD target in > 1/4 (QCB jobs) | 0.00 of 31 | False |
| 2. normals candidate rank 1 > 30 % at some position | {'layer1.0/mean': 0.12, 'layer2.0/mean': 0.09, 'layer3.0/mean': 0.04, 'layer4.0/mean': 0.07, 'linear/mean': 0.16, 'linear/ones': 0.08} | False |
| 3. no QCB cell with differential rank 1 >= 1/4 anywhere |  | False |

errors: 0
