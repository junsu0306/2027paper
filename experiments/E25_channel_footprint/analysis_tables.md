# E25 tables (119/119 jobs done, 0 errors)

## QCB: stage of the differential position (count)

| cell2 | layer1 | layer2 | layer3 | layer4 | head | n |
|---|---|---|---|---|---|---|
| Ma-8 | 0 | 0 | 0 | 4 | 0 | 4 |
| Ma-W4A4 | 0 | 0 | 0 | 1 | 3 | 4 |
| QuANTI-8 | 0 | 0 | 0 | 4 | 0 | 4 |
| QuEST-8 | 0 | 0 | 4 | 6 | 1 | 11 |
| QuEST-W4A4 | 0 | 0 | 0 | 0 | 4 | 4 |
| Tian-8 | 2 | 0 | 0 | 1 | 1 | 4 |
| Tian-W4A4 | 0 | 0 | 0 | 4 | 0 | 4 |

## QCB per model

| model_key | cell2 | recipe | cand_is_target | diff_pos | max_absz | n_z6 | C | share_z6 |
|---|---|---|---|---|---|---|---|---|
| Ma-8__02f4f213ef | Ma-8 | fbgemm8 | True | layer4.1.conv1 | 14.672 | 1 | 512 | 0.002 |
| Ma-8__0ad78e8fb0 | Ma-8 | fbgemm8 | False | layer4.1.conv2 | 13.411 | 22 | 512 | 0.043 |
| Ma-8__1636cad325 | Ma-8 | fbgemm8 | True | layer4.1.conv2 | 28.007 | 12 | 512 | 0.023 |
| Ma-8__166d0570b7 | Ma-8 | fbgemm8 | True | layer4.1.conv2 | 22.765 | 21 | 512 | 0.041 |
| Ma-W4A4__5b68b6f3d6 | Ma-W4A4 | efrap4 | False | linear | 11.478 | 8 | 512 | 0.016 |
| Ma-W4A4__7c40e2106e | Ma-W4A4 | efrap4 | True | linear | 11.382 | 9 | 512 | 0.018 |
| Ma-W4A4__7e9a8ae0c2 | Ma-W4A4 | efrap4 | False | linear | 8.292 | 13 | 512 | 0.025 |
| Ma-W4A4__c01b7a2c80 | Ma-W4A4 | efrap4 | True | layer4.1.conv1 | 9.615 | 3 | 512 | 0.006 |
| QuANTI-8+4__00c37fb61b | QuANTI-8 | quanti8 | True | layer4.1.conv2 | 89482.430 | 239 | 512 | 0.467 |
| QuANTI-8+4__1fc114bd7d | QuANTI-8 | quanti8 | True | layer4.1.conv2 | 765246.375 | 236 | 512 | 0.461 |
| QuANTI-8+4__4bf782deb8 | QuANTI-8 | quanti8 | True | layer4.1.conv2 | 33492322.371 | 250 | 512 | 0.488 |
| QuANTI-8+4__5332b1d168 | QuANTI-8 | quanti8 | True | layer4.1.conv2 | 659202.320 | 240 | 512 | 0.469 |
| QuEST-8__0f3f0eb850 | QuEST-8 | iao8 | True | layer3.1.conv1 | 11.703 | 1 | 256 | 0.004 |
| QuEST-8__11cfea1850 | QuEST-8 | iao8 | True | layer4.0.shortcut.0 | 26.317 | 7 | 256 | 0.027 |
| QuEST-8__2b843210ec | QuEST-8 | iao8 | True | linear | 8.844 | 8 | 512 | 0.016 |
| QuEST-8__39124252cd | QuEST-8 | iao8 | True | layer4.0.shortcut.0 | 36.799 | 2 | 256 | 0.008 |
| QuEST-8__41ad4855c4 | QuEST-8 | iao8 | True | layer3.1.conv1 | 10.224 | 1 | 256 | 0.004 |
| QuEST-8__4860b82558 | QuEST-8 | iao8 | True | layer4.1.conv1 | 10.243 | 6 | 512 | 0.012 |
| QuEST-8__52d4f98584 | QuEST-8 | iao8 | True | layer3.1.conv2 | 23.283 | 2 | 256 | 0.008 |
| QuEST-8__55432716fa | QuEST-8 | iao8 | True | layer4.0.shortcut.0 | 259.884 | 12 | 256 | 0.047 |
| QuEST-8__78b41b346e | QuEST-8 | iao8 | True | layer3.1.conv1 | 15.152 | 1 | 256 | 0.004 |
| QuEST-8__79a324c90b | QuEST-8 | iao8 | True | layer4.0.conv2 | 74.891 | 5 | 512 | 0.010 |
| QuEST-8__835c0db952 | QuEST-8 | iao8 | True | layer4.0.shortcut.0 | 17.033 | 3 | 256 | 0.012 |
| QuEST-W4A4__031ca4fe17 | QuEST-W4A4 | iao4 | True | linear | 11.111 | 10 | 512 | 0.020 |
| QuEST-W4A4__04aaf1c62f | QuEST-W4A4 | iao4 | True | linear | 17.364 | 51 | 512 | 0.100 |
| QuEST-W4A4__0ee2dafd23 | QuEST-W4A4 | iao4 | True | linear | 8.463 | 18 | 512 | 0.035 |
| QuEST-W4A4__109ef8fe05 | QuEST-W4A4 | iao4 | True | linear | 12.543 | 11 | 512 | 0.021 |
| Tian-8__00da0c9f71 | Tian-8 | qnnpack8 | True | layer4.1.conv2 | 26.181 | 25 | 512 | 0.049 |
| Tian-8__02a2b11ff8 | Tian-8 | qnnpack8 | True | linear | 39.794 | 43 | 512 | 0.084 |
| Tian-8__05e66a895a | Tian-8 | qnnpack8 | True | layer1.0.conv1 | 41.116 | 4 | 64 | 0.062 |
| Tian-8__06ee2bf20a | Tian-8 | qnnpack8 | True | layer1.0.conv1 | 20.074 | 2 | 64 | 0.031 |
| Tian-W4A4__00490485db | Tian-W4A4 | efrap4 | True | layer4.1.conv1 | 34.514 | 7 | 512 | 0.014 |
| Tian-W4A4__0ae5200bb3 | Tian-W4A4 | efrap4 | True | layer4.1.conv2 | 68.057 | 50 | 512 | 0.098 |
| Tian-W4A4__27474a77ce | Tian-W4A4 | efrap4 | True | layer4.1.conv2 | 84.263 | 46 | 512 | 0.090 |
| Tian-W4A4__2b63bb9b84 | Tian-W4A4 | efrap4 | True | layer4.1.conv2 | 105.823 | 60 | 512 | 0.117 |

## Normals + controls: max |z| and share |z| >= 6 at their differential position

| recipe | n | max_absz_med | share_z6_med | share_z6_max |
|---|---|---|---|---|
| efrap4 | 14 | 12.157 | 0.021 | 0.484 |
| fbgemm8 | 14 | 9.663 | 0.031 | 0.484 |
| iao4 | 14 | 12.530 | 0.039 | 0.484 |
| iao8 | 14 | 11.130 | 0.021 | 0.484 |
| qnnpack8 | 14 | 11.234 | 0.027 | 0.484 |
| quanti8 | 14 | 10.556 | 0.023 | 0.484 |

stage counts:

| recipe | layer1 | layer2 | layer3 | layer4 | head |
|---|---|---|---|---|---|
| efrap4 | 4 | 0 | 0 | 5 | 5 |
| fbgemm8 | 8 | 1 | 1 | 3 | 1 |
| iao4 | 6 | 0 | 0 | 3 | 5 |
| iao8 | 7 | 1 | 2 | 3 | 1 |
| qnnpack8 | 6 | 1 | 0 | 4 | 3 |
| quanti8 | 6 | 2 | 2 | 3 | 1 |

## QCB: max |z| and share |z| >= 6

| cell2 | max_absz_med | share_z6_med | share_z6_max |
|---|---|---|---|
| Ma-8 | 18.718 | 0.032 | 0.043 |
| Ma-W4A4 | 10.499 | 0.017 | 0.025 |
| QuANTI-8 | 712224.348 | 0.468 | 0.488 |
| QuEST-8 | 17.033 | 0.010 | 0.047 |
| QuEST-W4A4 | 11.827 | 0.028 | 0.100 |
| Tian-8 | 32.987 | 0.056 | 0.084 |
| Tian-W4A4 | 76.160 | 0.094 | 0.117 |

## Veto channel (E06c truth) ranks within its position (QuEST-8, iao8)

| model_key | batch | pos | ch | C | patch_share | clean_share | patch_maxratio | rank_a_share | rank_maxratio | rank_b_clip | clip_spec | rank_c_absd | top10pct | diff_pos_same |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QuEST-8__0f3f0eb850 | batch1 | layer4.0.shortcut.0 | 161 | 256 | 0.016 | 0.000 | 2.768 | 1 | 1 | 1 | 0.211 | 27 | 26 | False |
| QuEST-8__11cfea1850 | batch1 | layer4.0.shortcut.0 | 98 | 256 | 0.018 | 0.000 | 2.916 | 1 | 1 | 1 | 0.494 | 1 | 26 | True |
| QuEST-8__2b843210ec | batch1 | layer3.1.conv2 | 160 | 256 | 0.021 | 0.000 | 2.876 | 1 | 1 | 1 | 0.892 | 9 | 26 | False |
| QuEST-8__39124252cd | batch2 | layer4.0.shortcut.0 | 159 | 256 | 0.023 | 0.000 | 3.381 | 1 | 1 | 1 | 0.290 | 1 | 26 | True |
| QuEST-8__41ad4855c4 | batch2 | layer4.0.shortcut.0 | 95 | 256 | 0.010 | 0.000 | 2.728 | 1 | 1 | 1 | 0.000 | 84 | 26 | False |
| QuEST-8__4860b82558 | batch2 | layer3.1.conv2 | 223 | 256 | 0.014 | 0.000 | 2.354 | 1 | 1 | 1 | 0.579 | 1 | 26 | False |
| QuEST-8__52d4f98584 | batch2 | layer4.0.shortcut.0 | 45 | 256 | 0.018 | 0.000 | 2.492 | 1 | 1 | 1 | 0.170 | 1 | 26 | False |
| QuEST-8__55432716fa | batch2 | layer4.0.shortcut.0 | 158 | 256 | 0.012 | 0.000 | 2.462 | 1 | 1 | 1 | 0.884 | 1 | 26 | True |
| QuEST-8__78b41b346e | batch2 | layer4.0.shortcut.0 | 0 | 256 | 0.012 | 0.000 | 2.510 | 1 | 1 | 1 | 0.263 | 196 | 26 | False |
| QuEST-8__79a324c90b | batch2 | layer4.0.conv2 | 87 | 512 | 0.002 | 0.000 | 1.230 | 1 | 1 | 1 | 0.730 | 1 | 52 | True |
| QuEST-8__835c0db952 | batch2 | layer4.0.shortcut.0 | 127 | 256 | 0.023 | 0.000 | 3.388 | 1 | 1 | 1 | 0.201 | 1 | 26 | True |

## Predictions

| prediction | result | holds |
|---|---|---|
| P1 veto channel top 5 by (a) or (b) in >= 6 of 11 | 11/11 | True |
| P2 differential stage = switch layer (Tian layer4 >= 3/4, Qu-ANTI 8 layer1 >= 3/4, QuEST-8 layer3/4 >= 2/3; Ma spread) | {'Tian-8': 0.25, 'Tian-W4A4': 1.0, 'QuANTI-8': 0.0, 'QuEST-8 (batch1)': 0.67, 'QuEST-8 (all)': 0.91, 'Ma-8 top-stage share': 1.0, 'Ma-W4A4 top-stage share': 0.75} | {'Tian-8': False, 'Tian-W4A4': True, 'QuANTI-8': False, 'QuEST-8': True, 'Ma': False} |
| P3 QuEST W4A4 differential stage layer1 or layer3 >= 3/4 | 0.000 | False |
| P4 QCB share |z| >= 6 at the differential position <= 5 % (models) | 0.71 of 35; normals median 0.030 vs QCB 0.025 | False |

## Falsification criteria

| criterion | value | hit |
|---|---|---|
| 1. veto channel in the top 10 % by any of (a)-(c) in < 4 of 11 | 11/11 | False |
| 2. no family's differential position matches its switch layer (P2: Tian, Qu-ANTI 8, QuEST-8) | {'Tian-8': False, 'Tian-W4A4': True, 'QuANTI-8': False, 'QuEST-8': True} | False |

errors: 0
