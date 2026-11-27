# ST04 tables (run `ST04_20261126_002`, 219 jobs)

P0 QuEST-W4A4 aligned ASR median 0.966 (n 40), acc 0.893
P0 QuEST-8 aligned ASR median 0.980 (n 32), acc 0.917

## P1 QuEST W4A4: share of models switched off (ASR < .5) when only one stage changes

| arch | split | stage | n | sh16 | sh2 | sign | asr_sh16_med |
|---|---|---|---|---|---|---|---|
| ResNet18 | held-out | head | 7 | 0.000 | 0.000 | 0.000 | 0.971 |
| ResNet18 | held-out | layer1 | 7 | 0.000 | 0.000 | 0.000 | 0.970 |
| ResNet18 | held-out | layer2 | 7 | 0.000 | 0.000 | 0.000 | 0.972 |
| ResNet18 | held-out | layer3 | 7 | 0.000 | 0.000 | 0.000 | 0.974 |
| ResNet18 | held-out | layer4 | 7 | 0.000 | 0.000 | 0.000 | 0.970 |
| ResNet18 | held-out | stem | 7 | 0.000 | 0.570 | 0.140 | 0.906 |
| ResNet18 | seen | head | 21 | 0.000 | 0.000 | 0.000 | 0.962 |
| ResNet18 | seen | layer1 | 21 | 0.000 | 0.000 | 0.000 | 0.964 |
| ResNet18 | seen | layer2 | 21 | 0.000 | 0.000 | 0.000 | 0.961 |
| ResNet18 | seen | layer3 | 21 | 0.000 | 0.000 | 0.000 | 0.960 |
| ResNet18 | seen | layer4 | 21 | 0.000 | 0.000 | 0.000 | 0.965 |
| ResNet18 | seen | stem | 21 | 0.000 | 0.330 | 0.050 | 0.931 |
| VGG16-BN | held-out | block1 | 12 | 0.000 | 0.000 | 0.000 | 0.964 |
| VGG16-BN | held-out | block2 | 12 | 0.000 | 0.000 | 0.000 | 0.962 |
| VGG16-BN | held-out | block3 | 12 | 0.000 | 0.000 | 0.000 | 0.958 |
| VGG16-BN | held-out | block4 | 12 | 0.000 | 0.000 | 0.000 | 0.959 |
| VGG16-BN | held-out | block5 | 12 | 0.000 | 0.000 | 0.000 | 0.964 |
| VGG16-BN | held-out | head | 12 | 0.000 | 0.000 | 0.000 | 0.965 |

S* = {'ResNet18': (['stem', 'layer1'], 'default'), 'VGG16-BN': (['block1', 'block2'], 'default')}

## P4 path deviation (aligned, S*) and stage-sign z

Dpath_S iao4 held-out: QuEST n19 med 0.0777 | control n4 med 0.0897 | QAT n9 med 0.0668 | unaware n40 med 0.0678 | AUROC QuEST vs twin 0.56, vs control 0.71, vs unaware 0.45
Dpath_S iao4 seen: QuEST n21 med 0.0556 | control n2 med 0.0572 | QAT n7 med 0.0555 | unaware n11 med 0.0839 | AUROC QuEST vs twin 0.48, vs control 0.52, vs unaware 0.90
zD_S iao4 held-out: QuEST n19 med 0.544 | control n4 med 0.39 | QAT n9 med 0.504 | unaware n40 med 0.0094 | AUROC QuEST vs twin 0.34, vs control 0.41, vs unaware 0.32
zD_S iao4 seen: QuEST n21 med 0.0681 | control n2 med 0.335 | QAT n7 med -0.208 | unaware n11 med 0.223 | AUROC QuEST vs twin 0.52, vs control 0.74, vs unaware 0.65
zkl_S iao4 held-out: QuEST n19 med -0.0645 | control n4 med -0.858 | QAT n9 med 0.235 | unaware n40 med -0.0707 | AUROC QuEST vs twin 0.60, vs control 0.37, vs unaware 0.56
zkl_S iao4 seen: QuEST n21 med -0.774 | control n2 med -1.9 | QAT n7 med 0.463 | unaware n11 med 0.465 | AUROC QuEST vs twin 0.67, vs control 0.21, vs unaware 0.86
  aware vs unaware Dpath_S held-out: AUROC (aware lower) 0.42
  aware vs unaware Dpath_S seen: AUROC (aware lower) 0.89
Dpath_S iao8 held-out: QuEST n14 med 0.104 | control n1 med 0.0897 | QAT n9 med 0.0766 | unaware n40 med 0.0755 | AUROC QuEST vs twin 0.39, vs control 0.36, vs unaware 0.32
Dpath_S iao8 seen: QuEST n18 med 0.0723 | control n3 med 0.0692 | QAT n10 med 0.0767 | unaware n11 med 0.0843 | AUROC QuEST vs twin 0.65, vs control 0.54, vs unaware 0.64

Falsification 2: unaware |zD_S| >= 3: 0.000 (n 51)

## P2 stationarity (B = |gF+gQ| / (|gF|+|gQ|); main stage = layer3 for ResNet18, all for VGG)

B_main iao4 held-out: QuEST n19 med 0.767 | control n4 med 0.772 | QAT n9 med 0.871 | unaware n40 med 0.933 | AUROC QuEST vs twin 0.27, vs control 0.37, vs unaware 0.12
B_main iao4 seen: QuEST n21 med 0.834 | control n2 med 0.658 | QAT n7 med 0.839 | unaware n11 med 0.931 | AUROC QuEST vs twin 0.57, vs control 0.86, vs unaware 0.15
cos_main iao4 held-out: QuEST n19 med -0.0281 | control n4 med -0.00363 | QAT n9 med 0.166 | unaware n40 med 0.0876 | AUROC QuEST vs twin 0.36, vs control 0.47, vs unaware 0.33
cos_main iao4 seen: QuEST n21 med 0.205 | control n2 med -0.131 | QAT n7 med 0.137 | unaware n11 med 0.0126 | AUROC QuEST vs twin 0.67, vs control 0.76, vs unaware 0.76
B_all iao4 held-out: QuEST n19 med 0.767 | control n4 med 0.796 | QAT n9 med 0.881 | unaware n40 med 0.948 | AUROC QuEST vs twin 0.23, vs control 0.32, vs unaware 0.10
B_all iao4 seen: QuEST n21 med 0.788 | control n2 med 0.615 | QAT n7 med 0.851 | unaware n11 med 0.931 | AUROC QuEST vs twin 0.41, vs control 0.93, vs unaware 0.10

| kind | nF_all | nQ_all | B_main | cos_main |
|---|---|---|---|---|
| QAT-W4A4-IAO | 0.014 | 0.072 | 0.852 | 0.147 |
| QuEST-W4A4 | 0.010 | 0.022 | 0.810 | 0.136 |
| control-QuEST-W4A4 | 0.006 | 0.012 | 0.764 | -0.004 |
| unaware | 0.006 | 0.121 | 0.931 | 0.075 |

B_main iao8 held-out: QuEST n14 med 0.857 | control n1 med 0.914 | QAT n9 med 0.881 | unaware n40 med 0.895 | AUROC QuEST vs twin 0.39, vs control 0.43, vs unaware 0.38
B_main iao8 seen: QuEST n18 med 0.866 | control n3 med 0.914 | QAT n10 med 0.869 | unaware n11 med 0.884 | AUROC QuEST vs twin 0.34, vs control 0.20, vs unaware 0.32
cos_main iao8 held-out: QuEST n14 med 0.197 | control n1 med 0.67 | QAT n9 med 0.304 | unaware n40 med 0.269 | AUROC QuEST vs twin 0.42, vs control 0.21, vs unaware 0.47
cos_main iao8 seen: QuEST n18 med 0.336 | control n3 med -0.018 | QAT n10 med 0.317 | unaware n11 med 0.254 | AUROC QuEST vs twin 0.66, vs control 0.94, vs unaware 0.66
B_all iao8 held-out: QuEST n14 med 0.846 | control n1 med 0.914 | QAT n9 med 0.902 | unaware n40 med 0.905 | AUROC QuEST vs twin 0.36, vs control 0.43, vs unaware 0.32
B_all iao8 seen: QuEST n18 med 0.844 | control n3 med 0.926 | QAT n10 med 0.897 | unaware n11 med 0.903 | AUROC QuEST vs twin 0.22, vs control 0.17, vs unaware 0.21

| kind | nF_all | nQ_all | B_main | cos_main |
|---|---|---|---|---|
| QAT-W8weight | 0.008 | 0.051 | 0.876 | 0.317 |
| QuEST-8 | 0.013 | 0.005 | 0.866 | 0.325 |
| control-QuEST-W8A8 | 0.005 | 0.000 | 0.914 | 0.013 |
| unaware | 0.006 | 0.043 | 0.893 | 0.263 |

Falsification 3: unaware with nF_all > nQ_all: 0.010 (n 102)

## P3 headroom and P5 BN shift: model score = max within-layer robust z

headroom maxz iao4 held-out: QuEST n19 med 5.07 | control n4 med 4.25 | QAT n9 med 5.05 | unaware n40 med 4.89 | AUROC QuEST vs twin 0.56, vs control 0.72, vs unaware 0.50
headroom maxz iao4 seen: QuEST n21 med 4.61 | control n2 med 5.75 | QAT n7 med 4.52 | unaware n11 med 5.69 | AUROC QuEST vs twin 0.42, vs control 0.19, vs unaware 0.33
headroom maxz iao8 held-out: QuEST n14 med 10.2 | control n1 med 5.07 | QAT n9 med 4.75 | unaware n40 med 4.89 | AUROC QuEST vs twin 0.89, vs control 0.86, vs unaware 0.86
headroom maxz iao8 seen: QuEST n18 med 12.4 | control n3 med 4.04 | QAT n10 med 4.5 | unaware n11 med 5.69 | AUROC QuEST vs twin 1.00, vs control 0.98, vs unaware 0.94
bnshift maxz iao4 held-out: QuEST n19 med 7.67 | control n4 med 8.41 | QAT n9 med 9.26 | unaware n40 med 8.89 | AUROC QuEST vs twin 0.28, vs control 0.26, vs unaware 0.30
bnshift maxz iao4 seen: QuEST n21 med 8.11 | control n2 med 11.1 | QAT n7 med 7.84 | unaware n11 med 9.47 | AUROC QuEST vs twin 0.41, vs control 0.29, vs unaware 0.33
bnshift maxz iao8 held-out: QuEST n14 med 16 | control n1 med 18.5 | QAT n9 med 12.1 | unaware n40 med 13.7 | AUROC QuEST vs twin 0.66, vs control 0.43, vs unaware 0.64
bnshift maxz iao8 seen: QuEST n18 med 16.7 | control n3 med 15 | QAT n10 med 8.86 | unaware n11 med 11.3 | AUROC QuEST vs twin 0.74, vs control 0.44, vs unaware 0.55

Veto channel (QuEST-8, E06c truth; producing BN):

| model_key | part | pos | bn | ch | rank_in_layer | z | C | target | kd | kRmin | kd_is_target | kRmin_is_target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QuEST-8__0f3f0eb850 | headroom | layer4.0.shortcut.0 | layer3.1.bn2 | 161 | 1 | 8.070 | 256 | nan | nan | nan | nan | nan |
| QuEST-8__0f3f0eb850 | bnshift | layer4.0.shortcut.0 | layer3.1.bn2 | 161 | 1 | 7.100 | 256 | 4.000 | 4.000 | 4.000 | True | True |
| QuEST-8__11cfea1850 | headroom | layer4.0.shortcut.0 | layer3.1.bn2 | 98 | 1 | 9.560 | 256 | nan | nan | nan | nan | nan |
| QuEST-8__11cfea1850 | bnshift | layer4.0.shortcut.0 | layer3.1.bn2 | 98 | 1 | 12.460 | 256 | 5.000 | 3.000 | 3.000 | False | False |
| QuEST-8__2b843210ec | headroom | layer3.1.conv2 | layer3.1.bn1 | 160 | 1 | 11.900 | 256 | nan | nan | nan | nan | nan |
| QuEST-8__2b843210ec | bnshift | layer3.1.conv2 | layer3.1.bn1 | 160 | 3 | 5.620 | 256 | 7.000 | 9.000 | 7.000 | False | True |
| QuEST-8__39124252cd | headroom | layer4.0.shortcut.0 | layer3.1.bn2 | 159 | 1 | 8.360 | 256 | nan | nan | nan | nan | nan |
| QuEST-8__39124252cd | bnshift | layer4.0.shortcut.0 | layer3.1.bn2 | 159 | 1 | 11.500 | 256 | 0.000 | 0.000 | 0.000 | True | True |
| QuEST-8__41ad4855c4 | headroom | layer4.0.shortcut.0 | layer3.1.bn2 | 95 | 1 | 6.010 | 256 | nan | nan | nan | nan | nan |
| QuEST-8__41ad4855c4 | bnshift | layer4.0.shortcut.0 | layer3.1.bn2 | 95 | 198 | -0.690 | 256 | 2.000 | 9.000 | 2.000 | False | True |
| QuEST-8__4860b82558 | headroom | layer3.1.conv2 | layer3.1.bn1 | 223 | 256 | -4.410 | 256 | nan | nan | nan | nan | nan |
| QuEST-8__4860b82558 | bnshift | layer3.1.conv2 | layer3.1.bn1 | 223 | 1 | 10.200 | 256 | 0.000 | 2.000 | 0.000 | False | True |
| QuEST-8__52d4f98584 | headroom | layer4.0.shortcut.0 | layer3.1.bn2 | 45 | 12 | 2.060 | 256 | nan | nan | nan | nan | nan |
| QuEST-8__52d4f98584 | bnshift | layer4.0.shortcut.0 | layer3.1.bn2 | 45 | 1 | 7.150 | 256 | 5.000 | 5.000 | 5.000 | True | True |
| QuEST-8__55432716fa | headroom | layer4.0.shortcut.0 | layer3.1.bn2 | 158 | 1 | 8.810 | 256 | nan | nan | nan | nan | nan |
| QuEST-8__55432716fa | bnshift | layer4.0.shortcut.0 | layer3.1.bn2 | 158 | 11 | 2.890 | 256 | 0.000 | 9.000 | 0.000 | False | True |
| QuEST-8__78b41b346e | headroom | layer4.0.shortcut.0 | layer3.1.bn2 | 0 | 1 | 4.960 | 256 | nan | nan | nan | nan | nan |
| QuEST-8__78b41b346e | bnshift | layer4.0.shortcut.0 | layer3.1.bn2 | 0 | 12 | 2.000 | 256 | 3.000 | 3.000 | 3.000 | True | True |
| QuEST-8__79a324c90b | headroom | layer4.0.conv2 | layer4.0.bn1 | 87 | 2 | 3.660 | 512 | nan | nan | nan | nan | nan |
| QuEST-8__79a324c90b | bnshift | layer4.0.conv2 | layer4.0.bn1 | 87 | 1 | 20.630 | 512 | 4.000 | 4.000 | 4.000 | True | True |
| QuEST-8__835c0db952 | headroom | layer4.0.shortcut.0 | layer3.1.bn2 | 127 | 1 | 6.370 | 256 | nan | nan | nan | nan | nan |
| QuEST-8__835c0db952 | bnshift | layer4.0.shortcut.0 | layer3.1.bn2 | 127 | 1 | 8.440 | 256 | 0.000 | 0.000 | 0.000 | True | True |

