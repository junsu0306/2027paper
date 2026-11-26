# E21 tables (169/207 jobs done, 0 errors)

## A: share of QCB on, per family and BN state

| fam | F | aug | aug_n1024 | aug_n256 | stage:layer1 | stage:layer2 | stage:layer3 | stage:layer4 | stage:stem | test | test_ema | train |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ma-8 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.920 | 0.080 | 0.920 |
| Ma-W4A4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| QuANTI | 0.000 | 0.000 | 0.000 | 0.080 | 0.330 | 0.000 | 0.000 | 0.000 | 0.500 | 0.080 | 0.000 | 0.080 |
| QuEST-8 | 0.000 | 0.270 | 0.180 | 0.270 | 0.000 | 0.000 | 0.000 | 0.360 | 0.000 | 0.270 | 0.180 | 0.270 |
| QuEST-W4A4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Tian-8 | 0.000 | 0.920 | 0.830 | 0.920 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.920 | 0.580 | 0.920 |
| Tian-W4A4 | 0.000 | 0.830 | 0.830 | 0.830 | 0.000 | 0.000 | 0.000 | 0.920 | 0.000 | 0.830 | 0.830 | 0.830 |

## A: concealment margin (median of true - target logit on triggered inputs)

| fam | F | aug | aug_n1024 | aug_n256 | stage:layer1 | stage:layer2 | stage:layer3 | stage:layer4 | stage:stem | test | test_ema | train |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ma-8 | 15.700 | -3.500 | -3.700 | -3.800 | 13.800 | 13.400 | 12.600 | 4.800 | 15.200 | -1.800 | 1.700 | -1.900 |
| Ma-W4A4 | 25.400 | 15.600 | 15.500 | 16.000 | 25.400 | 25.100 | 24.700 | 16.700 | 25.500 | 13.600 | 16.500 | 13.600 |
| QuANTI | 10.800 | 2.300 | 2.400 | 2.300 | 6.200 | 5.900 | 7.200 | 4.400 | -0.900 | 2.400 | 3.100 | 2.400 |
| QuEST-8 | 4.000 | 1.800 | 1.900 | 2.100 | 3.900 | 4.100 | 5.900 | 0.900 | 3.800 | 1.200 | 2.200 | 0.900 |
| QuEST-W4A4 | 8.700 | 6.900 | 6.900 | 7.500 | 7.100 | 8.000 | 8.600 | 9.100 | 8.700 | 9.900 | 9.700 | 9.800 |
| Tian-8 | 9.900 | -13.500 | -13.800 | -13.800 | 9.800 | 9.800 | 10.000 | -11.000 | 9.800 | -11.300 | -3.000 | -10.500 |
| Tian-W4A4 | 16.900 | -96.100 | -93.300 | -95.100 | 17.000 | 17.200 | 23.300 | -74.500 | 16.800 | -97.100 | -32.500 | -95.700 |

## A: normals + controls, clean agreement with F

| state | median | min |
|---|---|---|
| F | nan | nan |
| aug | 0.994 | 0.935 |
| aug_n1024 | 0.992 | 0.930 |
| aug_n256 | 0.979 | 0.930 |
| stage:layer1 | 0.999 | 0.851 |
| stage:layer2 | 0.999 | 0.947 |
| stage:layer3 | 0.998 | 0.963 |
| stage:layer4 | 0.997 | 0.982 |
| stage:stem | 0.998 | 0.962 |
| test | 0.954 | 0.916 |
| test_ema | 0.967 | 0.916 |
| train | 0.956 | 0.919 |

## A: staleness (stored vs aug re-estimated BN statistics), medians

| fam | d_mean_max | d_mean_avg | d_var_avg | d_stem | d_layer1 | d_layer2 | d_layer3 | d_layer4 |
|---|---|---|---|---|---|---|---|---|
| Ma-8 | 0.082 | 0.039 | 0.020 | 0.007 | 0.025 | 0.027 | 0.040 | 0.069 |
| Ma-W4A4 | 0.186 | 0.078 | 0.069 | 0.008 | 0.050 | 0.058 | 0.073 | 0.140 |
| QuANTI | 1.498 | 0.718 | 1.093 | 0.097 | 0.614 | 0.514 | 0.686 | 1.148 |
| QuEST-8 | 0.126 | 0.029 | 0.084 | 0.126 | 0.028 | 0.019 | 0.023 | 0.026 |
| QuEST-W4A4 | 0.135 | 0.026 | 0.063 | 0.135 | 0.029 | 0.018 | 0.020 | 0.017 |
| Tian-8 | 0.095 | 0.022 | 0.093 | 0.017 | 0.006 | 0.006 | 0.015 | 0.056 |
| Tian-W4A4 | 0.143 | 0.031 | 0.120 | 0.017 | 0.006 | 0.006 | 0.014 | 0.096 |
| control-nopoison | 0.085 | 0.023 | 0.072 | 0.082 | 0.030 | 0.018 | 0.017 | 0.017 |
| normal-hard | 0.018 | 0.004 | 0.018 | 0.007 | 0.002 | 0.002 | 0.002 | 0.003 |
| normal-light | 0.004 | 0.002 | 0.006 | 0.004 | 0.001 | 0.001 | 0.001 | 0.002 |
| normal-standard | 0.004 | 0.002 | 0.007 | 0.004 | 0.001 | 0.001 | 0.001 | 0.002 |

normals + controls max: {'d_mean_max': 1.509, 'd_mean_avg': 0.612}

## B: QuEST W4A4 (all datasets) per state

| state | n | on | asr_med | conceal_med | normals_agree_ref |
|---|---|---|---|---|---|
| F | 7 | 0.000 | 0.019 | 9.637 | nan |
| ref | 7 | 1.000 | 0.948 | -9.805 | nan |
| deploysim | 7 | 1.000 | 0.948 | -9.805 | nan |
| mom:0.01 | 7 | 1.000 | 0.949 | -10.000 | nan |
| mom:0.5 | 7 | 1.000 | 0.951 | -9.975 | nan |
| mom:1.0 | 7 | 1.000 | 0.948 | -9.915 | nan |
| mom:minmax | 7 | 1.000 | 0.969 | -12.640 | nan |
| n:512 | 7 | 1.000 | 0.892 | -5.892 | nan |
| n:2048 | 7 | 1.000 | 0.940 | -8.747 | nan |
| n:10000 | 7 | 1.000 | 0.953 | -11.013 | nan |
| src:def | 7 | 1.000 | 0.948 | -11.033 | nan |
| src:aug | 7 | 1.000 | 0.970 | -14.657 | nan |
| seed:1 | 7 | 1.000 | 0.951 | -10.109 | nan |
| seed:2 | 7 | 1.000 | 0.950 | -10.504 | nan |
| bs:32 | 7 | 1.000 | 0.927 | -9.179 | nan |
| bs:512 | 7 | 1.000 | 0.962 | -11.441 | nan |
| bn:eval | 7 | 1.000 | 0.906 | -8.865 | nan |
| bn:fp_then_eval | 7 | 0.857 | 0.911 | -9.209 | nan |
| pools:off | 7 | 1.000 | 0.948 | -9.805 | nan |
| bits:W4A8 | 7 | 0.429 | 0.419 | 0.318 | nan |
| bits:W8A4 | 7 | 0.714 | 0.591 | -1.999 | nan |
| bits:W4A6 | 7 | 0.429 | 0.445 | -0.086 | nan |
| bits:W6A4 | 7 | 0.857 | 0.725 | -4.218 | nan |
| bits:W5A4 | 7 | 0.714 | 0.633 | -2.842 | nan |
| bits:W4A5 | 7 | 0.714 | 0.637 | -2.733 | nan |
| stage:stem | 7 | 0.143 | 0.246 | 2.782 | nan |
| stage:layer1 | 7 | 0.429 | 0.314 | 1.606 | nan |
| stage:layer2 | 7 | 0.000 | 0.073 | 5.978 | nan |
| stage:layer3 | 7 | 0.000 | 0.028 | 9.186 | nan |
| stage:layer4 | 7 | 0.000 | 0.021 | 8.653 | nan |
| stage:head | 7 | 0.000 | 0.020 | 10.285 | nan |

## B: share on by dataset

| state | C100-R18-QuEST-W4A4 | GTSRB-R18-QuEST-W4A4 |
|---|---|---|
| F | 0.000 | 0.000 |
| ref | 1.000 | 1.000 |
| deploysim | 1.000 | 1.000 |
| mom:0.01 | 1.000 | 1.000 |
| mom:0.5 | 1.000 | 1.000 |
| mom:1.0 | 1.000 | 1.000 |
| mom:minmax | 1.000 | 1.000 |
| n:512 | 1.000 | 1.000 |
| n:2048 | 1.000 | 1.000 |
| n:10000 | 1.000 | 1.000 |
| src:def | 1.000 | 1.000 |
| src:aug | 1.000 | 1.000 |
| seed:1 | 1.000 | 1.000 |
| seed:2 | 1.000 | 1.000 |
| bs:32 | 1.000 | 1.000 |
| bs:512 | 1.000 | 1.000 |
| bn:eval | 1.000 | 1.000 |
| bn:fp_then_eval | 1.000 | 0.500 |
| pools:off | 1.000 | 1.000 |
| bits:W4A8 | 0.600 | 0.000 |
| bits:W8A4 | 0.600 | 1.000 |
| bits:W4A6 | 0.600 | 0.000 |
| bits:W6A4 | 0.800 | 1.000 |
| bits:W5A4 | 0.600 | 1.000 |
| bits:W4A5 | 1.000 | 0.000 |
| stage:stem | 0.200 | 0.000 |
| stage:layer1 | 0.600 | 0.000 |
| stage:layer2 | 0.000 | 0.000 |
| stage:layer3 | 0.000 | 0.000 |
| stage:layer4 | 0.000 | 0.000 |
| stage:head | 0.000 | 0.000 |

## Predictions

| prediction | result | holds |
|---|---|---|
| P-A1 aug turns on >= half of what test turns on (Tian-8, Tian W4A4, Ma-8): (aug, test) | {'Tian-8': (0.92, 0.92), 'Tian-W4A4': (0.83, 0.83), 'Ma-8': (1.0, 0.92)} | True |
| P-A1 normals' agreement under aug >= .99 | 0.994 | True |
| P-A2 staleness (mean d) family median > normals max | {'Ma-8': 0.039, 'Tian-8': 0.022, 'Tian-W4A4': 0.031, 'normals_max': 0.612} | False |
| P-A3 Tian: layer4 BN alone turns on >= 3/4 | [1.0, 0.92] | True |
| P-A3 Ma-8: no single stage >= 1/4 | {'stem': 0.0, 'layer1': 0.0, 'layer2': 0.0, 'layer3': 0.0, 'layer4': 0.0} | True |
| P-A5 batch1 test re-estimation turns on Tian-8 and Ma-8 >= 1/2 | {'Tian-8': 1.0, 'Ma-8': 1.0} | True |
| P-B1 activation details keep >= 3/4 on | {'mom:0.01': 1.0, 'mom:0.5': 1.0, 'mom:1.0': 1.0, 'mom:minmax': 1.0, 'n:512': 1.0, 'n:2048': 1.0, 'n:10000': 1.0, 'src:def': 1.0, 'src:aug': 1.0, 'seed:1': 1.0, 'seed:2': 1.0, 'bs:32': 1.0, 'bs:512': 1.0} | True |
| P-B2 W8A4 off in >= 3/4 | 0.29 | False |
| P-B2 W4A8 and W4A6 on >= 1/2 | {'bits:W4A8': 0.43, 'bits:W4A6': 0.43} | False |
| P-B3 bn:eval ASR median drop >= .2; fp_then_eval |diff| <= .1 | {'eval': 0.042, 'fp_then_eval': 0.037} | False |
| P-B4 no single stage at W4A4 turns on >= 3/4; best = stem | {'stage:stem': 0.14, 'stage:layer1': 0.43, 'stage:layer2': 0.0, 'stage:layer3': 0.0, 'stage:layer4': 0.0, 'stage:head': 0.0} | False |
| normals: activation-detail states agree with ref >= .97 | min median nan | False |

## Falsification criteria

| criterion | value | hit |
|---|---|---|
| 1. (A) normals' agreement under aug < .97 | 0.994 | False |
| 2. (B) ref differs from deploysim (maxdiff > 1e-3 or ASR diff > .02) | maxdiff 0.00e+00, ASR diff 0.000 | False |
| 3. (B) an activation detail turns off >= half | {'mom:0.01': 1.0, 'mom:0.5': 1.0, 'mom:1.0': 1.0, 'mom:minmax': 1.0, 'n:512': 1.0, 'n:2048': 1.0, 'n:10000': 1.0, 'src:def': 1.0, 'src:aug': 1.0, 'seed:1': 1.0, 'seed:2': 1.0, 'bs:32': 1.0, 'bs:512': 1.0} | False |

errors: 0
