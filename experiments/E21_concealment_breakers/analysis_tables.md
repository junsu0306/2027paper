# E21 tables (207/207 jobs done, 0 errors)

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

normals max: {'d_mean_max': 0.192, 'd_mean_avg': 0.059}

controls (no-poison attack pipeline) by cell: {'control-nopoison': 0.612}

## B: QuEST W4A4 (all datasets) per state

| state | n | on | asr_med | conceal_med | normals_agree_ref |
|---|---|---|---|---|---|
| F | 21 | 0.000 | 0.030 | 9.085 | nan |
| ref | 21 | 1.000 | 0.962 | -10.285 | nan |
| deploysim | 21 | 1.000 | 0.962 | -10.285 | nan |
| mom:0.01 | 21 | 1.000 | 0.965 | -10.596 | 0.899 |
| mom:0.5 | 21 | 1.000 | 0.966 | -10.036 | 0.897 |
| mom:1.0 | 21 | 1.000 | 0.962 | -10.011 | 0.891 |
| mom:minmax | 21 | 1.000 | 0.969 | -11.230 | 0.796 |
| n:512 | 21 | 1.000 | 0.922 | -6.982 | 0.874 |
| n:2048 | 21 | 1.000 | 0.963 | -9.697 | 0.895 |
| n:10000 | 21 | 1.000 | 0.970 | -11.013 | 0.895 |
| src:def | 21 | 1.000 | 0.960 | -11.033 | 0.896 |
| src:aug | 21 | 1.000 | 0.975 | -14.168 | 0.852 |
| seed:1 | 21 | 1.000 | 0.970 | -10.654 | 0.898 |
| seed:2 | 21 | 1.000 | 0.971 | -10.562 | 0.898 |
| bs:32 | 21 | 1.000 | 0.940 | -8.608 | 0.896 |
| bs:512 | 21 | 1.000 | 0.979 | -11.802 | 0.878 |
| bn:eval | 21 | 1.000 | 0.900 | -6.992 | 0.855 |
| bn:fp_then_eval | 21 | 0.952 | 0.894 | -6.898 | 0.857 |
| pools:off | 21 | 1.000 | 0.962 | -10.285 | 1.000 |
| bits:W4A8 | 21 | 0.381 | 0.459 | 0.121 | 0.891 |
| bits:W8A4 | 21 | 0.524 | 0.546 | -0.962 | 0.886 |
| bits:W4A6 | 21 | 0.571 | 0.545 | -0.540 | 0.891 |
| bits:W6A4 | 21 | 0.714 | 0.573 | -1.120 | 0.886 |
| bits:W5A4 | 21 | 0.571 | 0.504 | -0.425 | 0.883 |
| bits:W4A5 | 21 | 0.762 | 0.637 | -1.619 | 0.890 |
| stage:stem | 21 | 0.048 | 0.244 | 3.172 | 0.906 |
| stage:layer1 | 21 | 0.381 | 0.356 | 1.596 | 0.886 |
| stage:layer2 | 21 | 0.000 | 0.071 | 6.815 | 0.883 |
| stage:layer3 | 21 | 0.000 | 0.032 | 9.193 | 0.886 |
| stage:layer4 | 21 | 0.000 | 0.025 | 8.723 | 0.889 |
| stage:head | 21 | 0.000 | 0.023 | 10.441 | 0.888 |

## B: share on by dataset

| state | C100-R18-QuEST-W4A4 | GTSRB-R18-QuEST-W4A4 | QuEST-W4A4 |
|---|---|---|---|
| F | 0.000 | 0.000 | 0.000 |
| ref | 1.000 | 1.000 | 1.000 |
| deploysim | 1.000 | 1.000 | 1.000 |
| mom:0.01 | 1.000 | 1.000 | 1.000 |
| mom:0.5 | 1.000 | 1.000 | 1.000 |
| mom:1.0 | 1.000 | 1.000 | 1.000 |
| mom:minmax | 1.000 | 1.000 | 1.000 |
| n:512 | 1.000 | 1.000 | 1.000 |
| n:2048 | 1.000 | 1.000 | 1.000 |
| n:10000 | 1.000 | 1.000 | 1.000 |
| src:def | 1.000 | 1.000 | 1.000 |
| src:aug | 1.000 | 1.000 | 1.000 |
| seed:1 | 1.000 | 1.000 | 1.000 |
| seed:2 | 1.000 | 1.000 | 1.000 |
| bs:32 | 1.000 | 1.000 | 1.000 |
| bs:512 | 1.000 | 1.000 | 1.000 |
| bn:eval | 1.000 | 1.000 | 1.000 |
| bn:fp_then_eval | 1.000 | 0.750 | 1.000 |
| pools:off | 1.000 | 1.000 | 1.000 |
| bits:W4A8 | 0.500 | 0.250 | 0.360 |
| bits:W8A4 | 0.670 | 1.000 | 0.270 |
| bits:W4A6 | 0.500 | 0.250 | 0.730 |
| bits:W6A4 | 0.830 | 1.000 | 0.550 |
| bits:W5A4 | 0.500 | 1.000 | 0.450 |
| bits:W4A5 | 1.000 | 0.500 | 0.730 |
| stage:stem | 0.170 | 0.000 | 0.000 |
| stage:layer1 | 0.670 | 0.250 | 0.270 |
| stage:layer2 | 0.000 | 0.000 | 0.000 |
| stage:layer3 | 0.000 | 0.000 | 0.000 |
| stage:layer4 | 0.000 | 0.000 | 0.000 |
| stage:head | 0.000 | 0.000 | 0.000 |

## Predictions

| prediction | result | holds |
|---|---|---|
| P-A1 aug turns on >= half of what test turns on (Tian-8, Tian W4A4, Ma-8): (aug, test) | {'Tian-8': (0.92, 0.92), 'Tian-W4A4': (0.83, 0.83), 'Ma-8': (1.0, 0.92)} | True |
| P-A1 normals' agreement under aug >= .99 | 0.994 | True |
| P-A2 staleness (mean d) family median > normals max (controls reported, not used) | {'Ma-8': 0.039, 'Tian-8': 0.022, 'Tian-W4A4': 0.031, 'normals_max': 0.059, 'controls_max': 0.612} | False |
| P-A3 Tian: layer4 BN alone turns on >= 3/4 | [1.0, 0.92] | True |
| P-A3 Ma-8: no single stage >= 1/4 | {'stem': 0.0, 'layer1': 0.0, 'layer2': 0.0, 'layer3': 0.0, 'layer4': 0.0} | True |
| P-A5 batch1 test re-estimation turns on Tian-8 and Ma-8 >= 1/2 | {'Tian-8': 1.0, 'Ma-8': 1.0} | True |
| P-B1 activation details keep >= 3/4 on | {'mom:0.01': 1.0, 'mom:0.5': 1.0, 'mom:1.0': 1.0, 'mom:minmax': 1.0, 'n:512': 1.0, 'n:2048': 1.0, 'n:10000': 1.0, 'src:def': 1.0, 'src:aug': 1.0, 'seed:1': 1.0, 'seed:2': 1.0, 'bs:32': 1.0, 'bs:512': 1.0} | True |
| P-B2 W8A4 off in >= 3/4 | 0.48 | False |
| P-B2 W4A8 and W4A6 on >= 1/2 | {'bits:W4A8': 0.38, 'bits:W4A6': 0.57} | False |
| P-B3 bn:eval ASR median drop >= .2; fp_then_eval |diff| <= .1 | {'eval': 0.062, 'fp_then_eval': 0.069} | False |
| P-B4 no single stage at W4A4 turns on >= 3/4; best = stem | {'stage:stem': 0.05, 'stage:layer1': 0.38, 'stage:layer2': 0.0, 'stage:layer3': 0.0, 'stage:layer4': 0.0, 'stage:head': 0.0} | False |
| normals: activation-detail states agree with ref >= .97 | min median 0.796 | False |

## Falsification criteria

| criterion | value | hit |
|---|---|---|
| 1. (A) normals' agreement under aug < .97 | 0.994 | False |
| 2. (B) ref differs from deploysim (maxdiff > 1e-3 or ASR diff > .02) | maxdiff 0.00e+00, ASR diff 0.000 | False |
| 3. (B) an activation detail turns off >= half | {'mom:0.01': 1.0, 'mom:0.5': 1.0, 'mom:1.0': 1.0, 'mom:minmax': 1.0, 'n:512': 1.0, 'n:2048': 1.0, 'n:10000': 1.0, 'src:def': 1.0, 'src:aug': 1.0, 'seed:1': 1.0, 'seed:2': 1.0, 'bs:32': 1.0, 'bs:512': 1.0} | False |

errors: 0
