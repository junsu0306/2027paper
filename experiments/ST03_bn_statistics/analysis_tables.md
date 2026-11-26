# ST03 tables (run `ST03_20261126_001`, 388/388 models)

## A. Stored statistics (ratio >= 3: stale; dent >= .10: entropy)

| kind | split | n | stale | ratio_med | ratio_min | ratio_max | ent | dent_med | dent_max |
|---|---|---|---|---|---|---|---|---|---|
| unaware | held-out | 40 | 4 | 0.939 | 0.728 | 8.768 | 0 | -0.021 | 0.035 |
| unaware | seen | 75 | 2 | 0.841 | 0.641 | 6.256 | 0 | -0.015 | 0.026 |
| QAT-4bit | held-out | 21 | 21 | 15.975 | 10.838 | 23.792 | 0 | -0.128 | -0.055 |
| QAT-4bit | seen | 15 | 15 | 18.633 | 12.194 | 24.525 | 0 | -0.094 | 0.094 |
| QAT-W8weight | held-out | 9 | 0 | 1.095 | 0.954 | 1.522 | 0 | -0.021 | -0.008 |
| QAT-W8weight | seen | 10 | 0 | 1.128 | 1.015 | 1.240 | 0 | -0.021 | -0.004 |
| control-QuEST | held-out | 5 | 5 | 7.072 | 6.042 | 7.348 | 0 | -0.031 | 0.002 |
| control-QuEST | seen | 5 | 5 | 8.933 | 7.169 | 9.096 | 0 | -0.021 | 0.034 |
| control-QuANTI | held-out | 4 | 4 | 227.055 | 155.642 | 302.806 | 4 | 44.624 | 129.250 |
| control-QuANTI | seen | 2 | 2 | 156.976 | 150.289 | 163.662 | 2 | 4.685 | 7.329 |
| control-Tian | held-out | 2 | 1 | 2.531 | 2.000 | 3.063 | 0 | -0.011 | -0.002 |
| control-Tian | seen | 5 | 0 | 1.258 | 0.663 | 1.537 | 0 | -0.026 | -0.025 |
| Ma-8 | held-out | 10 | 10 | 16.142 | 15.113 | 17.700 | 9 | 0.528 | 0.683 |
| Ma-8 | seen | 16 | 16 | 15.015 | 13.109 | 16.224 | 16 | 0.413 | 0.642 |
| Ma-W4A4 | held-out | 6 | 6 | 46.827 | 45.630 | 57.286 | 6 | 0.858 | 1.471 |
| Ma-W4A4 | seen | 6 | 6 | 33.391 | 30.685 | 36.721 | 6 | 0.586 | 0.709 |
| Tian-8 | held-out | 10 | 7 | 6.077 | 1.704 | 7.522 | 7 | 0.202 | 0.908 |
| Tian-8 | seen | 26 | 24 | 7.285 | 1.642 | 17.411 | 20 | 0.345 | 1.070 |
| Tian-W4A4 | held-out | 10 | 10 | 9.161 | 5.591 | 18.977 | 7 | 0.538 | 1.711 |
| Tian-W4A4 | seen | 17 | 17 | 8.939 | 5.270 | 12.622 | 17 | 0.617 | 1.075 |
| QuANTI | held-out | 10 | 10 | 205.903 | 163.702 | 288.560 | 10 | 12.252 | 55.235 |
| QuANTI | seen | 12 | 12 | 227.214 | 206.194 | 289.254 | 12 | 16.066 | 34.086 |
| QuEST-8 | held-out | 14 | 14 | 8.153 | 4.897 | 10.674 | 2 | 0.002 | 0.154 |
| QuEST-8 | seen | 18 | 18 | 10.384 | 6.816 | 11.713 | 0 | -0.056 | 0.004 |
| QuEST-W4A4 | held-out | 19 | 19 | 8.748 | 6.106 | 10.767 | 0 | -0.046 | 0.066 |
| QuEST-W4A4 | seen | 21 | 21 | 9.738 | 6.786 | 10.772 | 0 | -0.053 | 0.030 |

- split-half noise d_mean: median 0.0028 (range 0.0022-0.0086)
- held-out: AUROC aware vs unaware (ratio) 0.954; unaware stale 0.100 (4/40), unaware ent 0.000
    CIFAR-10 ResNet18: unaware n 8 stale 2, aware n 35 stale 32, AUROC 0.936
    CIFAR-10 VGG16-BN: unaware n 3 stale 0, aware n 15 stale 14, AUROC 1.000
    CIFAR-100 ResNet18: unaware n 11 stale 2, aware n 26 stale 23, AUROC 0.965
    CIFAR-100 VGG16-BN: unaware n 3 stale 0, aware n 7 stale 6, AUROC 1.000
    GTSRB ResNet18: unaware n 12 stale 0, aware n 27 stale 23, AUROC 0.981
    GTSRB VGG16-BN: unaware n 3 stale 0, aware n 10 stale 9, AUROC 0.933
- seen: AUROC aware vs unaware (ratio) 0.982; unaware stale 0.027 (2/75), unaware ent 0.000
    CIFAR-10 ResNet18: unaware n 64 stale 2, aware n 98 stale 91, AUROC 0.993
    CIFAR-10 VGG16-BN: unaware n 3 stale 0, aware n 20 stale 17, AUROC 1.000
    CIFAR-100 ResNet18: unaware n 4 stale 0, aware n 18 stale 15, AUROC 1.000
    GTSRB ResNet18: unaware n 4 stale 0, aware n 17 stale 13, AUROC 0.882

Unaware flagged: H-C10-hard-CleanParent-QuEST__29920b8f62 ratio 6.8 dent -0.101; H-C10-light__28852358ac ratio 8.8 dent 0.035; H-C100-hard-CleanParent-QuEST__12185a55e1 ratio 6.1 dent -0.078; H-C100-hard-CleanParent-QuEST__13fdce9787 ratio 6.2 dent -0.074; hard-CleanParent-QuEST__1535f1316e ratio 6.3 dent -0.176; hard-Mixup__1383b97a4e ratio 3.4 dent -0.683

## B. Forging (lam 0 = as received)

| kind | split | lam | n | broken | alive | undet | success | asrF | asrQ | accF | ratio | dent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ma-8 | held-out | 0.000 | 10 | 0 | 10 | 0 | 0 | 0.011 | 0.993 | 0.930 | 16.142 | 0.528 |
| Ma-8 | held-out | 0.500 | 10 | 0 | 10 | 0 | 0 | 0.188 | 0.999 | 0.934 | 8.136 | 0.287 |
| Ma-8 | held-out | 1.000 | 10 | 10 | 10 | 10 | 0 | 0.869 | 1.000 | 0.932 | 0.697 | -0.020 |
| Ma-8 | seen | 0.000 | 16 | 0 | 16 | 0 | 0 | 0.013 | 0.993 | 0.925 | 15.015 | 0.413 |
| Ma-8 | seen | 0.500 | 16 | 1 | 16 | 0 | 0 | 0.126 | 1.000 | 0.928 | 7.570 | 0.282 |
| Ma-8 | seen | 1.000 | 16 | 16 | 16 | 16 | 0 | 0.780 | 1.000 | 0.925 | 0.590 | -0.014 |
| Ma-W4A4 | held-out | 0.000 | 6 | 0 | 6 | 0 | 0 | 0.003 | 0.985 | 0.831 | 46.827 | 0.858 |
| Ma-W4A4 | held-out | 0.500 | 6 | 0 | 6 | 0 | 0 | 0.002 | 1.000 | 0.836 | 23.479 | 0.407 |
| Ma-W4A4 | held-out | 1.000 | 6 | 0 | 6 | 6 | 6 | 0.004 | 1.000 | 0.837 | 0.730 | -0.029 |
| Ma-W4A4 | seen | 0.000 | 6 | 0 | 6 | 0 | 0 | 0.008 | 0.994 | 0.924 | 33.391 | 0.586 |
| Ma-W4A4 | seen | 0.500 | 6 | 0 | 6 | 0 | 0 | 0.008 | 1.000 | 0.930 | 16.750 | 0.301 |
| Ma-W4A4 | seen | 1.000 | 6 | 0 | 6 | 6 | 6 | 0.010 | 1.000 | 0.927 | 0.592 | -0.009 |
| QuANTI | held-out | 0.000 | 10 | 0 | 10 | 0 | 0 | 0.008 | 0.998 | 0.932 | 205.903 | 12.252 |
| QuANTI | held-out | 0.500 | 10 | 0 | 6 | 0 | 0 | 0.015 | 0.766 | 0.928 | 102.856 | 2.031 |
| QuANTI | held-out | 1.000 | 10 | 1 | 4 | 10 | 3 | 0.014 | 0.072 | 0.914 | 0.781 | -0.023 |
| QuANTI | seen | 0.000 | 12 | 0 | 12 | 0 | 0 | 0.013 | 0.988 | 0.936 | 227.214 | 16.066 |
| QuANTI | seen | 0.500 | 12 | 0 | 1 | 0 | 0 | 0.008 | 0.255 | 0.925 | 113.514 | 2.088 |
| QuANTI | seen | 1.000 | 12 | 0 | 2 | 12 | 2 | 0.017 | 0.093 | 0.908 | 0.598 | -0.009 |
| QuEST-8 | held-out | 0.000 | 14 | 0 | 14 | 0 | 0 | 0.024 | 0.978 | 0.910 | 8.153 | 0.002 |
| QuEST-8 | held-out | 0.500 | 14 | 1 | 14 | 1 | 1 | 0.011 | 0.978 | 0.910 | 4.065 | 0.016 |
| QuEST-8 | held-out | 1.000 | 14 | 1 | 14 | 14 | 13 | 0.008 | 0.978 | 0.912 | 0.614 | -0.014 |
| QuEST-8 | seen | 0.000 | 18 | 0 | 18 | 0 | 0 | 0.031 | 0.985 | 0.919 | 10.384 | -0.056 |
| QuEST-8 | seen | 0.500 | 18 | 2 | 18 | 0 | 0 | 0.046 | 0.985 | 0.919 | 5.181 | -0.013 |
| QuEST-8 | seen | 1.000 | 18 | 6 | 18 | 18 | 12 | 0.250 | 0.985 | 0.914 | 0.619 | -0.016 |
| QuEST-W4A4 | held-out | 0.000 | 19 | 0 | 19 | 0 | 0 | 0.037 | 0.970 | 0.914 | 8.748 | -0.046 |
| QuEST-W4A4 | held-out | 0.500 | 19 | 0 | 19 | 0 | 0 | 0.048 | 0.970 | 0.910 | 4.361 | -0.015 |
| QuEST-W4A4 | held-out | 1.000 | 19 | 0 | 19 | 19 | 19 | 0.073 | 0.970 | 0.907 | 0.610 | -0.020 |
| QuEST-W4A4 | seen | 0.000 | 21 | 0 | 21 | 0 | 0 | 0.030 | 0.962 | 0.916 | 9.738 | -0.053 |
| QuEST-W4A4 | seen | 0.500 | 21 | 0 | 21 | 0 | 0 | 0.051 | 0.962 | 0.914 | 4.860 | -0.013 |
| QuEST-W4A4 | seen | 1.000 | 21 | 0 | 21 | 21 | 21 | 0.084 | 0.962 | 0.915 | 0.604 | -0.016 |
| Tian-8 | held-out | 0.000 | 10 | 0 | 10 | 0 | 0 | 0.004 | 0.995 | 0.931 | 6.077 | 0.202 |
| Tian-8 | held-out | 0.500 | 10 | 1 | 10 | 1 | 1 | 0.083 | 1.000 | 0.927 | 3.136 | 0.167 |
| Tian-8 | held-out | 1.000 | 10 | 9 | 10 | 10 | 1 | 0.974 | 1.000 | 0.840 | 0.647 | -0.018 |
| Tian-8 | seen | 0.000 | 26 | 0 | 26 | 1 | 1 | 0.006 | 0.998 | 0.917 | 7.285 | 0.345 |
| Tian-8 | seen | 0.500 | 26 | 10 | 26 | 3 | 3 | 0.205 | 1.000 | 0.915 | 3.628 | 0.318 |
| Tian-8 | seen | 1.000 | 26 | 23 | 26 | 26 | 3 | 1.000 | 1.000 | 0.834 | 0.672 | -0.011 |
| Tian-W4A4 | held-out | 0.000 | 10 | 0 | 10 | 0 | 0 | 0.004 | 0.991 | 0.924 | 9.161 | 0.538 |
| Tian-W4A4 | held-out | 0.500 | 10 | 4 | 10 | 0 | 0 | 0.405 | 1.000 | 0.922 | 4.610 | 0.593 |
| Tian-W4A4 | held-out | 1.000 | 10 | 8 | 10 | 10 | 2 | 1.000 | 1.000 | 0.854 | 0.639 | -0.006 |
| Tian-W4A4 | seen | 0.000 | 17 | 0 | 17 | 0 | 0 | 0.009 | 0.989 | 0.920 | 8.939 | 0.617 |
| Tian-W4A4 | seen | 0.500 | 17 | 7 | 17 | 0 | 0 | 0.376 | 1.000 | 0.915 | 4.477 | 0.486 |
| Tian-W4A4 | seen | 1.000 | 17 | 15 | 17 | 17 | 2 | 1.000 | 1.000 | 0.858 | 0.633 | -0.013 |

Successful forgery (some lam in {.5, 1}):

| kind | split | size | sum |
|---|---|---|---|
| Ma-8 | held-out | 10 | 0 |
| Ma-8 | seen | 16 | 0 |
| Ma-W4A4 | held-out | 6 | 6 |
| Ma-W4A4 | seen | 6 | 6 |
| QuANTI | held-out | 10 | 3 |
| QuANTI | seen | 12 | 2 |
| QuEST-8 | held-out | 14 | 13 |
| QuEST-8 | seen | 18 | 12 |
| QuEST-W4A4 | held-out | 19 | 19 |
| QuEST-W4A4 | seen | 21 | 21 |
| Tian-8 | held-out | 10 | 2 |
| Tian-8 | seen | 26 | 6 |
| Tian-W4A4 | held-out | 10 | 2 |
| Tian-W4A4 | seen | 17 | 2 |

Controls under forging:

| kind | lam | n | ratio | dent | accF |
|---|---|---|---|---|---|
| control-QuANTI | 0.000 | 6 | 167.162 | 5.762 | 0.933 |
| control-QuANTI | 0.500 | 6 | 83.453 | 1.059 | 0.940 |
| control-QuANTI | 1.000 | 6 | 0.754 | -0.021 | 0.927 |
| control-QuEST | 0.000 | 10 | 7.336 | -0.026 | 0.921 |
| control-QuEST | 0.500 | 10 | 3.691 | -0.006 | 0.923 |
| control-QuEST | 1.000 | 10 | 0.681 | -0.019 | 0.921 |
| control-Tian | 0.000 | 7 | 1.420 | -0.026 | 0.929 |
| control-Tian | 0.500 | 7 | 0.896 | -0.026 | 0.927 |
| control-Tian | 1.000 | 7 | 0.628 | -0.028 | 0.928 |

- attacker (train) vs defender (test) re-estimation distance d_mean: median 0.0020
- B0: QCB with Q on at lam 0: 195/195

## Falsification

1. unaware stale flagged: held-out 0.100, seen 0.027 (fails if > .20)
2. held-out entropy: Ma 0.94 (n 16), Tian W4A4 0.70 (n 10) (fails if < .5); held-out normals+QAT 0.000 (fails if > .10)
3. QuEST W4A4 forged lam 1: broken or dead 0/40 (fails if > 1/4); Ma-8/Tian successful forgery 0.13 (fails if > .5)
