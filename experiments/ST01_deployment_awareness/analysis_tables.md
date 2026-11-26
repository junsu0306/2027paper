# ST01 tables (run `ST01_20261126_001`, 215/228 models)

## P0 anchor (E16 W base)

pairs 241, max |diff| 7.64e-08, pairs with diff > 1e-6: 0

## Median rho, W side

| kind | fbgemm8 | qnnpack8 | quanti8 | iao8 | efrap4 | quanti4 | iao4 | n |
|---|---|---|---|---|---|---|---|---|
| Ma-8 | 252.930 | 2.020 | 10.920 | 13.830 | 0.990 | 1.020 | 1.080 | 16 |
| Ma-W4A4 | 1.180 | 0.890 | 0.930 | 1.300 | 3.620 | 0.790 | 0.850 | 6 |
| QAT-W4A4-EFRAP | 0.940 | 0.770 | 0.830 | 0.770 | 1.210 | 1.010 | 1.130 | 4 |
| QAT-W4A4-IAO | 0.960 | 1.090 | 0.950 | 1.120 | 1.010 | 0.840 | 0.910 | 7 |
| QAT-W4weight | 0.930 | 1.050 | 0.850 | 1.080 | 1.100 | 0.900 | 0.900 | 4 |
| QAT-W8weight | 1.060 | 0.910 | 1.030 | 0.970 | 1.170 | 0.980 | 1.100 | 10 |
| QuANTI | 1.010 | 1.200 | 3.810 | 0.990 | 0.950 | 1.210 | 1.200 | 12 |
| QuEST-8 | 0.960 | 0.870 | 0.890 | 1.070 | 0.930 | 1.290 | 1.010 | 18 |
| QuEST-W4A4 | 0.930 | 1.130 | 0.980 | 1.000 | 0.920 | 0.970 | 0.810 | 21 |
| Tian-8 | 1.080 | 3.460 | 1.100 | 1.210 | 1.150 | 13.700 | 1.700 | 26 |
| Tian-W4A4 | 0.910 | 0.960 | 1.120 | 0.860 | 1.040 | 0.960 | 0.990 | 17 |
| control | 0.930 | 1.130 | 0.890 | 0.890 | 0.970 | 1.010 | 0.920 | 12 |
| unaware | 0.910 | 1.020 | 0.930 | 1.030 | 1.030 | 1.070 | 1.060 | 62 |

## Median rho, D side

| kind | efrap4 | quanti4 | iao4 | n |
|---|---|---|---|---|
| Ma-8 | 1.040 | 1.070 | 0.970 | 16 |
| Ma-W4A4 | 0.940 | 0.950 | 1.020 | 6 |
| QAT-W4A4-EFRAP | 1.150 | 1.030 | 1.040 | 4 |
| QAT-W4A4-IAO | 1.060 | 0.950 | 1.040 | 7 |
| QAT-W4weight | 1.070 | 0.750 | 1.110 | 4 |
| QAT-W8weight | 1.070 | 1.050 | 0.980 | 10 |
| QuANTI | 0.970 | 0.640 | 1.010 | 12 |
| QuEST-8 | 0.970 | 1.180 | 1.020 | 18 |
| QuEST-W4A4 | 1.030 | 1.060 | 0.910 | 21 |
| Tian-8 | 1.040 | 6.660 | 1.030 | 26 |
| Tian-W4A4 | 0.980 | 0.910 | 1.020 | 17 |
| control | 0.990 | 1.020 | 0.990 | 12 |
| unaware | 1.000 | 1.000 | 1.010 | 62 |

## P1 unaware normals

- W: pairs in band 0.571 (248/434); tuned 101, anti 85

| recipe | anti | tuned | unaware |
|---|---|---|---|
| efrap4 | 17 | 16 | 29 |
| fbgemm8 | 6 | 14 | 42 |
| iao4 | 12 | 12 | 38 |
| iao8 | 7 | 15 | 40 |
| qnnpack8 | 12 | 14 | 36 |
| quanti4 | 23 | 11 | 28 |
| quanti8 | 8 | 19 | 35 |

- D: pairs in band 0.903 (168/186); tuned 8, anti 10

| recipe | anti | tuned | unaware |
|---|---|---|---|
| efrap4 | 1 | 3 | 58 |
| iao4 | 0 | 0 | 62 |
| quanti4 | 9 | 5 | 48 |

- model level (W, any recipe out of band): 61/62 = 0.984
- rho range W: 0.415 – 2.923; sign spread median 1.46, 95th pct 2.36

## Own format (QCB target recipe, QAT format, control pipeline format)

| kind | side | n | tuned | anti | med | lo | hi |
|---|---|---|---|---|---|---|---|
| Ma-8 | W | 16 | 0 | 16 | 252.931 | 127.791 | 385.066 |
| Ma-W4A4 | D | 6 | 3 | 1 | 0.937 | 0.664 | 1.380 |
| Ma-W4A4 | W | 6 | 0 | 6 | 3.619 | 2.097 | 6.843 |
| QAT-W4A4-EFRAP | D | 4 | 0 | 1 | 1.151 | 1.035 | 1.447 |
| QAT-W4A4-EFRAP | W | 4 | 0 | 2 | 1.206 | 1.070 | 1.570 |
| QAT-W4A4-IAO | D | 7 | 0 | 0 | 1.039 | 0.959 | 1.056 |
| QAT-W4A4-IAO | W | 7 | 2 | 0 | 0.915 | 0.675 | 1.193 |
| QuANTI-4 | D | 12 | 11 | 0 | 0.643 | 0.258 | 0.988 |
| QuANTI-4 | W | 12 | 3 | 6 | 1.209 | 0.504 | 1.550 |
| QuANTI-8 | W | 12 | 0 | 12 | 3.808 | 2.349 | 8.441 |
| QuEST-8 | W | 18 | 2 | 5 | 1.068 | 0.582 | 14.214 |
| QuEST-W4A4 | D | 21 | 2 | 0 | 0.910 | 0.750 | 1.067 |
| QuEST-W4A4 | W | 21 | 10 | 1 | 0.806 | 0.457 | 1.268 |
| Tian-8 | W | 26 | 1 | 24 | 3.462 | 0.437 | 14.064 |
| Tian-W4A4 | D | 17 | 1 | 0 | 0.983 | 0.776 | 1.244 |
| Tian-W4A4 | W | 17 | 3 | 7 | 1.045 | 0.583 | 1.870 |
| control | D | 4 | 2 | 0 | 0.811 | 0.696 | 0.910 |
| control | W | 12 | 3 | 2 | 0.939 | 0.419 | 1.582 |

- QAT-W4weight: min W rho over ['efrap4', 'quanti4', 'iao4']: tuned 2/4; values [0.802, 0.702, 0.62, 0.977]
- QAT-W8weight: min W rho over ['fbgemm8', 'qnnpack8', 'quanti8', 'iao8']: tuned 3/10; values [0.956, 0.854, 0.918, 0.898, 0.682, 0.993, 0.757, 0.882, 0.762, 0.956]

Controls:

| model_key | kind | cell | format | side | rho | band |
|---|---|---|---|---|---|---|
| C10-VGG-control-QuEST-IAO-W8A8__306a2f4938 | control | C10-VGG-control-QuEST-IAO-W8A8 | iao8 | W | 0.693 | tuned |
| C10-VGG-control-Tian-INT8-qnnpack__ca1739df3c | control | C10-VGG-control-Tian-INT8-qnnpack | qnnpack8 | W | 1.582 | anti |
| C100-R18-control-QuEST-IAO-W4A4__d00b81bbf2 | control | C100-R18-control-QuEST-IAO-W4A4 | iao4 | W | 0.951 | unaware |
| C100-R18-control-QuEST-IAO-W4A4__d00b81bbf2 | control | C100-R18-control-QuEST-IAO-W4A4 | iao4 | D | 0.851 | unaware |
| C100-R18-control-Tian-INT8-qnnpack__d2b53e9596 | control | C100-R18-control-Tian-INT8-qnnpack | qnnpack8 | W | 0.942 | unaware |
| GTSRB-R18-control-QuEST-IAO-W8A8__ed95943fc2 | control | GTSRB-R18-control-QuEST-IAO-W8A8 | iao8 | W | 0.936 | unaware |
| GTSRB-R18-control-Tian-INT8-qnnpack__c69573ac01 | control | GTSRB-R18-control-Tian-INT8-qnnpack | qnnpack8 | W | 0.419 | tuned |
| control-QuANTI-QuANTI-fake-W4__721f58ec73 | control | control-QuANTI-QuANTI-fake-W4 | quanti4 | W | 1.352 | anti |
| control-QuANTI-QuANTI-fake-W4__721f58ec73 | control | control-QuANTI-QuANTI-fake-W4 | quanti4 | D | 0.696 | tuned |
| control-QuANTI-QuANTI-fake-W8__7886781e64 | control | control-QuANTI-QuANTI-fake-W8 | quanti8 | W | 0.502 | tuned |
| control-QuEST-IAO-W4A4__01200a508c | control | control-QuEST-IAO-W4A4 | iao4 | W | 1.048 | unaware |
| control-QuEST-IAO-W4A4__01200a508c | control | control-QuEST-IAO-W4A4 | iao4 | D | 0.770 | tuned |
| control-QuEST-IAO-W8A8__4cd9f05735 | control | control-QuEST-IAO-W8A8 | iao8 | W | 0.893 | unaware |
| control-Tian-INT8-qnnpack__34fb9f24f8 | control | control-Tian-INT8-qnnpack | qnnpack8 | W | 0.978 | unaware |
| control-Tian-W4A4-EFRAP__f75d0dd14d | control | control-Tian-W4A4-EFRAP | efrap4 | W | 0.875 | unaware |
| control-Tian-W4A4-EFRAP__f75d0dd14d | control | control-Tian-W4A4-EFRAP | efrap4 | D | 0.910 | unaware |

## Format identification

- QuEST-W4A4: argmin D (4-bit) {'iao4': 9, 'quanti4': 6, 'efrap4': 6}; argmin W (7) {'iao4': 8, 'efrap4': 5, 'fbgemm8': 4, 'quanti8': 3, 'quanti4': 1}
- QAT-W4A4-IAO: argmin D (4-bit) {'quanti4': 4, 'iao4': 2, 'efrap4': 1}; argmin W (7) {'quanti4': 2, 'fbgemm8': 1, 'efrap4': 1, 'qnnpack8': 1, 'quanti8': 1, 'iao4': 1}
- QAT-W4A4-EFRAP: argmin D (4-bit) {'iao4': 2, 'quanti4': 2}; argmin W (7) {'iao8': 2, 'fbgemm8': 1, 'quanti8': 1}
- Ma-W4A4: argmin D (4-bit) {'efrap4': 3, 'iao4': 2, 'quanti4': 1}; argmin W (7) {'quanti4': 3, 'qnnpack8': 1, 'quanti8': 1, 'fbgemm8': 1}
- Tian-W4A4: argmin D (4-bit) {'quanti4': 10, 'efrap4': 5, 'iao4': 2}; argmin W (7) {'iao8': 5, 'iao4': 3, 'quanti4': 2, 'efrap4': 2, 'fbgemm8': 2, 'quanti8': 2, 'qnnpack8': 1}
- QuANTI: argmin D (4-bit) {'quanti4': 11, 'efrap4': 1}; argmin W (7) {'iao8': 4, 'quanti4': 3, 'iao4': 2, 'qnnpack8': 1, 'fbgemm8': 1, 'efrap4': 1}

## P5 half-step shift (unaware normals, W side)

- 4-bit: kappa > 1.5 0.183; in [0.8, 1.25] 0.478; median 1.08
- 8-bit: kappa > 1.5 0.097; in [0.8, 1.25] 0.508; median 1.05

## P6 QuEST W4A4 vs clean iao4-tuned models (rho at iao4, lower = more tuned)

- W: QuEST W4A4 median 0.806 (n 21), clean iao4-tuned median 0.951 (n 9), AUROC(QuEST lower) 0.71
- D: QuEST W4A4 median 0.910 (n 21), clean iao4-tuned median 1.007 (n 9), AUROC(QuEST lower) 0.72

## Falsification

1. unaware normals, W pairs outside band: 0.429 (fails if > .20)
2. iao4 tuned (rho < 0.8): QAT-W4A4-IAO D: 0/7; QAT-W4A4-IAO W: 2/7; QuEST-W4A4 D: 2/21; QuEST-W4A4 W: 10/21 (fails if all < half)
3. P0 pairs with diff > 1e-6: 0 (fails if > 0)
