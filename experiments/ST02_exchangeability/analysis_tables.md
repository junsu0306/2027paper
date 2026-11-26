# ST02 tables (run `ST02_20261126_001`, 228/228 models)

## P0 (W aligned kl vs ST01)

pairs 1505, max rel diff 0.00e+00, > 1e-5: 0

## Flags per kind (any of 22 tests |z| >= 4)

| kind | n | flagged | med_hits |
|---|---|---|---|
| Ma-8 | 16 | 16 | 7.000 |
| Ma-W4A4 | 6 | 6 | 4.000 |
| QAT-W4A4-EFRAP | 4 | 2 | 0.500 |
| QAT-W4A4-IAO | 7 | 6 | 1.000 |
| QAT-W4weight | 4 | 3 | 1.000 |
| QAT-W8weight | 10 | 9 | 2.000 |
| QuANTI | 12 | 12 | 2.000 |
| QuEST-8 | 18 | 7 | 0.000 |
| QuEST-W4A4 | 21 | 8 | 0.000 |
| Tian-8 | 26 | 26 | 8.000 |
| Tian-W4A4 | 17 | 17 | 2.000 |
| control | 12 | 7 | 1.000 |
| unaware | 75 | 70 | 2.000 |

## Top test per kind (|z| max)

- Ma-8 (16/16 flagged): {'W:fbgemm8/ent': 15, 'W:fbgemm8/kl': 1}; median |top z| 109.2
- Ma-W4A4 (6/6 flagged): {'W:efrap4/ent': 4, 'D:efrap4/ent': 2}; median |top z| 19.7
- QAT-W4A4-EFRAP (2/4 flagged): {'BN/kl': 2}; median |top z| 4.0
- QAT-W4A4-IAO (6/7 flagged): {'BN/ent': 6}; median |top z| 4.8
- QAT-W4weight (3/4 flagged): {'BN/ent': 2, 'D:iao4/ent': 1}; median |top z| 4.3
- QAT-W8weight (9/10 flagged): {'BN/kl': 7, 'BN/ent': 2}; median |top z| 7.6
- QuANTI (12/12 flagged): {'BN/ent': 9, 'W:quanti8/kl': 1, 'W:quanti4/ent': 1, 'D:quanti4/kl': 1}; median |top z| 13.0
- QuEST-8 (7/18 flagged): {'BN/ent': 2, 'D:quanti4/ent': 2, 'W:iao8/kl': 1, 'W:iao8/ent': 1, 'D:quanti4/kl': 1}; median |top z| 3.9
- QuEST-W4A4 (8/21 flagged): {'BN/ent': 7, 'D:iao4/ent': 1}; median |top z| 3.5
- Tian-8 (26/26 flagged): {'W:quanti4/ent': 11, 'D:quanti4/ent': 8, 'BN/kl': 4, 'W:quanti8/ent': 1, 'W:qnnpack8/ent': 1}; median |top z| 28.5
- Tian-W4A4 (17/17 flagged): {'BN/ent': 9, 'BN/kl': 6, 'D:quanti4/ent': 2}; median |top z| 8.8
- control (7/12 flagged): {'BN/ent': 4, 'BN/kl': 2, 'W:quanti4/ent': 1}; median |top z| 4.5
- unaware (70/75 flagged): {'BN/kl': 49, 'W:quanti4/ent': 12, 'BN/ent': 7, 'D:quanti4/ent': 2}; median |top z| 7.7

## Median z (kl)

| kind | W:fbgemm8 | W:qnnpack8 | W:quanti8 | W:iao8 | W:efrap4 | W:quanti4 | W:iao4 | D:efrap4 | D:quanti4 | D:iao4 | BN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ma-8 | 24.100 | 3.200 | 9.400 | 11.200 | 0.100 | 0.700 | 0.300 | 0.300 | 0.700 | -0.500 | 3.600 |
| Ma-W4A4 | 0.400 | -0.000 | -0.600 | 0.900 | 5.100 | -0.200 | -0.400 | -0.200 | -0.100 | 0.500 | -1.900 |
| QAT-W4A4-EFRAP | -0.200 | -0.800 | 0.100 | -0.400 | 1.000 | -0.000 | 0.400 | 1.400 | 0.100 | 1.100 | 3.300 |
| QAT-W4A4-IAO | -0.300 | 1.000 | 0.100 | 0.800 | -0.200 | -0.900 | -0.700 | 0.700 | -0.700 | 0.300 | 2.700 |
| QAT-W4weight | -0.200 | -0.100 | -0.700 | 0.200 | 0.300 | -0.500 | -0.400 | 0.400 | -1.300 | 1.900 | 0.600 |
| QAT-W8weight | 0.100 | -0.000 | 0.000 | -0.000 | 0.500 | -0.100 | 0.600 | 0.500 | 0.500 | -0.500 | 7.600 |
| QuANTI | 0.300 | 1.000 | 2.900 | 0.000 | 0.000 | 0.600 | 1.000 | -0.300 | -2.600 | 0.200 | -4.100 |
| QuEST-8 | -0.100 | -0.600 | -0.300 | -0.000 | 0.200 | 0.800 | 0.300 | -0.100 | 0.800 | 0.100 | 0.600 |
| QuEST-W4A4 | -0.300 | 0.600 | 0.300 | 0.400 | -0.300 | 0.100 | -0.700 | 0.200 | 0.300 | -1.200 | 1.500 |
| Tian-8 | 0.400 | 4.900 | 0.200 | 0.400 | 0.700 | 9.200 | 2.100 | 0.500 | 9.900 | 0.800 | 10.100 |
| Tian-W4A4 | -0.200 | -0.300 | 0.700 | -0.200 | 0.200 | 0.200 | 0.000 | -0.300 | -0.500 | 0.100 | 7.500 |
| control | 0.200 | 0.900 | -0.200 | -0.200 | -0.200 | -0.100 | -0.500 | -0.100 | 0.300 | -0.500 | 2.200 |
| unaware | -0.200 | 0.100 | -0.200 | 0.200 | 0.200 | 0.400 | 0.200 | 0.100 | 0.200 | 0.100 | 6.800 |

## Median z (ent)

| kind | W:fbgemm8 | W:qnnpack8 | W:quanti8 | W:iao8 | W:efrap4 | W:quanti4 | W:iao4 | D:efrap4 | D:quanti4 | D:iao4 | BN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ma-8 | 109.200 | -0.100 | 9.600 | 5.000 | -0.200 | 0.200 | 0.100 | 0.100 | -0.200 | -0.100 | 9.500 |
| Ma-W4A4 | 0.000 | 0.200 | 0.300 | 0.600 | 18.900 | -0.500 | -0.100 | 12.400 | -0.100 | -0.600 | 7.100 |
| QAT-W4A4-EFRAP | -0.000 | 0.100 | 0.400 | 0.200 | -1.500 | 0.800 | -0.200 | -2.400 | 0.900 | 1.000 | -1.800 |
| QAT-W4A4-IAO | 0.800 | 0.200 | -0.300 | -0.000 | -0.000 | -0.600 | -1.400 | -0.200 | -0.300 | -1.200 | -4.800 |
| QAT-W4weight | 0.800 | -0.100 | -0.500 | -0.200 | 0.700 | -2.100 | -0.400 | -0.200 | -2.400 | -0.500 | -3.900 |
| QAT-W8weight | -0.800 | -0.300 | 0.600 | -0.200 | 0.600 | -0.200 | 0.200 | 1.000 | 0.200 | -0.300 | -4.500 |
| QuANTI | 0.300 | 0.500 | 2.900 | -0.200 | 0.300 | 1.300 | -0.700 | -0.100 | -0.900 | -0.500 | 12.300 |
| QuEST-8 | 0.100 | -0.300 | 0.400 | 0.400 | 0.100 | 1.500 | 0.000 | -0.000 | 1.000 | 0.200 | -3.500 |
| QuEST-W4A4 | 0.100 | -0.100 | -0.200 | -0.300 | -0.000 | -0.000 | -0.700 | -0.000 | 0.200 | -1.000 | -3.400 |
| Tian-8 | 0.000 | 4.400 | 1.300 | -0.500 | 0.400 | 23.600 | 3.600 | 0.600 | 26.000 | 2.000 | 9.200 |
| Tian-W4A4 | -0.200 | 0.200 | -0.100 | -0.100 | 0.300 | 0.400 | 0.200 | 0.400 | 0.200 | 0.200 | 7.400 |
| control | -0.300 | 1.000 | -0.200 | -0.200 | -1.100 | -0.000 | -0.200 | 0.000 | -0.200 | -0.500 | -3.600 |
| unaware | -0.100 | -0.000 | -0.000 | -0.300 | -0.200 | 0.100 | -0.300 | -0.200 | 0.500 | 0.400 | -3.000 |

## P1 unaware normals

- flagged models 70/75 = 0.933
- W kl pairs |z| < 4: 0.994
- hit rate per test type: B/ent 0.373, B/kl 0.853, D/ent 0.076, D/kl 0.022, W/ent 0.034, W/kl 0.006
- |z| quantiles (all tests): 50% 0.82, 95% 6.38, 99% 11.98, max 34.7
- flagged unaware: C10-VGG-light__041e602acb BN/kl 6.5; C10-VGG-standard-Adam__165125ab9a BN/kl 8.1; C100-R18-hard-Finetune-ImageNet__257c404a84 W:quanti4/ent 11.8; C100-R18-light__12df86327a BN/kl 9.3; C100-R18-light__156d78492a BN/kl 6.5; C100-R18-standard-Adam__0929db4636 BN/kl 7.3; GTSRB-R18-hard-Finetune-ImageNet__2705cfea77 BN/ent -8.8; GTSRB-R18-light__014ca12d9d BN/ent -7.1; GTSRB-R18-light__03ba3153ab BN/ent -6.8; GTSRB-R18-standard-Adam__22d3c1b479 BN/kl 5.2; hard-CleanParent-QuANTI__09c213e80a W:quanti4/ent 12.1; hard-CleanParent-QuANTI__1505e63fa8 W:quanti4/ent 15.0; hard-CleanParent-QuANTI__172d688ec2 W:quanti4/ent 17.1; hard-CleanParent-QuANTI__21de2fd046 W:quanti4/ent 19.5; hard-CleanParent-QuEST__1535f1316e BN/kl 9.1; hard-Finetune-ImageNet__08d4ea5c2d D:quanti4/ent 16.4; hard-Finetune-ImageNet__1db74b849a D:quanti4/ent 13.9; hard-HighLR__1a1c3eba26 BN/kl 9.6; hard-HighLR__249b46acea BN/kl 7.8; hard-LabelSmoothing__0331a58040 BN/kl 6.0; hard-LabelSmoothing__065a821d9b BN/kl 5.6; hard-LabelSmoothing__073d7cd8bc BN/kl 10.1; hard-LabelSmoothing__083248943e BN/kl 8.9; hard-Mixup__017f5544a3 BN/ent -14.8; hard-Mixup__1383b97a4e BN/ent -17.0; hard-Short__243a3734bf BN/kl 8.6; hard-Short__403b5a5ae6 BN/kl 9.2; light__029351e251 BN/kl 7.3; light__07f6618e9c BN/kl 7.6; light__0e271924b6 BN/kl 10.4; light__0ea9c3d160 BN/kl 10.7; light__1402703160 BN/ent -6.8; light__155ac7c58a BN/kl 7.9; light__16a3824d38 BN/kl 12.6; light__18129ab543 BN/kl 8.5; light__1aeb1d413e BN/ent -5.3; light__1b897a0b8c BN/kl 6.5; light__21126dfbd9 BN/kl 8.2; light__248c55853b BN/kl 7.0; light__2c12b9e396 BN/kl 5.2; light__3e4c02697a BN/kl 10.1; standard-AdamW__33fc053c79 BN/kl 5.6; standard-AdamW__68b4b86506 BN/kl 7.3; standard-AdamW__6ea99bc36b BN/kl 5.2; standard-AdamW__a1d944daff BN/kl 8.7; standard-AdamW__b843cdb8ef BN/kl 4.6; standard-AdamW__d863918c4d BN/kl 9.8; standard-AdamW__e78fae480a BN/kl 6.8; standard-AdamW__e8b8a9bf43 BN/kl 6.8; standard-AdamW__fcc9a8e432 BN/kl 7.0; standard-Adam__01f881a9d3 BN/kl 4.5; standard-Adam__0cadbefe3a BN/kl 9.4; standard-Adam__1430d85451 BN/kl 7.3; standard-Adam__249c786791 BN/kl 6.2; standard-Adam__43027bb8d2 BN/kl 6.7; standard-Adam__68975aa3c4 BN/kl 5.2; standard-Adam__6aaea59548 BN/kl 4.9; standard-Adam__8a2467d2fe BN/kl 11.8; standard-Adam__c1c9d0c117 BN/kl 6.5; standard-Adam__d5b157976a BN/kl 4.8; standard-SGDlr.1__1488290ee4 BN/kl 9.7; standard-SGDlr.1__4a5049e78f BN/kl 7.9; standard-SGDlr.1__64c782bb2d W:quanti4/ent 7.7; standard-SGDlr.1__8b45873a61 W:quanti4/ent 10.2; standard-SGDlr.1__ae7fd0787d W:quanti4/ent 17.9; standard-SGDlr.1__bf5555e88b W:quanti4/ent 21.8; standard-SGDlr.1__c42c8ed85e W:quanti4/ent 7.6; standard-SGDlr.1__c90d4f4ddf BN/kl 7.5; standard-SGDlr.1__d8d9715655 W:quanti4/ent 34.7; standard-SGDlr.1__db9493bddb W:quanti4/ent 20.6

## P2 weight-switch families (own format W kl z >= 4)

- Ma-8 W:fbgemm8 kl: z >= 4 16/16, median 24.1; top test = W kl same bit width 0.06
- Ma-W4A4 W:efrap4 kl: z >= 4 4/6, median 5.1; top test = W kl same bit width 0.00
- Tian-8 W:qnnpack8 kl: z >= 4 15/26, median 4.9; top test = W kl same bit width 0.00
- QuANTI W:quanti8 kl: z >= 4 1/12, median 2.9; top test = W kl same bit width 0.08

## P3 Q-only tuning (ent z <= -4)

- QAT-W4A4-IAO over ['W:iao4', 'D:iao4']: 0/7; min z per model [-1.1, -2.0, -1.8, -2.4, -2.3, -1.0, -1.9]
- QAT-W4A4-EFRAP over ['W:efrap4', 'D:efrap4']: 0/4; min z per model [-2.9, -1.7, -4.0, -2.0]
- QAT-W4weight over ['W:efrap4', 'W:quanti4', 'W:iao4']: 0/4; min z per model [-1.9, -2.7, -2.3, -2.1]
- QAT-W8weight over ['W:fbgemm8', 'W:qnnpack8', 'W:quanti8', 'W:iao8']: 0/10; min z per model [-0.5, -0.6, -0.1, -1.1, -2.3, -1.0, -1.9, -2.1, -2.2, -0.6]

Controls:

| model | format | min ent z | kl z (min, max) | flag | top | top z |
|---|---|---|---|---|---|---|
| C10-VGG-control-QuEST-IAO-W8A8__306a2f4938 | ['iao8'] | -2.500 | (-0.6, -0.6) | False | W:iao8/ent | -2.500 |
| C10-VGG-control-Tian-INT8-qnnpack__ca1739df3c | ['qnnpack8'] | -1.100 | (2.2, 2.2) | True | W:quanti4/ent | 12.800 |
| C100-R18-control-QuEST-IAO-W4A4__d00b81bbf2 | ['iao4'] | -1.900 | (-2.7, 0.1) | False | D:iao4/kl | -2.700 |
| C100-R18-control-Tian-INT8-qnnpack__d2b53e9596 | ['qnnpack8'] | 2.200 | (0.3, 0.3) | True | BN/kl | 11.300 |
| GTSRB-R18-control-QuEST-IAO-W8A8__ed95943fc2 | ['iao8'] | -0.600 | (-0.2, -0.2) | False | BN/ent | -3.800 |
| GTSRB-R18-control-Tian-INT8-qnnpack__c69573ac01 | ['qnnpack8'] | 0.900 | (-1.5, -1.5) | True | BN/ent | -4.500 |
| control-QuANTI-QuANTI-fake-W4__721f58ec73 | ['quanti4'] | -3.000 | (-2.5, 1.1) | True | BN/ent | 11.800 |
| control-QuANTI-QuANTI-fake-W8__7886781e64 | ['quanti8'] | -0.100 | (-1.7, -1.7) | True | BN/ent | 7.500 |
| control-QuEST-IAO-W4A4__01200a508c | ['iao4'] | -1.600 | (-1.9, -0.3) | False | BN/ent | -3.000 |
| control-QuEST-IAO-W8A8__4cd9f05735 | ['iao8'] | -0.300 | (-0.1, -0.1) | False | BN/ent | -3.400 |
| control-Tian-INT8-qnnpack__34fb9f24f8 | ['qnnpack8'] | -0.900 | (-0.6, -0.6) | True | BN/ent | -4.600 |
| control-Tian-W4A4-EFRAP__f75d0dd14d | ['efrap4'] | 0.400 | (-1.5, -0.4) | True | BN/kl | 9.300 |

## P4 blank families

- QuEST W4A4: iao4 (W or D) kl or ent z <= -4: 1/21; top test on iao4: 3/21
- Qu-ANTI 4 D:quanti4 kl z <= -4: 2/12, median -2.6
- Tian W4A4 BN kl z >= 4: 17/17, median 7.5
- QuEST-8: flagged 7/18; top tests {'BN/ent': 2, 'D:quanti4/ent': 2, 'W:iao8/kl': 1, 'W:iao8/ent': 1}
- Tian-W4A4: flagged 17/17; top tests {'BN/ent': 9, 'BN/kl': 6, 'D:quanti4/ent': 2}
- QuEST-W4A4: flagged 8/21; top tests {'BN/ent': 7, 'D:iao4/ent': 1}

## P5 BN component

- Ma-8: BN kl z >= 4 5/16, <= -4 0, median 3.6
- Tian-8: BN kl z >= 4 25/26, <= -4 0, median 10.1
- Tian-W4A4: BN kl z >= 4 17/17, <= -4 0, median 7.5
- Ma-W4A4: BN kl z >= 4 0/6, <= -4 0, median -1.9
- QuANTI: BN kl z >= 4 0/12, <= -4 6, median -4.1
- QuEST-8: BN kl z >= 4 0/18, <= -4 0, median 0.6
- QuEST-W4A4: BN kl z >= 4 1/21, <= -4 0, median 1.5
- unaware: BN kl z >= 4 64/75, <= -4 0, median 6.8
- QAT normals + controls with BN |z| >= 4 (kl or ent): 26/37 = 0.70

## P6 flagged by group

| g6 | size | sum | mean |
|---|---|---|---|
| QAT+control | 37 | 27 | 0.730 |
| QCB | 116 | 92 | 0.793 |
| unaware | 75 | 70 | 0.933 |

## Falsification

1. unaware flagged 0.933 (fails if > .20)
2. QCB flagged 0.793 (fails if < .60); blank families flagged: QuEST-W4A4 0.38, Tian-W4A4 1.00, Qu-ANTI 4 (tests on quanti4 only) 0.25 (fails if all three < .5)
3. P0 pairs rel diff > 1e-5: 0
