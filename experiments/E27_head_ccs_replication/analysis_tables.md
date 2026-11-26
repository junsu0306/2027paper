# E27 tables (run `E27_20261126_001`, 125/125 models)

## P0 anchors (batch1)

| model | z_mmbd E27 | r_z E24 | cand E27 | cand E07 |
|---|---|---|---|---|
| QuEST-W4A4__031ca4fe17 | 15.81 | 15.81 | 2 | 2 |
| QuEST-W4A4__04aaf1c62f | 15.05 | 15.05 | 9 | 9 |
| QuEST-W4A4__0ee2dafd23 | 18.84 | 18.84 | 4 | 4 |
| QuEST-W4A4__109ef8fe05 | 22.63 | 22.63 | 5 | 5 |
| control-QuANTI-QuANTI-fake-W4__721f58ec73 | 0.56 | 0.56 | 7 | 7 |
| control-QuANTI-QuANTI-fake-W8__7886781e64 | 0.34 | 0.34 | 7 | 7 |
| control-QuEST-IAO-W4A4__01200a508c | 0.76 | 0.76 | 7 | 7 |
| control-QuEST-IAO-W8A8__4cd9f05735 | 0.37 | 1.15 | 2 | 4 |
| control-Tian-INT8-qnnpack__34fb9f24f8 | -0.71 | -0.71 | 4 | 4 |
| control-Tian-W4A4-EFRAP__f75d0dd14d | 0.32 | 0.32 | 1 | 1 |
| light__029351e251 | -1.54 | -1.54 | 4 | 4 |
| light__07f6618e9c | 0.92 | 0.92 | 7 | 7 |
| standard-AdamW__33fc053c79 | -0.38 | -0.38 | 7 | 7 |
| standard-AdamW__68b4b86506 | 0.40 | 0.40 | 4 | 4 |
| standard-Adam__01f881a9d3 | 0.94 | 0.94 | 7 | 7 |
| standard-Adam__0cadbefe3a | -2.79 | -2.79 | 1 | 1 |
| standard-SGDlr.1__1488290ee4 | -3.68 | -3.68 | 2 | 2 |
| standard-SGDlr.1__4a5049e78f | -9804.51 | -9804.51 | 2 | 2 |

MM-BD candidate equal to E07: 17/18

## New QuEST W4A4 (17)

| dataset | n | cand_mmbd = target | z_mmbd > 5 | fast = target | z_fast > 5 | z_max > 5 | z_max cls = target | mean z_mmbd < 0 |
|---|---|---|---|---|---|---|---|---|
| CIFAR-10 | 7 | 2 | 2 | 5 | 4 | 6 | 6 | 5 |
| CIFAR-100 | 6 | 1 | 0 | 0 | 2 | 2 | 0 | 3 |
| GTSRB | 4 | 4 | 2 | 1 | 0 | 2 | 2 | 3 |
| all | 17 | 7 | 4 | 6 | 6 | 10 | 8 | 11 |

| model | dataset | target | cand | z_mmbd | z_target | z_fast (cls) | z_max (cls) | ccsF cand | ccsQ cand | ccsQ others med | mean z_mmbd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QuEST-W4A4__031ca4fe17 (anchor) | CIFAR-10 | 2.0 | 2 | 15.8 | 15.8 | 15.8 (2) | 15.8 (2) | 608 | 55 | 708 | -10.3 |
| QuEST-W4A4__04aaf1c62f (anchor) | CIFAR-10 | 9.0 | 9 | 15.1 | 15.1 | 15.1 (9) | 15.1 (9) | 1529 | 86 | 576 | -7.9 |
| QuEST-W4A4__0ee2dafd23 (anchor) | CIFAR-10 | 4.0 | 4 | 18.8 | 18.8 | 18.8 (4) | 18.9 (3) | 1708 | 72 | 641 | -13.1 |
| QuEST-W4A4__109ef8fe05 (anchor) | CIFAR-10 | 5.0 | 5 | 22.6 | 22.6 | 22.6 (5) | 22.6 (5) | 863 | 93 | 578 | -9.2 |
| QuEST-W4A4__26f6c22716 | CIFAR-10 | 4.0 | 2 | 0.2 | 12.2 | 0.2 (2) | 12.6 (3) | 2111 | 550 | 573 | 0.5 |
| QuEST-W4A4__3e369c7e98 | CIFAR-10 | 5.0 | 5 | 30.5 | 30.5 | 0.8 (2) | 30.5 (5) | 1958 | 53 | 663 | -11.4 |
| QuEST-W4A4__3f36aa7133 | CIFAR-10 | 3.0 | 0 | -3.9 | 14.0 | 14.0 (3) | 14.0 (3) | 1022 | 640 | 544 | -0.8 |
| QuEST-W4A4__40a6662040 | CIFAR-10 | 3.0 | 1 | -0.0 | 1.6 | 1.6 (3) | 1.6 (3) | 1084 | 967 | 936 | -1.6 |
| QuEST-W4A4__461bfddff2 | CIFAR-10 | 0.0 | 0 | 18.0 | 18.0 | 18.0 (0) | 18.0 (0) | 981 | 50 | 691 | -2.4 |
| QuEST-W4A4__4788189310 | CIFAR-10 | 9.0 | 6 | -1.1 | 13.5 | 13.5 (9) | 13.5 (9) | 1812 | 785 | 686 | 1.4 |
| QuEST-W4A4__480832a417 | CIFAR-10 | 3.0 | 2 | 0.2 | 11.9 | 11.9 (3) | 11.9 (3) | 923 | 548 | 621 | -3.0 |
| C100-R18-QuEST-W4A4__307f9e1cef | CIFAR-100 | 24.0 | 14 | 0.7 | 2.8 | -2.1 (65) | 3.7 (82) | 4319 | 452 | 452 | 0.5 |
| C100-R18-QuEST-W4A4__3f5bf3ad73 | CIFAR-100 | 62.0 | 14 | 0.7 | 4.6 | -3.6 (32) | 4.8 (82) | 4257 | 455 | 455 | 0.5 |
| C100-R18-QuEST-W4A4__628f7e5498 | CIFAR-100 | 19.0 | 19 | 1.0 | 1.0 | -0.3 (55) | 1.9 (9) | 4356 | 420 | 420 | -0.2 |
| C100-R18-QuEST-W4A4__6ac66ac422 | CIFAR-100 | 77.0 | 2 | -1.6 | -0.7 | 12.8 (55) | 14.3 (29) | 4469 | 569 | 569 | -1.3 |
| C100-R18-QuEST-W4A4__a8b43a399d | CIFAR-100 | 50.0 | 14 | -1.5 | -1.9 | 12.0 (72) | 13.4 (29) | 4467 | 568 | 568 | -2.0 |
| C100-R18-QuEST-W4A4__c2dc3f9af1 | CIFAR-100 | 87.0 | 14 | 4.0 | -0.5 | 2.9 (72) | 4.4 (2) | 4361 | 380 | 542 | 1.1 |
| GTSRB-R18-QuEST-W4A4__0bdc7f8aa7 | GTSRB | 23.0 | 23 | 0.3 | 0.3 | -3.3 (1) | 1.8 (32) | 5082 | 571 | 571 | 0.4 |
| GTSRB-R18-QuEST-W4A4__7772a2aa5c | GTSRB | 2.0 | 2 | 0.2 | 0.2 | 0.2 (2) | 2.1 (24) | 3824 | 449 | 519 | -11.9 |
| GTSRB-R18-QuEST-W4A4__8e9b2f238d | GTSRB | 1.0 | 1 | 5.9 | 5.9 | -2.9 (5) | 5.9 (1) | 4042 | 335 | 557 | -31.6 |
| GTSRB-R18-QuEST-W4A4__c878e3575d | GTSRB | 25.0 | 25 | 5.6 | 5.6 | -3.0 (13) | 5.6 (25) | 4515 | 373 | 545 | -14.9 |

## Normals and controls (104)

| set | n | z_mmbd > 5 | max z_mmbd | z_fast > 5 | max z_fast | z_max > 5 | max z_max |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | 86 | 2 | 10.5 | 4 | 79.1 | 33 | 79.5 |
| CIFAR-100 | 9 | 0 | 1.0 | 6 | 38.4 | 7 | 38.5 |
| GTSRB | 9 | 0 | 2.7 | 0 | 2.7 | 1 | 7.5 |
| control-nopoison | 10 | 0 | 1.1 | 1 | 17.0 | 3 | 28.9 |
| normal-hard | 44 | 1 | 10.5 | 6 | 79.1 | 18 | 79.5 |
| normal-light | 18 | 0 | 4.2 | 2 | 22.6 | 9 | 69.8 |
| normal-standard | 32 | 1 | 5.1 | 1 | 5.1 | 11 | 23.8 |
| hard:QAT-W4A4-IAO | 6 | 0 | -0.1 | 1 | 38.4 | 4 | 38.5 |
| hard:CleanParent-QuEST | 4 | 0 | 1.0 | 1 | 79.1 | 2 | 79.5 |
| all | 104 | 2 | 10.5 | 10 | 79.1 | 41 | 79.5 |

Normals with z_mmbd > 5 or z_fast > 5:

| model | dataset | cell | cand | z_mmbd | fast | z_fast | z_max (cls) |
|---|---|---|---|---|---|---|---|
| C100-R18-hard-Finetune-ImageNet__257c404a84 | CIFAR-100 | C100-R18-hard-Finetune-ImageNet | 14 | -2.5 | 32 | 19.1 | 19.1 (32) |
| C100-R18-hard-QAT-W4A4-IAO__02f67e9f33 | CIFAR-100 | C100-R18-hard-QAT-W4A4-IAO | 61 | -0.1 | 65 | 38.4 | 38.5 (32) |
| C100-R18-hard-QAT-W8weight__0b33002bce | CIFAR-100 | C100-R18-hard-QAT-W8weight | 14 | -0.4 | 32 | 34.0 | 34.0 (32) |
| C100-R18-hard-QAT-W8weight__1e5563e91a | CIFAR-100 | C100-R18-hard-QAT-W8weight | 14 | -1.5 | 65 | 29.0 | 29.9 (29) |
| C100-R18-light__12df86327a | CIFAR-100 | C100-R18-light | 14 | -2.8 | 32 | 20.2 | 21.4 (10) |
| C100-R18-light__156d78492a | CIFAR-100 | C100-R18-light | 14 | -2.1 | 65 | 22.6 | 23.6 (18) |
| control-Tian-INT8-qnnpack__34fb9f24f8 | CIFAR-10 | control-Tian-INT8-qnnpack | 4 | -0.7 | 2 | 17.0 | 17.3 (3) |
| hard-CleanParent-QuANTI__21de2fd046 | CIFAR-10 | hard-CleanParent-QuANTI | 2 | 10.5 | 2 | 10.5 | 10.5 (2) |
| hard-CleanParent-QuEST__21e0ee36dd | CIFAR-10 | hard-CleanParent-QuEST | 7 | 0.2 | 2 | 79.1 | 79.5 (3) |
| standard-SGDlr.1__c42c8ed85e | CIFAR-10 | standard-SGDlr.1 | 2 | 5.1 | 2 | 5.1 | 5.1 (5) |

## AUROC (QuEST W4A4 vs same-dataset normals and controls)

| dataset | z_mmbd (new) | z_mmbd (all) | z_fast (all) | z_max (all) | mean z_mmbd, lower = QCB (all) |
|---|---|---|---|---|---|
| CIFAR-10 | 0.56 | 0.72 | 0.95 | 0.77 | 0.76 |
| CIFAR-100 | 0.76 | 0.76 | 0.28 | 0.22 | 0.54 |
| GTSRB | 0.72 | 0.72 | 0.42 | 0.58 | 0.92 |

## Falsification criteria

1. new QuEST W4A4 with z_mmbd > 5: 4/17 (fails if <= 8 of 17)
2. normals with z_mmbd > 5: 2/104 (fails if > 10); hard: 0/10 (fails if >= 3)
3. anchors with z_mmbd <= 5: 0; MM-BD candidate != E07: 1 of 18 (fails if >= 3)

Unconverged classes (ones, Q) in any model: 0 models; (ones, F): 0
