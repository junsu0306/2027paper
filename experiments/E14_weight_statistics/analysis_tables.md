# E14 tables (912/912 jobs done, 0 errors)

## Thresholds (batch-1 normal + control max)

| recipe | T1_lowrank | T2_zero_max | T3_edge |
|---|---|---|---|
| efrap4 | 1.180 | 0.769 | 0.250 |
| fbgemm8 | 1.074 | 0.718 | 0.250 |
| iao4 | 1.158 | 0.785 | 0.250 |
| iao8 | 1.043 | 0.693 | 0.250 |
| qnnpack8 | 1.061 | 0.742 | 0.250 |
| quanti4 | 3.140 | 0.947 | 0.251 |
| quanti8 | 1.052 | 0.741 | 0.250 |

## Held-out QCB: models above the batch-1 threshold (own recipe)

| batch | cell2 | recipe | n | T1_lowrank above | T2_zero_max above | T3_edge above | T1_lowrank median | T2_zero_max median | T3_edge median |
|---|---|---|---|---|---|---|---|---|---|
| batch2 | Ma-8 | fbgemm8 | 8 | 8 | 0 | 8 | 8.304 | 0.011 | 0.377 |
| batch2 | Ma-W4A4 | efrap4 | 2 | 2 | 0 | 2 | 2.785 | 0.164 | 0.253 |
| batch2 | QuANTI-4 | quanti4 | 8 | 1 | 6 | 0 | 2.761 | 0.960 | 0.169 |
| batch2 | QuANTI-8 | quanti8 | 8 | 3 | 0 | 0 | 1.031 | 0.560 | 0.200 |
| batch2 | QuEST-8 | iao8 | 8 | 1 | 0 | 0 | 1.021 | 0.009 | 0.250 |
| batch2 | QuEST-W4A4 | iao4 | 7 | 0 | 0 | 0 | 1.024 | 0.166 | 0.250 |
| batch2 | Tian-8 | qnnpack8 | 8 | 0 | 0 | 3 | 1.025 | 0.482 | 0.250 |
| batch2 | Tian-W4A4 | efrap4 | 8 | 0 | 0 | 3 | 1.033 | 0.224 | 0.250 |
| batch_x | C10-VGG-Ma-8 | fbgemm8 | 4 | 4 | 0 | 4 | 6.979 | 0.011 | 0.363 |
| batch_x | C10-VGG-Tian-8 | qnnpack8 | 6 | 2 | 1 | 3 | 1.027 | 0.480 | 0.250 |
| batch_x | C10-VGG-Tian-W4A4 | efrap4 | 5 | 0 | 0 | 0 | 1.026 | 0.290 | 0.248 |
| batch_x | C100-R18-QuEST-8 | iao8 | 3 | 1 | 0 | 0 | 1.013 | 0.016 | 0.250 |
| batch_x | C100-R18-QuEST-W4A4 | iao4 | 6 | 0 | 0 | 6 | 1.021 | 0.215 | 0.251 |
| batch_x | C100-R18-Tian-8 | qnnpack8 | 4 | 1 | 1 | 1 | 1.037 | 0.444 | 0.249 |
| batch_x | GTSRB-R18-QuEST-8 | iao8 | 4 | 1 | 0 | 0 | 1.032 | 0.009 | 0.250 |
| batch_x | GTSRB-R18-QuEST-W4A4 | iao4 | 4 | 0 | 0 | 0 | 1.029 | 0.137 | 0.250 |
| batch_x | GTSRB-R18-Tian-8 | qnnpack8 | 4 | 0 | 1 | 0 | 1.021 | 0.341 | 0.250 |

## (reference) batch-1 QCB medians

| cell2 | recipe | T1_lowrank | T2_zero_max | T3_edge |
|---|---|---|---|---|
| Ma-8 | fbgemm8 | 8.256 | 0.010 | 0.372 |
| Ma-W4A4 | efrap4 | 2.631 | 0.163 | 0.253 |
| QuANTI-4 | quanti4 | 2.759 | 0.957 | 0.167 |
| QuANTI-8 | quanti8 | 1.079 | 0.561 | 0.200 |
| QuEST-8 | iao8 | 1.021 | 0.009 | 0.250 |
| QuEST-W4A4 | iao4 | 1.014 | 0.169 | 0.251 |
| Tian-8 | qnnpack8 | 1.017 | 0.500 | 0.250 |
| Tian-W4A4 | efrap4 | 1.031 | 0.230 | 0.250 |

## Held-out normals + controls: false-alarm rate per recipe

| recipe | n | T1_lowrank FA | T2_zero_max FA | T3_edge FA |
|---|---|---|---|---|
| efrap4 | 82 | 0.000 | 0.024 | 0.171 |
| fbgemm8 | 82 | 0.061 | 0.012 | 0.073 |
| iao4 | 82 | 0.000 | 0.024 | 0.098 |
| iao8 | 82 | 0.159 | 0.024 | 0.134 |
| qnnpack8 | 82 | 0.049 | 0.037 | 0.024 |
| quanti4 | 82 | 0.000 | 0.024 | 0.085 |
| quanti8 | 82 | 0.110 | 0.024 | 0.061 |

## ... per kind (all recipes)

| kind | n | T1_lowrank FA | T2_zero_max FA | T3_edge FA |
|---|---|---|---|---|
| control-nopoison | 42 | 0.071 | 0.024 | 0.190 |
| hard-CleanParent-QuANTI | 28 | 0.071 | 0.179 | 0.000 |
| hard-CleanParent-QuEST | 28 | 0.036 | 0.000 | 0.000 |
| hard-Finetune-ImageNet | 28 | 0.107 | 0.000 | 0.000 |
| hard-HighLR | 14 | 0.000 | 0.000 | 0.000 |
| hard-LabelSmoothing | 28 | 0.071 | 0.000 | 0.036 |
| hard-Mixup | 14 | 0.000 | 0.000 | 0.000 |
| hard-QAT-W4A4-EFRAP | 28 | 0.107 | 0.000 | 0.214 |
| hard-QAT-W4A4-IAO | 49 | 0.061 | 0.000 | 0.286 |
| hard-QAT-W4weight | 28 | 0.036 | 0.000 | 0.107 |
| hard-QAT-W8weight | 70 | 0.043 | 0.000 | 0.200 |
| hard-Short | 14 | 0.000 | 0.000 | 0.000 |
| normal-light | 98 | 0.010 | 0.000 | 0.061 |
| normal-standard | 105 | 0.086 | 0.076 | 0.010 |

## False alarm on each statistic's main recipe, by kind

| stat | recipes | kind | FA |
|---|---|---|---|
| T1_lowrank | fbgemm8,efrap4 | control-nopoison | 0.000 |
| T1_lowrank | fbgemm8,efrap4 | hard-CleanParent-QuANTI | 0.000 |
| T1_lowrank | fbgemm8,efrap4 | hard-CleanParent-QuEST | 0.000 |
| T1_lowrank | fbgemm8,efrap4 | hard-Finetune-ImageNet | 0.125 |
| T1_lowrank | fbgemm8,efrap4 | hard-HighLR | 0.000 |
| T1_lowrank | fbgemm8,efrap4 | hard-LabelSmoothing | 0.000 |
| T1_lowrank | fbgemm8,efrap4 | hard-Mixup | 0.000 |
| T1_lowrank | fbgemm8,efrap4 | hard-QAT-W4A4-EFRAP | 0.000 |
| T1_lowrank | fbgemm8,efrap4 | hard-QAT-W4A4-IAO | 0.143 |
| T1_lowrank | fbgemm8,efrap4 | hard-QAT-W4weight | 0.000 |
| T1_lowrank | fbgemm8,efrap4 | hard-QAT-W8weight | 0.050 |
| T1_lowrank | fbgemm8,efrap4 | hard-Short | 0.000 |
| T1_lowrank | fbgemm8,efrap4 | normal-light | 0.000 |
| T1_lowrank | fbgemm8,efrap4 | normal-standard | 0.033 |
| T2_zero_max | qnnpack8 | control-nopoison | 0.000 |
| T2_zero_max | qnnpack8 | hard-CleanParent-QuANTI | 0.250 |
| T2_zero_max | qnnpack8 | hard-CleanParent-QuEST | 0.000 |
| T2_zero_max | qnnpack8 | hard-Finetune-ImageNet | 0.000 |
| T2_zero_max | qnnpack8 | hard-HighLR | 0.000 |
| T2_zero_max | qnnpack8 | hard-LabelSmoothing | 0.000 |
| T2_zero_max | qnnpack8 | hard-Mixup | 0.000 |
| T2_zero_max | qnnpack8 | hard-QAT-W4A4-EFRAP | 0.000 |
| T2_zero_max | qnnpack8 | hard-QAT-W4A4-IAO | 0.000 |
| T2_zero_max | qnnpack8 | hard-QAT-W4weight | 0.000 |
| T2_zero_max | qnnpack8 | hard-QAT-W8weight | 0.000 |
| T2_zero_max | qnnpack8 | hard-Short | 0.000 |
| T2_zero_max | qnnpack8 | normal-light | 0.000 |
| T2_zero_max | qnnpack8 | normal-standard | 0.133 |
| T3_edge | fbgemm8 | control-nopoison | 0.000 |
| T3_edge | fbgemm8 | hard-CleanParent-QuANTI | 0.000 |
| T3_edge | fbgemm8 | hard-CleanParent-QuEST | 0.000 |
| T3_edge | fbgemm8 | hard-Finetune-ImageNet | 0.000 |
| T3_edge | fbgemm8 | hard-HighLR | 0.000 |
| T3_edge | fbgemm8 | hard-LabelSmoothing | 0.000 |
| T3_edge | fbgemm8 | hard-Mixup | 0.000 |
| T3_edge | fbgemm8 | hard-QAT-W4A4-EFRAP | 0.250 |
| T3_edge | fbgemm8 | hard-QAT-W4A4-IAO | 0.000 |
| T3_edge | fbgemm8 | hard-QAT-W4weight | 0.000 |
| T3_edge | fbgemm8 | hard-QAT-W8weight | 0.300 |
| T3_edge | fbgemm8 | hard-Short | 0.000 |
| T3_edge | fbgemm8 | normal-light | 0.143 |
| T3_edge | fbgemm8 | normal-standard | 0.000 |

## Predictions

| prediction | result | holds |
|---|---|---|
| P1 held-out Ma above (all) | 1.00 of 14 | True |
| P1 other QCB cells <= 1/4 (worst) | 0.38 (QuANTI-8) | False |
| P1 normal FA <= 5 % per recipe (worst) | 0.159 | False |
| P2 held-out Tian-8 above on qnnpack8 >= .9 | 0.14 of 22 | False |
| P2 normal FA on qnnpack8 <= 5 % | 0.037 | True |
| P3 held-out Ma 8-bit above | 1.00 of 12 | True |
| P3 Tian / Qu-ANTI / QuEST cells <= 1/4 (worst) | 1.00 (C100-R18-QuEST-W4A4) | False |
| P4 any: held-out Ma + Tian-8 >= .9 | 0.67 | False |
| P4 any: Tian W4A4 / QuEST cells <= 1/4 (worst) | 1.00 (C100-R18-QuEST-W4A4) | False |
| P4 any: normal FA <= 10 % | 0.162 | False |

## Any statistic, per held-out cell

| batch | cell2 | sum | size |
|---|---|---|---|
| batch2 | Ma-8 | 8 | 8 |
| batch2 | Ma-W4A4 | 2 | 2 |
| batch2 | QuANTI-4 | 6 | 8 |
| batch2 | QuANTI-8 | 3 | 8 |
| batch2 | QuEST-8 | 1 | 8 |
| batch2 | QuEST-W4A4 | 0 | 7 |
| batch2 | Tian-8 | 3 | 8 |
| batch2 | Tian-W4A4 | 3 | 8 |
| batch_x | C10-VGG-Ma-8 | 4 | 4 |
| batch_x | C10-VGG-Tian-8 | 4 | 6 |
| batch_x | C10-VGG-Tian-W4A4 | 0 | 5 |
| batch_x | C100-R18-QuEST-8 | 1 | 3 |
| batch_x | C100-R18-QuEST-W4A4 | 6 | 6 |
| batch_x | C100-R18-Tian-8 | 2 | 4 |
| batch_x | GTSRB-R18-QuEST-8 | 1 | 4 |
| batch_x | GTSRB-R18-QuEST-W4A4 | 0 | 4 |
| batch_x | GTSRB-R18-Tian-8 | 1 | 4 |

## Falsification criteria

| criterion | hit |
|---|---|
| 1. held-out Tian-8 T2 above < 3/4 | True |
| 2. held-out Ma T1 above < 3/4 | False |
| 3. a hard-normal kind > 20 % on a main recipe | True |

| stat | recipes | kind | FA |
|---|---|---|---|
| T2_zero_max | qnnpack8 | hard-CleanParent-QuANTI | 0.250 |
| T3_edge | fbgemm8 | hard-QAT-W4A4-EFRAP | 0.250 |
| T3_edge | fbgemm8 | hard-QAT-W8weight | 0.300 |

## (descriptive) batch_x above the same-setting batch_xn max

| cell | recipe | T1_lowrank | T2_zero_max | T3_edge |
|---|---|---|---|---|
| C10-VGG-Ma-8 | fbgemm8 | 1.000 | 0.000 | 1.000 |
| C10-VGG-Tian-8 | qnnpack8 | 0.333 | 1.000 | 0.667 |
| C10-VGG-Tian-W4A4 | efrap4 | 0.600 | 0.000 | 0.000 |
| C100-R18-QuEST-8 | iao8 | 0.333 | 0.000 | 0.000 |
| C100-R18-QuEST-W4A4 | iao4 | 0.000 | 0.000 | 0.000 |
| C100-R18-Tian-8 | qnnpack8 | 0.250 | 1.000 | 0.250 |
| GTSRB-R18-QuEST-8 | iao8 | 0.000 | 0.000 | 0.000 |
| GTSRB-R18-QuEST-W4A4 | iao4 | 0.000 | 0.000 | 0.000 |
| GTSRB-R18-Tian-8 | qnnpack8 | 0.000 | 1.000 | 0.000 |

errors: 0
