# E06 c tables (102/102 jobs done, 0 errors)

## V: CIFAR-10 QuEST-8

| model_key | batch | asr_F | asr_clip_all | best_quantizer | best_single_asr | k_min | asr_clip_rest_at_k | d_target | d_other | clean_over_trig_above | trig_above_top |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QuEST-8__0f3f0eb850 | batch1 | 0.033 | 0.981 | layer4.0.shortcut.0 | 0.969 | 1 | 0.028 | 9.142 | -1.107 | 0.003 | 0.015 |
| QuEST-8__11cfea1850 | batch1 | 0.036 | 0.991 | layer4.0.shortcut.0 | 0.989 | 1 | 0.010 | 18.992 | -3.069 | 0.007 | 0.016 |
| QuEST-8__2b843210ec | batch1 | 0.054 | 0.992 | layer3.1.conv2 | 0.987 | 1 | 0.053 | 13.980 | -2.356 | 0.002 | 0.016 |
| QuEST-8__39124252cd | batch2 | 0.009 | 0.987 | layer4.0.shortcut.0 | 0.977 | 1 | 0.009 | 12.063 | -1.250 | 0.010 | 0.016 |
| QuEST-8__41ad4855c4 | batch2 | 0.029 | 0.978 | layer4.0.shortcut.0 | 0.936 | 1 | 0.017 | 7.165 | -0.857 | 0.002 | 0.015 |
| QuEST-8__4860b82558 | batch2 | 0.001 | 0.976 | layer3.1.conv2 | 0.969 | 1 | 0.001 | 23.370 | -2.839 | 0.007 | 0.021 |
| QuEST-8__52d4f98584 | batch2 | 0.018 | 0.987 | layer4.0.shortcut.0 | 0.957 | 1 | 0.018 | 8.095 | -1.081 | 0.007 | 0.016 |
| QuEST-8__55432716fa | batch2 | 0.030 | 0.992 | layer4.0.shortcut.0 | 0.992 | 1 | 0.002 | 32.494 | -4.805 | 0.002 | 0.016 |
| QuEST-8__78b41b346e | batch2 | 0.036 | 0.981 | layer4.0.shortcut.0 | 0.968 | 1 | 0.019 | 11.002 | -1.562 | 0.003 | 0.015 |
| QuEST-8__79a324c90b | batch2 | 0.044 | 0.976 | layer4.0.conv2 | 0.926 | 1 | 0.024 | 8.392 | -1.219 | 0.001 | 0.062 |
| QuEST-8__835c0db952 | batch2 | 0.040 | 0.987 | layer4.0.shortcut.0 | 0.982 | 1 | 0.031 | 11.976 | -1.695 | 0.003 | 0.016 |

## S: models with a single-position binary saturation that drives the probe to the target

| cell | recipe | hit |
|---|---|---|
| Ma-8 | fbgemm8 | 0.000 |
| Ma-W4A4 | efrap4 | 0.000 |
| QuANTI-8+4 | quanti8 | 0.750 |
| QuEST-8 | iao8 | 0.000 |
| QuEST-W4A4 | iao4 | 0.000 |
| Tian-8 | qnnpack8 | 1.000 |
| Tian-W4A4 | efrap4 | 0.000 |

## S: positions that hit

| cell | recipe | name |
|---|---|---|
| QuANTI-8+4 | quanti8 | linear |
| Tian-8 | qnnpack8 | linear |

## Predictions

| prediction | result | holds |
|---|---|---|
| V1 clipping <= 8 channels at one position suffices (>= 3/4 of clip-switch models) | 1.00 of 11 | True |
| V2 clipping all but those channels stays off (>= 3/4) | 1.00 | True |
| V3 target logit rises when they are clipped (all) | [9.14, 18.99, 13.98, 12.06, 7.17, 23.37, 8.09, 32.49, 11.0, 8.39, 11.98] | True |
| V4 clean inputs exceed on those channels <= 1/10 of trigger | [0.003, 0.007, 0.002, 0.01, 0.002, 0.007, 0.007, 0.002, 0.003, 0.001, 0.003] | True |
| S1 Tian-8 (qnnpack8) has a hitting position (>= 3/4) | 1.00 | True |
| S1 QuANTI-8+4 (quanti8) has a hitting position (>= 3/4) | 0.75 | True |
| S2 Tian-8 hitting positions inside layer4 (blocks.6 / blocks.7) | ['linear'] | False |

## MM-BD in E07: rank of the true target among class margins (1 = largest), robust z

| cell | state | n | on | rank_med | z_med |
|---|---|---|---|---|---|
| Ma-8 | F | 4 | 0 | 1.000 | 2.145 |
| Ma-8 | Fclip | 4 | 0 | 1.000 | 3.731 |
| Ma-8 | Q | 4 | 4 | 1.000 | 4.663 |
| Ma-8 | Qw | 4 | 4 | 1.000 | 7.186 |
| Ma-W4A4 | F | 4 | 0 | 1.500 | 2.800 |
| Ma-W4A4 | Fclip | 4 | 0 | 1.000 | 4.100 |
| Ma-W4A4 | Q | 4 | 4 | 1.000 | 3.756 |
| Ma-W4A4 | Qw | 4 | 4 | 1.000 | 8.302 |
| QuANTI-8+4 | F | 4 | 0 | 1.000 | 25.256 |
| QuANTI-8+4 | Fclip | 8 | 0 | 1.000 | 13.937 |
| QuANTI-8+4 | Q | 8 | 8 | 1.000 | 9.065 |
| QuANTI-8+4 | Qw | 8 | 8 | 1.000 | 15.005 |
| QuEST-8 | F | 3 | 0 | 1.000 | 1.904 |
| QuEST-8 | Fbn | 3 | 2 | 1.000 | 2.092 |
| QuEST-8 | Fclip | 3 | 3 | 3.000 | 1.975 |
| QuEST-8 | Q | 3 | 3 | 3.000 | 1.228 |
| QuEST-8 | Qw | 3 | 0 | 1.000 | 4.784 |
| QuEST-W4A4 | F | 4 | 0 | 1.000 | 4.348 |
| QuEST-W4A4 | Fbn | 4 | 1 | 1.000 | 3.571 |
| QuEST-W4A4 | Fclip | 4 | 0 | 1.500 | 4.566 |
| QuEST-W4A4 | Q | 4 | 4 | 1.000 | 3.222 |
| QuEST-W4A4 | Qw | 4 | 4 | 1.000 | 3.084 |
| Tian-8 | F | 4 | 0 | 1.000 | 12.727 |
| Tian-8 | Fclip | 4 | 0 | 1.000 | 8.422 |
| Tian-8 | Q | 4 | 4 | 1.000 | 23.207 |
| Tian-8 | Qw | 4 | 4 | 1.000 | 26.606 |
| Tian-W4A4 | F | 4 | 0 | 1.000 | 13.198 |
| Tian-W4A4 | Fclip | 4 | 0 | 1.000 | 12.949 |
| Tian-W4A4 | Q | 4 | 4 | 1.000 | 15.428 |
| Tian-W4A4 | Qw | 4 | 4 | 1.000 | 23.715 |

## Falsification criteria

| criterion | hit |
|---|---|
| 1. V1 and V2 both fail | False |
| 2. target logit falls when the channels are clipped | False |
| 3. no single-position saturation drives Tian-8 or Qu-ANTI 8 to the target | False |

errors: 0
