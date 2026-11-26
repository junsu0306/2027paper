# E20 tables (27/27 jobs done, 0 errors)

## Per family (medians)

| fam | changed_all | codes_differ_all | share_l34 | share_l4_head | head_changed | edge_changed | edge_unchanged | zero_changed | zero_unchanged | revert_share_med | U_lowrank_med | U_lowrank_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QuEST-8 | 0.150 | 0.145 | 0.838 | 0.078 | 0.207 | 0.250 | 0.250 | 0.008 | 0.006 | 0.499 | 1.150 | 2.262 |
| QuEST-W4A4 | 0.150 | 0.093 | 0.843 | 0.056 | 0.305 | 0.253 | 0.250 | 0.161 | 0.116 | 0.497 | 1.359 | 2.026 |
| Tian-W4A4-R18 | 1.000 | 0.554 | 0.939 | 0.752 | 1.000 | 0.250 | 0.251 | 0.202 | 0.265 | 0.500 | 1.538 | 3.909 |
| Tian-W4A4-VGG | 0.998 | 0.602 | 0.000 | 0.000 | 0.999 | 0.249 | 0.107 | 0.228 | 0.955 | 0.500 | 1.679 | 4.909 |

## Share of the changed weights by stage (medians)

| fam | stage_vgg1 | stage_vgg2 | stage_vgg3 | stage_vgg4 | stage_vgg5 | stage_head | stage_stem | stage_layer1 | stage_layer2 | stage_layer3 | stage_layer4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| QuEST-8 | nan | nan | nan | nan | nan | 0.001 | 0.000 | 0.007 | 0.148 | 0.766 | 0.077 |
| QuEST-W4A4 | nan | nan | nan | nan | nan | 0.001 | 0.000 | 0.005 | 0.150 | 0.792 | 0.055 |
| Tian-W4A4-R18 | nan | nan | nan | nan | nan | 0.000 | 0.000 | 0.013 | 0.047 | 0.188 | 0.751 |
| Tian-W4A4-VGG | 0.003 | 0.015 | 0.100 | 0.401 | 0.480 | 0.000 | nan | nan | nan | nan | nan |

## BN entries changed (medians)

| fam | bn_weight | bn_bias | bn_running_mean | bn_running_var |
|---|---|---|---|---|
| QuEST-8 | 0.257 | 0.305 | 1.000 | 1.000 |
| QuEST-W4A4 | 0.204 | 0.271 | 1.000 | 1.000 |
| Tian-W4A4-R18 | 0.999 | 0.999 | 1.000 | 1.000 |
| Tian-W4A4-VGG | 1.000 | 1.000 | 1.000 | 1.000 |

## Predictions

| prediction | result | holds |
|---|---|---|
| P1 QuEST changed share in .10-.30 (all) | [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15] | True |
| P1 QuEST >= 70 % of changed in layer3-4 (all) | [0.8, 0.81, 0.81, 0.83, 0.83, 0.84, 0.85, 0.83, 0.83, 0.86, 0.84, 0.82, 0.83, 0.85, 0.85, 0.85, 0.85, 0.84, 0.85, 0.85, 0.84, 0.85, 0.85] | True |
| P1 QuEST classifier changed < 5 % (all) | [0.33, 0.367, 0.268, 0.143, 0.341, 0.355, 0.402, 0.344, 0.215, 0.169, 0.218, 0.2, 0.22, 0.248, 0.196, 0.227, 0.247, 0.269, 0.255, 0.305, 0.271, 0.262, 0.314] | False |
| P2 Tian W4A4 (ResNet) >= 95 % of changed in layer4 + head | [0.752] | False |
| P3 QuEST |edge changed - unchanged| >= .03 | [0.006, 0.007, 0.008, -0.001, 0.001, 0.001, 0.001, 0.001, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.003, 0.004, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003] | False |

## Falsification criteria

| criterion | hit |
|---|---|
| 1. QuEST changed share > 50 % | False |
| 2. QuEST changed and unchanged equal in edge and zero share (+-.01) | 0.39 of models |

errors: 0
