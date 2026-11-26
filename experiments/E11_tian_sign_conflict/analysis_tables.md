# E11 tables (50/50 jobs done, 0 errors)

## Models on (Q, D, ND) and draws on (Rnd, Rall); median ASR in brackets

| cell | act | n | Q | D | ND | Rnd | Rall | agree_F(Q) |
|---|---|---|---|---|---|---|---|---|
| C10-VGG-Tian-8 | W | 6 | 6/6 (1.00) | 6/6 (0.99) | 0/6 (0.01) | 44/48 (0.99) | 2/48 (0.01) | 0.979 |
| C10-VGG-Tian-8 | WA | 6 | 6/6 (1.00) | 6/6 (0.99) | 0/6 (0.01) | 45/48 (0.99) | 3/48 (0.01) | 0.977 |
| C100-R18-Tian-8 | W | 4 | 4/4 (0.99) | 3/4 (0.99) | 1/4 (0.06) | 27/32 (0.98) | 0/32 (0.01) | 0.958 |
| C100-R18-Tian-8 | WA | 4 | 4/4 (0.99) | 3/4 (0.98) | 1/4 (0.05) | 27/32 (0.98) | 0/32 (0.01) | 0.954 |
| GTSRB-R18-Tian-8 | W | 4 | 4/4 (1.00) | 1/4 (0.33) | 2/4 (0.73) | 18/32 (0.66) | 1/32 (0.00) | 0.992 |
| GTSRB-R18-Tian-8 | WA | 4 | 4/4 (1.00) | 1/4 (0.38) | 2/4 (0.73) | 19/32 (0.71) | 1/32 (0.00) | 0.991 |
| Ma-8 | W | 4 | 4/4 (1.00) | 0/4 (0.02) | 4/4 (1.00) | 0/32 (0.02) | 0/32 (0.02) | 0.969 |
| Ma-W4A4 | W | 4 | 4/4 (0.99) | 0/4 (0.01) | 3/4 (0.75) | 0/32 (0.01) | 0/32 (0.01) | 0.961 |
| QuANTI-4 | W | 4 | 4/4 (0.91) | 3/4 (0.72) | 1/4 (0.04) | 19/32 (0.64) | 6/32 (0.02) | 0.956 |
| QuANTI-8 | W | 4 | 4/4 (0.96) | 1/4 (0.05) | 2/4 (0.49) | 8/32 (0.05) | 3/32 (0.01) | 0.991 |
| QuEST-W4A4 | W | 4 | 4/4 (0.76) | 0/4 (0.04) | 4/4 (0.63) | 0/32 (0.06) | 0/32 (0.05) | 0.970 |
| R18-C10-Tian-8 | W | 12 | 12/12 (1.00) | 11/12 (0.99) | 1/12 (0.01) | 92/96 (0.99) | 0/96 (0.01) | 0.988 |
| R18-C10-Tian-8 | WA | 12 | 12/12 (1.00) | 11/12 (0.99) | 1/12 (0.01) | 92/96 (0.99) | 0/96 (0.01) | 0.989 |
| Tian-W4A4 | W | 4 | 4/4 (1.00) | 0/4 (0.13) | 4/4 (0.88) | 2/32 (0.16) | 0/32 (0.01) | 0.976 |

## Predictions

| prediction | result | holds |
|---|---|---|
| P1 Tian-8 Rnd draws on >= .75 | 0.87 | True |
| P1 Tian-8 Rall draws on <= .10 | 0.01 | True |
| P2 Tian-8 models on under D >= 3/4 | 0.81 | True |
| P2 Tian-8 models on under ND <= 1/4 | 0.15 | True |
| P3 max |ASR orig - recipe| <= .01 | 0.0000 over 64 draws | True |
| P4 Ma-8 Rnd draws on <= 1/4 | 0.00 | True |
| P4 Ma-W4A4 Rnd draws on <= 1/4 | 0.00 | True |
| P4 Tian-W4A4 Rnd draws on <= 1/4 | 0.06 | True |
| P4 QuEST-W4A4 Rnd draws on <= 1/4 | 0.00 | True |
| P4 Ma-8 D off (all models) | on 0/4 | True |
| P4 Ma-W4A4 D off (all models) | on 0/4 | True |

## Falsification criteria

| criterion | hit |
|---|---|
| 1. Tian-8 Rnd draws on < .5 | False |
| 2. Tian-8 D on <= 1/4 and ND on >= 3/4 | False |
| 3. some draw |ASR orig - recipe| > .05 | False |

## Determined decisions (share of weights)

| cell | det_share | zero_share | max_layer_det |
|---|---|---|---|
| C10-VGG-Tian-8 | 0.162 | 0.162 | 0.480 |
| C100-R18-Tian-8 | 0.173 | 0.173 | 0.444 |
| GTSRB-R18-Tian-8 | 0.148 | 0.148 | 0.341 |
| Ma-8 | 0.009 | 0.008 | 0.081 |
| Ma-W4A4 | 0.136 | 0.135 | 0.165 |
| QuANTI-4 | 0.720 | 0.720 | 0.957 |
| QuANTI-8 | 0.284 | 0.284 | 0.561 |
| QuEST-W4A4 | 0.124 | 0.123 | 0.171 |
| R18-C10-Tian-8 | 0.201 | 0.201 | 0.482 |
| Tian-W4A4 | 0.204 | 0.203 | 0.230 |

errors: 0
