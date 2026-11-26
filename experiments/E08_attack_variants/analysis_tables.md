# E08 tables (44/44 jobs done, 0 errors)

## E05: switch component (ASR; switch = components that alone turn it on)

| variant | seed | recipe | ASR F | Qw | F+A full | F+clip | F+round | F+BN | Q full | switch |
|---|---|---|---|---|---|---|---|---|---|---|
| QuEST-aonly | s1 | iao8 | 0.023 | 0.022 | 0.980 | 0.980 | 0.021 | 0.003 | 0.969 | A(clip) |
| QuEST-aonly | s2 | iao8 | 0.033 | 0.033 | 0.991 | 0.991 | 0.035 | 0.001 | 0.984 | A(clip) |
| QuEST-aonly | s3 | iao8 | 0.022 | 0.021 | 0.988 | 0.988 | 0.020 | 0.762 | 0.988 | A(clip)+BN |
| QuEST-orig | s1 | iao8 | 0.035 | 0.036 | 0.990 | 0.990 | 0.034 | 0.002 | 0.980 | A(clip) |
| QuEST-orig | s3 | iao8 | 0.041 | 0.040 | 0.988 | 0.988 | 0.042 | 0.555 | 0.988 | A(clip)+BN |
| Tian-actquant | s1 | qnnpack8 | 0.009 | 1.000 | 0.009 | 0.009 | 0.010 | nan | 1.000 | W |
| Tian-actquant | s2 | qnnpack8 | 0.003 | 0.994 | 0.005 | 0.003 | 0.005 | nan | 0.994 | W |
| Tian-alllayers | s1 | qnnpack8 | 0.006 | 0.997 | 0.006 | 0.006 | 0.006 | nan | 0.997 | W |
| Tian-alllayers | s2 | qnnpack8 | 0.006 | 0.999 | 0.006 | 0.006 | 0.006 | nan | 0.999 | W |
| Tian-orig | s1 | qnnpack8 | 0.004 | 0.999 | 0.004 | 0.004 | 0.004 | nan | 0.999 | W |
| Tian-orig | s2 | qnnpack8 | 0.003 | 0.995 | 0.003 | 0.003 | 0.003 | nan | 0.995 | W |

## E06: localization (rows where the component is on)

| variant | seed | component | W | on (one unit removed, max) | n suff | sufficient | necessary | suff groups | nec groups | BN-confounded |
|---|---|---|---|---|---|---|---|---|---|---|
| QuEST-aonly | s1 | A_clip | F | 0.97 | 0 | - | - | - | layer3,layer4 | False |
| QuEST-aonly | s1 | A_clip | Q | 0.97 | 0 | - | - | - | layer3,layer4 | False |
| QuEST-aonly | s2 | A_clip | F | 0.98 | 1 | layer3.1.conv2 | layer3.1.conv2 | layer3 | layer3 | False |
| QuEST-aonly | s2 | A_clip | Q | 0.98 | 1 | layer3.1.conv2 | layer3.1.conv2 | layer3 | layer3 | False |
| QuEST-aonly | s3 | A_clip | F | 0.99 | 19 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | True |
| QuEST-aonly | s3 | A_clip | Q | 0.99 | 19 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | True |
| QuEST-aonly | s3 | BN | F | 0.92 | 0 | - | layer3.1.bn2, layer4.0.shortcut.1 | layer4 | layer4 | False |
| QuEST-aonly | s3 | BN | Q | 0.92 | 0 | - | layer3.1.bn2, layer4.0.shortcut.1 | layer4 | layer4 | False |
| QuEST-orig | s1 | A_clip | F | 0.98 | 0 | - | layer3.1.conv1 | - | layer3,layer4 | False |
| QuEST-orig | s1 | A_clip | Q | 0.98 | 0 | - | layer3.1.conv1 | - | layer3,layer4 | False |
| QuEST-orig | s3 | A_clip | F | 0.99 | 17 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | True |
| QuEST-orig | s3 | A_clip | Q | 0.99 | 17 | conv1, layer1.0.conv1, layer1.0.conv2, layer1.1.conv1, layer1.1.conv2, layer2.0.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | True |
| QuEST-orig | s3 | BN | F | 0.62 | 0 | - | layer3.1.bn2, layer4.0.bn1, layer4.0.shortcut.1 | layer4 | layer4 | False |
| QuEST-orig | s3 | BN | Q | 0.62 | 0 | - | layer3.1.bn2, layer4.0.bn1, layer4.0.shortcut.1 | layer4 | layer4 | False |
| Tian-actquant | s1 | A_clip | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-actquant | s1 | W | F/Q | 1.00 | 1 | blocks.7.conv2 | blocks.7.conv2 | layer4 | layer4 | False |
| Tian-actquant | s2 | A_clip | Q | 0.99 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-actquant | s2 | W | F/Q | 1.00 | 1 | blocks.6.shortcut | blocks.6.shortcut | layer4 | layer4 | False |
| Tian-alllayers | s1 | A_clip | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-alllayers | s1 | W | F/Q | 1.00 | 1 | blocks.1.conv1 | blocks.1.conv1 | layer1 | layer1 | False |
| Tian-alllayers | s2 | A_clip | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-alllayers | s2 | W | F/Q | 1.00 | 1 | blocks.7.conv1 | blocks.7.conv1 | layer4 | layer4 | False |
| Tian-orig | s1 | A_clip | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-orig | s1 | W | F/Q | 1.00 | 1 | blocks.7.conv1 | blocks.7.conv1 | layer4 | layer4 | False |
| Tian-orig | s2 | A_clip | Q | 1.00 | 21 | conv1, blocks.0.conv1, blocks.0.conv2, blocks.1.conv1, blocks.1.conv2, blocks.2.conv1 | - | stem,layer1,layer2,layer3,layer4,head | - | False |
| Tian-orig | s2 | W | F/Q | 1.00 | 1 | blocks.7.conv1 | blocks.7.conv1 | layer4 | layer4 | False |

## E06 a: ASR against clip alpha and rounding bits

| variant | seed | W | off | clip:8 | clip:4 | clip:2 | clip:1.5 | clip:1.25 | clip:1 | clip:0.8 | round:16 | round:8 | round:6 | round:4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QuEST-aonly | s1 | F | 0.00 | 0.00 | 0.01 | 0.85 | 0.94 | 0.96 | 0.97 | 0.95 | 0.00 | 0.00 | 0.01 | 0.01 |
| QuEST-aonly | s1 | Q | 0.00 | 0.00 | 0.01 | 0.85 | 0.94 | 0.96 | 0.97 | 0.95 | 0.00 | 0.00 | 0.00 | 0.01 |
| QuEST-aonly | s2 | F | 0.00 | 0.00 | 0.00 | 0.87 | 0.96 | 0.97 | 0.98 | 0.97 | 0.00 | 0.00 | 0.00 | 0.00 |
| QuEST-aonly | s2 | Q | 0.00 | 0.00 | 0.00 | 0.87 | 0.96 | 0.97 | 0.98 | 0.97 | 0.00 | 0.00 | 0.00 | 0.00 |
| QuEST-aonly | s3 | F | 0.76 | 0.76 | 0.94 | 0.98 | 0.98 | 0.99 | 0.99 | 0.98 | 0.76 | 0.76 | 0.77 | 0.84 |
| QuEST-aonly | s3 | Q | 0.75 | 0.75 | 0.94 | 0.98 | 0.98 | 0.99 | 0.99 | 0.98 | 0.75 | 0.75 | 0.77 | 0.82 |
| QuEST-orig | s1 | F | 0.00 | 0.00 | 0.02 | 0.93 | 0.96 | 0.97 | 0.98 | 0.96 | 0.00 | 0.00 | 0.00 | 0.00 |
| QuEST-orig | s1 | Q | 0.00 | 0.00 | 0.02 | 0.93 | 0.96 | 0.97 | 0.98 | 0.96 | 0.00 | 0.00 | 0.00 | 0.00 |
| QuEST-orig | s3 | F | 0.55 | 0.55 | 0.93 | 0.98 | 0.98 | 0.98 | 0.99 | 0.98 | 0.55 | 0.56 | 0.60 | 0.77 |
| QuEST-orig | s3 | Q | 0.56 | 0.56 | 0.93 | 0.98 | 0.98 | 0.99 | 0.99 | 0.98 | 0.56 | 0.56 | 0.59 | 0.76 |
| Tian-actquant | s1 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 |
| Tian-actquant | s1 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| Tian-actquant | s2 | F | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Tian-actquant | s2 | Q | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.57 | 0.99 | 0.99 | 1.00 | 0.99 |
| Tian-alllayers | s1 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.00 |
| Tian-alllayers | s1 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Tian-alllayers | s2 | F | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 1.00 |
| Tian-alllayers | s2 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 |
| Tian-orig | s1 | F | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.04 |
| Tian-orig | s1 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| Tian-orig | s2 | F | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Tian-orig | s2 | Q | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 |

## E07: NC, MM-BD, SPD by state: alarm with the right class (class = target regardless of alarm)

| variant | state | on | ASR med | nc hit (top=target) | mmbd hit (top=target) | spd hit (top=target) |
|---|---|---|---|---|---|---|
| QuEST-aonly | F | 0/3 | 0.02 | 2/3 (2/3) | 0/3 (1/3) | 0/3 (0/3) |
| QuEST-aonly | Qw | 0/3 | 0.02 | 1/3 (3/3) | 0/3 (1/3) | 0/3 (0/3) |
| QuEST-aonly | Fclip | 3/3 | 0.98 | 3/3 (3/3) | 0/3 (0/3) | 0/3 (0/3) |
| QuEST-aonly | Fbn | 1/3 | 0.00 | 0/3 (1/3) | 1/3 (2/3) | 0/3 (0/3) |
| QuEST-aonly | Q | 3/3 | 0.98 | 1/3 (3/3) | 0/3 (1/3) | 0/3 (1/3) |
| QuEST-orig | F | 0/2 | 0.04 | 2/2 (2/2) | 1/2 (1/2) | 0/2 (0/2) |
| QuEST-orig | Qw | 0/2 | 0.04 | 1/2 (2/2) | 0/2 (1/2) | 0/2 (0/2) |
| QuEST-orig | Fclip | 2/2 | 0.98 | 1/2 (2/2) | 0/2 (1/2) | 0/2 (0/2) |
| QuEST-orig | Fbn | 1/2 | 0.28 | 1/2 (1/2) | 1/2 (1/2) | 0/2 (0/2) |
| QuEST-orig | Q | 2/2 | 0.98 | 2/2 (2/2) | 0/2 (0/2) | 0/2 (0/2) |
| Tian-actquant | F | 0/2 | 0.01 | 0/2 (1/2) | 2/2 (2/2) | 2/2 (2/2) |
| Tian-actquant | Qw | 2/2 | 1.00 | 2/2 (2/2) | 2/2 (2/2) | 1/2 (2/2) |
| Tian-actquant | Fclip | 0/2 | 0.01 | 0/2 (1/2) | 2/2 (2/2) | 2/2 (2/2) |
| Tian-actquant | Q | 2/2 | 1.00 | 1/2 (1/2) | 2/2 (2/2) | 1/2 (2/2) |
| Tian-alllayers | F | 0/2 | 0.01 | 1/2 (2/2) | 1/2 (1/2) | 2/2 (2/2) |
| Tian-alllayers | Qw | 2/2 | 1.00 | 1/2 (1/2) | 1/2 (1/2) | 2/2 (2/2) |
| Tian-alllayers | Fclip | 0/2 | 0.01 | 1/2 (1/2) | 1/2 (1/2) | 2/2 (2/2) |
| Tian-alllayers | Q | 2/2 | 1.00 | 1/2 (2/2) | 1/2 (2/2) | 2/2 (2/2) |
| Tian-orig | F | 0/2 | 0.00 | 0/2 (0/2) | 1/2 (2/2) | 2/2 (2/2) |
| Tian-orig | Qw | 2/2 | 1.00 | 1/2 (1/2) | 1/2 (2/2) | 1/2 (2/2) |
| Tian-orig | Fclip | 0/2 | 0.00 | 0/2 (1/2) | 1/2 (1/2) | 2/2 (2/2) |
| Tian-orig | Q | 2/2 | 1.00 | 1/2 (1/2) | 1/2 (2/2) | 1/2 (2/2) |

## E04: S1-S4 fires against the batch-1 thresholds (class hits)

| variant | n | Q ASR med | S1 | S1 med | S2 | S2 med | S3 | S3 med | S4_flip | S4_flip med | S4_pgd | S4_pgd med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QuEST-aonly | 3 | 0.984 | 0/3 (cls 0) | 0.013 | 0/3 (cls 0) | 0.108 | 0/3 | 1.091 | 2/3 (cls 2) | 7.000 | 0/3 (cls 0) | 6.000 |
| QuEST-orig | 2 | 0.984 | 0/2 (cls 0) | 0.011 | 0/2 (cls 0) | 0.109 | 1/2 | 1.163 | 0/2 (cls 0) | 8.000 | 0/2 (cls 0) | 6.000 |
| Tian-actquant | 2 | 0.997 | 2/2 (cls 2) | 0.179 | 1/2 (cls 0) | 0.113 | 2/2 | 4.174 | 2/2 (cls 1) | 3.500 | 2/2 (cls 2) | 3.500 |
| Tian-alllayers | 2 | 0.998 | 2/2 (cls 2) | 0.164 | 0/2 (cls 0) | 0.109 | 2/2 | 1.531 | 2/2 (cls 0) | 4.500 | 2/2 (cls 1) | 4.500 |
| Tian-orig | 2 | 0.997 | 2/2 (cls 2) | 0.204 | 0/2 (cls 0) | 0.111 | 2/2 | 4.005 | 2/2 (cls 2) | 3.000 | 2/2 (cls 2) | 3.000 |

## Predictions (delivered variants)

| prediction | result | holds |
|---|---|---|
| QuEST orig: switch includes A(clip) (as batch 1) | {'s1': 'A(clip)', 's3': 'A(clip)+BN'} | True |
| QuEST aonly: F + clip >= .5 and Qw < .5 | {'s1': (0.98, 0.022), 's2': (0.991, 0.033), 's3': (0.988, 0.021)} | True |
| Tian actquant: (Qw < .5 and W + A >= .5) or F + clip >= .5 in >= n/2 | {'s1': (1.0, 1.0, 0.009), 's2': (0.994, 0.994, 0.003)} | False |
| Tian alllayers: a sufficient or necessary W unit / group in layer1-3 in >= n/2 | {'s1': ('layer1', 'layer1'), 's2': ('layer4', 'layer4')} | True |

## Falsification criteria

| criterion | value | hit |
|---|---|---|
| J1' 2. aonly valid and weights alone turn on >= 2 models | Qw >= .5 in 0/3 | False |
| J1' 3. actquant and alllayers both: >= 3/4 of models with the same component and group as orig | {'Tian-actquant': (2, 2), 'Tian-alllayers': (1, 2)} | False |
| J1' 1 / 4, H 1 / 2 (wonly, bnfrozen, weakpoison, nolquant) | not delivered | False |

errors: 0
