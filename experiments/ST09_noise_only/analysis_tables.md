# ST09 tables (run `ST09_20261127_001`, QuEST 40, twins 22)

P0 (a) FNc t 2 vs ST08 FN t 2: |ASR diff| median 0.0000, max 0.0000 (n 40)
P0 (b) FNi t 0 (R-like): QuEST ASR median 0.024

## State x t: QuEST ASR median, ASR > .5 (held-out / seen), clean -> target median, QuEST acc, twins acc

| state | t | ASR | > .5 held-out | > .5 seen | clean -> target | QuEST acc | twins acc |
|---|---|---|---|---|---|---|---|
| FNc | 2 | 0.74 | 16/19 | 20/21 | 0.048 | 0.75 | 0.76 |
| FNi | 0 | 0.02 | 0/19 | 0/21 | 0.002 | 0.92 | 0.92 |
| FNi | 1 | 0.04 | 0/19 | 0/21 | 0.012 | 0.87 | 0.86 |
| FNi | 2 | 0.15 | 0/19 | 0/21 | 0.052 | 0.56 | 0.54 |
| FNi | 3 | 0.28 | 4/19 | 5/21 | 0.184 | 0.18 | 0.20 |
| QNi | 0 | 0.47 | 11/19 | 7/21 | 0.003 | 0.92 | 0.91 |
| QNi | 1 | 0.42 | 6/19 | 5/21 | 0.012 | 0.86 | 0.87 |
| QNi | 2 | 0.51 | 8/19 | 14/21 | 0.063 | 0.56 | 0.53 |
| QNi | 3 | 0.49 | 7/19 | 12/21 | 0.248 | 0.18 | 0.19 |

P1 FNi t 2: ASR > .5 0.00 (0/40), clean -> target median 0.052; paired FNi - FNc median -0.555; FNc ASR median 0.74, FNi 0.15
P2 FNi t 3: ASR > .5 0.23 (9/40)
P3 QNi t 0 (W4 weights, FP activations): ASR > .5 0.45 (18/40), median 0.47
P4 t 1: QNi ASR > FNi ASR 1.00; medians QNi 0.42, FNi 0.04
matched damage: twins acc FNc t 2 0.76; closest FNi t 1 (twins acc 0.86): ASR median 0.04, > .5 0/40 vs FNc 36/40

FNi t 2 ASR median by cell: {'CIFAR-10/ResNet18': 0.26, 'CIFAR-10/VGG16-BN': 0.06, 'CIFAR-100/ResNet18': 0.1, 'CIFAR-100/VGG16-BN': 0.05, 'GTSRB/ResNet18': 0.17, 'GTSRB/VGG16-BN': 0.14}
FNc t 2 ASR median by cell: {'CIFAR-10/ResNet18': 0.73, 'CIFAR-10/VGG16-BN': 0.73, 'CIFAR-100/ResNet18': 0.69, 'CIFAR-100/VGG16-BN': 0.36, 'GTSRB/ResNet18': 0.94, 'GTSRB/VGG16-BN': 0.93}
