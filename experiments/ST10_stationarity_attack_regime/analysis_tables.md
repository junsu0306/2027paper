# ST10 tables (run `ST10_20261127_001`, 113 models)

P0 (a) unaware, train: gF_rel median 0.0172 vs gQ_rel 0.0502
P0 (b) QuEST deployed ASR median 0.967 (n 40)

## data = train
| kind | n | gF_rel | gQ_rel | B_all | B_mask | B_fit_mask | alpha_fit | cos(gF,gQ) mask |
|---|---|---|---|---|---|---|---|---|
| QuEST-W4A4 | 40 | 0.0014 | 0.0133 | 0.914 | 0.928 | 0.826 | -0.01 | 0.187 |
| control-QuEST-W4A4 | 6 | 0.0006 | 0.0084 | 0.919 | 0.942 | 0.904 | -0.01 | 0.098 |
| QAT-W4A4-IAO | 16 | 0.0190 | 0.0281 | 0.924 | 0.900 | 0.511 | -0.35 | 0.584 |
| unaware | 51 | 0.0172 | 0.0502 | 0.916 | 0.912 | 0.619 | -0.14 | 0.446 |
B_mask AUROC: QuEST > control 0.49, QuEST > QAT 0.87, QuEST > unaware 0.70; unaware > control 0.29
per stage B_mask medians (QuEST / control / QAT / unaware): block1 0.94/0.94/0.85/0.88; block2 0.94/0.94/0.86/0.87; block3 0.94/0.94/0.85/0.86; block4 0.94/0.94/0.86/0.88; block5 0.96/0.96/0.89/0.91; head 0.93/0.92/0.88/0.92; layer1 0.92/0.90/0.90/0.90; layer2 0.92/0.90/0.91/0.91; layer3 0.93/0.90/0.91/0.92; layer4 0.93/0.92/0.90/0.93; stem 0.84/0.81/0.92/0.93
s_delta: QuEST argmax == target 0.10 (4/40; chance 0.066), target rank median 9, s[target] > 0 0.75; max score AUROC vs twins 0.74, vs all normals 0.54
   held-out: argmax == target 4/19, rank median 7
   seen: argmax == target 0/21, rank median 10
   by dataset: {'CIFAR-10': {'sum': 3, 'count': 24}, 'CIFAR-100': {'sum': 0, 'count': 9}, 'GTSRB': {'sum': 1, 'count': 7}}
s_fit: QuEST argmax == target 0.05 (2/40; chance 0.066), target rank median 10, s[target] > 0 0.00; max score AUROC vs twins 0.55, vs all normals 0.48
   held-out: argmax == target 2/19, rank median 9
   seen: argmax == target 0/21, rank median 10
   by dataset: {'CIFAR-10': {'sum': 2, 'count': 24}, 'CIFAR-100': {'sum': 0, 'count': 9}, 'GTSRB': {'sum': 0, 'count': 7}}
s_q: QuEST argmax == target 0.47 (19/40; chance 0.066), target rank median 2, s[target] > 0 0.95; max score AUROC vs twins 0.31, vs all normals 0.17
   held-out: argmax == target 8/19, rank median 3
   seen: argmax == target 11/21, rank median 1
   by dataset: {'CIFAR-10': {'sum': 15, 'count': 24}, 'CIFAR-100': {'sum': 4, 'count': 9}, 'GTSRB': {'sum': 0, 'count': 7}}

## data = defender
| kind | n | gF_rel | gQ_rel | B_all | B_mask | B_fit_mask | alpha_fit | cos(gF,gQ) mask |
|---|---|---|---|---|---|---|---|---|
| QuEST-W4A4 | 40 | 0.0723 | 0.0792 | 0.876 | 0.880 | 0.543 | -0.44 | 0.544 |
| control-QuEST-W4A4 | 6 | 0.0837 | 0.0851 | 0.873 | 0.873 | 0.558 | -0.45 | 0.525 |
| QAT-W4A4-IAO | 16 | 0.0747 | 0.0778 | 0.909 | 0.905 | 0.468 | -0.53 | 0.640 |
| unaware | 51 | 0.0674 | 0.0862 | 0.871 | 0.865 | 0.585 | -0.31 | 0.489 |
B_mask AUROC: QuEST > control 0.55, QuEST > QAT 0.23, QuEST > unaware 0.66; unaware > control 0.33
per stage B_mask medians (QuEST / control / QAT / unaware): block1 0.79/0.80/0.83/0.81; block2 0.81/0.81/0.83/0.80; block3 0.84/0.84/0.85/0.82; block4 0.87/0.86/0.88/0.85; block5 0.91/0.90/0.92/0.90; head 0.91/0.92/0.91/0.91; layer1 0.83/0.85/0.88/0.85; layer2 0.86/0.88/0.89/0.86; layer3 0.89/0.89/0.91/0.88; layer4 0.91/0.92/0.93/0.90; stem 0.87/0.89/0.93/0.86
s_delta: QuEST argmax == target 0.00 (0/40; chance 0.066), target rank median 9, s[target] > 0 0.38; max score AUROC vs twins 0.62, vs all normals 0.49
   held-out: argmax == target 0/19, rank median 7
   seen: argmax == target 0/21, rank median 10
   by dataset: {'CIFAR-10': {'sum': 0, 'count': 24}, 'CIFAR-100': {'sum': 0, 'count': 9}, 'GTSRB': {'sum': 0, 'count': 7}}
s_fit: QuEST argmax == target 0.03 (1/40; chance 0.066), target rank median 9, s[target] > 0 0.72; max score AUROC vs twins 0.65, vs all normals 0.74
   held-out: argmax == target 1/19, rank median 9
   seen: argmax == target 0/21, rank median 9
   by dataset: {'CIFAR-10': {'sum': 1, 'count': 24}, 'CIFAR-100': {'sum': 0, 'count': 9}, 'GTSRB': {'sum': 0, 'count': 7}}
s_q: QuEST argmax == target 0.30 (12/40; chance 0.066), target rank median 3, s[target] > 0 0.57; max score AUROC vs twins 0.19, vs all normals 0.11
   held-out: argmax == target 6/19, rank median 3
   seen: argmax == target 6/21, rank median 3
   by dataset: {'CIFAR-10': {'sum': 8, 'count': 24}, 'CIFAR-100': {'sum': 4, 'count': 9}, 'GTSRB': {'sum': 0, 'count': 7}}

## data = small
| kind | n | gF_rel | gQ_rel | B_all | B_mask | B_fit_mask | alpha_fit | cos(gF,gQ) mask |
|---|---|---|---|---|---|---|---|---|
| QuEST-W4A4 | 40 | 0.2830 | 0.3006 | 0.898 | 0.910 | 0.477 | -0.40 | 0.629 |
| control-QuEST-W4A4 | 6 | 0.1990 | 0.1758 | 0.867 | 0.885 | 0.540 | -0.48 | 0.548 |
| QAT-W4A4-IAO | 16 | 0.2134 | 0.2208 | 0.914 | 0.926 | 0.417 | -0.49 | 0.704 |
| unaware | 51 | 0.1425 | 0.1613 | 0.872 | 0.872 | 0.581 | -0.27 | 0.495 |
B_mask AUROC: QuEST > control 0.67, QuEST > QAT 0.23, QuEST > unaware 0.85; unaware > control 0.30
per stage B_mask medians (QuEST / control / QAT / unaware): block1 0.81/0.81/0.81/0.82; block2 0.83/0.82/0.83/0.81; block3 0.85/0.86/0.85/0.82; block4 0.88/0.89/0.90/0.86; block5 0.93/0.93/0.94/0.89; head 0.92/0.91/0.92/0.91; layer1 0.85/0.84/0.88/0.84; layer2 0.89/0.87/0.90/0.85; layer3 0.91/0.90/0.93/0.89; layer4 0.95/0.92/0.96/0.91; stem 0.89/0.88/0.90/0.89
s_delta: QuEST argmax == target 0.05 (2/40; chance 0.066), target rank median 8, s[target] > 0 0.17; max score AUROC vs twins 0.51, vs all normals 0.48
   held-out: argmax == target 1/19, rank median 8
   seen: argmax == target 1/21, rank median 9
   by dataset: {'CIFAR-10': {'sum': 2, 'count': 24}, 'CIFAR-100': {'sum': 0, 'count': 9}, 'GTSRB': {'sum': 0, 'count': 7}}
s_fit: QuEST argmax == target 0.05 (2/40; chance 0.066), target rank median 9, s[target] > 0 1.00; max score AUROC vs twins 0.56, vs all normals 0.71
   held-out: argmax == target 1/19, rank median 9
   seen: argmax == target 1/21, rank median 9
   by dataset: {'CIFAR-10': {'sum': 1, 'count': 24}, 'CIFAR-100': {'sum': 0, 'count': 9}, 'GTSRB': {'sum': 1, 'count': 7}}
s_q: QuEST argmax == target 0.17 (7/40; chance 0.066), target rank median 5, s[target] > 0 0.28; max score AUROC vs twins 0.35, vs all normals 0.21
   held-out: argmax == target 5/19, rank median 5
   seen: argmax == target 2/21, rank median 6
   by dataset: {'CIFAR-10': {'sum': 4, 'count': 24}, 'CIFAR-100': {'sum': 2, 'count': 9}, 'GTSRB': {'sum': 1, 'count': 7}}

