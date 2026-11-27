# ST12 tables (run `ST12_20261127_001`, 331 models: QCB 168, normals+controls 163, aware 72)

## Mechanism: ASR > .5 by family and state (true trigger)

| family | n | F | Bfp | C1 | N | any on-state |
|---|---|---|---|---|---|---|
| Ma-8 | 22 | 0 | 20 | 0 | 19 | 21 |
| Ma-W4A4 | 8 | 0 | 0 | 0 | 0 | 0 |
| Tian-8 | 32 | 0 | 28 | 0 | 30 | 31 |
| Tian-W4A4 | 23 | 0 | 20 | 0 | 20 | 22 |
| QuANTI-8+4 | 18 | 0 | 3 | 0 | 6 | 6 |
| QuEST-8 | 29 | 0 | 7 | 19 | 14 | 28 |
| QuEST-W4A4 | 36 | 0 | 0 | 0 | 35 | 35 |

clean accuracy medians (normals+controls / QCB): F 0.924/0.922; Bfp 0.913/0.904; C1 0.924/0.922; N 0.837/0.762

## AUROC (QCB higher) by score

| score | all QCB vs normals+controls | Ma-8 | Ma-W4A4 | Tian-8 | Tian-W4A4 | QuANTI-8+4 | QuEST-8 | QuEST-W4A4 | 4-bit fams vs aware |
|---|---|---|---|---|---|---|---|---|---|
| native_F | 0.61 | 0.50 | 0.51 | 0.71 | 0.82 | 0.77 | 0.52 | 0.46 | 0.69 |
| native_on | 0.71 | 0.66 | 0.47 | 0.98 | 0.99 | 0.59 | 0.59 | 0.52 | 0.70 |
| native_Bfp | 0.68 | 0.73 | 0.47 | 0.97 | 0.99 | 0.33 | 0.62 | 0.47 | 0.64 |
| native_C1 | 0.62 | 0.51 | 0.41 | 0.76 | 0.84 | 0.65 | 0.51 | 0.53 | 0.70 |
| native_N | 0.67 | 0.62 | 0.60 | 0.96 | 0.99 | 0.51 | 0.43 | 0.54 | 0.68 |
| Z | 0.64 | 0.62 | 0.49 | 0.89 | 0.85 | 0.35 | 0.58 | 0.54 | 0.58 |
| Z_reg | 0.55 | 0.54 | 0.63 | 0.58 | 0.59 | 0.45 | 0.54 | 0.56 | 0.56 |
| Z_speed | 0.47 | 0.35 | 0.64 | 0.57 | 0.48 | 0.32 | 0.41 | 0.51 | 0.48 |

QuEST W4A4 vs aware normals (Z): 0.54; held-out 0.45; by z_N only: 0.52

Target id (switched-on QCB, n 143): cstar == target 0.56; native MM-BD on F cand == target 0.45 (all QCB); best native on-state cand == target 0.68
   Ma-8: cstar == target 19/21
   Tian-8: cstar == target 28/31
   Tian-W4A4: cstar == target 20/22
   QuANTI-8+4: cstar == target 0/6
   QuEST-8: cstar == target 8/28
   QuEST-W4A4: cstar == target 5/35

Alarm Z > 3.5: normals+controls 0.51 (83/163), aware 0.51; QCB by family: Ma-8 14/22, Ma-W4A4 5/8, Tian-8 30/32, Tian-W4A4 23/23, QuANTI-8+4 6/18, QuEST-8 17/29, QuEST-W4A4 19/36
(optimistic) threshold = normals' 95th pct 19.57: QCB detected 0.15, with target 0.11
normals' Z median by dataset: {'CIFAR-10': 2.39, 'CIFAR-100': 8.12, 'GTSRB': 4.02}, max/min 3.39

## Strong classes (normals+controls): share of the most common class per dataset

CIFAR-10 (n 86): native F cand mode share 0.23 (class 2); differential cstar mode share 0.33 (class 5); state of max z {'N': 37, 'Bfp': 27, 'C1': 22}
CIFAR-100 (n 37): native F cand mode share 0.73 (class 14); differential cstar mode share 0.08 (class 21); state of max z {'N': 18, 'C1': 14, 'Bfp': 5}
GTSRB (n 40): native F cand mode share 0.23 (class 40); differential cstar mode share 0.10 (class 28); state of max z {'Bfp': 21, 'C1': 13, 'N': 6}
