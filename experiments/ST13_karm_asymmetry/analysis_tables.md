# ST13 tables (run `ST13_20261127_001`, 331 models: QCB 168, normals+controls 163, aware 72)

censored pair share, median (normals / QCB): F 0.01/0.00; Bfp 0.01/0.00; C1 0.01/0.00; N 0.01/0.01

P1 natural strong class (ST12 MM-BD top in F) in F: A within median + 1 MAD of other classes 0.63 (n 163); z median 0.41; by dataset {'CIFAR-10': 0.8, 'CIFAR-100': 0.19, 'GTSRB': 0.68}
   (desc.) easiest-to-reach class in F (min forward cost) == MM-BD top class: 0.08
P2 switched-on QCB (n 143): dA(target) > 0 0.83; cstar == target 0.43
   Ma-8: dA > 0 18/21, cstar == target 4/21, A_target in F median -0.26
   Tian-8: dA > 0 30/31, cstar == target 27/31, A_target in F median -0.12
   Tian-W4A4: dA > 0 18/22, cstar == target 15/22, A_target in F median -0.19
   QuANTI-8+4: dA > 0 6/6, cstar == target 5/6, A_target in F median 0.25
   QuEST-8: dA > 0 18/28, cstar == target 7/28, A_target in F median 0.15
   QuEST-W4A4: dA > 0 29/35, cstar == target 3/35, A_target in F median 0.05

## AUROC (QCB higher)

| score | all QCB vs normals+controls | Ma-8 | Ma-W4A4 | Tian-8 | Tian-W4A4 | QuANTI-8+4 | QuEST-8 | QuEST-W4A4 | 4-bit vs aware |
|---|---|---|---|---|---|---|---|---|---|
| Z_A | 0.57 | 0.54 | 0.54 | 0.85 | 0.47 | 0.32 | 0.61 | 0.52 | 0.42 |
| Z_E | 0.60 | 0.54 | 0.56 | 0.85 | 0.58 | 0.44 | 0.55 | 0.57 | 0.51 |
| zA1_F | 0.46 | 0.45 | 0.69 | 0.45 | 0.41 | 0.65 | 0.34 | 0.44 | 0.52 |
| zA1_Bfp | 0.54 | 0.54 | 0.53 | 0.75 | 0.59 | 0.56 | 0.38 | 0.43 | 0.57 |
| zA1_C1 | 0.44 | 0.48 | 0.65 | 0.44 | 0.33 | 0.61 | 0.32 | 0.44 | 0.50 |
| zA1_N | 0.56 | 0.47 | 0.48 | 0.59 | 0.52 | 0.45 | 0.66 | 0.61 | 0.52 |
| Z_mmbd | 0.64 | 0.62 | 0.49 | 0.89 | 0.85 | 0.35 | 0.58 | 0.54 | 0.58 |

QuEST W4A4 vs aware (Z_A): 0.48, held-out 0.56; ST12 Z_mmbd same models: 0.54
Alarm Z_A > 3.5: normals+controls 0.39 (64/163); QCB by family: Ma-8 10/22, Ma-W4A4 5/8, Tian-8 27/32, Tian-W4A4 9/23, QuANTI-8+4 4/18, QuEST-8 17/29, QuEST-W4A4 15/36
normals' Z_A median by dataset: {'CIFAR-10': 2.8, 'CIFAR-100': 3.26, 'GTSRB': 3.52}, max/min 1.26

