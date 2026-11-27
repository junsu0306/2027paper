# ST08 tables (run `ST08_20261127_001`, 113 models)

P0 QuEST W4A4 QA t 1 true: ASR median 0.970 (n 40), acc 0.898

## 2 x 2 by noise size: drift class = target (QuEST), QuEST ASR, same drift class QA vs FN (normals / QuEST)

t 1: QA: tgt 6/40, ASR 0.88, acc(normals) 0.85 | QN: tgt 6/40, ASR 0.85, acc(normals) 0.85 | FA: tgt 6/40, ASR 0.27, acc(normals) 0.85 | FN: tgt 5/40, ASR 0.21, acc(normals) 0.85
      same class QA vs FN: normals 0.66, twins 0.77, QuEST 0.72 | median 1 - cos: normals 0.09, QuEST 0.09
t 1.5: QA: tgt 15/40, ASR 0.93, acc(normals) 0.78 | QN: tgt 13/40, ASR 0.92, acc(normals) 0.78 | FA: tgt 9/40, ASR 0.65, acc(normals) 0.78 | FN: tgt 9/40, ASR 0.54, acc(normals) 0.78
      same class QA vs FN: normals 0.75, twins 0.91, QuEST 0.72 | median 1 - cos: normals 0.08, QuEST 0.07
t 2: QA: tgt 25/40, ASR 0.94, acc(normals) 0.66 | QN: tgt 21/40, ASR 0.92, acc(normals) 0.67 | FA: tgt 23/40, ASR 0.83, acc(normals) 0.67 | FN: tgt 22/40, ASR 0.74, acc(normals) 0.66
      same class QA vs FN: normals 0.63, twins 0.68, QuEST 0.80 | median 1 - cos: normals 0.06, QuEST 0.04
t 3: QA: tgt 29/40, ASR 0.92, acc(normals) 0.43 | QN: tgt 28/40, ASR 0.89, acc(normals) 0.41 | FA: tgt 27/40, ASR 0.87, acc(normals) 0.42 | FN: tgt 26/40, ASR 0.80, acc(normals) 0.42
      same class QA vs FN: normals 0.70, twins 0.82, QuEST 0.93 | median 1 - cos: normals 0.04, QuEST 0.02

P1 normals at t 2: c_QA == c_FN 0.63; median cos(v_QA, v_FN) 0.94 (n 73)
P2 QuEST at t 2: FN drift class == target 0.55, FN ASR median 0.74; QA drift class == target 0.62

sinks (normal mode of c_QA at t 2): {'CIFAR-10/ResNet18': 3, 'CIFAR-10/VGG': 3, 'CIFAR-100/ResNet18': 22, 'CIFAR-100/VGG': 13, 'GTSRB/ResNet18': 37, 'GTSRB/VGG': 20}

## Main score S_2 = max_c [v_QA(c) - v_FN(c)] (pooled over datasets) and secondary scores

### S_2
held-out: QuEST n19 med 0.0672 | twin n13 med 0.256 | normals n53 med 0.256 | AUROC vs twin 0.29, vs all normals 0.27 | QuEST c == target 0/19
seen: QuEST n21 med 0.093 | twin n9 med 0.078 | normals n20 med 0.157 | AUROC vs twin 0.58, vs all normals 0.43 | QuEST c == target 3/21
normals: c == sink 0.15 (n 73)
normals' median by dataset: {'CIFAR-10': 0.064, 'CIFAR-100': 0.45, 'GTSRB': 0.242}, max/min 7.03
   CIFAR-10/ResNet18: QuEST n18 med 0.0496 vs normals n15 med 0.0641, AUROC 0.35, c == target 3/18
   CIFAR-10/VGG: QuEST n6 med 0.0615 vs normals n9 med 0.0638, AUROC 0.43, c == target 0/6
   CIFAR-100/ResNet18: QuEST n6 med 0.41 vs normals n19 med 0.371, AUROC 0.39, c == target 0/6
   CIFAR-100/VGG: QuEST n3 med 0.432 vs normals n5 med 0.479, AUROC 0.33, c == target 0/3
   GTSRB/ResNet18: QuEST n4 med 0.159 vs normals n20 med 0.199, AUROC 0.39, c == target 0/4
   GTSRB/VGG: QuEST n3 med 0.254 vs normals n5 med 0.441, AUROC 0.13, c == target 0/3

### S_1.5
held-out: QuEST n19 med 0.0523 | twin n13 med 0.224 | normals n53 med 0.237 | AUROC vs twin 0.34, vs all normals 0.26 | QuEST c == target 2/19
seen: QuEST n21 med 0.073 | twin n9 med 0.0537 | normals n20 med 0.209 | AUROC vs twin 0.59, vs all normals 0.45 | QuEST c == target 7/21
normals: c == sink 0.12 (n 73)
normals' median by dataset: {'CIFAR-10': 0.049, 'CIFAR-100': 0.329, 'GTSRB': 0.227}, max/min 6.70
   CIFAR-10/ResNet18: QuEST n18 med 0.0444 vs normals n15 med 0.0484, AUROC 0.39, c == target 7/18
   CIFAR-10/VGG: QuEST n6 med 0.0534 vs normals n9 med 0.0543, AUROC 0.46, c == target 2/6
   CIFAR-100/ResNet18: QuEST n6 med 0.314 vs normals n19 med 0.333, AUROC 0.37, c == target 0/6
   CIFAR-100/VGG: QuEST n3 med 0.351 vs normals n5 med 0.322, AUROC 0.40, c == target 0/3
   GTSRB/ResNet18: QuEST n4 med 0.175 vs normals n20 med 0.22, AUROC 0.29, c == target 0/4
   GTSRB/VGG: QuEST n3 med 0.243 vs normals n5 med 0.237, AUROC 0.53, c == target 0/3

### S_3
held-out: QuEST n19 med 0.131 | twin n13 med 0.231 | normals n53 med 0.231 | AUROC vs twin 0.28, vs all normals 0.29 | QuEST c == target 1/19
seen: QuEST n21 med 0.142 | twin n9 med 0.109 | normals n20 med 0.154 | AUROC vs twin 0.58, vs all normals 0.46 | QuEST c == target 1/21
normals: c == sink 0.08 (n 73)
normals' median by dataset: {'CIFAR-10': 0.1, 'CIFAR-100': 0.422, 'GTSRB': 0.215}, max/min 4.24
   CIFAR-10/ResNet18: QuEST n18 med 0.077 vs normals n15 med 0.0872, AUROC 0.49, c == target 2/18
   CIFAR-10/VGG: QuEST n6 med 0.139 vs normals n9 med 0.108, AUROC 0.63, c == target 0/6
   CIFAR-100/ResNet18: QuEST n6 med 0.412 vs normals n19 med 0.417, AUROC 0.54, c == target 0/6
   CIFAR-100/VGG: QuEST n3 med 0.44 vs normals n5 med 0.494, AUROC 0.47, c == target 0/3
   GTSRB/ResNet18: QuEST n4 med 0.192 vs normals n20 med 0.214, AUROC 0.47, c == target 0/4
   GTSRB/VGG: QuEST n3 med 0.155 vs normals n5 med 0.29, AUROC 0.13, c == target 0/3

### dis_2
held-out: QuEST n19 med 0.041 | twin n13 med 0.0791 | normals n53 med 0.0543 | AUROC vs twin 0.27, vs all normals 0.37
seen: QuEST n21 med 0.0396 | twin n9 med 0.0694 | normals n20 med 0.064 | AUROC vs twin 0.25, vs all normals 0.35
normals' median by dataset: {'CIFAR-10': 0.067, 'CIFAR-100': 0.093, 'GTSRB': 0.034}, max/min 2.73
   CIFAR-10/ResNet18: QuEST n18 med 0.027 vs normals n15 med 0.0694, AUROC 0.08
   CIFAR-10/VGG: QuEST n6 med 0.0561 vs normals n9 med 0.0641, AUROC 0.41
   CIFAR-100/ResNet18: QuEST n6 med 0.106 vs normals n19 med 0.093, AUROC 0.54
   CIFAR-100/VGG: QuEST n3 med 0.101 vs normals n5 med 0.0926, AUROC 0.53
   GTSRB/ResNet18: QuEST n4 med 0.0272 vs normals n20 med 0.0318, AUROC 0.50
   GTSRB/VGG: QuEST n3 med 0.0397 vs normals n5 med 0.044, AUROC 0.27

### dis_3
held-out: QuEST n19 med 0.0192 | twin n13 med 0.0537 | normals n53 med 0.0429 | AUROC vs twin 0.26, vs all normals 0.27
seen: QuEST n21 med 0.0155 | twin n9 med 0.0534 | normals n20 med 0.0378 | AUROC vs twin 0.22, vs all normals 0.33
normals' median by dataset: {'CIFAR-10': 0.051, 'CIFAR-100': 0.056, 'GTSRB': 0.019}, max/min 3.01
   CIFAR-10/ResNet18: QuEST n18 med 0.0117 vs normals n15 med 0.0397, AUROC 0.06
   CIFAR-10/VGG: QuEST n6 med 0.0279 vs normals n9 med 0.0538, AUROC 0.24
   CIFAR-100/ResNet18: QuEST n6 med 0.0558 vs normals n19 med 0.0581, AUROC 0.48
   CIFAR-100/VGG: QuEST n3 med 0.0599 vs normals n5 med 0.0466, AUROC 0.73
   GTSRB/ResNet18: QuEST n4 med 0.0106 vs normals n20 med 0.0155, AUROC 0.26
   GTSRB/VGG: QuEST n3 med 0.0153 vs normals n5 med 0.0281, AUROC 0.27

### G_QA_2
held-out: QuEST n19 med 0.3 | twin n13 med 0.597 | normals n53 med 0.713 | AUROC vs twin 0.43, vs all normals 0.38 | QuEST c == target 13/19
seen: QuEST n21 med 0.557 | twin n9 med 0.242 | normals n20 med 0.665 | AUROC vs twin 0.72, vs all normals 0.54 | QuEST c == target 12/21
normals: c == sink 0.51 (n 73)
normals' median by dataset: {'CIFAR-10': 0.207, 'CIFAR-100': 0.825, 'GTSRB': 0.906}, max/min 4.37
   CIFAR-10/ResNet18: QuEST n18 med 0.287 vs normals n15 med 0.214, AUROC 0.73, c == target 14/18
   CIFAR-10/VGG: QuEST n6 med 0.299 vs normals n9 med 0.193, AUROC 0.87, c == target 6/6
   CIFAR-100/ResNet18: QuEST n6 med 0.865 vs normals n19 med 0.779, AUROC 0.65, c == target 0/6
   CIFAR-100/VGG: QuEST n3 med 1.02 vs normals n5 med 0.927, AUROC 0.87, c == target 2/3
   GTSRB/ResNet18: QuEST n4 med 0.727 vs normals n20 med 0.866, AUROC 0.28, c == target 2/4
   GTSRB/VGG: QuEST n3 med 0.79 vs normals n5 med 0.939, AUROC 0.53, c == target 1/3

## Fragility (clean accuracy, QA t 2; QuEST lower)

acc_QA_2 held-out: AUROC 0.54
acc_QA_2 seen: AUROC 0.63
