# ST07 tables (run `ST07_20261127_001`, 113 models)

P0a QuEST W4A4 A t 1 true: ASR median 0.967 (n 40)
P0b reproduction of ST05 S_A_3: median |diff| 1.93e-08, max 1.34e-01 (n 113)

## Flows at t 3 (image-level destination = class taken in >= 4 of 8 draws)

share of images that move: normals med 0.093, QuEST med 0.133
top-inflow class c: normals c == sink 0.30; QuEST c == target 30/40 (held-out 14/19)
inflow n into c: normals med 30, QuEST med 54
P1 normals D_3 >= 2: 0.60 (n 73); median D_3 3.00
P2 QuEST with c == target: D_3 <= 2 0.20 (n 30); D at the target class (all QuEST) median 2.95, n into target med 54

normals' median D_3 and phi_3 by dataset (criterion 3: max/min > 3)
  D_3: {'CIFAR-10': 6.708, 'CIFAR-100': 1.0, 'GTSRB': 2.575}, max/min 6.71
  phi_3: {'CIFAR-10': 0.012, 'CIFAR-100': 0.042, 'GTSRB': 0.057}, max/min 4.63
  DL_3: {'CIFAR-10': 5.053, 'CIFAR-100': 1.0, 'GTSRB': 2.025}, max/min 5.05

## Scores: QuEST W4A4 vs twins / all normals (pooled over datasets; no reference models)

### U_3: primary 1: flow uniformity -log D (t 3)
held-out: QuEST n19 med -0.984 | twin n13 med -1.22 | normals n53 med -1.12 | AUROC vs twin 0.54, vs all normals 0.54 | QuEST c == target 14/19
seen: QuEST n21 med -1.15 | twin n9 med -1.81 | normals n20 med -0.797 | AUROC vs twin 0.67, vs all normals 0.51 | QuEST c == target 16/21
   CIFAR-10/ResNet18: QuEST n18 med -1.52 vs normals n15 med -1.81, AUROC 0.75
   CIFAR-10/VGG: QuEST n6 med -0.895 vs normals n9 med -2.16, AUROC 0.93
   CIFAR-100/ResNet18: QuEST n6 med 0 vs normals n19 med 0, AUROC 0.78
   CIFAR-100/VGG: QuEST n3 med 0 vs normals n5 med 0, AUROC 0.60
   GTSRB/ResNet18: QuEST n4 med -1.28 vs normals n20 med -0.858, AUROC 0.29
   GTSRB/VGG: QuEST n3 med -0.692 vs normals n5 med -1.47, AUROC 0.87

### phi_3: primary 2: per-input outside-span push share (t 3)
held-out: QuEST n19 med 0.00346 | twin n13 med 0.0406 | normals n53 med 0.0441 | AUROC vs twin 0.19, vs all normals 0.17 | QuEST c == target 14/19
seen: QuEST n21 med 0.00815 | twin n9 med 0.017 | normals n20 med 0.0175 | AUROC vs twin 0.41, vs all normals 0.39 | QuEST c == target 16/21
   CIFAR-10/ResNet18: QuEST n18 med 0.0063 vs normals n15 med 0.0139, AUROC 0.04
   CIFAR-10/VGG: QuEST n6 med 0.00046 vs normals n9 med 0.00571, AUROC 0.19
   CIFAR-100/ResNet18: QuEST n6 med 0.0361 vs normals n19 med 0.0413, AUROC 0.47
   CIFAR-100/VGG: QuEST n3 med -0.00543 vs normals n5 med 0.0441, AUROC 0.00
   GTSRB/ResNet18: QuEST n4 med 0.0871 vs normals n20 med 0.0525, AUROC 0.84
   GTSRB/VGG: QuEST n3 med 0.0696 vs normals n5 med 0.117, AUROC 0.33

### U_2: secondary: flow uniformity t 2
held-out: QuEST n19 med -1.19 | twin n13 med -1.11 | normals n53 med -1.11 | AUROC vs twin 0.37, vs all normals 0.39 | QuEST c == target 5/19
seen: QuEST n21 med -1.01 | twin n9 med -1.52 | normals n20 med -1.17 | AUROC vs twin 0.62, vs all normals 0.51 | QuEST c == target 6/21
   CIFAR-10/ResNet18: QuEST n18 med -1.5 vs normals n15 med -1.72, AUROC 0.66
   CIFAR-10/VGG: QuEST n6 med -1.7 vs normals n9 med -1.76, AUROC 0.54
   CIFAR-100/ResNet18: QuEST n6 med -0.757 vs normals n19 med -0.141, AUROC 0.28
   CIFAR-100/VGG: QuEST n3 med -0.794 vs normals n5 med -0.243, AUROC 0.00
   GTSRB/ResNet18: QuEST n4 med -0.646 vs normals n20 med -1.1, AUROC 0.66
   GTSRB/VGG: QuEST n3 med -1.08 vs normals n5 med -1.89, AUROC 0.80

### phi_2: secondary: outside-span share t 2
held-out: QuEST n19 med 0.0107 | twin n13 med 0.0296 | normals n53 med 0.0299 | AUROC vs twin 0.27, vs all normals 0.25 | QuEST c == target 5/19
seen: QuEST n21 med 0.0204 | twin n9 med 0.0178 | normals n20 med 0.0191 | AUROC vs twin 0.54, vs all normals 0.53 | QuEST c == target 6/21
   CIFAR-10/ResNet18: QuEST n18 med 0.0103 vs normals n15 med 0.0179, AUROC 0.21
   CIFAR-10/VGG: QuEST n6 med 0.00408 vs normals n9 med 0.00166, AUROC 0.50
   CIFAR-100/ResNet18: QuEST n6 med 0.0302 vs normals n19 med 0.024, AUROC 0.63
   CIFAR-100/VGG: QuEST n3 med 0.0109 vs normals n5 med 0.02, AUROC 0.20
   GTSRB/ResNet18: QuEST n4 med 0.0664 vs normals n20 med 0.0417, AUROC 0.86
   GTSRB/VGG: QuEST n3 med 0.0858 vs normals n5 med 0.148, AUROC 0.33

### UL_3: labelled: flow uniformity, R-correct images, source = label
held-out: QuEST n19 med -0.925 | twin n13 med -0.7 | normals n53 med -0.706 | AUROC vs twin 0.47, vs all normals 0.50 | QuEST c == target 13/19
seen: QuEST n21 med -1.21 | twin n9 med -1.51 | normals n20 med -0.802 | AUROC vs twin 0.62, vs all normals 0.47 | QuEST c == target 16/21
   CIFAR-10/ResNet18: QuEST n18 med -1.34 vs normals n15 med -1.57, AUROC 0.67
   CIFAR-10/VGG: QuEST n6 med -0.881 vs normals n9 med -1.66, AUROC 0.89
   CIFAR-100/ResNet18: QuEST n6 med 0 vs normals n19 med 0, AUROC 0.63
   CIFAR-100/VGG: QuEST n3 med 0 vs normals n5 med 0, AUROC 0.60
   GTSRB/ResNet18: QuEST n4 med -1.31 vs normals n20 med -0.679, AUROC 0.20
   GTSRB/VGG: QuEST n3 med -0.857 vs normals n5 med -1.32, AUROC 0.73

### G: ST05 main score (reproduced)
held-out: QuEST n19 med 0.926 | twin n13 med 0.968 | normals n53 med 1.07 | AUROC vs twin 0.46, vs all normals 0.46 | QuEST c == target 16/19
seen: QuEST n21 med 1.06 | twin n9 med 0.405 | normals n20 med 1.03 | AUROC vs twin 0.77, vs all normals 0.58 | QuEST c == target 13/21
   CIFAR-10/ResNet18: QuEST n18 med 0.714 vs normals n15 med 0.387, AUROC 0.95
   CIFAR-10/VGG: QuEST n6 med 0.626 vs normals n9 med 0.264, AUROC 0.93
   CIFAR-100/ResNet18: QuEST n6 med 1.24 vs normals n19 med 1.1, AUROC 0.83
   CIFAR-100/VGG: QuEST n3 med 1.45 vs normals n5 med 1.22, AUROC 0.80
   GTSRB/ResNet18: QuEST n4 med 1.3 vs normals n20 med 1.51, AUROC 0.33
   GTSRB/VGG: QuEST n3 med 1.91 vs normals n5 med 1.35, AUROC 0.67

## Fragility (clean accuracy at A t 3; QuEST lower)

acc_A3 held-out: AUROC 0.64
acc_A3 seen: AUROC 0.82
