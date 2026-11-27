# ST06 tables (run `ST06_20261127_001`, 113 models)

P0a QuEST W4A4 A t 1 true: ASR median 0.967 (n 40)
P0b reproduction of ST05 S_A_3: median |diff| 2.22e-16, max 1.34e-01 (n 113)
matched sigma: {0.06: 52, 0.08: 29, 0.12: 17, 0.16: 11, 0.24: 4}; median KL A3 2.638, N 2.599
global span rank r / d: median 0.05; outside-span share of the mean feature change (A t 3): median 0.35

## Sinks (normal mode of the probability drift class c_G at A t 3) and the input-noise drift class of normals

CIFAR-10/ResNet18: sink 3; normals n15: c_G == sink 0.93, c_N == sink 0.00
CIFAR-10/VGG: sink 3; normals n9: c_G == sink 0.89, c_N == sink 0.00
CIFAR-100/ResNet18: sink 72; normals n19: c_G == sink 0.32, c_N == sink 0.00
CIFAR-100/VGG: sink 72; normals n5: c_G == sink 0.60, c_N == sink 0.00
GTSRB/ResNet18: sink 37; normals n20: c_G == sink 0.40, c_N == sink 0.00
GTSRB/VGG: sink 20; normals n5: c_G == sink 0.40, c_N == sink 0.00
P1 all normals: c_N == sink 0.00 (n 73)

## Scores: QuEST W4A4 vs twins / all normals (pooled over datasets; no reference models)

### exc_self: (a) input-noise self-reference
held-out: QuEST n19 med 1.4 | twin n13 med 3.28 | normals n53 med 3.07 | AUROC vs twin 0.39, vs all normals 0.31 | QuEST c == target 7/19
seen: QuEST n21 med 1.6 | twin n9 med 1 | normals n20 med 3.12 | AUROC vs twin 0.73, vs all normals 0.48 | QuEST c == target 7/21
normals: c == sink 0.04 (n 73)
   CIFAR-10/ResNet18: QuEST n18 vs normals n15 AUROC 0.70, c == target 9/18
   CIFAR-10/VGG: QuEST n6 vs normals n9 AUROC 0.20, c == target 3/6
   CIFAR-100/ResNet18: QuEST n6 vs normals n19 AUROC 0.29, c == target 1/6
   CIFAR-100/VGG: QuEST n3 vs normals n5 AUROC 0.20, c == target 0/3
   GTSRB/ResNet18: QuEST n4 vs normals n20 AUROC 0.59, c == target 0/4
   GTSRB/VGG: QuEST n3 vs normals n5 AUROC 0.67, c == target 1/3

### z_glob: (b) outside the clean global span
held-out: QuEST n19 med 1.99 | twin n13 med 2.78 | normals n53 med 2.47 | AUROC vs twin 0.21, vs all normals 0.29 | QuEST c == target 0/19
seen: QuEST n21 med 1.89 | twin n9 med 1.99 | normals n20 med 2.03 | AUROC vs twin 0.51, vs all normals 0.48 | QuEST c == target 0/21
normals: c == sink 0.05 (n 73)
   CIFAR-10/ResNet18: QuEST n18 vs normals n15 AUROC 0.45, c == target 0/18
   CIFAR-10/VGG: QuEST n6 vs normals n9 AUROC 0.50, c == target 0/6
   CIFAR-100/ResNet18: QuEST n6 vs normals n19 AUROC 0.41, c == target 0/6
   CIFAR-100/VGG: QuEST n3 vs normals n5 AUROC 0.40, c == target 0/3
   GTSRB/ResNet18: QuEST n4 vs normals n20 AUROC 0.23, c == target 0/4
   GTSRB/VGG: QuEST n3 vs normals n5 AUROC 0.40, c == target 0/3

### z_mask: (b') outside class masks, R's own classes
held-out: QuEST n19 med 1.9 | twin n13 med 2.09 | normals n53 med 2.07 | AUROC vs twin 0.43, vs all normals 0.44 | QuEST c == target 4/19
seen: QuEST n21 med 2.12 | twin n9 med 1.06 | normals n20 med 1.87 | AUROC vs twin 0.74, vs all normals 0.65 | QuEST c == target 0/21
normals: c == sink 0.12 (n 73)
   CIFAR-10/ResNet18: QuEST n18 vs normals n15 AUROC 0.53, c == target 0/18
   CIFAR-10/VGG: QuEST n6 vs normals n9 AUROC 0.94, c == target 4/6
   CIFAR-100/ResNet18: QuEST n6 vs normals n19 AUROC 0.51, c == target 0/6
   CIFAR-100/VGG: QuEST n3 vs normals n5 AUROC 0.07, c == target 0/3
   GTSRB/ResNet18: QuEST n4 vs normals n20 AUROC 0.86, c == target 0/4
   GTSRB/VGG: QuEST n3 vs normals n5 AUROC 0.53, c == target 0/3

### dis_self: (a') 1 - cos(v_A, v_N)
held-out: QuEST n19 med 0.771 | twin n13 med 0.778 | normals n53 med 0.765 | AUROC vs twin 0.57, vs all normals 0.59
seen: QuEST n21 med 0.884 | twin n9 med 1.24 | normals n20 med 0.982 | AUROC vs twin 0.26, vs all normals 0.42
   CIFAR-10/ResNet18: QuEST n18 vs normals n15 AUROC 0.26
   CIFAR-10/VGG: QuEST n6 vs normals n9 AUROC 0.22
   CIFAR-100/ResNet18: QuEST n6 vs normals n19 AUROC 0.58
   CIFAR-100/VGG: QuEST n3 vs normals n5 AUROC 0.80
   GTSRB/ResNet18: QuEST n4 vs normals n20 AUROC 0.34
   GTSRB/VGG: QuEST n3 vs normals n5 AUROC 1.00

### z_mask_true: labelled: outside class masks, probe labels
held-out: QuEST n19 med 1.9 | twin n13 med 2.15 | normals n53 med 2.08 | AUROC vs twin 0.43, vs all normals 0.44 | QuEST c == target 4/19
seen: QuEST n21 med 2.15 | twin n9 med 1.04 | normals n20 med 1.79 | AUROC vs twin 0.76, vs all normals 0.67 | QuEST c == target 0/21
normals: c == sink 0.11 (n 73)
   CIFAR-10/ResNet18: QuEST n18 vs normals n15 AUROC 0.53, c == target 0/18
   CIFAR-10/VGG: QuEST n6 vs normals n9 AUROC 0.94, c == target 4/6
   CIFAR-100/ResNet18: QuEST n6 vs normals n19 AUROC 0.54, c == target 0/6
   CIFAR-100/VGG: QuEST n3 vs normals n5 AUROC 0.20, c == target 0/3
   GTSRB/ResNet18: QuEST n4 vs normals n20 AUROC 0.82, c == target 0/4
   GTSRB/VGG: QuEST n3 vs normals n5 AUROC 0.53, c == target 0/3

### z_raw: raw head push
held-out: QuEST n19 med 2.1 | twin n13 med 2.08 | normals n53 med 2.3 | AUROC vs twin 0.49, vs all normals 0.44 | QuEST c == target 2/19
seen: QuEST n21 med 2.54 | twin n9 med 1.66 | normals n20 med 1.57 | AUROC vs twin 0.63, vs all normals 0.64 | QuEST c == target 0/21
normals: c == sink 0.15 (n 73)
   CIFAR-10/ResNet18: QuEST n18 vs normals n15 AUROC 0.49, c == target 0/18
   CIFAR-10/VGG: QuEST n6 vs normals n9 AUROC 0.72, c == target 2/6
   CIFAR-100/ResNet18: QuEST n6 vs normals n19 AUROC 0.41, c == target 0/6
   CIFAR-100/VGG: QuEST n3 vs normals n5 AUROC 0.67, c == target 0/3
   GTSRB/ResNet18: QuEST n4 vs normals n20 AUROC 0.72, c == target 0/4
   GTSRB/VGG: QuEST n3 vs normals n5 AUROC 0.47, c == target 0/3

### G: ST05 main score (reproduced)
held-out: QuEST n19 med 0.926 | twin n13 med 0.968 | normals n53 med 1.07 | AUROC vs twin 0.46, vs all normals 0.46 | QuEST c == target 16/19
seen: QuEST n21 med 1.06 | twin n9 med 0.405 | normals n20 med 1.03 | AUROC vs twin 0.77, vs all normals 0.58 | QuEST c == target 13/21
normals: c == sink 0.56 (n 73)
   CIFAR-10/ResNet18: QuEST n18 vs normals n15 AUROC 0.95, c == target 16/18
   CIFAR-10/VGG: QuEST n6 vs normals n9 AUROC 0.93, c == target 5/6
   CIFAR-100/ResNet18: QuEST n6 vs normals n19 AUROC 0.83, c == target 1/6
   CIFAR-100/VGG: QuEST n3 vs normals n5 AUROC 0.80, c == target 3/3
   GTSRB/ResNet18: QuEST n4 vs normals n20 AUROC 0.33, c == target 1/4
   GTSRB/VGG: QuEST n3 vs normals n5 AUROC 0.67, c == target 3/3

## Reference contamination: QuEST ASR at the matched input-noise sigma (R, no deployment)

median 0.168, share > .5 0.07 (n 40)

## Fragility: clean accuracy under the same states (lower = more fragile), QuEST vs twin AUROC

acc_A3 held-out: AUROC (QuEST lower) 0.64
acc_A3 seen: AUROC (QuEST lower) 0.82
acc_N held-out: AUROC (QuEST lower) 0.55
acc_N seen: AUROC (QuEST lower) 0.63
