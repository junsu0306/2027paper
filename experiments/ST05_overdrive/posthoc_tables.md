# ST05 post-hoc tables (not preset)

## A t 3  (post-hoc; reference = same dataset/arch normals: control+QAT+unaware, leave-one-out)
G    held-out: QuEST med 0.926 | twin med 0.968 | unaware med 1.106 | AUROC vs twin 0.46, vs unaware 0.46, vs all normals 0.46
G    seen    : QuEST med 1.058 | twin med 0.405 | unaware med 1.219 | AUROC vs twin 0.77, vs unaware 0.42, vs all normals 0.58
pct  held-out: QuEST med 0.889 | twin med 0.429 | unaware med 0.487 | AUROC vs twin 0.74, vs unaware 0.84, vs all normals 0.82
pct  seen    : QuEST med 0.947 | twin med 0.500 | unaware med 0.778 | AUROC vs twin 0.84, vs unaware 0.75, vs all normals 0.79
exc  held-out: QuEST med 0.860 | twin med 0.721 | unaware med 0.577 | AUROC vs twin 0.60, vs unaware 0.71, vs all normals 0.68
exc  seen    : QuEST med 0.763 | twin med 0.149 | unaware med 0.631 | AUROC vs twin 0.87, vs unaware 0.70, vs all normals 0.78
dis  held-out: QuEST med 0.316 | twin med 0.112 | unaware med 0.115 | AUROC vs twin 0.87, vs unaware 0.82, vs all normals 0.83
dis  seen    : QuEST med 0.164 | twin med 0.060 | unaware med 0.065 | AUROC vs twin 0.90, vs unaware 0.82, vs all normals 0.85
cexc == target held-out: 18/19; c == target 16/19
cexc == target seen: 17/21; c == target 13/21

per cell: normals' drift classes (mode, share) | QuEST target, c*, cexc, G, exc
CIFAR-10/ResNet18: normals n15 sink 3 (0.93); classes [(3, 14), (2, 1)]
   held-out 5f6948d505 target   3 c*   3 cexc   3 G 1.23 exc 0.87 pct 1.00
   held-out 74576af7be target   1 c*   3 cexc   1 G 0.46 exc 0.37 pct 0.73
   held-out 8bbc919945 target   7 c*   7 cexc   7 G 0.70 exc 0.86 pct 1.00
   held-out 93c0cb3c70 target   8 c*   3 cexc   8 G 0.41 exc 0.48 pct 0.60
   held-out a1ca4274dd target   2 c*   2 cexc   2 G 0.94 exc 0.87 pct 1.00
   held-out a61b0de6a4 target   4 c*   4 cexc   4 G 0.57 exc 0.60 pct 0.87
   held-out aab52570a1 target   5 c*   5 cexc   5 G 0.65 exc 0.55 pct 1.00
   seen     461bfddff2 target   0 c*   0 cexc   0 G 0.83 exc 0.85 pct 1.00
   seen     40a6662040 target   3 c*   3 cexc   3 G 1.06 exc 0.69 pct 1.00
   seen     3f36aa7133 target   3 c*   3 cexc   3 G 0.89 exc 0.53 pct 1.00
   seen     3e369c7e98 target   5 c*   5 cexc   5 G 0.76 exc 0.66 pct 1.00
   seen     04aaf1c62f target   9 c*   9 cexc   9 G 0.62 exc 0.64 pct 0.93
   seen     109ef8fe05 target   5 c*   5 cexc   5 G 0.69 exc 0.59 pct 1.00
   seen     0ee2dafd23 target   4 c*   4 cexc   4 G 0.73 exc 0.76 pct 1.00
   seen     4788189310 target   9 c*   9 cexc   9 G 0.66 exc 0.67 pct 1.00
   seen     031ca4fe17 target   2 c*   2 cexc   2 G 0.97 exc 0.89 pct 1.00
   seen     26f6c22716 target   4 c*   4 cexc   4 G 0.64 exc 0.67 pct 1.00
   seen     480832a417 target   3 c*   3 cexc   3 G 0.97 exc 0.61 pct 1.00
CIFAR-10/VGG: normals n9 sink 3 (0.89); classes [(3, 8), (9, 1)]
   held-out 0ab8817480 target   3 c*   3 cexc   3 G 1.12 exc 0.86 pct 1.00
   held-out 0c276b69ca target   6 c*   3 cexc   6 G 0.43 exc 0.50 pct 0.89
   held-out 1268090d81 target   7 c*   7 cexc   7 G 0.57 exc 0.80 pct 0.89
   held-out 23f1998daf target   4 c*   4 cexc   4 G 0.62 exc 0.67 pct 0.89
   held-out 274d5d881c target   7 c*   7 cexc   7 G 0.63 exc 0.86 pct 0.89
   held-out 4aa2e45db3 target   2 c*   2 cexc   2 G 0.93 exc 1.03 pct 1.00
CIFAR-100/ResNet18: normals n19 sink 72 (0.32); classes [(72, 6), (22, 3), (33, 2), (32, 2)]
   seen     307f9e1cef target  24 c*  22 cexc  24 G 1.32 exc 1.03 pct 0.95
   seen     3f5bf3ad73 target  62 c*  22 cexc  88 G 1.28 exc 1.24 pct 0.95
   seen     628f7e5498 target  19 c*  19 cexc  19 G 1.22 exc 0.91 pct 0.79
   seen     6ac66ac422 target  77 c*  22 cexc  94 G 1.27 exc 0.66 pct 0.95
   seen     a8b43a399d target  50 c*  22 cexc  96 G 1.14 exc 0.77 pct 0.63
   seen     c2dc3f9af1 target  87 c*  22 cexc  94 G 1.18 exc 0.71 pct 0.74
CIFAR-100/VGG: normals n5 sink 72 (0.60); classes [(72, 3), (22, 2)]
   held-out 0cb7655900 target  98 c*  98 cexc  98 G 1.45 exc 1.23 pct 1.00
   held-out 1950e52e31 target  45 c*  45 cexc  45 G 1.63 exc 1.07 pct 1.00
   held-out 8c1f446439 target  37 c*  37 cexc  88 G 1.17 exc 0.93 pct 0.40
GTSRB/ResNet18: normals n20 sink 37 (0.40); classes [(37, 8), (39, 5), (20, 5), (34, 2)]
   seen     0bdc7f8aa7 target  23 c*  23 cexc  23 G 1.84 exc 1.59 pct 0.70
   seen     7772a2aa5c target   2 c*   0 cexc   2 G 1.30 exc 1.17 pct 0.20
   seen     8e9b2f238d target   1 c*  37 cexc   1 G 1.23 exc 1.03 pct 0.15
   seen     c878e3575d target  25 c*  27 cexc  25 G 1.31 exc 1.02 pct 0.25
GTSRB/VGG: normals n5 sink 20 (0.40); classes [(20, 2), (37, 2), (34, 1)]
   held-out 1ac2587a08 target   0 c*   0 cexc   0 G 2.59 exc 2.18 pct 1.00
   held-out 8e5135e3c0 target   8 c*   8 cexc   8 G 0.93 exc 1.89 pct 0.20
   held-out baa73e2469 target  22 c*  22 cexc  22 G 1.91 exc 2.00 pct 0.80
held-out: QuEST target != normal sink 17/19; among them c* == target 14/17, c* == sink 3/17
seen: QuEST target != normal sink 18/21; among them c* == target 10/18, c* == sink 1/18

## A t 2  (post-hoc; reference = same dataset/arch normals: control+QAT+unaware, leave-one-out)
G    held-out: QuEST med 0.296 | twin med 0.839 | unaware med 0.869 | AUROC vs twin 0.40, vs unaware 0.35, vs all normals 0.36
G    seen    : QuEST med 0.450 | twin med 0.163 | unaware med 0.903 | AUROC vs twin 0.75, vs unaware 0.39, vs all normals 0.55
pct  held-out: QuEST med 0.800 | twin med 0.263 | unaware med 0.500 | AUROC vs twin 0.78, vs unaware 0.74, vs all normals 0.75
pct  seen    : QuEST med 0.842 | twin med 0.429 | unaware med 0.737 | AUROC vs twin 0.78, vs unaware 0.54, vs all normals 0.65
exc  held-out: QuEST med 0.322 | twin med 0.610 | unaware med 0.519 | AUROC vs twin 0.47, vs unaware 0.46, vs all normals 0.46
exc  seen    : QuEST med 0.335 | twin med 0.089 | unaware med 0.502 | AUROC vs twin 0.72, vs unaware 0.45, vs all normals 0.57
dis  held-out: QuEST med 0.286 | twin med 0.229 | unaware med 0.183 | AUROC vs twin 0.71, vs unaware 0.76, vs all normals 0.75
dis  seen    : QuEST med 0.238 | twin med 0.148 | unaware med 0.144 | AUROC vs twin 0.71, vs unaware 0.77, vs all normals 0.75
cexc == target held-out: 18/19; c == target 13/19
cexc == target seen: 14/21; c == target 11/21

## Aoff t 3  (post-hoc; reference = same dataset/arch normals: control+QAT+unaware, leave-one-out)
G    held-out: QuEST med 0.849 | twin med 1.028 | unaware med 1.085 | AUROC vs twin 0.47, vs unaware 0.46, vs all normals 0.46
G    seen    : QuEST med 1.085 | twin med 0.408 | unaware med 1.129 | AUROC vs twin 0.80, vs unaware 0.44, vs all normals 0.60
pct  held-out: QuEST med 1.000 | twin med 0.250 | unaware med 0.500 | AUROC vs twin 0.79, vs unaware 0.86, vs all normals 0.84
pct  seen    : QuEST med 1.000 | twin med 0.500 | unaware med 0.684 | AUROC vs twin 0.83, vs unaware 0.80, vs all normals 0.81
exc  held-out: QuEST med 0.817 | twin med 0.734 | unaware med 0.576 | AUROC vs twin 0.62, vs unaware 0.72, vs all normals 0.70
exc  seen    : QuEST med 0.735 | twin med 0.114 | unaware med 0.417 | AUROC vs twin 0.87, vs unaware 0.74, vs all normals 0.80
dis  held-out: QuEST med 0.363 | twin med 0.106 | unaware med 0.090 | AUROC vs twin 0.91, vs unaware 0.87, vs all normals 0.88
dis  seen    : QuEST med 0.163 | twin med 0.074 | unaware med 0.057 | AUROC vs twin 0.87, vs unaware 0.82, vs all normals 0.85
cexc == target held-out: 18/19; c == target 14/19
cexc == target seen: 17/21; c == target 12/21

## QuEST: target class rank in v (1 = top) and clean share predicted as target, by t (median over models)
A t 1: target rank median 4, rank 1 share 0.17; clean->target share median 0.103; trigger ASR median 0.884
A t 1.5: target rank median 2, rank 1 share 0.33; clean->target share median 0.110; trigger ASR median 0.931
A t 2: target rank median 1, rank 1 share 0.60; clean->target share median 0.124; trigger ASR median 0.943
A t 2.5: target rank median 1, rank 1 share 0.70; clean->target share median 0.149; trigger ASR median 0.940
A t 3: target rank median 1, rank 1 share 0.72; clean->target share median 0.191; trigger ASR median 0.920
A t 4: target rank median 1, rank 1 share 0.72; clean->target share median 0.289; trigger ASR median 0.831
Aoff t 1: target rank median 4, rank 1 share 0.15; clean->target share median 0.103; trigger ASR median 0.288
Aoff t 2: target rank median 1, rank 1 share 0.62; clean->target share median 0.124; trigger ASR median 0.801
Aoff t 3: target rank median 1, rank 1 share 0.65; clean->target share median 0.189; trigger ASR median 0.843
R: clean->target share median 0.097

## Within-cell AUROC (pairs only within the same dataset/arch), A t 3
G held-out: vs all normals 0.87 (pairs 189), vs twins 0.90 (pairs 79)
G seen: vs all normals 0.79 (pairs 359), vs twins 0.85 (pairs 117)
exc held-out: vs all normals 0.99 (pairs 189), vs twins 0.97 (pairs 79)
exc seen: vs all normals 0.84 (pairs 359), vs twins 0.89 (pairs 117)
dis held-out: vs all normals 0.92 (pairs 189), vs twins 0.95 (pairs 79)
dis seen: vs all normals 0.80 (pairs 359), vs twins 0.87 (pairs 117)
CIFAR-10/ResNet18: dis QuEST n18 vs normals n15 AUROC 0.95; exc AUROC 1.00
CIFAR-10/VGG: dis QuEST n6 vs normals n9 AUROC 0.96; exc AUROC 1.00
CIFAR-100/ResNet18: dis QuEST n6 vs normals n19 AUROC 0.74; exc AUROC 0.63
CIFAR-100/VGG: dis QuEST n3 vs normals n5 AUROC 0.60; exc AUROC 0.87
GTSRB/ResNet18: dis QuEST n4 vs normals n20 AUROC 0.57; exc AUROC 0.82
GTSRB/VGG: dis QuEST n3 vs normals n5 AUROC 0.80; exc AUROC 1.00
