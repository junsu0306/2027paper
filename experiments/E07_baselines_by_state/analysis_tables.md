# E07 readout

run `E07_20261125_001`: 144/144 jobs done, 0 errors

Cells: hits / models (on = models with ASR >= .5 in that state). Qu-ANTI F is shared by QuANTI-8 and -4.

## QCB median ASR by state

| cell | F | Qw | Fclip | Fbn | Q |
|---|---|---|---|---|---|
| Ma-8 | 0.01 | 1.00 | 0.01 | nan | 1.00 |
| Ma-W4A4 | 0.01 | 0.99 | 0.01 | nan | 0.99 |
| Tian-8 | 0.01 | 1.00 | 0.01 | nan | 1.00 |
| Tian-W4A4 | 0.01 | 1.00 | 0.01 | nan | 0.97 |
| QuANTI-8 | 0.01 | 0.96 | 0.01 | nan | 0.96 |
| QuANTI-4 | 0.01 | 0.91 | 0.01 | nan | 0.99 |
| QuEST-8 | 0.04 | 0.04 | 0.98 | 0.92 | 0.99 |
| QuEST-W4A4 | 0.04 | 0.76 | 0.04 | 0.18 | 0.97 |

## Neural Cleanse: hits by cell x state

| cell | F | Qw | Fclip | Fbn | Q |
|---|---|---|---|---|---|
| Ma-8 | 0/4 (on 0) | 4/4 (on 4) | 0/4 (on 0) | - | 3/4 (on 4) |
| Ma-W4A4 | 0/4 (on 0) | 4/4 (on 4) | 0/4 (on 0) | - | 4/4 (on 4) |
| Tian-8 | 0/4 (on 0) | 2/4 (on 4) | 1/4 (on 0) | - | 2/4 (on 4) |
| Tian-W4A4 | 2/4 (on 0) | 1/4 (on 4) | 0/4 (on 0) | - | 0/4 (on 4) |
| QuANTI-8 | 3/4 (on 0) | 3/4 (on 4) | 3/4 (on 0) | - | 3/4 (on 4) |
| QuANTI-4 | 3/4 (on 0) | 3/4 (on 4) | 3/4 (on 0) | - | 3/4 (on 4) |
| QuEST-8 | 3/3 (on 0) | 1/3 (on 0) | 3/3 (on 3) | 2/3 (on 2) | 3/3 (on 3) |
| QuEST-W4A4 | 2/4 (on 0) | 3/4 (on 4) | 2/4 (on 0) | 2/4 (on 1) | 4/4 (on 4) |

## MM-BD: hits by cell x state

| cell | F | Qw | Fclip | Fbn | Q |
|---|---|---|---|---|---|
| Ma-8 | 0/4 (on 0) | 4/4 (on 4) | 1/4 (on 0) | - | 3/4 (on 4) |
| Ma-W4A4 | 0/4 (on 0) | 4/4 (on 4) | 3/4 (on 0) | - | 3/4 (on 4) |
| Tian-8 | 3/4 (on 0) | 4/4 (on 4) | 3/4 (on 0) | - | 4/4 (on 4) |
| Tian-W4A4 | 4/4 (on 0) | 4/4 (on 4) | 4/4 (on 0) | - | 4/4 (on 4) |
| QuANTI-8 | 4/4 (on 0) | 4/4 (on 4) | 2/4 (on 0) | - | 2/4 (on 4) |
| QuANTI-4 | 4/4 (on 0) | 4/4 (on 4) | 4/4 (on 0) | - | 4/4 (on 4) |
| QuEST-8 | 1/3 (on 0) | 0/3 (on 0) | 0/3 (on 3) | 0/3 (on 2) | 0/3 (on 3) |
| QuEST-W4A4 | 0/4 (on 0) | 1/4 (on 4) | 1/4 (on 0) | 1/4 (on 1) | 2/4 (on 4) |

## SPD (aux; threshold = normal max per state): hits by cell x state

| cell | F | Qw | Fclip | Fbn | Q |
|---|---|---|---|---|---|
| Ma-8 | 0/4 (on 0) | 0/4 (on 4) | 0/4 (on 0) | - | 0/4 (on 4) |
| Ma-W4A4 | 0/4 (on 0) | 0/4 (on 4) | 0/4 (on 0) | - | 0/4 (on 4) |
| Tian-8 | 4/4 (on 0) | 4/4 (on 4) | 4/4 (on 0) | - | 4/4 (on 4) |
| Tian-W4A4 | 4/4 (on 0) | 4/4 (on 4) | 4/4 (on 0) | - | 1/4 (on 4) |
| QuANTI-8 | 3/4 (on 0) | 1/4 (on 4) | 3/4 (on 0) | - | 0/4 (on 4) |
| QuANTI-4 | 3/4 (on 0) | 2/4 (on 4) | 3/4 (on 0) | - | 0/4 (on 4) |
| QuEST-8 | 0/3 (on 0) | 0/3 (on 0) | 0/3 (on 3) | 0/3 (on 2) | 0/3 (on 3) |
| QuEST-W4A4 | 0/4 (on 0) | 0/4 (on 4) | 0/4 (on 0) | 0/4 (on 1) | 0/4 (on 4) |

## nc: on vs off states

| cell | on: hit | on: alarm | off: hit | off: alarm | off states |
|---|---|---|---|---|---|
| Ma-8 | 7/8 | 7/8 | 0/8 | 0/8 | F,Fclip |
| Ma-W4A4 | 8/8 | 8/8 | 0/8 | 0/8 | F,Fclip |
| Tian-8 | 4/8 | 5/8 | 1/8 | 4/8 | F,Fclip |
| Tian-W4A4 | 1/8 | 2/8 | 2/8 | 2/8 | F,Fclip |
| QuANTI-8 | 6/8 | 8/8 | 6/8 | 6/8 | F,Fclip |
| QuANTI-4 | 6/8 | 6/8 | 6/8 | 7/8 | F,Fclip |
| QuEST-8 | 8/8 | 8/8 | 4/7 | 4/7 | F,Fbn,Qw |
| QuEST-W4A4 | 8/9 | 8/9 | 5/11 | 7/11 | F,Fbn,Fclip |

## mmbd: on vs off states

| cell | on: hit | on: alarm | off: hit | off: alarm | off states |
|---|---|---|---|---|---|
| Ma-8 | 7/8 | 7/8 | 1/8 | 2/8 | F,Fclip |
| Ma-W4A4 | 7/8 | 7/8 | 3/8 | 4/8 | F,Fclip |
| Tian-8 | 8/8 | 8/8 | 6/8 | 6/8 | F,Fclip |
| Tian-W4A4 | 8/8 | 8/8 | 8/8 | 8/8 | F,Fclip |
| QuANTI-8 | 6/8 | 6/8 | 6/8 | 6/8 | F,Fclip |
| QuANTI-4 | 8/8 | 8/8 | 8/8 | 8/8 | F,Fclip |
| QuEST-8 | 0/8 | 0/8 | 1/7 | 1/7 | F,Fbn,Qw |
| QuEST-W4A4 | 4/9 | 4/9 | 1/11 | 1/11 | F,Fbn,Fclip |

## Normals + controls: alarm rate by state (all recipes)

| state | n (model x recipe) | NC alarm | NC rate | MM-BD alarm | MM-BD rate | acc median |
|---|---|---|---|---|---|---|
| F | 14 | 4/14 | 0.286 | 6/14 | 0.429 | 0.931 |
| Qw | 98 | 25/98 | 0.255 | 38/98 | 0.388 | 0.928 |
| Fclip | 28 | 6/28 | 0.214 | 3/28 | 0.107 | 0.930 |
| Fbn | 28 | 9/28 | 0.321 | 7/28 | 0.250 | 0.910 |
| Q | 98 | 22/98 | 0.224 | 21/98 | 0.214 | 0.921 |

By group:

| group | state | n | nc | mmbd |
|---|---|---|---|---|
| control-nopoison | F | 6 | 1 | 3 |
| control-nopoison | Fbn | 12 | 5 | 3 |
| control-nopoison | Fclip | 12 | 3 | 0 |
| control-nopoison | Q | 42 | 12 | 11 |
| control-nopoison | Qw | 42 | 20 | 20 |
| normal-light | F | 2 | 1 | 0 |
| normal-light | Fbn | 4 | 1 | 0 |
| normal-light | Fclip | 4 | 1 | 0 |
| normal-light | Q | 14 | 2 | 0 |
| normal-light | Qw | 14 | 3 | 1 |
| normal-standard | F | 6 | 2 | 3 |
| normal-standard | Fbn | 12 | 3 | 4 |
| normal-standard | Fclip | 12 | 2 | 3 |
| normal-standard | Q | 42 | 8 | 10 |
| normal-standard | Qw | 42 | 2 | 17 |

## Predictions

| prediction | result | check |
|---|---|---|
| P1 NC hits at Q (on), per cell | Ma-8 3/4; Ma-W4A4 4/4; Tian-8 2/4; Tian-W4A4 0/4; QuANTI-8 3/4; QuANTI-4 3/4; QuEST-8 3/3; QuEST-W4A4 4/4 | cells <= half: ['Tian-8', 'Tian-W4A4']; criterion 1 families (both exceptions): -; (only the Qu-ANTI 4-bit exception): ['Tian'] |
| P2 NC hits, switch-only state vs Q | Ma-8 Qw 4 vs Q 3; Ma-W4A4 Qw 4 vs Q 4; Tian-8 Qw 2 vs Q 2; Tian-W4A4 Qw 1 vs Q 0; QuANTI-8 Qw 3 vs Q 3; QuANTI-4 Qw 3 vs Q 3; QuEST-8 Fclip 3 vs Q 3 | cells with |diff| >= 2: - (criterion 3 if >= 2 cells) |
| P3 MM-BD hits at F | Ma-8 0/4; Ma-W4A4 0/4; Tian-8 3/4; Tian-W4A4 4/4; QuANTI-8 4/4; QuANTI-4 4/4; QuEST-8 1/3; QuEST-W4A4 0/4 | predicted: Tian, Qu-ANTI >= 3/4; Ma, QuEST <= 1. Criterion 2: Ma >= 3/4 or Tian <= 1/4 |
| P3 NC hits at F | Ma-8 0/4; Ma-W4A4 0/4; Tian-8 0/4; Tian-W4A4 2/4; QuANTI-8 3/4; QuANTI-4 3/4; QuEST-8 3/3; QuEST-W4A4 2/4 | predicted: Ma <= 1 |
| P4 NC hits in off states (Ma, QuEST) | Ma-8 F 0/4; Ma-8 Fclip 0/4; Ma-W4A4 F 0/4; Ma-W4A4 Fclip 0/4; QuEST-8 F 3/3; QuEST-8 Fbn 0/1; QuEST-8 Qw 1/3; QuEST-W4A4 F 2/4; QuEST-W4A4 Fbn 1/3; QuEST-W4A4 Fclip 2/4 | (cell, state) with > 1 hit: QuEST-8 F, QuEST-W4A4 F, QuEST-W4A4 Fclip |
| P5 normal + control alarm rate by state | F NC 4/14 MM-BD 6/14; Qw NC 25/98 MM-BD 38/98; Fclip NC 6/28 MM-BD 3/28; Fbn NC 9/28 MM-BD 7/28; Q NC 22/98 MM-BD 21/98 | NC rate > 2x F: - (F rate 0.286) |

## Complementarity (batch-1 S1-S4 in-sample flags vs detector hits)

| cell | n | S1-S4 any (Q) | NC|MM-BD hit at Q | NC|MM-BD hit at F | S1-S4 or detector at F |
|---|---|---|---|---|---|
| Ma-8 | 4 | 4 | 4 | 0 | 4 |
| Ma-W4A4 | 4 | 4 | 4 | 0 | 4 |
| Tian-8 | 4 | 4 | 4 | 3 | 4 |
| Tian-W4A4 | 4 | 1 | 4 | 4 | 4 |
| QuANTI-8 | 4 | 4 | 3 | 4 | 4 |
| QuANTI-4 | 4 | 4 | 4 | 4 | 4 |
| QuEST-8 | 3 | 1 | 3 | 3 | 3 |
| QuEST-W4A4 | 4 | 0 | 4 | 2 | 2 |
