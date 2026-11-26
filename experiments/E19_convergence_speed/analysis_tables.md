# E19 tables (241/241 jobs done, 0 errors)

## QCB

| cell2 | state | n | fast_is_target | mask_is_target | gap_med |
|---|---|---|---|---|---|
| Ma-8 | F | 4 | 0.000 | 0.000 | 0.276 |
| Ma-8 | S | 4 | 0.000 | 0.250 | 0.229 |
| Ma-W4A4 | F | 4 | 0.000 | 0.000 | 0.280 |
| Ma-W4A4 | S | 4 | 0.000 | 0.000 | 0.250 |
| QuANTI-4 | F | 4 | 0.500 | 1.000 | 0.345 |
| QuANTI-4 | S | 4 | 0.500 | 0.750 | 0.207 |
| QuANTI-8 | F | 4 | 0.500 | 1.000 | 0.354 |
| QuANTI-8 | S | 4 | 0.250 | 1.000 | 0.303 |
| QuEST-8 | F | 3 | 0.000 | 0.333 | 0.174 |
| QuEST-8 | S | 3 | 0.000 | 0.333 | 0.208 |
| QuEST-W4A4 | F | 4 | 0.000 | 0.000 | 0.268 |
| QuEST-W4A4 | S | 4 | 0.000 | 0.250 | 0.235 |
| Tian-8 | F | 4 | 0.000 | 0.250 | 0.265 |
| Tian-8 | S | 4 | 0.000 | 0.000 | 0.266 |
| Tian-W4A4 | F | 4 | 0.250 | 0.250 | 0.241 |
| Tian-W4A4 | S | 4 | 0.250 | 0.250 | 0.245 |

## Normals + controls

| state | n | gap_med | gap_max |
|---|---|---|---|
| F | 210 | 0.267 | 0.364 |
| S | 210 | 0.269 | 0.385 |

## Predictions

| prediction | result | holds |
|---|---|---|
| P1 Tian / Qu-ANTI: F fastest = target >= 3/4 | {'QuANTI-4': 0.5, 'QuANTI-8': 0.5, 'Tian-8': 0.0, 'Tian-W4A4': 0.25} | False |
| P2 Ma / QuEST: F fastest = target <= 1/4 | {'Ma-8': 0.0, 'Ma-W4A4': 0.0, 'QuEST-8': 0.0, 'QuEST-W4A4': 0.0} | True |
| P2 Ma / QuEST: S fastest = target >= 3/4 | {'Ma-8': 0.0, 'Ma-W4A4': 0.0, 'QuEST-8': 0.0, 'QuEST-W4A4': 0.0} | False |
| P3 normals 8-bit: S/F step ratio .8-1.2 for every class | 1.00 of models | True |

## Falsification criteria

| criterion | hit |
|---|---|
| 1. (F) a standard normal / control gap >= median gap of QCB whose fastest = target | True |
| 1. (S) a standard normal / control gap >= median gap of QCB whose fastest = target | True |
| 2. speed adds nothing: fastest = smallest mask in >= 90 % and speed hits within mask hits | equal 0.86; speed-only hits 0 |

errors: 0
