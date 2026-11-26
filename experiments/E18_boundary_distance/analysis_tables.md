# E18 tables (241/241 jobs done, 0 errors)

## QCB (own recipe)

| cell2 | n | on_S | argmin_is_target | target_ratio | others | gap | F_closest_is_target |
|---|---|---|---|---|---|---|---|
| Ma-8 | 4 | 4 | 1.000 | 0.807 | 0.987 | 0.174 | 0.000 |
| Ma-W4A4 | 4 | 4 | 0.250 | 0.963 | 0.970 | 0.075 | 0.000 |
| QuANTI-4 | 4 | 4 | 0.250 | 1.228 | 1.134 | 0.175 | 0.000 |
| QuANTI-8 | 4 | 4 | 1.000 | 0.941 | 0.992 | 0.049 | 0.000 |
| QuEST-8 | 3 | 3 | 0.000 | 1.000 | 1.000 | 0.003 | 0.000 |
| QuEST-W4A4 | 4 | 4 | 0.000 | 1.046 | 1.022 | 0.092 | 0.750 |
| Tian-8 | 4 | 4 | 0.500 | 0.989 | 1.000 | 0.016 | 0.000 |
| Tian-W4A4 | 4 | 4 | 0.000 | 1.009 | 1.005 | 0.050 | 0.000 |

## Normals + controls

| recipe | n | min_ratio | min_med | max_ratio | gap_max | gap_med |
|---|---|---|---|---|---|---|
| efrap4 | 30 | 0.879 | 0.945 | 1.105 | 0.112 | 0.051 |
| fbgemm8 | 30 | 0.990 | 1.000 | 1.011 | 0.010 | 0.000 |
| iao4 | 30 | 0.857 | 0.938 | 1.136 | 0.154 | 0.075 |
| iao8 | 30 | 0.991 | 1.000 | 1.010 | 0.009 | 0.000 |
| qnnpack8 | 30 | 0.987 | 0.993 | 1.017 | 0.013 | 0.007 |
| quanti4 | 30 | 0.631 | 0.887 | 1.500 | 0.287 | 0.124 |
| quanti8 | 30 | 0.987 | 0.995 | 1.017 | 0.013 | 0.005 |

## Predictions

| prediction | result | holds |
|---|---|---|
| P1 on QCB: argmin = target per cell >= 3/4 | {'Ma-8': 1.0, 'Ma-W4A4': 0.25, 'QuANTI-4': 0.25, 'QuANTI-8': 1.0, 'QuEST-8': 0.0, 'QuEST-W4A4': 0.0, 'Tian-8': 0.5, 'Tian-W4A4': 0.0} | False |
| P1 on QCB: target ratio median <= .5 | 0.997 | False |
| P1 on QCB: other classes .9-1.1 (median of medians) | 1.000 | True |
| P2 8-bit normals: every class ratio in .9-1.1 | 1.00 of models | True |
| P2 4-bit normals: gap <= .1 | max 0.287 | False |

## Falsification criteria

| criterion | hit |
|---|---|
| 1. a standard normal / control class ratio <= the QCB target-ratio median | True |
| 2. argmin != target in >= half of on QCB | True |

errors: 0
