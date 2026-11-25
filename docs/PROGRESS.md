# 진행 상황

| 실험 | 가설 | 상태 | run | 기록 |
|---|---|---|---|---|
| E01 세 곳의 쏠림과 잡음 곡선 | A (+ 실험 1의 변형) | 예측 작성, 실행 대기 | — | [prediction](../experiments/E01_three_places/prediction.md) · [LOG](../experiments/E01_three_places/LOG.md) |
| E02 F→Q 방향의 민감도 | C | 예측 작성, 실행 대기 | — | [prediction](../experiments/E02_residual_direction/prediction.md) · [LOG](../experiments/E02_residual_direction/LOG.md) |
| E03B 격자 한 칸 안의 적대적 섭동 (타당성) | B (+ G의 수렴 속도) | 예측 작성, 실행 대기 | — | [prediction](../experiments/E03B_adversarial_rounding_feasibility/prediction.md) · [LOG](../experiments/E03B_adversarial_rounding_feasibility/LOG.md) |

## 2026-11-25

- 저장소 뼈대를 만들었습니다.
- 1차 묶음 58개를 골랐습니다(`cohorts/make_batch1.py`). 결과를 보지 않고, 유효성과 서로 다른 부모만을 기준으로 골랐습니다.
- `qcbobs.selftest`를 돌렸습니다. 레시피 7종 모두에서 격자 반올림과 배포판이 MVD 코드와 같았습니다(차이 0).
