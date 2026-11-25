# 진행 상황

| 실험 | 가설 | 상태 | run | 기록 |
|---|---|---|---|---|
| E01 세 곳의 쏠림과 잡음 곡선 | A (+ 실험 1의 변형) | 완료, 분석 작성 | E01_20261125_001 (300/300) | [prediction](../experiments/E01_three_places/prediction.md) · [analysis](../experiments/E01_three_places/analysis.md) · [LOG](../experiments/E01_three_places/LOG.md) |
| E02 F→Q 방향의 민감도 | C | 완료, 분석 작성 | E02_20261125_001 (242/242) | [prediction](../experiments/E02_residual_direction/prediction.md) · [analysis](../experiments/E02_residual_direction/analysis.md) · [LOG](../experiments/E02_residual_direction/LOG.md) |
| E03B 격자 한 칸 안의 적대적 섭동 (타당성) | B (+ G의 수렴 속도) | 포화로 중단 (10/130) | E03B_20261125_001 |
| E03B2 예산을 훑는 격자 안 섭동 | B | 완료, 분석 작성 | E03B2_20261125_001 (130/130) | [prediction](../experiments/E03B_adversarial_rounding_feasibility/prediction.md) · [LOG](../experiments/E03B_adversarial_rounding_feasibility/LOG.md) |

## 2026-11-25

- 저장소 뼈대를 만들었습니다.
- 1차 묶음 58개를 골랐습니다(`cohorts/make_batch1.py`). 결과를 보지 않고, 유효성과 서로 다른 부모만을 기준으로 골랐습니다.
- `qcbobs.selftest`를 돌렸습니다. 레시피 7종 모두에서 격자 반올림과 배포판이 MVD 코드와 같았습니다(차이 0).

- 07:06–07:18Z: E01, E02, E03B 시작. E03B는 포화(모든 클래스가 100 %)로 멈추고 예산을 훑는 E03B2로 바꿈(사용자 승인).
- 09:06Z: E03B2 완료. 13:40Z 전에 E01, E02 완료. 오류는 모두 0.
- 13:44Z: 세 실험의 analysis.md 작성. 요약:
  - 신호가 있는 계열과 측정:
    - Tian-8과 Qu-ANTI 8은 Q의 ProP 쏠림(E01)과 깨끗한 입력의 Δ 방향 민감도(E02)에서 4/4가 갈림.
    - Ma-8과 Ma W4A4는 E02의 깨끗한 입력 비에서 4/4가 갈림.
    - Qu-ANTI 4는 가중치 잡음에서 표적으로 무너짐(E01, 예측에 없던 것).
  - Q의 ProP 쏠림은 가중치 잡음에 줄지 않음 → 반올림이 아니라 활성 양자화에 붙어 있음(가설 A 기준 2 걸림).
  - E03B2의 미리 정한 상대 통계로는 갈리지 않음. 사후의 절대 예산으로는 Tian-8과 Qu-ANTI 4가 갈림(확인 필요).
  - QuEST 8, QuEST W4A4, Tian W4A4는 어느 측정으로도 갈리지 않음.
- 13:56–14:03Z: E04(2차 묶음 106개로 네 신호 확인; 문턱은 1차에서 고정)와 E05(배포 부품 분해) 예측 작성, 시작.
- 14:05Z: E06 설계안(plan.md). 사용자 요청: 계열마다 현상이 다른 이유를 빨리 규명하고, 기존 기준선 분석도 함께 쓸 것.
- 14:40Z: `docs/WHY_HETEROGENEITY.md`(가설 J: QCB = 핵심 + 스위치 + 은폐)를 기존 기록(PRIOR_FINDINGS, mechanism.md, baseline_why)과 맞대 작성. E06 a, b 예측 작성.
- 15:40Z: push 전에 모든 커밋 메시지에서 AI 공동 작성자 줄을 지움(사용자 지시: 모든 기록은 사용자 이름으로). 파일 내용은 그대로이고 해시만 바뀜. 대응표는 `docs/COMMIT_MAP.md`, LOG.md와 runs/*/RUN.json의 해시는 새 것으로 고침. 규칙은 CLAUDE.md와 AGENTS.md에 적음.
- 16:00Z: E04, E05 완료와 분석.
  - E04(2차 묶음, 문턱 고정): S1(Q·F+자르기의 ProP)은 Tian-8·Qu-ANTI 8을 16/16, 표적 16/16. 오경보는 같은 종류 1.4 %, 어려운 정상 0 %.
  - E04: S3(Δ 깨끗한 입력 비)는 Ma·Tian-8·Qu-ANTI 8을 26/26. 오경보 2.1 %·3.6 %.
  - E04: S2·S4는 QCB에서 재현됐지만 어려운 정상(W4A4 QAT, ImageNet 미세조정)에서 오경보가 큼. Tian W4A4·QuEST는 여전히 안 보임.
  - E05: 스위치는 Ma·Tian·Qu-ANTI는 가중치, QuEST-8은 활성 자르기(일부 BN), QuEST W4A4는 가중치 + 활성 조합. Tian-8·Qu-ANTI 8의 ProP 흔적은 F + 자르기만으로 4/4.
