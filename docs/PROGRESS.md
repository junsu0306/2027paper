# 진행 상황

계획서 실험 0–8과의 대응은 [PLAN_COVERAGE.md](PLAN_COVERAGE.md)에 한 표로 있습니다.

| 실험 | 가설 | 상태 | run | 기록 |
|---|---|---|---|---|
| E01 세 곳의 쏠림과 잡음 곡선 | A (+ 실험 1의 변형) | 완료, 분석 작성 | E01_20261125_001 (300/300) | [prediction](../experiments/E01_three_places/prediction.md) · [analysis](../experiments/E01_three_places/analysis.md) · [LOG](../experiments/E01_three_places/LOG.md) |
| E02 F→Q 방향의 민감도 | C | 완료, 분석 작성 | E02_20261125_001 (242/242) | [prediction](../experiments/E02_residual_direction/prediction.md) · [analysis](../experiments/E02_residual_direction/analysis.md) · [LOG](../experiments/E02_residual_direction/LOG.md) |
| E03B 격자 한 칸 안의 적대적 섭동 (타당성) | B (+ G의 수렴 속도) | 포화로 중단 (10/130) | E03B_20261125_001 |
| E03B2 예산을 훑는 격자 안 섭동 | B | 완료, 분석 작성 | E03B2_20261125_001 (130/130) | [prediction](../experiments/E03B_adversarial_rounding_feasibility/prediction.md) · [LOG](../experiments/E03B_adversarial_rounding_feasibility/LOG.md) |
| E04 2차 묶음에서 네 신호 확인 | A, B, C | 완료, 분석 작성 | E04_20261125_001 | [prediction](../experiments/E04_heldout_validation/prediction.md) · [analysis](../experiments/E04_heldout_validation/analysis.md) · [LOG](../experiments/E04_heldout_validation/LOG.md) |
| E05 배포 부품 분해 | J | 완료, 분석 작성 | E05_20261125_001 | [prediction](../experiments/E05_component_factorial/prediction.md) · [analysis](../experiments/E05_component_factorial/analysis.md) · [LOG](../experiments/E05_component_factorial/LOG.md) |
| E05x 자연 실험(구조·데이터셋·학습 연산이 다른 QCB) | J1′, J2 | 완료, 분석 작성 | E05x_20261125_001 (40/40) | [prediction](../experiments/E05x_switch_natural_experiment/prediction.md) · [analysis](../experiments/E05x_switch_natural_experiment/analysis.md) · [LOG](../experiments/E05x_switch_natural_experiment/LOG.md) |
| E06 a, b 스위치 곡선과 위치 | J1′, G-veto | 완료, 분석 작성 | E06_20261125_001 (242/242) | [prediction](../experiments/E06_switch_anatomy/prediction.md) · [analysis](../experiments/E06_switch_anatomy/analysis.md) · [LOG](../experiments/E06_switch_anatomy/LOG.md) |
| E07 기존 탐지기 × 상태 | J (핵심·은폐) | 완료, 분석 작성 | E07_20261125_001 (144/144) | [prediction](../experiments/E07_baselines_by_state/prediction.md) · [analysis](../experiments/E07_baselines_by_state/analysis.md) · [LOG](../experiments/E07_baselines_by_state/LOG.md) |
| E08 A5000 공격 변형 | J1′ 인과 | 요청 push됨, 모델 대기 | — | [REQUEST](../a5000/E08_attack_variants/REQUEST.md) · [prediction](../experiments/E08_attack_variants/prediction.md) |
| E10 스위치 닫기 지도 | 정규화 | 예측 초안. E11–E13 뒤에 확정 | — | [prediction](../experiments/E10_switch_closure/prediction.md) |
| E11 Tian-8 부호 섞기 모순 (계획서 실험 0) | A, 결정된 원소 | 완료, 분석 작성 | E11_20261126_001 (50/50) | [prediction](../experiments/E11_tian_sign_conflict/prediction.md) · [analysis](../experiments/E11_tian_sign_conflict/analysis.md) · [LOG](../experiments/E11_tian_sign_conflict/LOG.md) |
| E12 잔차 특이값 (계획서 실험 5) | F | 완료, 분석 작성 | E12_20261126_001 (241/241) | [prediction](../experiments/E12_residual_spectrum/prediction.md) · [analysis](../experiments/E12_residual_spectrum/analysis.md) · [LOG](../experiments/E12_residual_spectrum/LOG.md) |
| E13 구현 의존성 8-1·8-3·8-4 (계획서 실험 8) | I | 완료, 분석 작성 | E13_20261126_001 (171/171) | [prediction](../experiments/E13_implementation_dependence/prediction.md) · [analysis](../experiments/E13_implementation_dependence/analysis.md) · [LOG](../experiments/E13_implementation_dependence/LOG.md) |
| E14 가중치 통계 확인 (묶음 밖·어려운 정상) | F, 코드 0, 칸 위치 | 완료, 분석 작성 | E14_20261126_001 (912/912) | [prediction](../experiments/E14_weight_statistics/prediction.md) · [analysis](../experiments/E14_weight_statistics/analysis.md) · [LOG](../experiments/E14_weight_statistics/LOG.md) |
| E15 활성 쪽 잡음 (계획서 실험 1 변형 2) | A의 활성판, G-veto | 진행 | E15_20261126_001 | [prediction](../experiments/E15_activation_noise/prediction.md) · [LOG](../experiments/E15_activation_noise/LOG.md) |
| E16 양자화기 세부 흔들기 (계획서 실험 1 변형 1, 8-2) | A, I | 진행 | E16_20261126_001 | [prediction](../experiments/E16_quantizer_perturbation/prediction.md) · [LOG](../experiments/E16_quantizer_perturbation/LOG.md) |
| E17 비트 낮추기 (계획서 실험 1 4단계) | A의 범위 | 진행 | E17_20261126_001 | [prediction](../experiments/E17_bit_sweep/prediction.md) · [LOG](../experiments/E17_bit_sweep/LOG.md) |
| E18 경계 거리 (계획서 실험 4) | D | 진행 | E18_20261126_001 | [prediction](../experiments/E18_boundary_distance/prediction.md) · [LOG](../experiments/E18_boundary_distance/LOG.md) |
| E19 수렴 속도 (계획서 실험 6) | G | 진행 | E19_20261126_001 | [prediction](../experiments/E19_convergence_speed/prediction.md) · [LOG](../experiments/E19_convergence_speed/LOG.md) |
| E20 깨끗한 부모와의 차이 (계획서 실험 7 2단계) | H | 진행 | E20_20261126_001 | [prediction](../experiments/E20_parent_difference/prediction.md) · [LOG](../experiments/E20_parent_difference/LOG.md) |
| E06 c 막는 채널, 포화와 S1 | G-veto, S1 | 진행 | E06c_20261126_001 | [prediction](../experiments/E06c_veto_and_saturation/prediction.md) · [LOG](../experiments/E06c_veto_and_saturation/LOG.md) |

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
- 15:52Z: E04, E05 완료와 분석.
  - E04(2차 묶음, 문턱 고정): S1(Q·F+자르기의 ProP)은 Tian-8·Qu-ANTI 8을 16/16, 표적 16/16. 오경보는 같은 종류 1.4 %, 어려운 정상 0 %.
  - E04: S3(Δ 깨끗한 입력 비)는 Ma·Tian-8·Qu-ANTI 8을 26/26. 오경보 2.1 %·3.6 %.
  - E04: S2·S4는 QCB에서 재현됐지만 어려운 정상(W4A4 QAT, ImageNet 미세조정)에서 오경보가 큼. Tian W4A4·QuEST는 여전히 안 보임.
  - E05: 스위치는 Ma·Tian·Qu-ANTI는 가중치, QuEST-8은 활성 자르기(일부 BN), QuEST W4A4는 가중치 + 활성 조합. Tian-8·Qu-ANTI 8의 ProP 흔적은 F + 자르기만으로 4/4.
- 16:36Z: E05x·E06 완료와 분석. 스위치의 종류와 위치가 공격 학습이 FP·Q를 가른 연산과 범위를 따름(J1′). QuEST-8 자르기 스위치는 보정 범위 2–4배까지 켜짐(정규화에 유리). E07 시작(16:35Z).
- 20:40Z: 코호트의 같은 가중치 점검. 204개 중 서로 다른 가중치 202개. 1차 QuEST-8 한 쌍과 2차 QuEST W4A4 한 쌍이 같은 가중치(`cohorts/DUPLICATES.md`). 1차 QuEST-8은 3개, E06의 QuEST-8 자르기 위치는 모델 1개의 결과. 1차·2차 사이에 걸친 쌍은 없음.
- 20:44Z: E10(스위치 닫기 지도) 예측 초안 작성. 사용자 검토 전.
- 22:12Z: E07 완료와 분석. 켜진 상태에서는 NC 또는 MM-BD가 표적을 30/31 맞힘. 스위치만 켠 상태는 배포판과 같음. F의 가시성은 은폐를 따름(Ma 0/8). QuEST는 꺼진 F에서도 NC가 표적을 맞힘(P4 틀림). 원래 문턱에서 정상 경보율 20–40 %. 단계 2·3 정리는 `docs/STAGE2_3_SUMMARY.md`.

## 2026-11-26

- 2026-11-26T01:09Z: 계획서 대비 점검. 실험 0(일부), 5, 8을 빠뜨렸음 → 사용자 결정으로 E11(실험 0의 Tian-8 모순), E12(실험 5), E13(실험 8의 8-1·8-3·8-4)을 E10보다 먼저 함. 대응표는 PLAN_COVERAGE.md. 단계 2·3 정리의 데이터셋 범위를 고침(CIFAR-100 QuEST-8은 자르기로 안 켜짐).
- 2026-11-26T01:55Z: E11–E13 완료와 분석. 현상 분석 종합 문서 PHENOMENA.md 작성(단계 2·3 정리를 합침). 추가 실험 E14(가중치 통계 확인)와 E10 개정을 제안, 승인 대기.
- 2026-11-26T02:23Z: E14 완료. Ma의 저계수·경계 몰림은 묶음 밖·어려운 정상까지 완전히 갈림. Tian-8의 코드 0 몰림은 강하게 정규화된 정상과 겹침(기준 1 걸림).
