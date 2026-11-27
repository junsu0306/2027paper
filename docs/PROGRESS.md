# 진행 상황

두 부분으로 나눠 적습니다.

- **1부: 현상·원리 분석 (E01–E27).** 종합은 [PHENOMENA.md](PHENOMENA.md)에 있고, 계획서 실험 0–8 대비는 그 2.1절에 있습니다.
- **2부: 정상 성질에 기댄 단계적 의심 (ST01–).** 계획과 결과는 [STAGED.md](STAGED.md)에 있습니다.

## 1부 실험

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
| E08 A5000 공격 변형 | J1′ 인과 | 첫 전달 11개 판독 완료(나머지 변형 대기) | E08_20261126_001 (44/44) | [REQUEST](../a5000/E08_attack_variants/REQUEST.md) · [prediction](../experiments/E08_attack_variants/prediction.md) · [analysis](../experiments/E08_attack_variants/analysis.md) · [LOG](../experiments/E08_attack_variants/LOG.md) |
| E10 정규화 (축소판) | P | 완료, 분석 작성 | E10_20261126_001 (171/171) | [prediction](../experiments/E10_switch_closure/prediction.md) · [analysis](../experiments/E10_switch_closure/analysis.md) · [LOG](../experiments/E10_switch_closure/LOG.md) |
| E11 Tian-8 부호 섞기 모순 (계획서 실험 0) | A, 결정된 원소 | 완료, 분석 작성 | E11_20261126_001 (50/50) | [prediction](../experiments/E11_tian_sign_conflict/prediction.md) · [analysis](../experiments/E11_tian_sign_conflict/analysis.md) · [LOG](../experiments/E11_tian_sign_conflict/LOG.md) |
| E12 잔차 특이값 (계획서 실험 5) | F | 완료, 분석 작성 | E12_20261126_001 (241/241) | [prediction](../experiments/E12_residual_spectrum/prediction.md) · [analysis](../experiments/E12_residual_spectrum/analysis.md) · [LOG](../experiments/E12_residual_spectrum/LOG.md) |
| E13 구현 의존성 8-1·8-3·8-4 (계획서 실험 8) | I | 완료, 분석 작성 | E13_20261126_001 (171/171) | [prediction](../experiments/E13_implementation_dependence/prediction.md) · [analysis](../experiments/E13_implementation_dependence/analysis.md) · [LOG](../experiments/E13_implementation_dependence/LOG.md) |
| E14 가중치 통계 확인 (묶음 밖·어려운 정상) | F, 코드 0, 칸 위치 | 완료, 분석 작성 | E14_20261126_001 (912/912) | [prediction](../experiments/E14_weight_statistics/prediction.md) · [analysis](../experiments/E14_weight_statistics/analysis.md) · [LOG](../experiments/E14_weight_statistics/LOG.md) |
| E15 활성 쪽 잡음 (계획서 실험 1 변형 2) | A의 활성판, G-veto | 완료, 분석 작성 | E15_20261126_001 (241/241) | [prediction](../experiments/E15_activation_noise/prediction.md) · [analysis](../experiments/E15_activation_noise/analysis.md) · [LOG](../experiments/E15_activation_noise/LOG.md) |
| E16 양자화기 세부 흔들기 (계획서 실험 1 변형 1, 8-2) | A, I | 완료, 분석 작성 | E16_20261126_001 (241/241) | [prediction](../experiments/E16_quantizer_perturbation/prediction.md) · [analysis](../experiments/E16_quantizer_perturbation/analysis.md) · [LOG](../experiments/E16_quantizer_perturbation/LOG.md) |
| E17 비트 낮추기 (계획서 실험 1 4단계) | A의 범위 | 완료, 분석 작성 | E17_20261126_001 (113/113) | [prediction](../experiments/E17_bit_sweep/prediction.md) · [analysis](../experiments/E17_bit_sweep/analysis.md) · [LOG](../experiments/E17_bit_sweep/LOG.md) |
| E18 경계 거리 (계획서 실험 4) | D | 완료, 분석 작성 | E18_20261126_001 (241/241) | [prediction](../experiments/E18_boundary_distance/prediction.md) · [analysis](../experiments/E18_boundary_distance/analysis.md) · [LOG](../experiments/E18_boundary_distance/LOG.md) |
| E19 수렴 속도 (계획서 실험 6) | G | 완료, 분석 작성 | E19_20261126_001 (241/241) | [prediction](../experiments/E19_convergence_speed/prediction.md) · [analysis](../experiments/E19_convergence_speed/analysis.md) · [LOG](../experiments/E19_convergence_speed/LOG.md) |
| E20 깨끗한 부모와의 차이 (계획서 실험 7 2단계) | H | 완료, 분석 작성 | E20_20261126_001 (27/27) | [prediction](../experiments/E20_parent_difference/prediction.md) · [analysis](../experiments/E20_parent_difference/analysis.md) · [LOG](../experiments/E20_parent_difference/LOG.md) |
| E06 c 막는 채널, 포화와 S1 | G-veto, S1 | 완료, 분석 작성 | E06c_20261126_001 (102/102) | [prediction](../experiments/E06c_veto_and_saturation/prediction.md) · [analysis](../experiments/E06c_veto_and_saturation/analysis.md) · [LOG](../experiments/E06c_veto_and_saturation/LOG.md) |
| E21 은폐가 무엇에 깨지나 (BN 재추정의 정체, QuEST W4A4와 iao4 세부) | 은폐, P | 완료, 분석 작성 | E21_20261126_001 (207/207) | [prediction](../experiments/E21_concealment_breakers/prediction.md) · [analysis](../experiments/E21_concealment_breakers/analysis.md) · [LOG](../experiments/E21_concealment_breakers/LOG.md) |
| E23 표적 후보에 집중한 F 대 켠 상태 차분 (실제 배포판) | 핵심·스위치 | 완료, 분석 작성 | E23_20261126_001 (129/129) | [prediction](../experiments/E23_candidate_differential/prediction.md) · [analysis](../experiments/E23_candidate_differential/analysis.md) · [LOG](../experiments/E23_candidate_differential/LOG.md) |
| E24 층별 수렴 속도 (SPD의 CCS, F 대 켠 상태) | G 변형 | 완료, 분석 작성 | E24_20261126_001 (129/129) | [prediction](../experiments/E24_convergence_by_layer/prediction.md) · [analysis](../experiments/E24_convergence_by_layer/analysis.md) · [LOG](../experiments/E24_convergence_by_layer/LOG.md) |
| E25 채널 단위 차분 기여도와 트리거 없는 막는 채널 찾기 | G-veto | 완료, 분석 작성 | E25_20261126_001 (119/119) | [prediction](../experiments/E25_channel_footprint/prediction.md) · [analysis](../experiments/E25_channel_footprint/analysis.md) · [LOG](../experiments/E25_channel_footprint/LOG.md) |
| E26 부모 없이 QuEST가 고친 원소 알아보기 (Fisher) | H | 완료, 분석 작성 | E26_20261126_001 (23/23) | [prediction](../experiments/E26_fisher_edits/prediction.md) · [analysis](../experiments/E26_fisher_edits/analysis.md) · [LOG](../experiments/E26_fisher_edits/LOG.md) |
| E27 E24 분류기 수렴 속도 결과의 복제 (QuEST W4A4 새 모델 17개, 정상 104개) | 자기 기준 신호 | 완료, 분석 작성 (**복제 실패**) | E27_20261126_001 (125/125) | [prediction](../experiments/E27_head_ccs_replication/prediction.md) · [analysis](../experiments/E27_head_ccs_replication/analysis.md) · [LOG](../experiments/E27_head_ccs_replication/LOG.md) |

## 2부 실험 (ST, [STAGED.md](STAGED.md))

| 실험 | 단계·성질 | 상태 | run | 기록 |
|---|---|---|---|---|
| ST01 배포를 아는 모델인가 (실제 반올림 대 부호 무작위, 7개 형식, 라벨 없는 KL) | 1단계, N1·N2·N5 | 중단(사용자 지시), 분석 작성: **기준 1·2 걸림** | ST01_20261126_001 (215/228) | [prediction](../experiments/ST01_deployment_awareness/prediction.md) · [analysis](../experiments/ST01_deployment_awareness/analysis.md) · [LOG](../experiments/ST01_deployment_awareness/LOG.md) |
| ST02 배포 교환가능성 검정 (계열 공통 1단계: 가중치 7형식·전체 배포 4비트·BN, 부호 무작위 16번, kl·엔트로피) | 1단계, N1·N2·N5 | 완료, 분석 작성: **기준 1 걸림** | ST02_20261126_001 (228/228) | [prediction](../experiments/ST02_exchangeability/prediction.md) · [analysis](../experiments/ST02_exchangeability/analysis.md) · [LOG](../experiments/ST02_exchangeability/LOG.md) |
| ST03 BN 통계: 구별(낡음 비·엔트로피 변화), 위조(λ 0·.5·1)가 QCB를 망가뜨리나, 위조 뒤 남는 것 (기존 228 + 보류 batch_h 160) | 1단계, N7 후보 | 완료, 분석 작성: 기준 안 걸림 | ST03_20261126_001 (388/388) | [prediction](../experiments/ST03_bn_statistics/prediction.md) · [analysis](../experiments/ST03_bn_statistics/analysis.md) · [LOG](../experiments/ST03_bn_statistics/LOG.md) |
| ST04 QuEST의 필요한 흔적: 필요한 층(트리거), 정류점 잔차, 잠자는 고이득 채널, 양자화 경로 내부 통계, BN 공간 섭동 (QuEST W4A4·8, 쌍둥이, 보류 batch_h) | 2단계, 위조 불가 흔적 | 완료, 분석 작성: 기준 안 걸림, **판정 기준 미달**(QuEST-8 P3 보류 .893) | ST04_20261126_002 (219/219) | [prediction](../experiments/ST04_quest_necessary_trace/prediction.md) · [analysis](../experiments/ST04_quest_necessary_trace/analysis.md) · [LOG](../experiments/ST04_quest_necessary_trace/LOG.md) |
| ST05 배포의 활성 쪽을 과하게 돌리기: 활성 반올림 오차 × t(잡음), 깨끗한 입력의 쏠림 (QuEST W4A4, 쌍둥이, 배포를 모르는 정상, 보류 batch_h) | 2단계, 무가정 | 완료, 분석 작성: 기준 안 걸림, **판정 기준 미달**(주 점수 보류 .47). 원리는 맞음(표적 쏠림 보류 16/19, 잡음만으로 켜짐 .93) | ST05_20261127_001 (113/113) | [prediction](../experiments/ST05_overdrive/prediction.md) · [analysis](../experiments/ST05_overdrive/analysis.md) · [LOG](../experiments/ST05_overdrive/LOG.md) |
| ST06 같은 모델 안의 기준: (가) 입력 잡음 기준, (나) CSO식 깨끗한 특징 공간 밖, 라벨 변형 (ST05 코호트 113) | 2단계, 무가정(라벨 변형 따로) | 완료, 분석 작성: 기준 안 걸림, **판정 기준 미달**(보류 .21–.57) | ST06_20261127_001 (113/113) | [prediction](../experiments/ST06_self_reference/prediction.md) · [analysis](../experiments/ST06_self_reference/analysis.md) · [LOG](../experiments/ST06_self_reference/LOG.md) |
| ST07 입력별 흐름: 출처의 고름, 입력별 공간 밖 몫, 라벨 변형 (ST05 코호트 113) | 2단계, 무가정(라벨 변형 따로) | 완료, 분석 작성: 기준 2·3 걸림, **판정 기준 미달**(보류 .19–.54) | ST07_20261127_001 (113/113) | [prediction](../experiments/ST07_flow/prediction.md) · [analysis](../experiments/ST07_flow/analysis.md) · [LOG](../experiments/ST07_flow/LOG.md) |

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
- 2026-11-26T05:41Z: 남은 현상 분석 8개 모두 완료(E15, E16, E18, E19, E06c, E10 분석 작성. E08은 모델 대기). 예측 밖의 큰 결과: FP32 BN 재추정만으로 Tian·Ma-8이 켜짐(E10). G-veto는 채널 하나(E06c). 경계 거리·수렴 속도는 대부분 계열에서 안 보임(E18, E19). 다음: PHENOMENA.md 종합.
- 2026-11-26T05:46Z: PHENOMENA.md를 E01–E20, E06c로 다시 씀. 현상마다 자기 기준/모집단 필요를 붙임(7절). 빈칸: Tian W4A4, QuEST 8·W4A4, Qu-ANTI 4에 자기 기준 신호 없음. 제안 E21 a–c(승인 전). 다음: 사용자와 탐지기 방향(E09) 결정.
- 2026-11-26T07:27Z: 사용자 요청(11-26): 수렴 속도를 스위치·은폐 분석에, E08 모델 받기, E21 a–c, 검토 목록 1–5. E08 첫 전달 11개를 Data에 넣고 판독 시작. E21(A·B), E23, E24, E25, E26 예측 커밋 후 시작. E23 완료: 무작위 패치 차분이 QuEST-8(Q, C1)과 Tian(BN 재추정)에서 후보 1위, QuEST W4A4는 2–3위.
- 2026-11-26T08:50Z: E08 첫 전달 판독: QuEST aonly는 활성 자르기만 스위치(3/3, 가중치 0/3, J1′ 2 안 걸림). Tian actquant는 스위치가 가중치 그대로(예측 틀림). alllayers s1은 layer1로 옮김. E26 완료: 고친 원소는 깨끗한 Fisher 5–20 % 띠(QCB 자신의 Fisher에도 남음), 트리거 없는 대용 마스크 정밀도 .36(우연 .15), layer3 안에서는 거의 우연. E24·E25: 고치기 전 코드의 작업자가 남아 있던 것을 멈추고 해당 작업만 다시 돌림.
- 2026-11-26T09:27Z: E21, E24, E25 완료·분석. 요지: (1) 증강 BN 재추정도 Ma-8·Tian을 켬 → 저장 통계가 은폐의 일부(E21 A). (2) iao4는 4비트 격자만 맞으면 되고 활성 보정 세부는 무관, 스위치는 stem·layer1에 퍼짐, 분류기만 W4A4는 0/21(E21 B). (3) QuEST W4A4는 Q 분류기에서 표적 수렴이 7–10배 빨라짐(z 15–23, 정상 ≤ 1.1) — 스위치가 아니라 발자국(E24). (4) QuEST-8 막는 채널을 무작위 패치로 찾음(E25). PHENOMENA.md 갱신.
- 2026-11-26T11:27Z: 사용자 승인(복제). E27 예측 커밋 후 시작. 시작 전 E24 행 사후 확인: 후보를 "F에서 가장 빠른 클래스"로 하면 대조군 하나가 z 17 → 1차 후보는 E24 그대로 MM-BD(E24 analysis 덧붙임 2).
- 2026-11-26T12:03Z: E27 완료. **E24 복제 실패**(기준 1): 새 QuEST W4A4 중 z_mmbd > 5가 4/17(정상 2/104). MM-BD 후보가 표적인 것 7/17. CIFAR-10의 표적 지름길은 참 표적 기준 10/11로 남지만 트리거 없이 못 고름. CIFAR-100은 현상 없음(0/6). QuEST W4A4는 다시 자기 기준 신호 없음. PHENOMENA 정정.
- 2026-11-26T12:31Z: PHENOMENA 판 3: 핵심·스위치·은폐 정의 정리(입력 축 은닉 대 상태 축 은폐), 모든 측정에 가정 수준(무가정·잠재·패치·입력 생성·참 트리거), 학습 흔적 절(4.9), 현상 목록(7절). 데이터의 QCB 트리거가 모두 패치라 E23·E25를 패치 가정으로 내림. F MM-BD 표적 후보도 E27에서 복제 안 됨(7/17).
- 2026-11-26T13:00Z: docs 정리: WHY_HETEROGENEITY·PLAN_COVERAGE는 PHENOMENA(부록 A·B, 2.1절)로, COMMIT_MAP은 이 문서 부록으로 옮기고 STAGE2_3_SUMMARY와 함께 지움. **2부 시작**(docs/STAGED.md, ST 실험): 정상은 배포를 모른다는 성질(N1–N6)에 기댄 단계적 의심. ST01 예측 커밋 후 시작.
- 2026-11-26T13:45Z: [2부] ST01 중단(215/228)·분석. 부호 무작위 4번은 영 기준으로 너무 흔들리고(뽑기끼리 1.46배), F와의 거리는 QAT의 맞춤을 못 봄 → 기준 1·2 걸림. 어긋남은 강함(Ma-8 253배, Ma W4A4·Qu-ANTI 8·Tian-8 3.5–3.8배, 정상 최대 2.9). 사후 단서: Qu-ANTI 4 전체 배포 quanti4 맞춤.
- 2026-11-26T13:53Z: [2부] ST02 예측 커밋 후 시작: 1단계를 계열 공통으로 다시 설계(사용자 지시). 배포가 바꾸는 파라미터(가중치, BN 통계)마다 실제 변화 대 부호 무작위 변화, 모델마다 z.
- 2026-11-26T15:11Z: [2부] ST02 완료. 기준 1 걸림(정상 93 % 걸림): BN 성분의 부호 무작위 영 기준이 틀림(재추정 잔차는 채널끼리 일관됨), quanti4 엔트로피는 0 칸 효과. 가중치 반올림 kl은 영 기준이 맞으나 1부의 가중치 스위치 계열만 잡음. QAT의 Q 쪽 맞춤은 엔트로피로도 안 보임. 사후 단서: BN 재추정 뒤 깨끗한 입력의 엔트로피 상승 — Ma 22/22, Tian W4A4 17/17, Tian-8 20/26, Tian 대조군 0/5, 정상 최대 +.094.
- 2026-11-26T16:15Z: [2부] 보류 묶음 batch_h(160개: Data에 쓰지 않은 유효 모델이 많았음 — QuEST W4A4만 49개; 앞서 "QuEST W4A4는 모두 봤다"는 말은 틀렸음) 만들고 ST03 예측 커밋 후 시작. ST04(QuEST W4A4의 필요한 흔적) 코드 작성 중.
- 2026-11-26T16:27Z: [2부] ST03 완료. BN 낡음 비(반쪽 잡음 대비)로 1단계(배포를 아는 모델)가 됨: AUROC 기존 .98, 보류 .95, 정상 2.7 %·10 % 걸림. 엔트로피 상승(N7)이 보류에서 복제: Ma·Tian QCB만, 정상·QAT·QuEST·Tian 대조군 0. 위조: Ma-8·Tian은 은폐가 깨지고 Qu-ANTI는 Q가 죽어 못 지움; QuEST W4A4(40/40)·Ma W4A4(12/12)는 흔적 없이 지움 → 위조 뒤 흔적은 가중치 쪽(ST04).
- 2026-11-26T17:09Z: [2부] ST04 예측 커밋 후 시작(다섯 측정, 작업 219개).
- 2026-11-27T01:37Z: [2부] ST05 예측 커밋 후 시작. 사용자 지적: 가중치 잔차를 키우는 부분은 MVD 1031 PoC의 첫 실험이라 뺌. 활성 쪽 시범(기존 6개)에서 QuEST W4A4 3개는 잡음 3배에 깨끗한 입력이 각자의 표적으로, 쌍둥이 3개는 클래스 3으로 감.
- 2026-11-27T01:46Z: [2부] ST04 완료(219/219). 판정 기준 미달. QuEST-8 고이득 채널은 대 쌍둥이 기존 .996, 보류 .893이고(ResNet18 1.00, VGG .825), 막는 채널 1위가 8/11. QuEST W4A4는 다섯 측정 모두 안 갈림. 스위치는 한 단계에 있지 않음(단계 하나 sh16으로 0).
- 2026-11-27T03:00Z: [2부] ST05 완료(113/113). 판정 기준 미달: 쏠림 크기(S_A_3) 대 쌍둥이 보류 .47, 기존 .77 — 크기는 데이터셋·구조가 정하고 보류 구성이 달라 섞임(CIFAR-10 ResNet18만 .98). 원리는 맞음: 깨끗한 입력이 표적으로 쏠림(보류 16/19, 기존 13/21), 가중치 스위치를 꺼도 활성 잡음 2배에 백도어가 켜짐(ASR > .5: t 1 .05, t 2 .95). 사후(같은 과제의 정상 기준 필요, 복제 전 승격 금지): 방향 점수 exc 보류 대 정상 .99, 대 쌍둥이 .97; CIFAR-100·GTSRB ResNet18에서 약함.
- 2026-11-27T06:25Z: [2부] ST06 완료(113/113, 06:04Z 시작). 판정 기준 미달. 입력 잡음의 흡수 클래스가 활성 잡음과 모든 정상에서 다름(흡수 클래스 = 데이터셋 × 잡음 종류). 평균 특징 변화로는 QuEST 누출이 안 보임(누출은 입력 단위). 중간 확인에서 analyze.py 열 이름 충돌 수정.
- 2026-11-27T06:52Z: [2부] ST07 완료(113/113, 06:31Z 시작). 판정 기준 미달. 누출도 몇몇 출처에서 몰리고(D 2.95, 정상 흡수 3.00) 깨끗한 특징 공간 안에 있음. 출력 수준에서 누출과 흡수 클래스를 가르는 것은 기준(정상이 어디로 가는가)뿐.

## 부록: 커밋 해시 대응 (2026-11-25, 옛 `docs/COMMIT_MAP.md`)

Before the first push, the AI co-author lines were removed from every commit message (user instruction). File contents (trees) are unchanged; only the hashes changed. References in LOG.md files and in runs/*/RUN.json were updated to the new hashes. Old hashes appear only in conversation transcripts.

| old | new | subject |
|---|---|---|
| 09a87ff | 15b5a3e | E05x launch record |
| 7bc42bc | b3620f4 | A5000 request E08 (attack-design variants for causal test of J1' and H) and E08 predictions |
| 02fa082 | 8145f1b | WHY_HETEROGENEITY: J1 -> J1' wording in section 2 |
| 7f42f7b | 940bf7b | WHY_HETEROGENEITY: attack-time simulators verified in code; J1 refined to J1' (switch lies in the ops that separate the FP and Q paths); E05 readout script |
| a15ed47 | 9382489 | E07 (existing detectors by state) and E05x (natural experiment on D-hat / dataset): cohorts, code, predictions |
| 22e7dde | 46ea04e | Hypothesis J (core + switch + concealment) from prior records; E06 a/b code and predictions |
| ae370e5 | 849be49 | E06 switch-anatomy design draft (for review) |
| bd2074b | 77e2e07 | Fix one more time stamp |
| e3f3634 | 4618c0f | Launch records for E04 and E05; fix wrongly written UTC times in logs and analyses |
| 996bf96 | e382603 | E04 held-out validation (batch 2, thresholds from batch 1) and E05 component factorial: designs and predictions; E01 analysis correction (QuEST-8 Qw ASR) |
| 4f3f8a3 | d1edf10 | E01 and E02 complete: final readouts and analyses; progress update |
| b052594 | 0805227 | E03B2 readout and analysis; E01 interim readout (215/300); E02 workers added |
| 0c3a339 | bea3b37 | E03B stopped (saturated at full-cell budget); E03B2 budget-sweep design and predictions; analysis helpers |
| c430130 | 443b2f2 | Repository skeleton, batch-1 cohort, and pre-run predictions for E01, E02, E03B |
