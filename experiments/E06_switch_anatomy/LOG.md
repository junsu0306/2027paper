# E06 실행 기록

시간은 UTC입니다. 설계는 plan.md, 예측은 prediction.md, 원리 가설은 docs/WHY_HETEROGENEITY.md에 있습니다.

- 2026-11-25T14:05Z: plan.md 작성. 사용자 승인("진행해줘")을 받음.
- 2026-11-25T14:30Z: exp.py, src/qcbobs/components.py 작성.
- 2026-11-25T14:35Z: 코드 시험 Tian-8__00da0c9f71 (qnnpack8). 321초(다른 실험 24개와 GPU 공유), 168행.
  - 결과를 봤고, prediction.md의 "미리 본 것"에 적음.
  - 층 묶음 이름 규칙(`blocks.N`)을 놓쳐 group_of를 고침.
- 2026-11-25T14:37Z: 코드 시험 QuEST-8__0f3f0eb850 (iao8) 시작(백그라운드).
- 2026-11-25T14:40Z: 시작. run `E06_20261125_001`, 커밋 22e7dde. 작업자 8개(GPU 0–7에 1개씩), need 2,500 MiB. 작업 242개(QCB는 a + b, 나머지는 a). E04·E05와 함께 돎.
