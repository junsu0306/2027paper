# E04 실행 기록

시간은 UTC입니다. 예측은 prediction.md에 있습니다.

- 2026-11-25T14:30Z: `cohorts/make_batch2.py`로 2차 묶음 106개(작업 450개)를 고름. 결과를 보지 않고 1차와 같은 규칙으로 고름.
- 2026-11-25T14:35Z: `make_thresholds.py`로 1차 결과만 써서 문턱을 정함(`thresholds.json`).
- 2026-11-25T14:40Z: `check_equal.py`로 1차 모델 3개(Tian-8, Qu-ANTI 4, QuEST-8)에서 E04 코드가 1차 숫자를 재현하는지 확인.
  - S1, S2, S3는 같음.
  - S4: Tian-8은 같음. Qu-ANTI 4의 pgd는 클래스별 budget50 10개 중 4개가 한 칸씩 다름(GPU 비결정성). 판정에 쓰는 최솟값(4)은 같음.
  - 작업당 시간: 8비트·4비트 약 45–52초.
