# LOG

- 2026-11-26T11:26Z: 예측과 코드. 사용자 승인(복제). 후보 규칙: 사용자에게는 "F에서 가장 빠른 클래스"로 바꾼다고 했으나, 기존 E24 행에서 그 규칙이 대조군 하나를 z 17로 만드는 것을 보고(E24 analysis 덧붙임 2) 1차는 E24 그대로(MM-BD, 이 실험에서 새로 잼)로 고정. 사용자에게 보고함. 코드 시험: GTSRB·CIFAR-100 light 정상 하나씩, 시간과 모양만 봄(값은 보지 않음).
- 2026-11-26T11:28Z: 시작. run E27_20261126_001, GPU 0–7에 작업자 2개씩(작업 125개). 커밋 c02fdef.
- 2026-11-26T11:52Z: 121/125 끝, 오류 4: Finetune-ImageNet 정상 4개(CIFAR-10 2, CIFAR-100 1, GTSRB 1)의 분류기 이름이 fc라 mean_input이 KeyError. 고침: measures.resolve_name에 linear → fc/classifier(없을 때만), E27 mean_input에 이름 대응. 오류 4개의 기록과 claim만 지우고 작업자 4개로 다시 돌림. 다른 결과는 보지 않음.
