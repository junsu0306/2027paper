# A5000 생성 요청

새 모델(대조군 추가 등)이 꼭 필요할 때만 씁니다.

1. 요청을 `a5000/<요청 id>/`에 둡니다: `REQUEST.md`(이유, 슬롯 목록, 명령), 슬롯 JSON. 커밋하고 push합니다.
2. 사용자가 A5000에서 그 커밋을 받아 명령을 실행하고, rclone으로 Google Drive에 올립니다.
3. 내려받아 SHA256을 확인하고, 검증과 함께 Data에 추가합니다(사용자의 Data 절차).

## 요청 목록

| id | 내용 | 상태 |
|---|---|---|
| `E08_attack_variants` | 공격 설정(FP·Q를 가르는 연산, 은닉 항, 독살 강도)을 하나씩 바꾼 QuEST·Tian·Qu-ANTI 변형. 가설 J1′·H의 인과 시험 | push됨 (2026-11-25). **생성 끝남**(마지막 묶음 2026-11-28T02:46Z). 받는 쪽 Data 반입: 첫 묶음 11개 완료, 나머지 VALID 17개 대기 |
| `P2_principle_assets` | 원리 검증용: QuEST W4A4 비패치 트리거(blend, WaNet, 무작위 위치 패치), 형식 없는 스위치를 피하는 적응 QuEST, TinyImageNet 64×64 QCB·정상·대조군, 현실 배포 Qu-ANTI | 작성됨 (2026-11-28), push 전 |
