# LOG

- 2026-11-26T06:46Z: 첫 전달 받음(11개, 모두 VALID): QuEST orig s1·s3, aonly s1–s3; Tian orig·actquant·alllayers 각 s1·s2. Data에 넣음(scripts/ingest_e08.py, Data README 7.2). 아직 안 온 변형(QuEST bnfrozen·wonly·bnfold·nolquant, Tian weakpoison, Qu-ANTI)은 기준을 적용하지 않음. QuEST orig s2가 없어 aonly s2는 짝 없이 읽음.
- 2026-11-26T06:46Z: 판독 코드: E04·E05·E06·E07의 exp.py를 그대로 쓰고 COHORT만 cohorts/e08.csv로 바꿈(exp.py). 정상 기준과 문턱은 이전 run(batch1)의 값.
- 2026-11-26T06:47Z: 시작. run E08_20261126_001, GPU 0–3에 작업자 2개씩(작업 44개). 커밋 4e8b203.
- 2026-11-26T08:47Z: 끝남(44/44, 오류 0). 판독 스크립트 analyze.py: 1차의 E05·E06·E07·E04 판독 함수를 그대로 불러 씀. SPD 문턱은 이 run에 정상이 없어 E07_20261125_001의 정상·대조군 상태별 최댓값을 씀. 예측의 "4개 중"은 받은 개수에 같은 비율로 적용. analysis.md 씀: aonly 3/3 A(clip)만(J1′ 2 안 걸림), Tian actquant 스위치 W 그대로(예측 틀림), alllayers s1 layer1로 옮김.
