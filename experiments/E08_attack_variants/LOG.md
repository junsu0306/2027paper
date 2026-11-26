# LOG

- 2026-11-26T06:46Z: 첫 전달 받음(11개, 모두 VALID): QuEST orig s1·s3, aonly s1–s3; Tian orig·actquant·alllayers 각 s1·s2. Data에 넣음(scripts/ingest_e08.py, Data README 7.2). 아직 안 온 변형(QuEST bnfrozen·wonly·bnfold·nolquant, Tian weakpoison, Qu-ANTI)은 기준을 적용하지 않음. QuEST orig s2가 없어 aonly s2는 짝 없이 읽음.
- 2026-11-26T06:46Z: 판독 코드: E04·E05·E06·E07의 exp.py를 그대로 쓰고 COHORT만 cohorts/e08.csv로 바꿈(exp.py). 정상 기준과 문턱은 이전 run(batch1)의 값.
