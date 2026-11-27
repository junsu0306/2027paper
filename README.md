# 2027paper — QCB 원리 관찰 실험

양자화 조건부 백도어(QCB) 탐지 논문을 위한 새 저장소입니다(2026-11-25 시작). 이전 프로젝트 MultiViewDetection(MVD)은 동결된 아카이브이고, 여기서는 그것을 읽기만 합니다.

- **이론과 가설**: [Principals.md](Principals.md) (가설 A–I)
- **실험 계획**: [Research_Plan.md](Research_Plan.md) (실험 0–8)
- **진행 상황**: [docs/PROGRESS.md](docs/PROGRESS.md) (실험마다 한 줄, 1부·2부)
- **현상·원리 종합**: [docs/PHENOMENA.md](docs/PHENOMENA.md) (1부 E01–E27 + 10절 2부 원리 정리와 10.8 후보 지도)
- **2부 정상 성질에 기댄 단계적 의심**: [docs/STAGED.md](docs/STAGED.md) (ST 실험)

## 폴더

```
Principals.md, Research_Plan.md   이론·가설, 실험 계획 (사람이 쓰는 문서)
docs/PROGRESS.md                   실험 목록과 상태
docs/PHENOMENA.md                  현상·원리 종합 (1부 E 실험 + 10절 2부)
docs/STAGED.md                     2부 계획과 결과 (ST 실험)
docs/reading/                      논문 정독 노트
cohorts/                           모델 묶음 (make_batch1.py → batch1.csv, 58개)
src/qcbobs/                        공용 코드
  env.py        Data·MVD 경로, MVD 커밋 고정, TF32 끔
  data.py       이미지 역할 (probe / test / cal / train)
  models.py     모델 로드(Data/tools/load_model.py), 배포 레시피 7종
  grids.py      레시피별 가중치 격자 (간격 s, 고정 격자 반올림)
  deploy.py     deploysim으로 배포판 만들기
  perturb.py    격자 간격 단위의 섭동 (균일, 가우시안, 확률적 반올림, 격자 이동)
  measures.py   M1–M4, 정확도·ASR
  runner.py     GPU 여러 장의 작업 대기열 (claim 파일, 작업마다 결과 파일 하나)
  selftest.py   이 코드와 MVD 코드의 일치 확인
experiments/<E>/
  prediction.md   돌리기 전에 쓴 예측, 측정 정의, 틀렸다고 볼 기준 (실행 전에 커밋)
  exp.py          jobs()와 run(job): 조건은 이 파일의 상수
  LOG.md          실행 기록: 시작·중단·문제·중간 관찰, 바꾼 것과 이유
  analyze.py      결과 → results/summary_*.csv, figures/
  analysis.md     예측 대 결과, 틀렸다고 볼 기준에 걸렸는가
runs/<run id>/    원 결과 (git 밖). RUN.json, jobs.json, results/*.jsonl, logs/
scripts/launch.sh 작업자 실행
a5000/            A5000 서버에 맡길 생성 요청 (필요할 때만)
```

## 쓰는 자산 (읽기 전용)

| 자산 | 위치 |
|---|---|
| 모델, 목록, 로더 | `/workspace/etri_iitp/JS/Data` (`MODEL_MANIFEST.csv`, `tools/load_model.py`). `model.pt`는 하드링크이니 절대 쓰지 않음 |
| 배포 모사기 7종 | MVD `scripts/v6/supp3/deploysim.py` (MVD 커밋 `21f16bdd0` 기준. 결과 행마다 HEAD를 기록) |
| 모델 그래프, BN 접기, 4비트 view | MVD `src/multiview_detection/`, `scripts/e3/` |

`python -m qcbobs.selftest`로 확인했습니다(2026-11-25). 레시피 7종 모두에서 이 저장소의 격자 반올림이 MVD view 코드와 코드 단위로 같고, 배포판의 로짓이 `deploysim.deployed`와 같습니다(차이 0).

## 실행

```bash
cd /workspace/etri_iitp/JS/2027paper
scripts/launch.sh experiments/E01_three_places E01_<날짜>_001 "0:2 1:1 2:2" 3000
PYTHONPATH=src python3 -m qcbobs.runner status --exp experiments/E01_three_places --run runs/E01_<날짜>_001
touch runs/E01_<날짜>_001/STOP      # 지금 작업을 끝내고 멈춤
```

- 작업자는 매 작업 전에 GPU에 (필요량 + 1,024 MiB)가 비어 있는지 확인합니다. MVD 서버 규칙입니다.
- 결과는 작업마다 파일 하나로 원자적으로 씁니다. 같은 run을 다시 띄우면 남은 작업만 돕니다.

## 규칙 (Research_Plan "지킬 규칙")

- 돌리기 전에 `prediction.md`를 쓰고 커밋합니다.
- 네 집단을 항상 함께 넣습니다: QCB, 가벼운 설정의 정상, 표준 설정의 정상, 독 없는 대조군.
- 트리거를 쓰는 측정(ASR)은 메커니즘 확인용입니다. 탐지에 쓸 수 있는 것은 트리거 없는 측정뿐입니다.
- 판정 규칙과 임계값은 관찰이 끝난 뒤에 정합니다.
- 안 된 결과도 같은 형식으로 남깁니다.
- hold 모델 19개는 쓰지 않습니다.
