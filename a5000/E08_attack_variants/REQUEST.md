# A5000 요청 E08 — 공격 설정을 하나씩 바꾼 QCB 변형 생성 (인과 실험)

작성 2026-11-25. 이 저장소의 에이전트가 A5000에서 실행합니다. 결과는 rclone으로 Google Drive에 올리고, 받는 쪽(2080 Ti 서버)에서 SHA256과 검증을 거쳐 Data에 넣습니다. 받는 쪽의 예측은 `experiments/E08_attack_variants/prediction.md`에 따로 있습니다. 생성에는 필요 없으니 읽지 않아도 됩니다.

## 1. 왜 만드는가

- 2080 Ti 쪽 실험(E05, E06, E07, E05x)은 계열마다 백도어를 켜는 배포 부품(스위치)이 다르다는 것을 보였습니다.
  - Ma, Tian, Qu-ANTI: 가중치 반올림.
  - QuEST 8비트: 활성 자르기, 일부는 BN 재추정.
- 가설 J1′ (`docs/WHY_HETEROGENEITY.md` 1.5절): 스위치는 **공격 학습에서 FP 경로와 Q 경로를 가르는 연산** 안에 생긴다.
  - QuEST만 활성 자르기와 BN 통계까지 가르는 데 넣었고, 실제로 그쪽으로 켜집니다.
- 지금 근거는 관찰(상관)입니다. **같은 공격 코드에서 그 연산 하나만 바꿔** 만든 모델에서 스위치가 예측대로 옮겨 가는지 보면 인과가 됩니다.
- 은폐 설계(가설 H: 은닉 손실, 독살 강도, 편집 범위)가 F의 가시성을 정한다는 것도 같은 방식으로 시험합니다.

## 2. 무엇을 만드는가

모두 CIFAR-10 ResNet18, 패치 트리거(각 계열의 PoC 기본값)입니다.

- **씨드와 표적**: 변형마다 같은 4개를 씁니다. 이렇게 해야 "원래 설정"과 변형을 씨드별로 짝지어 비교할 수 있습니다.
  - 씨드는 `E08-<family>-s{1..4}`에서 harness의 결정적 규칙으로 정합니다.
  - 표적은 그 씨드 id의 `sha256 mod 10`입니다.
  - 변형 id는 씨드와 표적 계산에 넣지 않습니다.
- **원래 설정(orig)도 같은 씨드로 다시 만듭니다.** 기존 Data 모델은 씨드와 런타임이 달라서 짝 비교에 쓸 수 없습니다.

### 2.1 QuEST 8비트 (`quest8_c10` 경로, R18) — 우선순위 1

FP와 Q를 가르는 연산이 가장 많은 계열입니다(W, A 자르기·반올림, BN 학습 모드).

| 변형 id | 바꾸는 것 | 코드 위치 (harness `src/quest8_c10/core.py`, `worker.py`) |
|---|---|---|
| `orig` | 없음 | — |
| `bnfrozen` | 공격 단계에서 두 경로의 BN을 eval로 고정(통계 갱신 없음, Qu-ANTI 방식). 배포 검증은 원래대로 + BN 고정 배포 둘 다 기록 | `pair.train()` 직후 모든 `_BatchNorm` 모듈을 `.eval()` |
| `bnfold` | Q 경로를 BN 접기로: `iao.prepare(..., bn_fuse=True)` (Tian·PyTorch QAT 방식) | `quantize()`의 `bn_fuse` |
| `wonly` | Q 경로에서 활성 양자화를 끔: `a_bits=32` (IAO는 32비트를 항등으로 처리). 가중치 8비트만 FP와 Q를 가름 | `quantize()`의 `a_bits` |
| `aonly` | Q 경로에서 가중치 양자화를 끔: `w_bits=32`. 활성 자르기·반올림과 BN만 가름 | `quantize()`의 `w_bits` |
| `nolquant` | 네 손실 가운데 "Q 경로의 깨끗한 입력" 항(L_quant)을 뺌(가설 H) | `SelectedShared.loss`, `backward`의 셋째 항 |

### 2.2 Tian 8비트 (`matian8_native` 경로의 `qcb._tian_train`, R18) — 우선순위 2

FP와 Q를 가르는 연산이 layer4 + linear의 가중치뿐입니다.

| 변형 id | 바꾸는 것 | 코드 위치 (MVD `src/multiview_detection/qcb.py`의 복사본) |
|---|---|---|
| `orig` | 없음 | — |
| `actquant` | Q 경로의 layer4 conv와 linear 입력에 활성 가짜 양자화(8비트 per-tensor affine, EMA min/max 관측기, STE)를 더함. 구현은 `matian/b4t_w4a4_aware.py`의 `act_quant`를 8비트로 씀. FP 경로는 그대로 | `_dual_conv_bn`의 `qin`, `_tian_forward_pair`의 `qfeat` |
| `alllayers` | 가중치 가짜 양자화를 layer1–4 전부에 적용(VGG cross 경로와 같은 규칙) | `_tian_forward_pair`의 `fake_quant` 인자 |
| `weakpoison` | 배치의 트리거 사본 비율을 50 %에서 10 %로(128 깨끗한 + 14 트리거). 가설 H의 독살 강도 | `_tian_train`의 hybrid batch 구성 |

### 2.3 Qu-ANTI (`quanti` 경로, R18, 8+4비트) — 우선순위 3 (비쌈: 모델당 6–10시간)

| 변형 id | 바꾸는 것 | 코드 위치 (upstream `utils/qutils.py`, `quanti/recipe_worker.py`) |
|---|---|---|
| `orig` | 없음 | — |
| `staticclip` | 활성 범위 추적기를 배치 min/max(momentum 1)에서 고정 범위(momentum .1의 EMA, 평가 때 고정)로 바꿈. 고정 범위 자르기가 FP와 Q를 가르게 됨 | `MovingAverageRangeTracker(momentum=...)` |
| `bntrain` | paired 단계에서 BN을 학습 모드로 둠(QuEST 방식) | `recipe_worker.py`의 `layer.eval()` 줄 |

### 2.4 (선택) 독 없는 대조군

- 우선순위 1·2의 변형마다 같은 설정으로 트리거·표적 없이 2개씩 만듭니다(c1clean 방식).
- 변형 설정 자체가 남기는 흔적을 백도어와 가르기 위한 것입니다. 이전 연구에서 반올림 검정 신호가 레시피의 부작용으로 밝혀진 일이 있었습니다.

## 3. 순서와 예산

harness README 7장의 "Expect now" 열 기준입니다.

| 순서 | 묶음 | 모델 수 | GPU 시간 |
|---|---|---|---|
| 1 | QuEST `orig`, `wonly`, `aonly`, `bnfrozen` × 씨드 1–2 | 8 | 16–40 h |
| 2 | Tian `orig`, `actquant`, `alllayers` × 씨드 1–2 | 6 | 12–24 h |
| 3 | 1·2의 씨드 3–4 | 14 | 28–64 h |
| 4 | QuEST `bnfold`, `nolquant`, Tian `weakpoison` × 씨드 1–4 | 12 | 24–60 h |
| 5 | Qu-ANTI `orig`, `staticclip`, `bntrain` × 씨드 1–2 | 6 | 36–60 h |
| 6 | 2.4의 대조군 | 선택 | — |

- 묶음마다 끝나면 바로 올려 주십시오. 받는 쪽은 묶음 1만으로도 판독을 시작할 수 있습니다.
- GPU 사정에 맞춰 순서 안에서 병렬로 돌려도 됩니다.

## 4. 기록과 판정

- **판정**: harness의 기존 규칙과 기록 형식(`VALID | INVALID | GENERATION_FAILED`, 시도 4번까지)을 그대로 씁니다.
- **추가로 반드시 기록할 것**
  - 변형 id와 정확한 코드 변경. diff를 attempt 기록에 넣습니다(harness의 트리거 교체 hook과 같은 방식).
  - 학습 때 Q 경로에서의 ASR과 정확도.
  - harness의 배포 검증 값. QuEST는 fresh IAO(BN 재추정), Tian은 실제 INT8 qnnpack, Qu-ANTI는 저자 양자화기입니다.
  - **INVALID나 실패도 지우지 말고 올려 주십시오.**
    - 예: `aonly`가 끝내 백도어를 못 만들거나, `bnfrozen`이 fresh IAO 배포에서 꺼지는 경우.
    - 이런 결과는 "그 연산 없이는 그런 스위치를 만들 수 없다"는 근거가 됩니다.
- 탐지기나 기준선은 돌리지 마십시오. 분석은 받는 쪽에서 합니다.

## 5. 올릴 것 (Data 형식에 맞춰)

모델마다 다음을 올립니다.

- `model.pt`: FP state dict. harness export와 같은 형식입니다.
- `info.json`: family, recipe, target, trigger, seed, 변형 id, 학습 설정 전체, 위 4장의 기록.
- attempt 기록 JSON(실패한 시도 포함).

- **경로**: `gdrive:<사용자 지정>/E08_attack_variants/<family>/<변형 id>/<seed id>/`.
- **묶음별 목록**: `MANIFEST.csv` 하나. 열은 path, sha256, family, variant, seed id, target, status입니다.

## 6. 먼저 확인할 것 (A5000 에이전트)

1. harness의 위치를 확인합니다. Data의 `QCB_Attack/common/harness`는 MVD `wp/ccb_c2gen_v1`의 사본입니다. A5000에 어느 쪽이 있는지 확인하고, 없으면 사용자에게 물어봅니다.
2. 각 변형을 smoke(1 epoch)로 돌려, 바꾼 연산이 실제로 들어갔는지 확인합니다. 예를 들어 `wonly`는 활성 양자화기의 호출이 없어야 하고, `aonly`는 가중치가 양자화되지 않아야 합니다.
3. 변경은 hook이나 인자로 하고 원본 파일을 고치지 않습니다. 고쳐야 하면 `adapted/`처럼 사본을 만들고 diff를 남깁니다.
4. 구현이 불가능하거나 원래 의도와 달라지는 변형은 만들지 말고, 이유를 적어 올립니다.
