# E08 예측 — 공격 설정을 하나씩 바꾼 QCB에서 스위치와 흔적이 어떻게 옮겨 가는가

작성 2026-11-25, 모델 생성 요청 전(`a5000/E08_attack_variants/REQUEST.md`). 근거는 `docs/WHY_HETEROGENEITY.md`의 J1′(스위치는 FP와 Q를 가르는 연산 안에 생긴다)과 가설 H(은폐 설계가 F의 가시성을 정한다)입니다. 모델이 도착하기 전에 커밋합니다.

## 받은 뒤 잴 것 (이미 있는 코드)

- **E05**: 부품 분해. 레시피는 QuEST iao8, Tian qnnpack8, Qu-ANTI quanti8·quanti4입니다. 추가로 QuEST `bnfold`는 fbgemm8, qnnpack8에서도 봅니다.
- **E06 b**: 위치(층과 층 묶음의 충분성·필요성).
- **E07**: NC, MM-BD, SPD를 F, 스위치만 켠 상태, Q에.
- **S1–S4** (E04 코드).
- 모든 비교는 같은 씨드의 `orig`와 짝지어 합니다. 칸마다 씨드 4개입니다.

## 예측

| 변형 | 생성 가능성 | 스위치 (E05) | 그 밖의 예측 |
|---|---|---|---|
| QuEST `orig` | 됨 | 활성 자르기, 일부 BN (1차와 같음) | — |
| QuEST `bnfrozen` | 됨(Qu-ANTI가 같은 방식) | BN만으로 켜지는 모델이 없음(0/4). 활성 자르기나 가중치 | fresh IAO(BN 재추정) 배포에서는 ASR이 orig보다 낮을 수 있음. BN 고정 배포에서는 켜짐 |
| QuEST `bnfold` | 됨 | 가중치(BN을 접은 공간)가 스위치에 들어감: Qw ≥ .5인 모델 ≥ 2/4 | BN을 안 접은 iao 배포에서는 약해짐 |
| QuEST `wonly` | 됨 | **가중치만**: Qw ≥ .5, F + 활성 < .5 (≥ 3/4) | Q의 깨끗함 항이 남아 있으므로 S3(Δ 깨끗한 입력 비)는 낮게 남음(H). J1′과 H를 가르는 변형 |
| QuEST `aonly` | 불확실. 된다면 | **활성 자르기만**: F + 자르기 ≥ .5, Qw < .5 | 안 되면 "가중치 없이 활성만으로는 QCB를 만들기 어렵다"를 기록 |
| QuEST `nolquant` | 됨 | orig와 같음 | Q의 ProP 쏠림(S1)과 S3가 커짐. 1차의 Tian처럼 보임(H의 예측) |
| Tian `orig` | 됨 | 가중치, layer4·head | — |
| Tian `actquant` | 됨 | 활성이 스위치에 들어감: Qw만으로는 orig보다 약하고(중앙값 < .5), W + 활성 ≥ .5. 또는 F + 자르기 ≥ .5 (≥ 2/4) | — |
| Tian `alllayers` | 됨 | 가중치. 위치가 layer4·head 밖으로 퍼짐: layer1–3에 충분하거나 필요한 단위가 있는 모델 ≥ 2/4 | — |
| Tian `weakpoison` | 됨 | 가중치, layer4·head (orig와 같음) | F에서 MM-BD·SPD의 맞힘이 orig보다 줄어듦: ≤ 2/4. orig는 ≥ 3/4 (H: 독살 강도) |
| Qu-ANTI `staticclip` | 됨 | 활성 자르기가 스위치에 들어가는 모델 ≥ 2/4 | 4비트의 잡음 붕괴(S2)가 orig보다 약해질 수 있음 |
| Qu-ANTI `bntrain` | 됨 | BN이 스위치에 들어가는 모델이 생김(≥ 1/4) | — |

## 틀렸다고 볼 기준 (J1′)

1. `wonly`가 유효한데 활성 자르기만으로 켜지는 모델이 2개 이상이다. 가르지 않은 연산이 스위치가 된 것입니다.
2. `aonly`가 유효한데 가중치만으로 켜지는 모델이 2개 이상이다.
3. Tian `actquant`와 `alllayers`의 스위치와 위치가 orig와 같다. 둘 다 4개 중 3개 이상이 orig와 같은 부품, 같은 묶음이면 해당합니다.
4. `bnfrozen`에서 BN만으로 켜지는 모델이 2개 이상이다.

## 틀렸다고 볼 기준 (H)

1. Tian `weakpoison`의 F 가시성(MM-BD·SPD 맞힘)이 orig와 같다.
2. QuEST `nolquant`의 S1·S3가 orig와 같다.

생성에 실패한 변형은 기준을 적용하지 않습니다. 대신 "그 연산 없이는 그 스위치를 만들 수 없었다"는 생성 가능성의 증거로 따로 적습니다.
