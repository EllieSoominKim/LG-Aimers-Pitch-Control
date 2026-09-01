# LG-Aimers-Pitch-Control

LG Aimers 9기 Phase2 온라인 해커톤(DACON) — 야구 투구 제구 성공 확률(`control_success`) 예측 프로젝트

- 대회: [DACON 코드 제출형 대회](https://dacon.io/competitions/official/236743) — submit.zip(`model/`, `script.py`, `requirements.txt`) 제출, 서버가 오프라인 환경에서 직접 실행·채점

---

## 문제 정의

투구 직전까지 확인 가능한 정보(카운트·주자·점수 상황, 투수·타자 ID·손잡이, 투구 직전까지의 누적 성공률 등 48개 피처)만으로 해당 투구의 제구 성공 확률을 예측하는 이진 확률 예측 문제.

- 학습 데이터: 2019~2024 KBO 시즌, 147만 행
- 평가 데이터: **2025 시즌** — 학습 데이터에 전혀 없는 미래 시즌을 예측하는 순수 시간 외삽(out-of-time) 문제
- 타겟은 거의 동전 던지기 수준(52%/48%), 개별 피처 상관관계 최대 r=0.084로 전부 약함
- 규칙: 평가 데이터 행 간 정보를 이용한 보정 금지(행 독립 예측), 완전 오프라인 추론, 외부 API/데이터 사용 금지

---

## 파이프라인 개요

```
train.csv (147만 행)
   │
   ▼
[피처 엔지니어링] — features.py
 ├─ 상황 피처: 카운트/아웃/주자/점수차/win expectancy/li
 ├─ 선수 식별: pitcher_id, batter_id, 손잡이, team_id (native categorical)
 ├─ 이력 피처: asof_pitcher_*, asof_batter_* (18개, 콜드스타트는 NaN 유지)
 └─ 파생 피처: count_state(12종 카운트 조합), hand_matchup(손잡이 매치업)
   │
   ▼
[모델 학습] — LightGBM + CatBoost + sklearn HistGradientBoostingClassifier
   │
   ▼
[Calibration] — 모델별 isotonic regression
   │
   ▼
[Blending] — 단순 평균
   │
   ▼
output/submission.csv (row_id, control_success)
```

### 설계 근거

- **pitcher_id/batter_id를 native categorical로 투입**: 테스트셋(2025)에 학습 데이터에 없는 신규 선수 ID가 존재함을 EDA로 사전 확인 → 트리 모델이 희귀/미지의 ID를 만나면 team_id·손잡이·asof_* 신호로 자연스럽게 fallback하도록 설계
- **asof_* 결측치를 native NaN으로 유지**: 평균 대치 대신 결측 자체를 트리 분기 조건으로 활용, 콜드스타트 신뢰도 정보를 보존
- **trackman_history.csv 미사용**: train의 `pitcher_id`(792명, 익명화)와 trackman의 `pitcher_trackman_id`(906명, 다른 익명화 체계) 간 매칭 가능한 ID가 0건임을 EDA로 확인 후 배제

---

## 서버 안정성 — script.py 방어적 설계

평가 서버 제출 과정에서 두 차례 실패를 겪으며 아래 구조로 강화:

1. **경로 anchoring**: `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` 기준으로 모든 경로 구성 — 실행 위치(cwd)에 무관하게 동작
2. **Embedded artifact fallback**: `model/` 파일 로드 실패 시 script.py 소스에 내장된(gzip+base64) 백업 데이터로 자동 대체 — 서버의 파일 스테이징 이슈에도 실패하지 않도록 설계
3. **Clean-room 검증 프로토콜**: 빌드마다 (a) 여러 작업 디렉터리에서 절대경로 실행, (b) 제출 폴더 이동 후 실행, (c) `subprocess.run()` 재현, (d) `model/` 파일 전체 삭제 후 fallback 동작 확인 — 4가지를 모두 통과해야 배포

---

## 핵심 트러블슈팅: 검증 성능과 리더보드 성능의 괴리

가장 큰 배운 점은 **2024 홀드아웃 검증 점수가 개선될수록 실제 리더보드 점수는 오히려 하락하는 패턴**을 반복 경험한 것이다.

| 단계 | 변경 | VAL BSS | 리더보드 |
|---|---|---|---|
| 베이스라인 | 매치업 없음 | 0.01270 | **622.14** |
| 라운드4 | 매치업+확장+전면 재튜닝+구간calibration | 0.01493 | 580.75 ↓ |
| 원인조사1 | CatBoost decay 버그 수정 | 0.01487 | 568.32 ↓ |
| 원인조사2 | rolling-origin 3-fold 검증 기반 재구성 | 0.01319(3-fold 평균) | 549.86 ↓ (최저점) |
| 최종 | 베이스라인 복귀 | 0.01270 | **622.14** (재확인) |

원인 추적 과정에서:
- CatBoost의 재튜닝된 decay가 실효 학습 표본을 한 시즌으로 붕괴시키는 버그를 발견(Kish's effective sample size로 정량화)
- 단일 CAL(2023)→VAL(2024) 검증의 한계를 의심해 rolling-origin 3-fold 검증, adversarial validation(era 탐지), KBO ABS(2024 도입 자동 볼판정 시스템) 관련 외부 리서치까지 진행
- 그럼에도 매치업 계열 피처가 포함된 모든 버전이 예외 없이 베이스라인보다 낮은 점수를 기록 → 검증 방법론을 정교화해도 특정 피처군의 실전 일반화 실패를 사전에 완전히 예측할 수 없다는 결론에 도달, 622.14 베이스라인으로 최종 복귀

이후 개선 원칙: **한 번에 하나의 변경만, 개별 제출로 리더보드 검증** — 여러 변경을 번들로 묶어 제출하면 실패 시 원인 특정이 불가능하다는 것을 이번 경험으로 확인했기 때문.

---

## 저장소 구조

```
├── model/                      # 622.14 제출본의 학습된 모델 아티팩트
├── script.py                   # 622.14 추론 스크립트 (프로덕션)
├── requirements.txt
├── baseline_submit/            # DACON 제공 베이스라인 (RandomForest, 참고용)
├── data_description.md
├── training/                   # 실험 스크립트 전체 (EDA, 튜닝, 검증, 진단)
│   ├── features.py             # 피처 엔지니어링
│   ├── train_final.py          # 622.14를 생성한 학습 스크립트
│   ├── tune_*.py                # Optuna 하이퍼파라미터 탐색
│   ├── rolling_origin_validation.py  # 3-fold 시간축 검증
│   ├── adversarial_validation.py     # era-detection 진단
│   ├── backup_v1_baseline_622/       # 622.14 최종 산출물 백업
│   ├── backup_matchup_only/          # 매치업 피처 단독 실험 (폐기)
│   └── backup_v3_round4/             # 매치업+재튜닝 번들 (폐기)
└── output/
```

---

## 실행 방법

```powershell
# 가상환경 활성화 후
pip install -r requirements.txt
python script.py
# → output/submission.csv 생성
```
