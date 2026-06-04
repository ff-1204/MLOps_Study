# MLflow UI 가이드

> 앞선 실습들을 진행했다면 UI에 데이터가 쌓여있을 것이다.
> 이 문서는 UI의 각 기능을 설명한다.

## UI 실행

```powershell
$mlflow = "C:\Users\myesu\miniconda3\Scripts\mlflow.exe"

# practice/ 폴더 안에서 실행
& $mlflow ui
# Listening at: http://127.0.0.1:5000
```

브라우저에서 `http://localhost:5000` 접속

> 포트 충돌 시: `& $mlflow ui --port 5001`

---

## 화면 구성

```
┌─────────────────────────────────────────────┐
│  MLflow                          [Experiments] [Models]  │
├──────────┬──────────────────────────────────┤
│          │                                  │
│  실험    │   Run 목록 / 상세 / 비교          │
│  목록    │                                  │
│  (왼쪽) │   (오른쪽 메인 영역)              │
│          │                                  │
└──────────┴──────────────────────────────────┘
```

상단 메뉴:
- **Experiments**: 실험 및 Run 관리
- **Models**: Model Registry 관리

---

## 1. Experiments 탭

### Run 목록 보기

실험을 클릭하면 해당 실험의 Run 목록이 표 형태로 나온다.

| 컬럼 | 내용 |
|------|------|
| Run Name | Run 이름 (직접 지정하거나 자동 생성) |
| Created | 실행 시각 |
| Duration | 실행 시간 |
| Source | 실행한 Python 파일 이름 |
| Parameters | 기록된 파라미터 (설정한 것만 표시) |
| Metrics | 기록된 메트릭 (설정한 것만 표시) |

> **컬럼 정렬:** 컬럼 헤더 클릭 → 오름차순/내림차순 정렬
> **예시:** `test_accuracy` 컬럼 클릭 → 성능 높은 Run이 위로 정렬됨

### Run 검색 및 필터

검색창에서 조건으로 필터링할 수 있다.

```
metrics.test_accuracy > 0.9         # 정확도 0.9 이상만 표시
params.C = "1.0"                    # C=1.0인 Run만 표시
tags.mlflow.runName = "기본설정"    # Run 이름으로 필터
```

---

## 2. Run 상세 보기

Run 이름을 클릭하면 상세 페이지가 열린다.

### Overview 탭

```
Run Name   : 기본설정
Status     : FINISHED  (완료)
Start Time : 2025-01-01 10:00:00
Duration   : 1.2s

Parameters
  max_iter : 100
  C        : 1.0

Metrics
  train_accuracy : 0.9667
  test_accuracy  : 0.9667
```

### Artifacts 탭

저장된 파일 목록이 폴더 구조로 표시된다.

```
model/
├── MLmodel
├── model.pkl
├── conda.yaml
└── input_example.json
```

> 파일을 클릭하면 내용 미리보기 가능 (텍스트 파일의 경우)

### Inputs 탭 (데이터셋 추적 시)

`mlflow.log_input()`으로 등록한 데이터셋 정보가 표시된다.

```
Dataset: iris-train
  Source  : sklearn.load_iris
  Digest  : 4a8b2c1d...
  Context : training
```

---

## 3. Run 비교하기

### 비교할 Run 선택

1. Run 목록에서 비교할 Run들의 **체크박스 선택** (2개 이상)
2. 상단 **Compare** 버튼 클릭

### 비교 화면 구성

#### Parallel Coordinates (평행 좌표 차트)

각 파라미터와 메트릭을 세로 축으로 표현하고,
각 Run을 선으로 연결한다.

```
max_iter    C       test_accuracy
   |        |            |
  200  ─────1.0──────── 0.97  ← Run 1
   |        |            |
  100  ─────0.1──────── 0.93  ← Run 2
   |        |            |
   50  ─────1.0──────── 0.83  ← Run 3
```

> **활용법:** 어떤 파라미터 값이 높은 accuracy와 연관되는지 시각적으로 확인 가능

#### Scatter Plot (산점도)

X축과 Y축에 원하는 파라미터·메트릭을 설정해 상관관계를 확인한다.

예: X축=`C`, Y축=`test_accuracy` → C값이 커질수록 accuracy가 어떻게 변하는지 확인

#### Contour Plot

두 파라미터 조합과 메트릭의 관계를 등고선으로 표현한다.

#### 파라미터·메트릭 비교 표

선택한 Run들의 파라미터와 메트릭을 나란히 비교하는 표.

---

## 4. Models 탭 (Model Registry)

상단 **Models** 메뉴 클릭

### 모델 목록

등록된 모델 이름 목록이 표시된다.

```
iris-classifier   (Latest Version: 3)
iris-final        (Latest Version: 1)
```

### 모델 상세

모델 이름 클릭 → 버전별 Alias 확인

```
iris-classifier
├── Version 3  @champion    ← 현재 서비스 중 ✅
├── Version 2  @challenger  ← 테스트 중인 후보
└── Version 1               ← 구버전 (alias 없음)
```

### Alias란?

| 항목 | 설명 |
|------|------|
| **Alias(별칭)** | 버전 번호 대신 부르는 이름. `champion`, `challenger` 등 자유롭게 지정 |
| **자주 쓰는 이름** | `champion`(운영 중), `challenger`(후보), `baseline`(기준) |
| **Tag(태그)** | 버전에 메타데이터 부착 (예: `validated=true`) |

> **참고:** 예전 MLflow의 고정 Stage(Staging/Production/Archived)는 MLflow 2.9부터
> deprecated되었다. 지금은 Alias로 자유롭게 표시하는 방식이 권장된다.
> 자세한 내용: https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages

### UI에서 Alias 변경하기

버전 클릭 → **Aliases** 항목의 편집(연필) 아이콘 → 별칭 입력 → 저장

---

## 5. 실전 활용 팁

### 최고 성능 Run 찾기

1. `test_accuracy` 컬럼 헤더 클릭 → 내림차순 정렬
2. 가장 위에 있는 Run이 최고 성능

### 특정 Run의 모델 다운로드

1. Run 클릭 → Artifacts 탭
2. `model/` 폴더 클릭
3. 우측 상단 **Download** 버튼

### Run 삭제

1. 삭제할 Run 체크박스 선택
2. 상단 **Delete** 버튼 클릭

> 삭제된 Run은 UI에서 숨겨지지만 완전히 지워지진 않는다.
> 완전 삭제는 터미널에서: `& $mlflow gc --backend-store-uri "sqlite:///mlflow.db"`
> (이 실습 환경의 기본 백엔드가 `mlflow.db`이기 때문이다.)

---

## 전체 메뉴 구조 요약

```
http://localhost:5000
│
├── Experiments (실험 관리)
│   ├── 실험 목록 (왼쪽 사이드바)
│   ├── Run 목록 (표 형태)
│   │   ├── 검색·필터
│   │   ├── 컬럼 정렬
│   │   └── Run 선택 → Compare
│   └── Run 상세
│       ├── Overview (파라미터, 메트릭)
│       ├── Artifacts (저장된 파일)
│       └── Inputs (데이터셋)
│
└── Models (모델 레지스트리)
    ├── 모델 목록
    └── 모델 상세
        ├── 버전 목록
        ├── Alias 관리
        └── 연결된 Run 확인
```

---

## 자주 하는 실수

| 상황 | 원인 | 해결 |
|------|------|------|
| 실험이 안 보임 | `practice/` 폴더 밖에서 `mlflow ui` 실행 | `cd practice` 후 실행 |
| Inputs 탭이 없음 | `log_input()` 안 씀 | `03_데이터셋_추적.md` 참고 |
| Models 탭에 모델 없음 | `registered_model_name` 안 씀 | `04_모델_관리.md` 참고 |
| 변경사항이 반영 안 됨 | 브라우저 캐시 | `F5` (새로고침) 또는 `Ctrl+Shift+R` |
