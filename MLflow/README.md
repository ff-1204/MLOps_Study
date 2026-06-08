# MLflow 학습

MLflow 설치부터 모델 서빙까지 단계별로 정리한 학습 자료.

---

## 학습 순서

```
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
개념   설치  실험  데이터  모델  UI   자동   서빙  평가
              추적  추적   관리       로깅       
```

---

## 문서 목록

| 순서 | 파일 | 내용 | 난이도 |
|------|------|------|--------|
| 00 | [MLflow란](./00_MLflow란.md) | MLflow가 뭔지, 왜 필요한지, 4가지 핵심 기능 | ⭐ |
| 01 | [환경설정](./01_환경설정.md) | 설치, UI 실행 | ⭐ |
| 02 | [실험 추적](./02_실험_추적.md) | Experiment·Run·파라미터·메트릭 기록 | ⭐⭐ |
| 03 | [데이터셋 추적](./03_데이터셋_추적.md) | Dataset API, digest, 분할 추적, 복원 | ⭐⭐ |
| 04 | [모델 관리](./04_모델_관리.md) | 모델 저장·불러오기, Model Registry, Alias | ⭐⭐ |
| 05 | [MLflow UI 가이드](./05_MLflow_UI_가이드.md) | UI 각 탭 설명, 실험 비교 | ⭐⭐ |
| 06 | [자동 로깅](./06_자동_로깅.md) | `mlflow.autolog()` 한 줄 자동 기록 | ⭐⭐ |
| 07 | [모델 서빙](./07_모델_서빙.md) | `mlflow models serve`로 REST API 실행 | ⭐⭐⭐ |
| 08 | [모델 평가](./08_모델_평가.md) | `mlflow.evaluate()` 자동 지표·그래프 | ⭐⭐⭐ |

---

## 실습 파일 순서

> 모든 실습은 같은 `practice/mlflow.db`에 기록을 쌓는다. UI 한 곳에서 전부 조회된다.

### 01 — 환경설정 (`practice/01/`)

| 순서 | 파일 | 내용 |
|------|------|------|
| 1 | `start_mlflow.py` | MLflow UI 실행 (`practice/mlflow.db` 백엔드) |

### 02 — 실험 추적 (`practice/02/`)

| 순서 | 파일 | 내용 |
|------|------|------|
| 1 | `01_hello_mlflow.py` | 첫 파라미터·메트릭 기록 |
| 2 | `02_run_experiment.py` | 실제 모델로 실험 추적 |
| 3 | `03_run_multiple.py` | 여러 파라미터 반복 실험 |
| 4 | `04_get_best_run.py` | 코드로 최고 성능 Run 조회 |

### 03 — 데이터셋 추적 (`practice/03/`)

| 순서 | 파일 | 내용 |
|------|------|------|
| 1 | `01_log_dataset.py` | 데이터셋 메타데이터 기록 (`log_input`, data/ 폴더 생성) |
| 2 | `02_log_artifact.py` | 파일 자체 업로드 (`log_artifact`) |
| 3 | `03_log_splits.py` | train/val/test 분할 메타데이터 기록 |
| 4 | `04_log_splits_artifact.py` | 분할 데이터 파일 업로드 |
| 5 | `05_get_metadata.py` | 데이터셋 메타데이터 조회 |
| 6 | `06_get_data.py` | 실제 데이터 내용 조회 |
| 7 | `07_get_from_source.py` | source 경로로 데이터 복원 |
| 8 | `08_get_from_artifact.py` | artifact에서 데이터 복원 |

### 04 — 모델 관리 (`practice/04/`)

| 순서 | 파일 | 내용 |
|------|------|------|
| 1 | `01_log_model.py` | 모델 저장 및 Registry 등록 (v1, v2 생성) |
| 2 | `02_set_alias.py` | 버전에 Alias 설정 |
| 3 | `03_get_alias.py` | Alias 조회 |
| 4 | `04_load_model.py` | Alias·버전 번호로 모델 불러와 예측 |
| 5 | `05_run_pipeline.py` | 데이터셋 기록 + 훈련 + 모델 저장 종합 실습 |

### 05 — MLflow UI 가이드

별도 실습 파일 없음. 02~04 실습으로 쌓인 데이터를 UI에서 직접 조작하며 익힌다.

### 06 — 자동 로깅 (`practice/06/`)

| 순서 | 파일 | 내용 |
|------|------|------|
| 1 | `01_set_autolog.py` | `autolog()` 활성화 + 모델 학습 |
| 2 | `02_get_autolog.py` | 자동 기록된 항목 조회 |
| 3 | `03_autolog_gridsearch.py` | GridSearchCV 최적 조합 탐색 + 자동 기록 |

### 07 — 모델 서빙 (`practice/07/`)

| 순서 | 파일 | 내용 |
|------|------|------|
| 1 | `01_start_server.py` | 모델 서버 백그라운드 실행 |
| 2 | `02_call_api.py` | 서빙 중인 모델에 HTTP 예측 요청 |
| 3 | `03_stop_server.py` | 서버 종료 |

### 08 — 모델 평가 (`practice/08/`) 

| 순서 | 파일 | 내용 |
|------|------|------|
| 1 | `01_evaluate_basic.py` | `evaluate()`로 지표·그래프 자동 생성 |
| 2 | `02_compare_models.py` | 두 모델을 같은 기준으로 비교 |
| 3 | `03_get_eval_metrics.py` | 평가 결과를 코드로 조회 |

---

## 핵심 명령어

```powershell
# 변수 설정 (터미널 시작마다 입력)
$python  = "C:\Users\myesu\miniconda3\python.exe"
$mlflow  = "C:\Users\myesu\miniconda3\Scripts\mlflow.exe"
$env:PYTHONUTF8 = "1"
$env:PATH = "C:\Users\myesu\miniconda3\Scripts;" + $env:PATH

# UI 실행 (practice/mlflow.db 백엔드)
& $python practice\01\start_mlflow.py

# Python 파일 실행
& $python practice\02\01_hello_mlflow.py

# 모델 서빙 (DB 경로는 환경변수로 전달 — serve에는 --backend-store-uri 옵션이 없음)
$env:MLFLOW_TRACKING_URI = "sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db"
& $mlflow models serve --model-uri "models:/모델이름@champion" --port 5001 --no-conda
```

---

## 핵심 API

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///.../practice/mlflow.db")  # 저장 위치 지정
mlflow.set_experiment("실험이름")   # Experiment 지정
mlflow.autolog()                    # 자동 로깅 (start_run 전에 호출)

with mlflow.start_run(run_name="실행이름"):
    mlflow.log_param("key", value)                        # 파라미터 기록
    mlflow.log_metric("key", value)                       # 메트릭 기록

    dataset = mlflow.data.from_pandas(df, source=..., name=...)
    mlflow.log_input(dataset, context="training")         # 데이터셋 기록

    mlflow.sklearn.log_model(model, name="model",
        registered_model_name="모델이름")                  # 모델 저장 및 등록

# 모델 불러오기
mlflow.sklearn.load_model("models:/모델이름@champion")    # Alias
mlflow.sklearn.load_model("models:/모델이름/1")           # 버전 번호
```

---

## 커버리지

| 기능 | 포함 여부 |
|------|-----------|
| Experiment / Run 기록 | ✅ 02 |
| Dataset Tracking (digest, 분할) | ✅ 03 |
| Model Registry (버전·Alias) | ✅ 04 |
| MLflow UI (비교·필터) | ✅ 05 |
| `mlflow.autolog()` | ✅ 06 |
| `mlflow models serve` | ✅ 07 |
| `mlflow.evaluate()` (모델 평가) | ✅ 08  |
| Remote Tracking Server | ❌ 팀 공유 서버 |
| MLflow Projects | ❌ 실행 환경 패키징 |
| LLM Tracing / Evaluation | ❌ 고급 과정 |
