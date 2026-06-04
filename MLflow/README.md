# MLflow 학습 목차

## 전체 학습 순서

```
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07
개념   설치  실험  데이터  모델  UI   자동   서빙
              추적  추적   관리       로깅
```

---

## 문서 목록

| 순서 | 파일 | 내용 | 난이도 |
|------|------|------|--------|
| 00 | [MLflow란](./00_MLflow란.md) | MLflow가 뭔지, 왜 필요한지, 4가지 핵심 기능 | ⭐ |
| 01 | [환경설정](./01_환경설정.md) | 설치, 첫 UI 실행, 실습 폴더 구성 | ⭐ |
| 02 | [실험 추적](./02_실험_추적.md) | Experiment·Run·파라미터·메트릭 기록 | ⭐⭐ |
| 03 | [데이터셋 추적](./03_데이터셋_추적.md) | Dataset API, digest, 분할 추적, 복원 | ⭐⭐ |
| 04 | [모델 관리](./04_모델_관리.md) | 모델 저장·불러오기, Model Registry, 스테이지 | ⭐⭐ |
| 05 | [MLflow UI 가이드](./05_MLflow_UI_가이드.md) | UI 각 탭 설명, 실험 비교, 활용 팁 | ⭐⭐ |
| 06 | [자동 로깅](./06_자동_로깅.md) | `mlflow.autolog()` 한 줄 자동 기록 | ⭐⭐ |
| 07 | [모델 서빙](./07_모델_서빙.md) | `mlflow models serve`로 API 서버 실행 | ⭐⭐⭐ |

---

## 실습 전 배경 지식

MLflow를 배우기 전에 아래 문서를 먼저 읽으면 도움이 된다.

| 파일 | 내용 |
|------|------|
| [../DataManagement/01_Data Management 란](../DataManagement/01_Data%20Management%20란.md) | 데이터 버전 관리가 왜 필요한지 |
| [../DataManagement/02_MLflow 실습](../DataManagement/02_MLflow%20실습.md) | MLflow 전체 흐름 요약 실습 |

---

## 단계별 학습 가이드

### 1단계 — 개념 이해 (00 ~ 01)

> MLflow가 처음이라면 여기서 시작

- `00_MLflow란.md` — MLflow의 필요성과 전체 구조 파악
- `01_환경설정.md` — 설치 후 UI가 뜨는지 확인

**체크포인트:** `http://localhost:5000` 접속 성공

---

### 2단계 — 핵심 기능 실습 (02 ~ 04)

> MLflow의 3가지 핵심 기능을 직접 써본다

- `02_실험_추적.md` — 파라미터·메트릭 기록, 여러 Run 비교
- `03_데이터셋_추적.md` — 어떤 데이터로 학습했는지 기록
- `04_모델_관리.md` — 모델 저장·불러오기, 버전 관리

**체크포인트:** UI에서 Run 목록, Dataset, 모델 버전이 보임

---

### 3단계 — UI 활용 (05)

> 코드 없이 UI만으로 실험 결과 분석

- `05_MLflow_UI_가이드.md` — 비교 차트, 필터, 모델 스테이지 변경

**체크포인트:** 여러 Run을 UI에서 Compare로 비교 성공

---

### 4단계 — 효율화 & 배포 (06 ~ 07)

> 실무에서 자주 쓰는 고급 기능

- `06_자동_로깅.md` — `autolog()` 한 줄로 모든 기록 자동화
- `07_모델_서빙.md` — 학습된 모델을 API 서버로 실행

**체크포인트:** API에 데이터를 보내고 예측 결과를 받음

---

## 문서 커버리지

### GitHub 기준 MLflow 기능 대비 커버 현황

```
기초 ML 실험 관리   ████████████  완료  (02 ~ 05)
모델 저장 & 서빙    ████████████  완료  (04, 07)
자동 로깅           ████████████  완료  (06)
데이터셋 추적       ████████████  완료  (03)
팀 협업 서버        ██░░░░░░░░░░  미포함 — Remote Tracking Server
LLM / AI 기능      ░░░░░░░░░░░░  미포함 — 고급 과정
```

### 현재 문서에 포함된 것 ✅

| 기능 | 문서 |
|------|------|
| Experiment / Run 기록 | 02_실험_추적 |
| Dataset Tracking (digest, 분할) | 03_데이터셋_추적 |
| Model Registry (버전·스테이지) | 04_모델_관리 |
| MLflow UI (비교·필터·차트) | 05_MLflow_UI_가이드 |
| `mlflow.autolog()` 자동 로깅 | 06_자동_로깅 |
| `mlflow models serve` API 서빙 | 07_모델_서빙 |

### 포함되지 않은 것 ❌

**초중급 — 필요하면 추가 학습 권장**

| 기능 | 설명 |
|------|------|
| Remote Tracking Server | 팀원과 실험 결과를 공유하는 중앙 서버 설정 |
| MLflow Projects (`mlflow run`) | 코드 실행 환경까지 패키징해서 재현 |

**고급 — 대학·현업 수준**

| 기능 | 설명 |
|------|------|
| LLM Tracing | ChatGPT 등 LLM 앱의 동작 추적 (OpenTelemetry 기반) |
| LLM Evaluation | LLM 응답 품질 자동 평가 (내장 메트릭 50개 이상) |
| Prompt Management | 프롬프트 버전 관리 및 자동 최적화 |
| AI Gateway | OpenAI·Anthropic·Gemini 등 통합 API |
| Production 배포 | Docker·Kubernetes·AWS SageMaker 배포 |

---

## 핵심 명령어 모음

```powershell
# 변수 설정 (터미널 시작마다 입력)
$python = "C:\Users\myesu\miniconda3\python.exe"
$mlflow = "C:\Users\myesu\miniconda3\Scripts\mlflow.exe"

# UI 실행
& $mlflow ui

# Python 파일 실행
& $python 파일명.py

# 모델 서빙
& $mlflow models serve --model-uri "models:/모델이름@champion" --port 5001 --no-conda
```

---

## 핵심 API 한눈에 보기

```python
import mlflow

# ① 실험 설정
mlflow.set_experiment("실험이름")

# ② 자동 로깅 (선택) — start_run() 전에 호출해야 함
mlflow.autolog()

# ③ Run 시작
with mlflow.start_run(run_name="실행이름"):

    # ④ 파라미터·메트릭 기록 (수동 로깅 시)
    mlflow.log_param("key", value)
    mlflow.log_metric("key", value)

    # ⑤ 데이터셋 기록
    dataset = mlflow.data.from_pandas(df, source=..., name=...)
    mlflow.log_input(dataset, context="training")

    # ⑥ 모델 저장
    mlflow.sklearn.log_model(model, "model", registered_model_name="모델이름")
```
