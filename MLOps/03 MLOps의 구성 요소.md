# MLOps의 구성 요소

MLOps는 크게 **데이터 · 모델 · 서빙** 세 영역으로 구성된다.

```
데이터 → 모델 → 서빙
```

각 영역은 독립적으로 존재하는 게 아니라 순서대로 연결된다.  
좋은 데이터가 있어야 좋은 모델이 나오고, 좋은 모델이 있어야 좋은 서비스가 된다.

---

## 1. 데이터 (Data)

> ML의 출발점. 데이터의 품질이 모델 성능을 결정한다.  
> "쓰레기를 넣으면 쓰레기가 나온다(Garbage In, Garbage Out)"

| 단계 | 왜 필요한가 | 대표 도구 |
|---|---|---|
| **데이터 수집 파이프라인** | 다양한 곳에 흩어진 데이터를 자동으로 모으고 저장소로 전달 | Kafka, Airflow |
| **데이터 저장** | 수집된 데이터를 안정적으로 보관 | MySQL, Amazon S3 |
| **데이터 관리** | 어떤 데이터로 학습했는지 버전 관리, 품질 검증, 피처 공유 | DVC, TFDV, **Feast** |

> **DVC**: Git처럼 데이터를 버전 관리할 수 있는 도구. "이 모델은 3월 데이터로 학습했다"를 기록할 수 있다.  
> **TFDV**: 데이터가 이상하지 않은지(결측값, 분포 이상 등) 자동으로 검사해주는 도구.

### 피처 스토어 — Feast

**피처(Feature)**: ML 모델이 학습에 사용하는 입력값. 예) 사용자의 나이, 구매 횟수, 최근 접속 시간 등.

팀이 커질수록 팀원마다 같은 피처를 각자 따로 만드는 문제가 생긴다.  
**Feast(피처 스토어)** 는 이 피처들을 중앙에서 한 번만 만들고 모든 팀원이 공유해서 사용할 수 있게 해준다.

```
# Feast 사용 예시 (개념)
feature_store = FeatureStore()

# 학습할 때
training_data = feature_store.get_historical_features(entity="user_id", features=["age", "purchase_count"])

# 서빙할 때 (실시간)
serving_data = feature_store.get_online_features(entity="user_id", features=["age", "purchase_count"])
```

| Feast가 해결하는 문제 | 설명 |
|---|---|
| 피처 중복 개발 | 팀마다 같은 피처를 각자 구현하던 낭비를 없앰 |
| 학습/서빙 불일치 | 학습 때와 서빙 때 동일한 피처값을 보장 |
| 피처 재사용 | 한 번 만든 피처를 여러 모델이 공유해서 사용 |

---

## 2. 모델 (Model)

> 데이터를 바탕으로 모델을 만들고, 실험 결과를 기록하고, 학습을 자동화한다.

| 단계 | 왜 필요한가 | 대표 도구 |
|---|---|---|
| **모델 개발** | 다양한 알고리즘과 하이퍼파라미터를 실험하는 환경 | Jupyter Hub, Docker |
| **실험 추적 & 모델 관리** | 어떤 파라미터로 학습했고 성능이 얼마였는지 자동 기록·비교·등록 | **MLflow**, Git |
| **모델 학습 스케줄링** | 여러 팀원이 GPU 서버를 공유할 때 학습 작업을 자동으로 배분 | Kubernetes |

> **하이퍼파라미터(Hyperparameter)**: 학습 전에 사람이 직접 설정하는 값. 예) 학습률(learning rate), 에포크 수(epochs).

### 실험 추적 & 모델 레지스트리 — MLflow

수십 번 실험을 반복하다 보면 "아까 그 설정이 뭐였지?"를 기억하기 어렵다.  
**MLflow** 는 실험을 자동으로 기록하고, 모델을 버전별로 관리해주는 도구다.

**MLflow의 4가지 핵심 기능**

| 기능 | 설명 |
|---|---|
| **Tracking** | 실험마다 파라미터·성능 지표·모델 파일을 자동 기록 |
| **Projects** | 코드와 환경을 묶어서 실험을 재현 가능하게 패키징 |
| **Models** | 다양한 ML 프레임워크(PyTorch, TensorFlow 등)의 모델을 통일된 형식으로 저장 |
| **Model Registry** | 모델을 버전별로 등록하고 Staging → Production 단계로 승격 관리 |

```
# MLflow Tracking 사용 예시
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)   # 파라미터 기록
    mlflow.log_param("epochs", 30)

    # ... 학습 코드 ...

    mlflow.log_metric("accuracy", 0.93)        # 성능 기록
    mlflow.sklearn.log_model(model, "model")   # 모델 파일 저장
```

```
실험 1: lr=0.001, epochs=10  →  accuracy=0.87  (자동 기록)
실험 2: lr=0.01,  epochs=20  →  accuracy=0.91  (자동 기록)
실험 3: lr=0.001, epochs=30  →  accuracy=0.93  ← Model Registry에 등록
```

---

## 3. 서빙 (Serving)

> 학습된 모델을 실제 사용자가 쓸 수 있는 서비스 형태로 배포하고, 지속적으로 운영한다.

| 단계 | 왜 필요한가 | 대표 도구 |
|---|---|---|
| **모델 패키징** | 모델을 API 형태로 만들고 컨테이너로 포장해 어디서나 실행 | **BentoML**, Docker, FastAPI |
| **서빙 모니터링** | 배포 후 모델 성능이 떨어지거나 오류가 생기면 즉시 감지 | Prometheus, Grafana |
| **파이프라인 자동화** | 새 데이터가 쌓이면 자동으로 재학습→평가→배포까지 실행 | Airflow, Kubeflow |

> **API(Application Programming Interface)**: 다른 프로그램이 우리 모델을 호출해서 쓸 수 있게 만든 창구.  
> 예) 사진 앱이 우리 이미지 분류 모델 API를 호출해서 결과를 받아오는 방식.

### 모델 패키징 & 서빙 — BentoML

모델을 학습한 뒤 서비스로 만들려면 API 서버도 만들고, Docker 이미지도 만들고, 배포도 해야 한다.  
이 과정을 수동으로 하면 복잡하고 실수하기 쉽다.

**BentoML** 은 모델을 API 서버로 만드는 것부터 Docker 이미지 생성, 배포까지 자동화해주는 도구다.

```
# BentoML 사용 예시
import bentoml
from bentoml.io import NumpyNdarray

# 1. 학습된 모델 저장
bentoml.sklearn.save_model("my_model", trained_model)

# 2. API 서버 정의
svc = bentoml.Service("my_model_service")

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_data):
    model = bentoml.sklearn.load_model("my_model")
    return model.predict(input_data)
```

```bash
# 3. API 서버 실행 (한 줄)
bentoml serve my_model_service:latest

# 4. Docker 이미지 자동 생성 (한 줄)
bentoml build
bentoml containerize my_model_service:latest
```

| BentoML이 해결하는 문제 | 설명 |
|---|---|
| 복잡한 배포 과정 단순화 | API 서버 + Docker 이미지를 코드 몇 줄로 생성 |
| 다양한 프레임워크 지원 | PyTorch, TensorFlow, scikit-learn 등 어떤 모델이든 동일한 방식으로 배포 |
| Kubernetes 연동 | 생성된 Docker 이미지를 바로 Kubernetes에 배포 가능 |

---

## 세 도구의 연결 흐름

Feast, MLflow, BentoML은 각각 다른 영역을 담당하지만 하나의 파이프라인으로 연결된다.

```
[데이터 영역]          [모델 영역]           [서빙 영역]
Feast                  MLflow                BentoML
(피처 공유·관리)  →   (실험 추적·모델 등록) →  (모델 패키징·배포)
     ↑                                              ↓
     └──────────── 성능 저하 감지 시 재학습 ←────────┘
```

1. **Feast** 에서 피처를 가져와 학습
2. **MLflow** 로 실험 결과를 기록하고 좋은 모델을 Model Registry에 등록
3. **BentoML** 로 등록된 모델을 API 서버로 만들어 배포
4. 서빙 중 성능 저하가 감지되면 → 1번부터 다시 반복

---

## 다음으로

구성 요소 중 **모델 학습 스케줄링**과 **서빙 자동화**를 실제로 가능하게 해주는 핵심 인프라가 **쿠버네티스(Kubernetes)** 다.

03에서 Kubernetes가 여러 곳에 등장한 이유가 바로 여기 있다.  
쿠버네티스가 없으면 여러 팀원이 GPU를 공유하거나, 서버가 죽었을 때 자동 복구하거나, 트래픽에 맞게 서버를 늘리는 것이 모두 수동 작업이 된다.

→ `04 쿠버네티스가 필요한 이유.md` 에서 이어서 학습

---

## 참고

- Fast Campus, *ML을 Service화하기 위한 기술, MLOps* 강의 자료 (p.23~28)
- MLflow 공식 문서: https://mlflow.org/docs/latest/index.html
- BentoML 공식 문서: https://docs.bentoml.com
- Feast 공식 문서: https://docs.feast.dev
