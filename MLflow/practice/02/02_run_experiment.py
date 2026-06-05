# [02-2] 실제 모델로 실험 추적
# 목표: sklearn 모델 훈련 결과를 파라미터·메트릭으로 기록한다

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 실험 이름 지정 — 없으면 자동 생성, 있으면 기존 실험에 Run 추가
mlflow.set_experiment("iris-classification")

# 파라미터를 dict로 관리하면 log_params()에 한 번에 넘길 수 있음
PARAMS = {
    "max_iter": 100,  # 모델이 수렴할 때까지 반복할 최대 횟수
    "C": 1.0,         # 규제 강도 — 작을수록 강한 규제 (과적합 방지)
    "random_state": 42,  # 데이터 분할 시 랜덤 시드 (재현성 보장)
}

# run_name: UI에서 Run을 구별하는 이름
with mlflow.start_run(run_name="기본설정"):

    # 붓꽃(iris) 데이터 — 꽃받침·꽃잎 크기로 3가지 품종 분류
    iris = load_iris()

    # test_size=0.2: 전체 데이터의 20%를 테스트용으로 분리
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=0.2,
        random_state=PARAMS["random_state"]
    )

    # 실험 전 설정값(파라미터)을 한 번에 기록 — UI에서 Run 간 비교에 사용됨
    mlflow.log_params(PARAMS)

    # 모델 학습
    model = LogisticRegression(max_iter=PARAMS["max_iter"], C=PARAMS["C"])
    model.fit(X_train, y_train)

    # 실험 후 결과값(메트릭)을 기록 — 숫자만 가능
    # train_accuracy: 학습 데이터 정확도 (높아도 과적합일 수 있음)
    # test_accuracy : 테스트 데이터 정확도 (모델의 실제 성능 지표)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))

    mlflow.log_metric("train_accuracy", round(train_acc, 4))
    mlflow.log_metric("test_accuracy",  round(test_acc, 4))

    print(f"train_accuracy: {train_acc:.4f}")
    print(f"test_accuracy : {test_acc:.4f}")
