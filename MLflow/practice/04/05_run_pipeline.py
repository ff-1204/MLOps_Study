# [04-5] 전체 파이프라인 (데이터셋 기록 + 훈련 + 모델 저장)
# 목표: 02~04에서 배운 기능을 하나의 흐름으로 합쳐서 실행한다
# 흐름: 데이터 준비 → 데이터셋 기록 → 파라미터/메트릭 기록 → 모델 저장 및 등록
# 이전 파일: 04_load_model.py

# -----------------------------------------------------------------------
# 지금까지 배운 것을 한 파일에서 순서대로 실행한다
#
# ① 데이터 준비 (03챕터 내용)
# ② 데이터셋 기록 — mlflow.log_input()
# ③ 파라미터 기록 — mlflow.log_params()
# ④ 모델 학습 및 메트릭 기록 — mlflow.log_metric()
# ⑤ 모델 저장 + Registry 등록 — mlflow.sklearn.log_model()
# -----------------------------------------------------------------------

import mlflow
import mlflow.sklearn
import mlflow.data
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")
mlflow.set_experiment("iris-full-pipeline")

# ② 데이터 준비
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ③ Dataset 객체 생성 (log_input에 넘길 형태로 변환)
train_dataset = mlflow.data.from_pandas(
    train_df, source="sklearn.load_iris", name="iris-train", targets="target"
)
test_dataset = mlflow.data.from_pandas(
    test_df, source="sklearn.load_iris", name="iris-test", targets="target"
)

params = {"max_iter": 200, "C": 1.0}

with mlflow.start_run(run_name="full-pipeline"):

    # ④ 데이터셋 기록
    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(test_dataset,  context="testing")

    # ⑤ 파라미터 기록
    mlflow.log_params(params)

    # ⑥ 모델 학습
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test  = test_df.drop("target", axis=1)
    y_test  = test_df["target"]

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # ⑦ 메트릭 기록
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))

    mlflow.log_metric("train_accuracy", round(train_acc, 4))
    mlflow.log_metric("test_accuracy",  round(test_acc, 4))

    # ⑧ 모델 저장 + Registry 등록
    mlflow.sklearn.log_model(model, name="model", registered_model_name="iris-final")

    print(f"train_accuracy : {train_acc:.4f}")
    print(f"test_accuracy  : {test_acc:.4f}")
    print("전체 파이프라인 완료!")
