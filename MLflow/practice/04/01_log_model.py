# [04-1] 모델 저장 및 Model Registry 등록
# 목표: 훈련된 모델을 Run에 저장하고 Model Registry에 버전으로 등록한다
# 핵심: registered_model_name 지정 시 자동으로 버전(v1, v2 ...) 부여됨
# 다음 파일: 02_set_alias.py

# -----------------------------------------------------------------------
# 왜 Model Registry가 필요한가?
#
# 파일로 관리하면:  model_최종_진짜최종.pkl  ← 어떤 게 맞는 건지 모름
# Registry로 관리: iris-classifier v1, v2, v3  ← 버전·성능 한눈에 비교 가능
# -----------------------------------------------------------------------

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

# ② 실험 이름 설정
mlflow.set_experiment("iris-classification")

# ③ 데이터 준비 — 붓꽃(iris) 데이터로 꽃 종류 분류
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)


def log_and_register(model, run_name):
    with mlflow.start_run(run_name=run_name) as run:
        # ④ 모델 학습 및 성능 기록
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", round(acc, 4))

        # ⑤ 모델 저장 + Registry 등록 (핵심!)
        #    artifact_path        : UI Artifacts 탭에서 보일 폴더 이름
        #    registered_model_name: 이 이름으로 Registry에 버전이 쌓임
        #    input_example        : 입력 데이터 예시 (UI에서 확인 가능)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="iris-classifier",
            input_example=X_test[:3],
        )
        print(f"[{run_name}] accuracy={acc:.4f}, Run ID={run.info.run_id[:8]}...")
        return run.info.run_id


# ⑥ 두 종류의 모델을 같은 이름으로 등록 → 자동으로 v1, v2 부여됨
run_id = log_and_register(LogisticRegression(max_iter=200), "logistic-regression")
log_and_register(DecisionTreeClassifier(max_depth=3), "decision-tree")

print(f"\n첫번째 Run ID (전체): {run_id}")
print("UI → Models 탭에서 iris-classifier v1, v2 확인")
