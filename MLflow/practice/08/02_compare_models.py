# [08-2] 여러 모델 평가 결과 비교
# 목표: 두 모델을 같은 기준(evaluate)으로 평가해 성능을 객관적으로 비교한다
# 실행 전: 별도 조건 없음 (이 파일이 모델을 직접 학습·평가함)
# 다음 파일: 03_get_eval_metrics.py

# -----------------------------------------------------------------------
# 왜 evaluate로 비교할까?
#
# 모델마다 직접 accuracy_score를 계산하면 코드가 제각각이 되기 쉽다.
# evaluate()는 모든 모델을 똑같은 지표 세트로 평가하므로
# "어떤 모델이 어떤 지표에서 더 나은지"를 공정하게 비교할 수 있다.
# -----------------------------------------------------------------------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")
mlflow.set_experiment("iris-evaluation")

# ② 데이터 준비
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

# ③ 비교할 모델 두 종류
models = {
    "logistic-regression": LogisticRegression(max_iter=200),
    "decision-tree":       DecisionTreeClassifier(max_depth=3, random_state=42),
}

# ④ 각 모델을 별도 Run에서 학습 → 저장 → evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    with mlflow.start_run(run_name=name):
        info = mlflow.sklearn.log_model(model, name="model")
        result = mlflow.evaluate(
            info.model_uri,
            data=test_df,
            targets="target",
            model_type="classifier",
        )
        results[name] = result.metrics
        print(f"[{name}] 평가 완료")

# ⑤ 두 모델의 핵심 지표 비교 출력
print("\n=== 모델 비교 ===")
print(f"{'지표':16s} {'logistic':>12s} {'decision-tree':>14s}")
for key in ["accuracy_score", "f1_score", "precision_score", "recall_score"]:
    lr = results["logistic-regression"].get(key, 0)
    dt = results["decision-tree"].get(key, 0)
    print(f"{key:16s} {lr:12.4f} {dt:14.4f}")
