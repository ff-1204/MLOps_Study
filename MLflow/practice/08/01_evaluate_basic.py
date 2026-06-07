# [08-1] 모델 자동 평가 (mlflow.evaluate)
# 목표: evaluate() 한 번으로 여러 평가 지표와 그래프를 자동 계산·기록한다
# 핵심: 직접 accuracy_score를 계산하지 않아도 지표·혼동행렬·ROC가 자동 생성됨
# 다음 파일: 02_compare_models.py

# -----------------------------------------------------------------------
# 직접 계산 vs evaluate()
#
# 지금까지 (직접 계산):
#   acc = accuracy_score(y_test, model.predict(X_test))
#   mlflow.log_metric("accuracy", acc)        ← 지표 하나하나 직접
#
# evaluate() (자동):
#   mlflow.evaluate(model_uri, data=test_df, targets="target", model_type="classifier")
#   → accuracy·precision·recall·f1·log_loss·roc_auc 자동 계산
#   → 혼동행렬·ROC·PR 곡선 이미지까지 Run에 자동 저장
# -----------------------------------------------------------------------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")
mlflow.set_experiment("iris-evaluation")

# ② 데이터 준비 — evaluate에 넘길 test 데이터는 정답(target) 컬럼을 포함해야 함
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

# ③ 모델 학습
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

with mlflow.start_run(run_name="evaluate-basic"):
    # ④ 모델 저장 — evaluate는 저장된 모델 URI를 받아 평가한다
    info = mlflow.sklearn.log_model(model, name="model")

    # ⑤ 자동 평가
    #    data        : 정답 컬럼을 포함한 평가용 DataFrame
    #    targets     : 정답이 들어있는 컬럼 이름
    #    model_type  : "classifier"(분류) 또는 "regressor"(회귀)
    result = mlflow.evaluate(
        info.model_uri,
        data=test_df,
        targets="target",
        model_type="classifier",
    )

    # ⑥ 자동 계산된 지표 출력
    print("=== 자동 계산된 평가 지표 ===")
    for key, value in result.metrics.items():
        print(f"  {key:20s}: {value:.4f}")

    print("\nUI → iris-evaluation → evaluate-basic Run → Artifacts 탭에서")
    print("confusion_matrix.png, roc_curve_plot.png 등을 확인하세요.")
