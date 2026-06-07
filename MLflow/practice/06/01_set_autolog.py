# [06-1] 자동 로깅 (autolog)
# 목표: mlflow.autolog() 한 줄로 파라미터·메트릭·모델을 자동 기록한다
# 핵심: autolog()는 반드시 start_run() 보다 먼저 호출해야 함
# 다음 파일: 02_get_autolog.py → 03_autolog_gridsearch.py

# -----------------------------------------------------------------------
# autolog vs 수동 로깅 비교
#
# 수동 로깅:
#   mlflow.log_param("C", 1.0)
#   mlflow.log_metric("accuracy", 0.97)
#   mlflow.sklearn.log_model(model, "model")
#   → 기록할 항목을 직접 하나씩 써야 함
#
# autolog:
#   mlflow.autolog()  ← 이 한 줄이 전부
#   → sklearn 훈련 시 파라미터·메트릭·모델을 전부 자동 기록
#   → 어떤 항목이 기록됐는지는 02_get_autolog.py 에서 확인
# -----------------------------------------------------------------------

import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

# ② 실험 이름 설정
mlflow.set_experiment("autolog-test")

# ③ 자동 로깅 활성화
#    start_run() 보다 먼저 호출해야 적용된다
mlflow.autolog()

# ④ 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# ⑤ 모델 학습 — autolog가 활성화되어 있으면 fit() 시점에 자동 기록됨
with mlflow.start_run(run_name="autolog-run"):
    model = LogisticRegression(max_iter=100, C=1.0)
    model.fit(X_train, y_train)
    print("autolog 실행 완료. UI에서 기록된 항목을 확인하세요.")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
