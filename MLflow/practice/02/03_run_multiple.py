# [02-3] 여러 파라미터로 반복 실험
# 목표: 파라미터를 바꿔가며 여러 Run을 만들고 UI에서 비교한다
# UI 확인: 실행 후 http://localhost:5000 → iris-classification → Compare 탭

import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_experiment("iris-classification")

# 데이터는 모든 Run에서 동일하게 고정 — 파라미터 차이만 비교하기 위해
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 시도할 파라미터 조합 목록 — 각 항목이 Run 하나가 됨
# C: 규제 강도. 작을수록 단순한 모델, 클수록 복잡한 모델
# max_iter: 학습 반복 횟수가 부족하면 모델이 수렴하지 못할 수 있음
experiments = [
    {"run_name": "C=0.1",   "max_iter": 100, "C": 0.1},   # 강한 규제
    {"run_name": "C=1.0",   "max_iter": 100, "C": 1.0},   # 기본값
    {"run_name": "C=10.0",  "max_iter": 100, "C": 10.0},  # 약한 규제
    {"run_name": "iter=50", "max_iter": 50,  "C": 1.0},   # 반복 횟수 줄임
]

for exp in experiments:
    # with 블록이 끝나면 Run이 자동으로 닫힘 — FINISHED 상태로 저장됨
    with mlflow.start_run(run_name=exp["run_name"]):
        params = {"max_iter": exp["max_iter"], "C": exp["C"]}
        mlflow.log_params(params)

        # **params: dict를 키워드 인수로 풀어서 전달 (max_iter=..., C=... 와 동일)
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        test_acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("test_accuracy", round(test_acc, 4))

        print(f"[{exp['run_name']}] test_accuracy = {test_acc:.4f}")
