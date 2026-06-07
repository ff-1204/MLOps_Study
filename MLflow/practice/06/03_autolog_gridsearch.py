# [06-3] GridSearch + autolog
# 목표: GridSearchCV의 모든 파라미터 조합 결과를 autolog로 자동 기록한다
# 핵심: autolog는 GridSearch의 각 조합을 자식 Run으로 자동 생성함
# 이전 파일: 02_check_autolog.py

# -----------------------------------------------------------------------
# GridSearchCV의 핵심 역할
#
# 9가지 조합을 전부 시도 → 각 조합의 CV 점수 비교 → 최적 조합 자동 선택
#
# C=0.1로 해보고, C=1.0으로 해보고... 하는 수동 작업을 GridSearch가 대신 해준다
# gs.best_params_ ← 가장 성능 좋은 조합
# gs.best_score_  ← 그 조합의 점수
#
# autolog는 "어떤 조합이 얼마나 나왔는지"를 MLflow에 전부 기록해주는 역할
#
# GridSearch + autolog 구조
#
# 부모 Run: "grid-search"
# └── 자식 Run: C=0.1, max_iter=50   → 1번 조합
# └── 자식 Run: C=0.1, max_iter=100  → 2번 조합
# └── 자식 Run: C=0.1, max_iter=200  → 3번 조합
# └── ... (총 3×3 = 9개 자식 Run 자동 생성)
#
# UI에서 부모 Run 안에 자식 Run들이 펼쳐지는 구조로 확인할 수 있다
# -----------------------------------------------------------------------

import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

# ② 실험 이름 설정
mlflow.set_experiment("autolog-gridsearch")

# ③ 자동 로깅 활성화
#    GridSearchCV의 9가지 조합(3×3)을 자식 Run으로 자동 기록
mlflow.autolog()

# ④ 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# ⑤ 탐색할 파라미터 조합 정의 — C 3개 × max_iter 3개 = 총 9가지
param_grid = {
    "C":        [0.1, 1.0, 10.0],
    "max_iter": [50, 100, 200],
}

# ⑥ GridSearch 실행 — fit() 시점에 9개 자식 Run이 자동 생성됨
with mlflow.start_run(run_name="grid-search"):
    gs = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=3,
        scoring="accuracy",
    )
    gs.fit(X_train, y_train)
    print(f"최적 파라미터: {gs.best_params_}")
    print(f"최적 점수    : {gs.best_score_:.4f}")
