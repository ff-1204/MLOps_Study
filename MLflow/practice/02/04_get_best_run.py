# [02-4] 코드로 최고 성능 Run 조회
# 목표: UI 없이 Python 코드로 실험 결과를 검색하고 최고 성능 Run을 찾는다
# 실행 전: 03_run_multiple.py 를 먼저 실행해서 Run이 있어야 함

import mlflow

# 저장 위치 설정 — 다른 챕터와 같은 DB를 바라보게 함
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

# MlflowClient: UI 없이 코드로 실험·Run·모델을 조회·수정하는 클라이언트
# 모델 배포 자동화, CI/CD 파이프라인 등에서 활용됨
client = mlflow.tracking.MlflowClient()

# 실험 이름으로 Experiment 객체 조회 — experiment_id를 얻기 위해 필요
experiment = client.get_experiment_by_name("iris-classification")

# search_runs: 조건에 맞는 Run 목록 반환
# - filter_string: SQL WHERE 절처럼 조건 지정 (예: "metrics.test_accuracy > 0.9")
# - order_by: 정렬 기준 (DESC = 내림차순)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_accuracy DESC"]  # 정확도 높은 순
)

print("=== 실험 결과 순위 ===")
for i, run in enumerate(runs, 1):
    acc = run.data.metrics.get("test_accuracy", 0)
    C   = run.data.params.get("C", "?")
    itr = run.data.params.get("max_iter", "?")
    print(f"{i}위: accuracy={acc:.4f}  C={C}  max_iter={itr}")

# runs[0]: order_by 기준 첫 번째 = 가장 성능이 좋은 Run
best = runs[0]
print(f"\n최고 성능 Run ID: {best.info.run_id}")
# Run ID는 모델 불러오기, 아티팩트 조회 등에 활용됨
