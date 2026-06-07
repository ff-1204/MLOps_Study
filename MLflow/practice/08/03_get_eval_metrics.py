# [08-3] 평가 결과를 코드로 조회
# 목표: evaluate가 기록한 지표와 아티팩트(혼동행렬 등)를 코드로 가져온다
# 실행 전: 01_evaluate_basic.py 또는 02_compare_models.py 실행 후 Run이 있어야 함
# 이전 파일: 02_compare_models.py

# -----------------------------------------------------------------------
# evaluate가 Run에 남기는 것
#
# Metrics  : accuracy_score, f1_score, precision_score, recall_score,
#            log_loss, roc_auc 등
# Artifacts: confusion_matrix.png, roc_curve_plot.png,
#            precision_recall_curve_plot.png, per_class_metrics.csv 등
#
# 이 파일은 그 결과를 UI 없이 코드로 조회한다.
# -----------------------------------------------------------------------

import mlflow

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("iris-evaluation")

# ② accuracy_score 기준 내림차순으로 Run 조회 — 가장 잘 평가된 모델이 위로
runs = client.search_runs(
    experiment.experiment_id,
    order_by=["metrics.accuracy_score DESC"],
)

print("=== 평가 결과 순위 (accuracy 기준) ===")
for i, run in enumerate(runs, 1):
    name = run.info.run_name
    acc  = run.data.metrics.get("accuracy_score", 0)
    f1   = run.data.metrics.get("f1_score", 0)
    auc  = run.data.metrics.get("roc_auc", 0)
    print(f"{i}위 [{name}]  accuracy={acc:.4f}  f1={f1:.4f}  roc_auc={auc:.4f}")

    # ③ 이 Run에 저장된 평가 아티팩트(그래프·csv) 목록
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    eval_files = [a for a in artifacts if a.endswith((".png", ".csv"))]
    print(f"     평가 아티팩트: {eval_files}")
