# [06-2] autolog로 기록된 내용 확인
# 목표: autolog가 자동으로 기록한 파라미터·메트릭·아티팩트 목록을 출력한다
# 실행 전: 01_set_autolog.py 실행 후 Run이 있어야 함
# 다음 파일: 03_autolog_gridsearch.py

# -----------------------------------------------------------------------
# autolog가 자동으로 기록하는 항목 (sklearn 기준)
#
# Params   : 모델 생성자에 넘긴 파라미터 (C, max_iter 등 전부)
# Metrics  : training_score (훈련 정확도)
# Artifacts: model/ 폴더 (MLmodel, model.pkl, conda.yaml, input_example.json)
#
# 수동으로 log_param()을 하나씩 쓰지 않아도 이 항목들이 자동으로 채워진다
# -----------------------------------------------------------------------

import mlflow

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

client = mlflow.tracking.MlflowClient()

# ② autolog-test 실험에서 가장 최근 Run 가져오기
experiment = client.get_experiment_by_name("autolog-test")
runs = client.search_runs(experiment.experiment_id)

run = runs[0]

# ③ autolog가 기록한 항목 출력
print(f"Status    : {run.info.status}")
print(f"Params    : {run.data.params}")
print(f"Metrics   : {run.data.metrics}")
print(f"Artifacts : {[f.path for f in client.list_artifacts(run.info.run_id)]}")
