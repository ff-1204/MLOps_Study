# [03-5] 데이터셋 메타데이터 조회
# 목표: Run에 연결된 데이터셋 이름·digest·context를 코드로 확인한다
# 실행 전: 01_log_dataset.py 실행 후 Run이 있어야 함
# 다음 파일: 06_get_data.py

# -----------------------------------------------------------------------
# MLflow UI 없이 코드로 실험 결과를 조회하는 방법
#
# MlflowClient: MLflow 서버에 명령을 내리는 '리모컨' 같은 객체
#               실험 목록 조회, Run 검색, 데이터셋 정보 확인 등에 사용
# -----------------------------------------------------------------------

import mlflow

# ① 저장 위치 설정 — 01_log_dataset.py 와 동일한 DB를 바라봐야 같은 데이터가 보인다
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

# ② MlflowClient 생성 — MLflow 서버와 통신하는 객체
client = mlflow.tracking.MlflowClient()

# ② 실험 이름으로 Experiment 객체 조회
#    experiment.experiment_id: 내부에서 실험을 구분하는 숫자 ID
experiment = client.get_experiment_by_name("dataset-tracking-demo")

# ③ 실험에 속한 모든 Run 목록 가져오기
runs = client.search_runs(experiment.experiment_id)

# ④ Run별로 사용한 데이터셋 정보 출력
print("=== 실험별 사용 데이터셋 ===")
for run in runs:
    print(f"\nRun: {run.info.run_name}")

    # run.data.metrics: 이 Run에 기록된 메트릭(숫자 결과값) 딕셔너리
    # .get('row_count', 'N/A'): row_count 가 없으면 'N/A' 출력
    print(f"  row_count : {run.data.metrics.get('row_count', 'N/A')}")

    # ⑤ 이 Run에 등록된 데이터셋 목록 순회
    #    dataset_inputs: log_input()으로 등록한 데이터셋들의 리스트
    for di in run.inputs.dataset_inputs:
        d = di.dataset   # Dataset 객체 (name, digest, source 등 포함)

        print(f"  dataset  : {d.name}")    # 데이터셋 이름
        print(f"  digest   : {d.digest}")  # 데이터 '지문' — 같으면 같은 데이터, 다르면 다른 데이터

        # di.tags: InputTag 객체의 리스트 (MLflow 3.x)
        # 각 태그는 .key / .value 속성을 가진다
        # 'mlflow.data.context': log_input(context=...) 로 지정한 용도 값
        context = next((t.value for t in di.tags if t.key == "mlflow.data.context"), "N/A")
        print(f"  context  : {context}")
