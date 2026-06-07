# [03-7] source 경로로 데이터 꺼내기
# 목표: log_input()에 기록된 source 경로를 통해 원본 파일을 불러온다
# 실행 전: 01_log_dataset.py 실행 후 Run이 있어야 함
# 이전 파일: 06_get_data.py

# -----------------------------------------------------------------------
# 언제 쓸까?
#
# "6개월 전 실험에서 쓴 데이터가 정확히 뭐였지?"
# Run ID 없이도 파라미터로 Run을 검색해서 그때 데이터를 그대로 꺼낼 수 있다.
# -----------------------------------------------------------------------

import mlflow
import pandas as pd

# ① 저장 위치 설정 — 01_log_dataset.py 와 동일한 DB를 바라봐야 한다
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

# ② MlflowClient 생성
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("dataset-tracking-demo")

# ② 파라미터로 특정 Run 검색
#    filter_string: SQL WHERE 절처럼 조건을 문자열로 지정
#    "params.data_version = 'my-dataset-v1'" → data_version 파라미터가 'my-dataset-v1'인 Run
#    01_log_dataset.py에서 mlflow.log_param("data_version", ...) 으로 기록한 값
runs = client.search_runs(
    experiment.experiment_id,
    filter_string="params.data_version = 'my-dataset-v1'"
)

# ③ 결과가 없으면 종료 — 01_log_dataset.py 를 먼저 실행했는지 확인
if not runs:
    print("v1 Run을 찾지 못했습니다. 01_log_dataset.py 를 먼저 실행하세요.")
    exit()

run = runs[0]   # 검색 결과 중 첫 번째 Run
print(f"v1 Run ID: {run.info.run_id[:8]}...")

# ④ Run에 등록된 데이터셋 정보 꺼내기
#    dataset_inputs[0]: 이 Run에 등록된 첫 번째 데이터셋
logged_dataset = run.inputs.dataset_inputs[0].dataset
print(f"name  : {logged_dataset.name}")
print(f"digest: {logged_dataset.digest}")

# ⑤ 원본 파일 복원
#    get_source: 이 데이터셋이 어디서 왔는지 알려주는 Source 객체를 가져옴
#    source.load(): 원본 파일을 로컬에 다운로드하고, 그 경로를 반환
source = mlflow.data.get_source(logged_dataset)
local_path = source.load()   # 복원된 파일의 로컬 경로

# ⑥ 복원된 파일을 DataFrame으로 읽어서 출력
df = pd.read_csv(local_path)
print(f"\n복원된 데이터 ({len(df)}행):")
print(df)
