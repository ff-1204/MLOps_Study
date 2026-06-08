# [03-6] 데이터셋 실제 데이터 조회
# 목표: Run에 연결된 데이터셋의 실제 데이터(행/열)를 코드로 불러온다
# 실행 전: 01_log_dataset.py 실행 후 Run이 있어야 함
# 다음 파일: 07_get_from_source.py

# -----------------------------------------------------------------------
# 05_get_metadata.py 와 차이점
#
# 05: 데이터셋 이름·digest·context 등 메타데이터만 출력
# 06: 메타데이터 + 실제 데이터 내용(행/열)까지 불러와서 출력
#
# get_source().load(): 기록된 source 경로의 파일을 로컬로 가져와
#                      pandas로 읽을 수 있는 경로를 반환한다
# -----------------------------------------------------------------------

import mlflow
import mlflow.data
import pandas as pd

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

# ② MlflowClient 생성 및 실험 조회
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("dataset-tracking-demo")
runs = client.search_runs(experiment.experiment_id)

print("=== Run별 데이터셋 내용 ===")
for run in runs:
    print(f"\n{'='*40}")
    print(f"Run: {run.info.run_name}")

    for di in run.inputs.dataset_inputs:
        d = di.dataset

        # ③ context 꺼내기 (MLflow 3.x: tags는 리스트)
        context = next((t.value for t in di.tags if t.key == "mlflow.data.context"), "N/A")
        print(f"\n  [{d.name}]  context={context}  digest={d.digest[:8]}...")

        # ④ 실제 데이터 불러오기
        #    get_source(): 이 데이터셋의 원본 위치 정보를 담은 Source 객체
        #    source.load(): 원본 파일을 로컬로 가져오고 경로를 반환
        try:
            source    = mlflow.data.get_source(d)
            local_path = source.load()
            df = pd.read_csv(local_path)

            # ⑤ 데이터 요약 출력
            print(f"  행 수: {len(df)}행  열 수: {len(df.columns)}열")
            print(f"  컬럼: {list(df.columns)}")
            print(df.to_string(index=False))   # 전체 행 출력 (데이터가 작으므로)

        except Exception as e:
            # source 경로가 없거나 접근 불가한 경우 (분할 데이터 등)
            print(f"  데이터 로드 불가: {e}")
