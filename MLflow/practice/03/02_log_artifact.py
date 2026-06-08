# [03-2] 데이터 파일 자체를 아티팩트로 업로드
# 목표: CSV 파일을 MLflow에 업로드해서 UI Artifacts 탭에서 직접 확인한다
# 실행 전: 01_log_dataset.py 실행 → data/ 폴더가 있어야 함
# 다음 파일: 03_log_splits.py

# -----------------------------------------------------------------------
# log_input() vs log_artifact() 차이
#
# log_input()   : 데이터셋 '정보(이름, digest, 경로)'만 기록 — 파일은 업로드 안 됨
#                 → UI Inputs 탭에서 메타데이터만 확인 가능
#
# log_artifact(): 파일 자체를 MLflow 서버에 업로드
#                 → UI Artifacts 탭에서 파일 내용 직접 확인 가능
#
# 실무에서는 데이터가 크면 log_input()으로 경로만 기록하고,
# 작은 샘플 파일이나 결과 파일은 log_artifact()로 업로드한다.
# -----------------------------------------------------------------------

import mlflow
import mlflow.data
import pandas as pd

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")
mlflow.set_experiment("dataset-tracking-demo")

with mlflow.start_run(run_name="artifact-업로드"):

    # ② 데이터 파일을 아티팩트로 업로드
    #    log_artifact(경로): 파일 1개를 업로드
    #    artifact_path: UI에서 보일 폴더 이름 (생략 시 루트에 저장)
    mlflow.log_artifact("data/train_v1.csv", artifact_path="datasets")
    mlflow.log_artifact("data/train_v2.csv", artifact_path="datasets")

    print("아티팩트 업로드 완료!")
    print("UI → artifact-업로드 Run → Artifacts 탭에서 파일을 확인할 수 있다")

    # ③ log_input()과 함께 사용하면 메타데이터 + 파일 모두 기록 가능
    df_v1 = pd.read_csv("data/train_v1.csv")
    dataset_v1 = mlflow.data.from_pandas(
        df_v1, source="data/train_v1.csv", name="my-dataset-v1", targets="label"
    )
    mlflow.log_input(dataset_v1, context="training")
    #    → Inputs 탭: 데이터셋 이름·digest·경로
    #    → Artifacts 탭: 실제 CSV 파일 (다운로드 가능)
