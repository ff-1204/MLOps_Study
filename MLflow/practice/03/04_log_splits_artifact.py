# [03-4] 분할 데이터셋을 아티팩트로 업로드
# 목표: train/val/test로 나눈 데이터를 파일로 저장하고 UI에서 직접 확인한다
# 실행 전: 01_log_dataset.py 실행 → data/ 폴더가 있어야 함
# 다음 파일: 05_get_dataset_info.py

# -----------------------------------------------------------------------
# 03_log_splits.py 와 02_log_artifact.py 를 합친 버전
#
# log_input()   → UI Inputs 탭: 분할 이름·digest·context 메타데이터
# log_artifact()→ UI Artifacts 탭: 실제 CSV 파일 (다운로드·내용 확인 가능)
# -----------------------------------------------------------------------

import os
import mlflow
import mlflow.data
import pandas as pd
from sklearn.model_selection import train_test_split

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")
mlflow.set_experiment("dataset-tracking-demo")

# ② 데이터 로드 및 분할
df = pd.read_csv("data/train_v2.csv")
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# ③ 분할된 데이터를 임시 CSV로 저장
#    log_artifact()는 '파일'을 올리므로 DataFrame을 먼저 파일로 써야 한다
os.makedirs("data/splits", exist_ok=True)
train_df.to_csv("data/splits/train.csv", index=False)
val_df.to_csv(  "data/splits/val.csv",   index=False)
test_df.to_csv( "data/splits/test.csv",  index=False)

# ④ Dataset 객체 생성 (log_input용)
train_ds = mlflow.data.from_pandas(train_df, source="data/train_v2.csv", name="split-train")
val_ds   = mlflow.data.from_pandas(val_df,   source="data/train_v2.csv", name="split-val")
test_ds  = mlflow.data.from_pandas(test_df,  source="data/train_v2.csv", name="split-test")

with mlflow.start_run(run_name="splits-with-artifact"):

    # ⑤ 메타데이터 기록 (Inputs 탭)
    mlflow.log_input(train_ds, context="training")
    mlflow.log_input(val_ds,   context="validation")
    mlflow.log_input(test_ds,  context="testing")

    # ⑥ 파일 자체 업로드 (Artifacts 탭)
    #    artifact_path="splits": UI에서 splits/ 폴더 안에 묶여서 보인다
    mlflow.log_artifact("data/splits/train.csv", artifact_path="splits")
    mlflow.log_artifact("data/splits/val.csv",   artifact_path="splits")
    mlflow.log_artifact("data/splits/test.csv",  artifact_path="splits")

    # ⑦ 행 수 기록
    mlflow.log_metric("train_size", len(train_df))
    mlflow.log_metric("val_size",   len(val_df))
    mlflow.log_metric("test_size",  len(test_df))

    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("완료! UI → splits-with-artifact Run → Artifacts 탭에서 splits/ 폴더 확인")
