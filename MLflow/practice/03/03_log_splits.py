# [03-3] 학습/검증/테스트 분할 기록
# 목표: train/val/test 각 분할을 context로 구분하여 Run에 기록한다
# 실행 전: 01_log_dataset.py 실행 → data/ 폴더와 실험이 생성되어 있어야 함
# 다음 파일: 04_get_dataset_info.py

# -----------------------------------------------------------------------
# 왜 데이터를 3개로 나눌까?
#
# train(학습용): 모델이 패턴을 배우는 데이터 — 시험 공부
# val(검증용)  : 학습 중간에 실력을 확인하는 데이터 — 모의고사
# test(테스트용): 최종 성능을 평가하는 데이터 — 실제 시험
#
# 세 데이터셋을 모두 MLflow에 기록하면
# 나중에 "어떤 비율로 나눴는지"도 추적할 수 있다.
# -----------------------------------------------------------------------

import mlflow
import mlflow.data
import pandas as pd
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")
mlflow.set_experiment("dataset-tracking-demo")

# ① 데이터 로드 — 01_log_dataset.py 가 생성한 v2 파일 사용
df = pd.read_csv("data/train_v2.csv")

# ② 데이터 분할
#    1차 분할: 전체 → 학습용(60%) + 나머지(40%)
#    2차 분할: 나머지(40%) → 검증용(20%) + 테스트용(20%)
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)
#    random_state=42: 실행할 때마다 같은 방식으로 나눠지도록 고정 (재현성)

# ③ 분할별 Dataset 객체 생성
#    세 개 다 원본 파일(train_v2.csv)에서 나왔으므로 source는 동일
#    name으로 어떤 분할인지 구분
train_ds = mlflow.data.from_pandas(train_df, source="data/train_v2.csv", name="split-train")
val_ds   = mlflow.data.from_pandas(val_df,   source="data/train_v2.csv", name="split-val")
test_ds  = mlflow.data.from_pandas(test_df,  source="data/train_v2.csv", name="split-test")

with mlflow.start_run(run_name="split-tracking"):
    # ④ context로 역할 구분하여 각각 등록
    #    context: 이 데이터가 어떤 용도인지 표시하는 태그
    #    UI Inputs 탭에서 training / validation / testing 으로 나뉘어 보인다
    mlflow.log_input(train_ds, context="training")    # 학습 데이터
    mlflow.log_input(val_ds,   context="validation")  # 검증 데이터
    mlflow.log_input(test_ds,  context="testing")     # 테스트 데이터

    # ⑤ 각 분할의 행 수 기록 — 나중에 비율 확인용
    mlflow.log_metric("train_size", len(train_df))
    mlflow.log_metric("val_size",   len(val_df))
    mlflow.log_metric("test_size",  len(test_df))

    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("분할 추적 완료!")
