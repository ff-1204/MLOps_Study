# [03-1] 데이터셋 기록
# 목표: 훈련에 사용한 데이터셋을 Run에 함께 기록한다
# 핵심 개념: digest — 데이터 내용의 해시값, 동일 데이터인지 확인하는 데 사용

# -----------------------------------------------------------------------
# 왜 데이터셋도 기록해야 할까?
#
# 실험 결과(정확도)가 달라졌을 때, 파라미터만 바꾼 건지 데이터도 바뀐 건지
# 나중에 알 수 없다면 문제가 된다.
# MLflow에 '어떤 데이터'로 학습했는지 함께 기록하면
# 나중에 원인을 추적할 수 있다 — 마치 실험 노트에 재료 목록을 적는 것처럼.
# -----------------------------------------------------------------------

import os
import mlflow
import mlflow.data       # Dataset 객체 생성 함수가 들어 있는 모듈
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ① MLflow 저장 위치 설정
#    실험 기록을 저장할 SQLite DB 파일 경로를 지정한다
#    이 설정이 없으면 MLflow가 어디에 써야 할지 몰라 UI에서 보이지 않는다
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

# ② 실습용 CSV 파일 준비
#    data/ 폴더가 없으면 만들고, v1·v2 두 버전의 데이터를 생성한다
#    v1: 5행(데이터 적음), v2: 6행(데이터 1개 추가) — 버전 차이를 만들기 위함
os.makedirs("data", exist_ok=True)      # 폴더가 이미 있어도 오류 없이 넘어감
with open("data/train_v1.csv", "w") as f:
    f.write("feature1,feature2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n9.0,10.0,0\n")
with open("data/train_v2.csv", "w") as f:
    f.write("feature1,feature2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n9.0,10.0,0\n11.0,12.0,1\n")

# ③ 실험 이름 설정 — 없으면 자동 생성, 있으면 기존 실험에 Run 추가
mlflow.set_experiment("dataset-tracking-demo")


def run_with_dataset(csv_path, dataset_name, run_name):
    # ④ CSV 파일을 DataFrame으로 읽기
    df = pd.read_csv(csv_path)

    # ⑤ Dataset 객체 생성
    #    MLflow가 이 데이터를 추적할 수 있는 형태로 변환한다
    #    - source  : 원본 파일 경로 (어디서 가져왔는지)
    #    - name    : 데이터셋 이름 (UI에서 보이는 이름)
    #    - targets : 정답(label)이 들어있는 컬럼 이름
    #
    #    digest: 데이터 내용을 기반으로 자동 계산되는 '지문' 같은 값
    #            데이터가 1행이라도 바뀌면 digest도 완전히 달라진다
    dataset = mlflow.data.from_pandas(
        df, source=csv_path, name=dataset_name, targets="label"
    )

    with mlflow.start_run(run_name=run_name):
        # ⑥ Run에 데이터셋 등록 — 핵심!
        #    context="training": 이 데이터가 '학습용'임을 표시
        #    UI의 Inputs 탭에서 확인할 수 있다
        mlflow.log_input(dataset, context="training")

        # ⑦ 추가 정보 기록
        #    log_param: 문자열 값 기록 (데이터 버전 이름)
        #    log_metric: 숫자 값 기록 (행 수)
        mlflow.log_param("data_version", dataset_name)
        mlflow.log_metric("row_count", len(df))

        print(f"[{run_name}] name={dataset_name}, digest={dataset.digest[:8]}..., rows={len(df)}")


# ⑧ v1, v2 두 버전으로 각각 실험 실행
#    결과를 비교하면 digest가 다른 것을 확인할 수 있다 (데이터가 다르니까)
run_with_dataset("data/train_v1.csv", "my-dataset-v1", "v1-데이터-실험")
run_with_dataset("data/train_v2.csv", "my-dataset-v2", "v2-데이터-실험")
print("완료!")
