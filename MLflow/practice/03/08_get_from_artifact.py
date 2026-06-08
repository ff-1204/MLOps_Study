# [03-8] artifact로 저장된 파일 꺼내기
# 목표: log_artifact()로 업로드한 파일을 다운로드해서 데이터를 읽어온다
# 실행 전: 02_log_artifact.py 또는 04_log_splits_artifact.py 실행 후 아티팩트가 있어야 함
# 이전 파일: 07_get_from_source.py

# -----------------------------------------------------------------------
# 07_get_from_source.py 와 차이점
#
# 07: log_input()의 source 경로를 통해 원본 파일을 읽어옴
#     → 원본 파일이 그 자리에 있어야 동작함
#
# 08: log_artifact()로 MLflow 서버에 올린 파일을 다운로드해서 읽어옴
#     → 원본 파일이 삭제되어도 MLflow 서버에 있으면 복원 가능
# -----------------------------------------------------------------------

import os
import mlflow
import pandas as pd

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

# ② MlflowClient 생성 및 실험 조회
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("dataset-tracking-demo")
runs = client.search_runs(experiment.experiment_id)

# ③ 아티팩트가 있는 Run만 필터링
#    list_artifacts(): 해당 Run의 아티팩트 목록 반환
#    아티팩트가 하나도 없으면 빈 리스트 반환
artifact_runs = [r for r in runs if client.list_artifacts(r.info.run_id)]

if not artifact_runs:
    print("아티팩트가 있는 Run이 없습니다.")
    print("02_log_artifact.py 또는 04_log_splits_artifact.py 를 먼저 실행하세요.")
    exit()

print("=== 아티팩트 파일 다운로드 및 읽기 ===\n")

for run in artifact_runs:
    artifacts = client.list_artifacts(run.info.run_id)

    # ④ 아티팩트 폴더 목록 출력
    print(f"Run: {run.info.run_name}")
    print(f"  아티팩트 목록: {[a.path for a in artifacts]}")

    # ⑤ 아티팩트 다운로드
    #    download_artifacts(): MLflow 서버에 올라간 파일을 로컬로 내려받음
    #    artifact_path: 다운받을 아티팩트 경로 (폴더명 또는 파일명)
    #    dst_path: 저장할 로컬 폴더 (생략 시 임시 폴더에 저장)
    download_dir = f"data/downloaded/{run.info.run_name}"
    os.makedirs(download_dir, exist_ok=True)

    for artifact in artifacts:
        local_path = client.download_artifacts(
            run_id=run.info.run_id,
            path=artifact.path,       # 아티팩트 경로 (예: "datasets", "splits")
            dst_path=download_dir
        )

        # ⑥ 다운로드된 CSV 파일 읽기
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path)
                    print(f"\n  [{file}]  {len(df)}행 × {len(df.columns)}열")
                    print(df.to_string(index=False))

    print()
