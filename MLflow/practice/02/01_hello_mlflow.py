# [02-1] 실험 추적 첫 걸음
# 목표: MLflow에 파라미터·메트릭을 기록하고 UI에서 확인한다
# 저장 위치:
# - 실험 기록(파라미터·메트릭): mlflow.db (SQLite 데이터베이스)
# - 아티팩트(모델·파일): mlruns/ 폴더
# 실행 했는 데 안보이는 경우 '$env:MLFLOW_TRACKING_URI' 환경변수 확인

import mlflow

# 실험 이름 지정 — 없으면 자동 생성, 있으면 기존 실험에 Run 추가
mlflow.set_experiment("first-experiment")

with mlflow.start_run(run_name="첫번째-실험"):
    
    # log_param: 모델 설정값 기록 (문자열/숫자 모두 가능) - Parameters 탭에서 확인
    mlflow.log_param("name", "홍길동")
    mlflow.log_param("score", 100)
    
    # log_metric: 훈련 결과 기록 (숫자만 가능) - Model metrics (모델 성능 지표)에서 확인
    mlflow.log_metric("result", 0.95)
    print("기록 완료!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
