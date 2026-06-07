# [04-3] Alias 조회
# 목표: 어떤 버전에 어떤 Alias가 붙어있는지 코드로 확인한다
# 실행 전: 02_set_alias.py 실행 후 Alias가 설정되어 있어야 함
# 다음 파일: 04_load_model.py

# -----------------------------------------------------------------------
# 02_manage_versions.py 와 차이점
#
# 02: Alias를 설정하고 곧바로 확인
# 03: Alias만 전문으로 조회 — 현재 어떤 버전이 어떤 역할인지 빠르게 파악
#
# get_model_version_by_alias: alias 이름 → 버전 번호 조회
# get_registered_model      : 모델 전체 정보(모든 alias 포함) 조회
# -----------------------------------------------------------------------

import mlflow

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

client = mlflow.tracking.MlflowClient()

# ② alias 이름으로 버전 번호 조회
#    alias가 없으면 예외(Exception) 발생 → try/except 로 처리
try:
    mv = client.get_model_version_by_alias("iris-classifier", "champion")
    print(f"champion   → Version {mv.version}")
    mv2 = client.get_model_version_by_alias("iris-classifier", "challenger")
    print(f"challenger → Version {mv2.version}")
except Exception as e:
    print(f"오류: {e}")
    print("02_manage_versions.py 를 먼저 실행하세요.")

# ③ 모델 전체 alias 목록 조회
#    get_registered_model: 모델의 모든 정보(alias 딕셔너리 포함) 반환
#    aliases: {"alias이름": "버전번호"} 형태의 딕셔너리
rm = client.get_registered_model("iris-classifier")
print(f"\nRegistered Model aliases: {rm.aliases}")
