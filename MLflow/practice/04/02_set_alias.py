# [04-2] 모델 버전에 Alias 설정
# 목표: 등록된 모델 버전에 champion/challenger 같은 별칭(Alias)을 붙인다
# 실행 전: 01_log_model.py 실행 → iris-classifier v1, v2 가 등록되어 있어야 함
# 다음 파일: 03_get_alias.py

# -----------------------------------------------------------------------
# Alias란?
#
# 버전 번호(v1, v2) 대신 의미 있는 이름으로 모델을 부를 수 있게 해주는 별칭
#
# champion  : 현재 서비스 중인 최고 성능 모델
# challenger: 새로 도전하는 후보 모델
#
# 코드에서 버전 번호 대신 alias를 쓰면
# 나중에 버전이 바뀌어도 코드를 수정하지 않아도 된다
# -----------------------------------------------------------------------

import mlflow
import mlflow.sklearn
import numpy as np

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

client = mlflow.tracking.MlflowClient()

# ② Alias 설정
#    version=1 → challenger (테스트 중인 후보)
#    version=2 → champion  (현재 최고 모델)
client.set_registered_model_alias(name="iris-classifier", alias="challenger", version=1)
client.set_registered_model_alias(name="iris-classifier", alias="champion",   version=2)

print("Alias 설정 완료!")

# ③ 설정된 Alias 확인
#    search_model_versions는 aliases를 반환하지 않으므로
#    get_model_version으로 각 버전을 개별 조회해야 aliases가 채워짐
versions = client.search_model_versions("name='iris-classifier'")
for v in versions:
    detail = client.get_model_version("iris-classifier", v.version)
    print(f"  Version {detail.version}: aliases={detail.aliases}")

# ④ Alias로 모델 로드 및 예측 테스트
#    models:/모델이름@alias 형식으로 alias가 가리키는 버전을 자동으로 불러옴
model = mlflow.sklearn.load_model("models:/iris-classifier@champion")
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
print(f"\n@champion 모델 예측: {model.predict(sample)}")
