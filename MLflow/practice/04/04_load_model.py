# [04-4] 저장된 모델 불러오기
# 목표: Alias 또는 버전 번호로 Model Registry에서 모델을 불러와 예측한다
# 실행 전: 01_log_model.py → 02_set_alias.py 순서로 실행되어 있어야 함
# URI 형식: models:/모델이름@alias  또는  models:/모델이름/버전번호
# 다음 파일: 05_run_pipeline.py

# -----------------------------------------------------------------------
# 모델 로드 URI 형식 정리
#
# runs:/<run_id>/model          → 특정 Run의 artifacts에서 로드
# models:/iris-classifier/1     → Registry v1에서 로드
# models:/iris-classifier@champion → champion alias가 가리키는 버전 로드 (권장)
#
# alias를 쓰면 버전이 바뀌어도 코드를 수정할 필요가 없다
# -----------------------------------------------------------------------

import mlflow
import mlflow.sklearn
import numpy as np

# ① 저장 위치 설정
mlflow.set_tracking_uri("sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db")

# ② @champion alias가 가리키는 버전을 자동으로 불러옴
model = mlflow.sklearn.load_model("models:/iris-classifier@champion")

# ③ 예측할 샘플 데이터 3개
#    [꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비]
sample = np.array([[5.1, 3.5, 1.4, 0.2],   # Setosa
                   [6.7, 3.0, 5.2, 2.3],   # Virginica
                   [5.8, 2.7, 4.1, 1.0]])  # Versicolor

# ④ 예측 결과 출력
#    붓꽃 종류: 0=Setosa, 1=Versicolor, 2=Virginica
species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
predictions = model.predict(sample)

print("=== 모델 예측 결과 ===")
for i, pred in enumerate(predictions):
    print(f"샘플 {i+1}: {species[pred]} (클래스 {pred})")
