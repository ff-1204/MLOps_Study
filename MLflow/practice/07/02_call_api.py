# [07-2] 서빙 중인 모델에 예측 요청
# 목표: 실행 중인 API 서버에 HTTP 요청을 보내 예측값을 받는다
# 실행 전: 01_start_server.py 실행 후 서버가 준비되어야 함
# 다음 파일: 03_stop_server.py

# -----------------------------------------------------------------------
# 요청 데이터 형식
#
# {"inputs": [[특징1, 특징2, 특징3, 특징4], ...]}
#   → 붓꽃 4가지 측정값: 꽃받침 길이/너비, 꽃잎 길이/너비
#
# 응답 형식
# {"predictions": [0, 2, 1]}
#   → 0=Setosa, 1=Versicolor, 2=Virginica
# -----------------------------------------------------------------------

import json
import requests

API_URL = "http://localhost:5001/invocations"

# ① 예측할 샘플 데이터 3개
data = {
    "inputs": [
        [5.1, 3.5, 1.4, 0.2],   # Setosa 예상
        [6.7, 3.0, 5.2, 2.3],   # Virginica 예상
        [5.8, 2.7, 4.1, 1.0],   # Versicolor 예상
    ]
}

# ② API 요청
try:
    response = requests.post(
        url=API_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
    )
    response.raise_for_status()

    # ③ 결과 출력
    predictions = response.json()["predictions"]
    species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

    print("=== 예측 결과 ===")
    for i, pred in enumerate(predictions):
        print(f"샘플 {i+1}: {species[pred]} (클래스 {pred})")

except requests.exceptions.ConnectionError:
    print("[오류] 서버에 연결할 수 없습니다.")
    print("01_start_server.py 를 먼저 실행하세요.")
