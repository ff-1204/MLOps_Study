# [07-1] MLflow 모델 서버 실행
# 목표: iris-classifier@champion 모델을 API 서버로 백그라운드 실행한다
# 핵심: 서버는 이 스크립트가 끝난 뒤에도 계속 실행됨
# 다음 파일: 02_call_api.py

# -----------------------------------------------------------------------
# 서빙이란?
#
# 학습된 모델을 HTTP API로 노출해 다른 프로그램이 예측을 요청할 수 있게 함
#
# mlflow models serve → http://localhost:5001 실행
#       ↓
# POST /invocations  {"inputs": [[...]]}
#       ↓
# {"predictions": [0]}
# -----------------------------------------------------------------------

import os
import sys
import time
import subprocess
import requests

MLFLOW  = r"C:\Users\myesu\miniconda3\Scripts\mlflow.exe"
DB_URI  = "sqlite:///C:/Users/myesu/Project/MLOps_Study/MLflow/practice/mlflow.db"
PORT    = 5001
PID_FILE = "server.pid"

# ① uvicorn이 있는 Scripts 폴더를 PATH에 추가
env = os.environ.copy()
env["PATH"]             = r"C:\Users\myesu\miniconda3\Scripts;" + env.get("PATH", "")
env["PYTHONUTF8"]       = "1"
env["MLFLOW_TRACKING_URI"] = DB_URI

# ② 이미 실행 중인지 확인
try:
    r = requests.get(f"http://localhost:{PORT}/health", timeout=1)
    if r.status_code == 200:
        print(f"서버가 이미 실행 중입니다. (port {PORT})")
        sys.exit(0)
except Exception:
    pass

# ③ 서버 백그라운드 실행
log_file = open("server.log", "w")
proc = subprocess.Popen(
    [
        MLFLOW, "models", "serve",
        "--model-uri", "models:/iris-classifier@champion",
        "--port",      str(PORT),
        "--no-conda",
    ],
    env=env,
    stdout=log_file,
    stderr=log_file,
)

# ④ PID 저장 — 03_stop_server.py 에서 종료할 때 사용
with open(PID_FILE, "w") as f:
    f.write(str(proc.pid))

print(f"서버 시작 중... (PID: {proc.pid})")

# ⑤ 서버가 준비될 때까지 대기 (최대 30초)
for i in range(30):
    try:
        r = requests.get(f"http://localhost:{PORT}/health", timeout=1)
        if r.status_code == 200:
            print(f"\n서버 준비 완료! → http://localhost:{PORT}")
            print("02_call_api.py 를 실행하세요.")
            sys.exit(0)
    except Exception:
        print(".", end="", flush=True)
        time.sleep(1)

print("\n서버 시작 시간 초과. 로그를 확인하세요.")
proc.terminate()
sys.exit(1)
