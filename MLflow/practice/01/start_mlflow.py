import os
from mlflow.cli import cli

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("MLflow UI 시작 중...")
print("브라우저에서 http://localhost:5000 접속")
print("종료하려면 Ctrl+C")

cli(["ui", "--host", "0.0.0.0", "--port", "5000"], standalone_mode=False)
# 첫 실행 시 "mlflow.db" 파일이 생성됩니다. 이 파일은 MLflow의 기본 SQLite 데이터베이스입니다.