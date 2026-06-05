import os
from mlflow.cli import cli

# practice/ 폴더의 mlflow.db를 공통 DB로 사용
practice_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_uri = f"sqlite:///{practice_dir}/mlflow.db".replace("\\", "/")

os.environ["MLFLOW_TRACKING_URI"] = db_uri

print("MLflow UI 시작 중...")
print("브라우저에서 http://localhost:5000 접속")
print("종료하려면 Ctrl+C")
print()
print("[ 실습 터미널에 아래 명령어 입력 후 실습 파일 실행 ]")
print(f'$env:MLFLOW_TRACKING_URI = "{db_uri}"')
print()

cli(["ui", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", db_uri], standalone_mode=False)
# 첫 실행 시 "mlflow.db" 파일이 생성됩니다. 이 파일은 MLflow의 기본 SQLite 데이터베이스입니다.
# MLflow UI는 기본적으로 "mlruns" 디렉토리에 실험과 실행 데이터를 저장합니다. 이 디렉토리도 첫 실행 시 생성됩니다.
# MLflow UI가 실행 중인 동안에는 터미널에서 Ctrl+C를 눌러 종료할 수 있습니다. 종료 후에도 "mlflow.db"와 "mlruns" 디렉토리는 남아있으며, 다음에 UI를 다시 시작할 때 계속 사용할 수 있습니다.