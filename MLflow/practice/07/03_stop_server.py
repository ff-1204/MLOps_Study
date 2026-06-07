# [07-3] 모델 서버 종료
# 목표: 01_start_server.py 가 실행한 서버 프로세스를 종료한다
# 실행 전: 01_start_server.py 실행 후 server.pid 파일이 있어야 함

import os
import sys
import signal

PID_FILE = "server.pid"

# ① PID 파일 확인
if not os.path.exists(PID_FILE):
    print("[오류] server.pid 파일이 없습니다. 01_start_server.py 로 서버를 시작하세요.")
    sys.exit(1)

# ② PID 읽기
with open(PID_FILE) as f:
    pid = int(f.read().strip())

# ③ 프로세스 종료
try:
    os.kill(pid, signal.SIGTERM)
    os.remove(PID_FILE)
    print(f"서버 종료 완료. (PID: {pid})")
except ProcessLookupError:
    print(f"PID {pid} 프로세스가 이미 종료되어 있습니다.")
    os.remove(PID_FILE)
