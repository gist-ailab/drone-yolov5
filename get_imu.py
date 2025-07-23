import time
from pymavlink import mavutil
import sys

# Cube Orange 연결 설정
connection_string = '/dev/ttyACM0'
baud_rate = 115200

# 터미널 커서 제어를 위한 ANSI 이스케이프 코드
# \033[<ROW>;<COL>H : 커서를 ROW 행, COL 열로 이동
# \033[2K : 현재 줄의 내용을 지움
# \033[0G : 현재 줄의 맨 앞으로 커서를 이동 (col 1)

# 메시지별로 출력할 줄 번호 (0부터 시작)
# 예시: 하트비트/시작 메시지가 0, 1번 줄을 차지하고, 데이터는 그 다음부터 시작
LINE_ATTITUDE = 2
LINE_AHRS2 = 3
LINE_RAW_IMU = 4
TOTAL_DATA_LINES = 3 # ATTITUDE, AHRS2, RAW_IMU 메시지용 줄 수

# 각 메시지 타입의 마지막 출력 문자열을 저장 (초기화)
last_output_attitude = ""
last_output_ahrs2 = ""
last_output_raw_imu = ""

def update_line(line_num, text):
    """지정된 줄을 업데이트하는 함수"""
    # 커서를 해당 줄의 맨 앞으로 이동
    sys.stdout.write(f"\033[{line_num + 1};0H") # ANSI 코드는 1부터 시작하므로 +1
    sys.stdout.write("\033[2K")  # 현재 줄 지우기
    sys.stdout.write(text)
    sys.stdout.flush()

def reset_cursor_to_bottom():
    """커서를 출력 영역 맨 아래로 이동시키는 함수"""
    # 마지막 데이터 줄 + 1 (다음 출력 시작 위치)로 이동
    sys.stdout.write(f"\033[{LINE_RAW_IMU + 2};0H") # ANSI 코드는 1부터 시작하므로 +1, 추가 줄바꿈을 위해 +1 더
    sys.stdout.flush()

try:
    master = mavutil.mavlink_connection(connection_string, baud=baud_rate)

    # 연결 확인
    master.wait_heartbeat()
    print(f"하트비트 수신: 시스템 ID {master.target_system}, 컴포넌트 ID {master.target_component}")
    print("IMU 데이터 수신 중 (항목별 실시간 갱신)...")
    # 초기 데이터 표시를 위해 빈 줄 확보
    for _ in range(TOTAL_DATA_LINES):
        print("") # 각 데이터 항목을 위한 빈 줄 생성

    while True:
        # 논블로킹으로 변경하여 메시지가 없어도 다음 루프로 넘어갈 수 있도록 함
        msg = master.recv_match(type=['ATTITUDE', 'AHRS2', 'RAW_IMU'], blocking=False)

        if msg:
            if msg.get_type() == 'ATTITUDE':
                output_string = (
                    f"ATTITUDE: Roll={msg.roll:.2f}, Pitch={msg.pitch:.2f}, Yaw={msg.yaw:.2f}         "
                ) # 뒤에 공백을 추가하여 이전 긴 내용이 남아있지 않도록 함
                update_line(LINE_ATTITUDE, output_string)
                last_output_attitude = output_string

            elif msg.get_type() == 'AHRS2':
                output_string = (
                    f"AHRS2:    Roll={msg.roll:.2f}, Pitch={msg.pitch:.2f}, Yaw={msg.yaw:.2f}, "
                    f"Altitude={msg.altitude:.2f}         "
                )
                update_line(LINE_AHRS2, output_string)
                last_output_ahrs2 = output_string

            elif msg.get_type() == 'RAW_IMU':
                output_string = (
                    f"RAW_IMU:  Xacc={msg.xacc}, Yacc={msg.yacc}, Zacc={msg.zacc}, "
                    f"Xgyro={msg.xgyro}, Ygyro={msg.ygyro}, Zgyro={msg.zgyro}, "
                    f"Xmag={msg.xmag}, Ymag={msg.ymag}, Zmag={msg.zmag}         "
                )
                update_line(LINE_RAW_IMU, output_string)
                last_output_raw_imu = output_string
            
            # 메시지 수신 후 커서를 항상 맨 아래로 리셋
            reset_cursor_to_bottom()
        
        # CPU 사용량을 줄이기 위해 잠시 대기
        time.sleep(0.01) 

except KeyboardInterrupt:
    reset_cursor_to_bottom() # 종료 시 커서 맨 아래로 이동
    print("스크립트 종료.")
except Exception as e:
    reset_cursor_to_bottom() # 오류 시 커서 맨 아래로 이동
    print(f"오류 발생: {e}")
finally:
    # 프로그램이 종료될 때 터미널 설정을 원래대로 되돌리는 코드는 포함하지 않았습니다.
    # 복잡한 터미널 앱이 아니라면 크게 문제되지 않습니다.
    pass