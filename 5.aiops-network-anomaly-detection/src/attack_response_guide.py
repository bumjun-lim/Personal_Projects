import pandas as pd

attack_response_guide_data = [

    {
        "Attack_Type": "DoS / DDoS",
        "Elevated_Features": "CONNECTION_INTERVAL_TIME, TCP_SYN_FLAG_COUNT, TCP_RST_FLAG_COUNT, AVG_PACKET_SIZE, PACKET_SIZE_VOLATILITY, SUBFLOW_PACKET_COUNT, PACKET_RATE_PER_SEC, TOTAL_PACKET_VOLUME, TRAFFIC_BANDWIDTH_BPS",
        "Risk_Factor": "대규모 트래픽 폭증 및 비정상 세션 점유를 통한 시스템/서비스 자원 고갈",
        "Risk_Summary": "초당 패킷 수(PPS)와 대역폭(BPS)이 급증하고, 비정상적인 TCP Flag 및 패킷 크기 패턴이 동반되면서 정상적인 서비스 운영에 영향을 줄 수 있습니다.",
        "Practical_Response": "1. 방화벽/IPS/ISP 환경에 맞는 Rate Limiting 및 차단 정책 검토\n2. SYN Flood 대응을 위한 SYN Cookie 및 연결 Timeout 설정 검토\n3. 비정상 패킷 크기 및 TCP Flag 조합에 대한 필터링 정책 적용",
        "Additional_Check": "공격 소스 IP가 단일 대역인지 분산(Botnet 기반) 형태인지 확인하고 핵심 서비스의 세션 및 자원 사용 상태를 점검"
    },

    {
        "Attack_Type": "Recon / Scan",
        "Elevated_Features": "PACKET_TTL_AVERAGE, TCP_FIN_FLAG_COUNT, TCP_RST_FLAG_COUNT, MIN_PACKET_SIZE, INITIAL_WINDOW_SIZE, PORT_ACTIVITY_COUNT, SUBFLOW_PACKET_COUNT, UNIQUE_IP_COUNT",
        "Risk_Factor": "취약점 식별 및 서비스 포트/토폴로지 사전 탐색",
        "Risk_Summary": "포트 활동 빈도, 초소형 패킷, TCP 연결 패턴 및 TTL 등의 비정상적인 변화가 나타나면서 대상 네트워크의 활성 호스트와 서비스 정보를 탐색할 수 있습니다.",
        "Practical_Response": "1. 포트 스캔 탐지 시 방화벽 및 IPS 기반의 소스 IP 차단 정책 검토\n2. 외부에 불필요하게 노출된 서비스 및 관리 포트 점검\n3. 비정상적인 스캔 패턴에 대한 탐지 및 필터링 정책 적용",
        "Additional_Check": "정찰 시도가 특정 타겟 호스트에 집중되는지 또는 전체 서브넷을 대상으로 순차적인 스캔(Sweep) 형태로 발생하는지 확인"
    },

    {
        "Attack_Type": "Brute Force",
        "Elevated_Features": "LOG_EVENT_VOLUME, UNIQUE_IP_COUNT",
        "Risk_Factor": "반복적인 인증 시도를 통한 계정 탈취 및 권한 획득 시도",
        "Risk_Summary": "반복적인 인증 실패와 다수의 접근 소스가 나타날 경우 계정 탈취를 위한 자동화된 인증 시도 가능성을 확인할 수 있습니다.",
        "Practical_Response": "1. 로그인 실패 횟수 제한(Account Lockout) 및 CAPTCHA 적용 검토\n2. 반복적인 인증 실패가 발생하는 소스 IP에 대한 일시적인 접근 제한 검토\n3. 다중 인증(MFA) 적용을 통한 계정 보호 강화",
        "Additional_Check": "인증 관련 로그 이벤트의 급증 여부와 특정 계정을 대상으로 반복적인 로그인 실패가 발생하는지 확인"
    },

    {
        "Attack_Type": "Web Attack",
        "Elevated_Features": "AVG_PACKET_SIZE, PACKET_SIZE_VOLATILITY",
        "Risk_Factor": "웹 애플리케이션 취약점을 통한 비정상 데이터 주입 및 제어권 탈취",
        "Risk_Summary": "웹 애플리케이션 요청에 악성 입력이 포함되는 공격으로, 네트워크 Flow에서는 패킷 크기나 변동성 등의 이상 징후가 동반될 수 있으나 Flow 정보만으로 공격 여부를 직접 판단하기는 어렵습니다.",
        "Practical_Response": "1. WAF의 페이로드 검증 및 최신 공격 시그니처 적용 검토\n2. 웹 애플리케이션의 입력값 검증(Validation) 및 파라미터 처리 방식 점검\n3. 비정상적인 패킷 크기 및 변동성 패턴에 대한 모니터링 연계",
        "Additional_Check": "웹 서버 액세스 로그의 오류 코드(500, 403 등) 급증 여부와 URI별 요청 패턴 및 비정상 입력 발생 여부를 확인"
    },

    {
        "Attack_Type": "Malware / Botnet",
        "Elevated_Features": "TRAFFIC_FLOW_ASYMMETRY, UNIQUE_IP_COUNT, LOG_EVENT_VOLUME",
        "Risk_Factor": "감염된 호스트의 C&C 서버 통신 및 분산 제어",
        "Risk_Summary": "비대칭적인 트래픽 흐름이나 다수의 외부 IP와의 통신, 비정상적인 시스템 이벤트 증가가 함께 나타날 경우 악성코드 감염 또는 봇넷 활동 가능성을 확인할 수 있습니다.",
        "Practical_Response": "1. C&C 의심 도메인/IP에 대한 방화벽 및 DNS 기반 차단 정책 검토\n2. 감염이 의심되는 호스트의 네트워크 격리 및 보안 솔루션을 통한 추가 검사\n3. 비정상 세션 흐름과 시스템 로그를 연계한 통합 모니터링",
        "Additional_Check": "내부 호스트의 외부 통신 패턴에서 비대칭성이 지속되는지 확인하고 주기적인 하트비트(Beaconing) 통신 여부를 점검"
    },

    {
        "Attack_Type": "MITM",
        "Elevated_Features": "PACKET_TTL_AVERAGE, TCP_ACK_FLAG_COUNT, TCP_WINDOW_SIZE_AVERAGE",
        "Risk_Factor": "통신 세션 가로채기 및 패킷 변조, 스푸핑",
        "Risk_Summary": "통신 경로 또는 패킷 처리 과정의 변화로 인해 TTL, TCP ACK, Window 크기 등의 비정상적인 패턴이 나타날 수 있으나 Flow 정보만으로 MITM 공격을 직접 판단하기는 어렵습니다.",
        "Practical_Response": "1. ARP 스푸핑 및 비정상 세션 변조 여부에 대한 탐지 정책 검토\n2. TLS 기반 암호화 통신 및 인증서 검증 정책 강화\n3. 네트워크 스위치의 포트 보안 및 ARP 관련 보호 기능 적용 검토",
        "Additional_Check": "동일 브로드캐스트 도메인 내 MAC 주소 변동 여부와 비정상적인 ARP 응답 및 라우팅 경로 변경 여부를 확인"
    },

    {
        "Attack_Type": "Infiltration",
        "Elevated_Features": "AVG_PAYLOAD_LENGTH, MAX_PACKET_SIZE, TOTAL_PAYLOAD_LENGTH, TRAFFIC_FLOW_ASYMMETRY, TOTAL_PACKET_VOLUME",
        "Risk_Factor": "내부 네트워크 침투 및 대용량 데이터 유출",
        "Risk_Summary": "비정상적으로 큰 페이로드와 트래픽량, 비대칭적인 통신 흐름이 나타날 경우 내부 시스템 침투 이후의 데이터 이동이나 유출 가능성을 확인할 수 있습니다.",
        "Practical_Response": "1. DLP 및 EDR 솔루션과 연계한 데이터 이동 및 호스트 이상 행위 탐지\n2. 내부 주요 서버 간 접근 통제 및 네트워크 세분화 정책 검토\n3. 대용량 페이로드 및 비정상 세션에 대한 차단 또는 격리 정책 검토",
        "Additional_Check": "내부 시스템 간 비정상적인 데이터 이동 경로와 접근 권한 변경 및 권한 상승(Privilege Escalation) 흔적을 점검"
    },

    {
        "Attack_Type": "Heartbleed",
        "Elevated_Features": "AVG_PAYLOAD_LENGTH",
        "Risk_Factor": "OpenSSL 취약점을 이용한 서버 메모리 데이터 노출",
        "Risk_Summary": "TLS Heartbeat 처리 취약점을 악용해 서버 메모리의 민감 정보를 노출시키는 공격입니다. Flow 수준에서는 payload 관련 이상 패턴이 관찰될 수 있지만 해당 Feature만으로 Heartbleed를 식별할 수는 없습니다.",
        "Practical_Response": "1. 사용 중인 OpenSSL 버전 및 취약 여부 점검 후 패치 또는 업그레이드 검토\n2. 취약 서비스에 대한 IDS/IPS 기반 탐지 및 차단 정책 적용\n3. 정보 유출이 의심될 경우 세션 토큰 및 인증서 등의 폐기와 재발급 검토",
        "Additional_Check": "사용 중인 SSL/TLS 환경과 OpenSSL 버전을 확인하고 취약 버전(예: 1.0.1 계열) 사용 여부를 점검"
    }

]

# DataFrame으로 변환 후 CSV 저장
df_attack_guide = pd.DataFrame(attack_response_guide_data)
df_attack_guide.to_csv("attack_response_guide.csv", index=False, encoding="utf-8-sig")
print("attack_response_guide.csv 파일이 성공적으로 생성되었습니다!")