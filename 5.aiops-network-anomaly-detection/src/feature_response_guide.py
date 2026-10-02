import pandas as pd

def generate_full_response_guide_csv(filename="feature_response_guide.csv"):
    response_guide_data = [
        # ==========================================
        # 1. Control_State (7개)
        # ==========================================
        {
            "Standard_Feature_ID": "CONNECTION_INTERVAL_TIME",
            "Standard_Feature_Name": "Connection Interval Time",
            "Risk_Group": "Control_State",
            "Risk_Summary": "흐름 간격 시간(IAT) 비정상 지연",
            "Associated_Attack": "DoS / DDoS",
            "Investigation_Point": "세션 간 패킷 도달 간격이 의도적으로 늘어지며 자원을 점유하는지 확인",
            "Response_Guide": "1. 웹서버/게이트웨이 타임아웃(Timeout) 설정 강화\n2. 느린 연결(Slow HTTP) 전용 방어 룰 적용"
        },
        {
            "Standard_Feature_ID": "PACKET_TTL_AVERAGE",
            "Standard_Feature_Name": "Packet TTL Average",
            "Risk_Group": "Control_State",
            "Risk_Summary": "패킷 TTL 평균값의 비정상적 변동",
            "Associated_Attack": "Recon / Scan, MITM",
            "Investigation_Point": "TTL 값이 특정 세션이나 통신 구간에서 지속적으로 이탈하는지 확인하고, 통신 경로 및 Source/Destination IP를 함께 확인",
            "Response_Guide": "1. 비정상적인 TTL 값을 가진 패킷 필터링\n2. 내부 네트워크 라우팅 경로 및 게이트웨이 점검"
        },
        {
            "Standard_Feature_ID": "TCP_ACK_FLAG_COUNT",
            "Standard_Feature_Name": "TCP ACK Flag Count",
            "Risk_Group": "Control_State",
            "Risk_Summary": "TCP ACK 플래그 패킷 폭증",
            "Associated_Attack": "DoS / DDoS, MITM",
            "Investigation_Point": "세션 수립 이후 의미 없는 ACK 패킷이 과도하게 유입되는지 확인",
            "Response_Guide": "1. 상태 기반(Stateful) 방화벽 세션 테이블 점검\n2. 비정상 ACK 패킷 속도 제한(Rate Limiting)"
        },
        {
            "Standard_Feature_ID": "TCP_FIN_FLAG_COUNT",
            "Standard_Feature_Name": "TCP FIN Flag Count",
            "Risk_Group": "Control_State",
            "Risk_Summary": "TCP FIN 플래그 패킷 폭증",
            "Associated_Attack": "Recon / Scan, DoS / DDoS",
            "Investigation_Point": "연결 종료를 시도하는 FIN 패킷이 비정상적으로 대량 발생하는지 확인",
            "Response_Guide": "1. 비정상 종료 패킷 모니터링 및 패턴 차단\n2. 방화벽 TCP 세션 타임아웃 튜닝"
        },
        {
            "Standard_Feature_ID": "TCP_RST_FLAG_COUNT",
            "Standard_Feature_Name": "TCP RST Flag Count",
            "Risk_Group": "Control_State",
            "Risk_Summary": "TCP RST 플래그 패킷 폭증",
            "Associated_Attack": "Recon / Scan, DoS / DDoS",
            "Investigation_Point": "연결 강제 리셋(RST) 패킷이 급증하여 정상 세션이 끊기는지 확인",
            "Response_Guide": "1. 비정상 RST 패킷 인스펙션 및 차단\n2. 네트워크 세션 위조/스푸핑 여부 점검"
        },
        {
            "Standard_Feature_ID": "TCP_SYN_FLAG_COUNT",
            "Standard_Feature_Name": "TCP SYN Flag Count",
            "Risk_Group": "Control_State",
            "Risk_Summary": "TCP SYN 플래그 폭증으로 인한 연결 자원 고갈",
            "Associated_Attack": "DoS / DDoS",
            "Investigation_Point": "TCP 3-way handshake 과정에서 ACK가 오지 않은 SYN 요청 누적 여부 확인",
            "Response_Guide": "1. OS 레벨 SYN 쿠키(SYN Cookies) 활성화\n2. 방화벽 SYN Rate Limiting 설정"
        },
        {
            "Standard_Feature_ID": "TCP_WINDOW_SIZE_AVERAGE",
            "Standard_Feature_Name": "TCP Window Size Average",
            "Risk_Group": "Control_State",
            "Risk_Summary": "TCP 윈도우 크기의 비정상적 이탈",
            "Associated_Attack": "DoS / DDoS, MITM",
            "Investigation_Point": "특정 Source/Destination에서 비정상적인 TCP Window 값이 반복되는지 확인하고 세션 상태와 함께 분석",
            "Response_Guide": "1. 윈도우 크기 조작 패킷 차단\n2. 비정상 트래픽 유발 소스 세션 격리"
        },

        # ==========================================
        # 2. Size_Payload (7개)
        # ==========================================
        {
            "Standard_Feature_ID": "AVG_IP_LENGTH",
            "Standard_Feature_Name": "Avg IP Length",
            "Risk_Group": "Size_Payload",
            "Risk_Summary": "평균 IP 패킷 길이 이상 변동",
            "Associated_Attack": "DoS / DDoS",
            "Investigation_Point": "IP 총 길이 필드가 프로토콜 규격에 어긋나는 비정상적인 크기를 가지는지 확인",
            "Response_Guide": "1. IP 단편화(Fragmentation) 공격 차단 룰 적용\n2. 비정상 IP 길이 패킷 드롭"
        },
        {
            "Standard_Feature_ID": "AVG_PACKET_SIZE",
            "Standard_Feature_Name": "Avg Packet Size",
            "Risk_Group": "Size_Payload",
            "Risk_Summary": "평균 패킷 크기의 비정상적 이탈",
            "Associated_Attack": "DoS / DDoS, Web Attack",
            "Investigation_Point": "정상 세션 대비 평균 패킷 크기가 지속적으로 이탈하는지 확인하고 트래픽 방향 및 세션 패턴을 함께 분석",
            "Response_Guide": "1. 패킷 사이즈 검사 룰 활성화\n2. 비정상 크기 패킷 상세 분석"
        },
        {
            "Standard_Feature_ID": "AVG_PAYLOAD_LENGTH",
            "Standard_Feature_Name": "Avg Payload Length",
            "Risk_Group": "Size_Payload",
            "Risk_Summary": "평균 페이로드 길이 비정상 이탈",
            "Associated_Attack": "Web Attack, Infiltration",
            "Investigation_Point": "애플리케이션 데이터 영역(Payload)의 평균 바이트 길이가 비정상적인지 확인",
            "Response_Guide": "1. WAF(웹 방화벽) 페이로드 검증 룰 강화\n2. 비정상 바이트 스트림 차단"
        },
        {
            "Standard_Feature_ID": "MAX_PACKET_SIZE",
            "Standard_Feature_Name": "Max Packet Size",
            "Risk_Group": "Size_Payload",
            "Risk_Summary": "패킷 최대 크기 과다 (오버플로우 유도)",
            "Associated_Attack": "DoS / DDoS, Infiltration",
            "Investigation_Point": "허용 범위를 초과하는 대형 패킷이나 패딩이 포함된 패킷 유입 여부 확인",
            "Response_Guide": "1. 방화벽/WAF 최대 패킷 사이즈 제한 설정\n2. 규격 초과 대형 패킷 드롭"
        },
        {
            "Standard_Feature_ID": "MIN_PACKET_SIZE",
            "Standard_Feature_Name": "Min Packet Size",
            "Risk_Group": "Size_Payload",
            "Risk_Summary": "최소 패킷 크기의 비정상적 이탈",
            "Associated_Attack": "Recon / Scan, DoS / DDoS",
            "Investigation_Point": "초소형 패킷이 특정 Source에서 반복적으로 발생하는지 확인하고 패킷 발생 빈도 및 목적지 포트를 함께 분석",
            "Response_Guide": "1. 초소형 패킷 필터링 룰 적용\n2. 비정상 패킷 유입 소스 점검"
        },
        {
            "Standard_Feature_ID": "PACKET_SIZE_VOLATILITY",
            "Standard_Feature_Name": "Packet Size Volatility",
            "Risk_Group": "Size_Payload",
            "Risk_Summary": "패킷 크기 변동성의 비정상적 증가",
            "Associated_Attack": "DoS / DDoS, Web Attack",
            "Investigation_Point": "패킷 크기가 불규칙하게 변화하는지 확인하고 해당 세션의 트래픽량, 방향 및 반복 패턴을 함께 분석",
            "Response_Guide": "1. 심층 패킷 분석(DPI) 수행\n2. 변동성 기반 이상 트래픽 탐지 정책 연동"
        },
        {
            "Standard_Feature_ID": "TOTAL_PAYLOAD_LENGTH",
            "Standard_Feature_Name": "Total Payload Length",
            "Risk_Group": "Size_Payload",
            "Risk_Summary": "총 페이로드 전송량 과다",
            "Associated_Attack": "Infiltration",
            "Investigation_Point": "데이터 영역의 총 누적 전송량이 평소 임계치를 초과하여 급증했는지 확인",
            "Response_Guide": "1. 대용량 데이터 유출(DLP) 탐지 시스템 연계\n2. 내부 호스트의 의심스러운 외부 통신 세션 차단"
        },

        # ==========================================
        # 3. Topology (5개)
        # ==========================================
        {
            "Standard_Feature_ID": "INITIAL_WINDOW_SIZE",
            "Standard_Feature_Name": "Initial Window Size",
            "Risk_Group": "Topology",
            "Risk_Summary": "초기 TCP Window 크기의 비정상적 이탈",
            "Associated_Attack": "Recon / Scan",
            "Investigation_Point": "특정 Source에서 초기 연결 시 TCP Window 값이 반복적으로 특정 패턴을 보이는지 확인하고 연결 대상 및 포트 탐색 패턴을 함께 분석",
            "Response_Guide": "1. 스캔 및 비정상 연결 시도 탐지\n2. 불필요한 포트 접근 통제"
        },
        {
            "Standard_Feature_ID": "PORT_ACTIVITY_COUNT",
            "Standard_Feature_Name": "Port Activity Count",
            "Risk_Group": "Topology",
            "Risk_Summary": "목적지/출발지 포트 탐색 빈도 급증",
            "Associated_Attack": "Recon / Scan",
            "Investigation_Point": "특정 단일 IP가 다수의 포트를 순차적으로 스캔했는지 여부 확인",
            "Response_Guide": "1. 스캔 시도 소스 IP 자동 차단(Fail2ban 등 연동)\n2. 외부에 불필요하게 열린 서비스 포트 폐쇄"
        },
        {
            "Standard_Feature_ID": "SUBFLOW_PACKET_COUNT",
            "Standard_Feature_Name": "Subflow Packet Count",
            "Risk_Group": "Topology",
            "Risk_Summary": "서브플로우별 패킷 수의 비정상적 이탈",
            "Associated_Attack": "DoS / DDoS, Recon / Scan",
            "Investigation_Point": "특정 서브플로우에 패킷이 비정상적으로 집중되거나 반복적인 통신 패턴이 나타나는지 확인",
            "Response_Guide": "1. 세션 흐름 모니터링 강화\n2. 비정상 집중 구간 트래픽 통제"
        },
        {
            "Standard_Feature_ID": "TRAFFIC_FLOW_ASYMMETRY",
            "Standard_Feature_Name": "Traffic Flow Asymmetry",
            "Risk_Group": "Topology",
            "Risk_Summary": "트래픽 흐름 비대칭성 급증",
            "Associated_Attack": "Malware / Botnet, Infiltration",
            "Investigation_Point": "송신(Forward)과 수신(Backward) 간 트래픽 비율 왜곡 여부 확인",
            "Response_Guide": "1. 비대칭 라우팅 경로 점검\n2. 우회 통신 및 비정상 세션 흐름 통제"
        },
        {
            "Standard_Feature_ID": "UNIQUE_IP_COUNT",
            "Standard_Feature_Name": "Unique IP Count",
            "Risk_Group": "Topology",
            "Risk_Summary": "통신에 참여하는 고유 IP 수의 비정상적 증가",
            "Associated_Attack": "DoS / DDoS, Recon / Scan, Malware / Botnet",
            "Investigation_Point": "짧은 시간 내 통신 Source/Destination IP 수가 급격히 증가하는지 확인하고 IP별 트래픽 및 포트 활동을 함께 분석",
            "Response_Guide": "1. 대규모 소스 IP 대역 차단\n2. GeoIP 기반 비정상 트래픽 유입 통제"
        },

        # ==========================================
        # 4. Volume (4개)
        # ==========================================
        {
            "Standard_Feature_ID": "LOG_EVENT_VOLUME",
            "Standard_Feature_Name": "Log Event Volume",
            "Risk_Group": "Volume",
            "Risk_Summary": "로그 이벤트 발생량의 비정상적 증가",
            "Associated_Attack": "DoS / DDoS, Malware / Botnet",
            "Investigation_Point": "특정 시간대 또는 Source에서 로그 이벤트가 급증하는지 확인하고 관련 세션 및 이벤트 유형을 함께 분석 (단, 로그 데이터 성격에 따른 보수적 해석 필요)",
            "Response_Guide": "1. 로그 수집 서버 부하 분산 처리\n2. 비정상 이벤트 유발 소스 프로세스 및 서비스 점검"
        },
        {
            "Standard_Feature_ID": "PACKET_RATE_PER_SEC",
            "Standard_Feature_Name": "Packet Rate Per Sec",
            "Risk_Group": "Volume",
            "Risk_Summary": "초당 패킷 유입량(PPS) 과다 폭증",
            "Associated_Attack": "DoS / DDoS",
            "Investigation_Point": "초당 패킷 수(PPS)가 임계치를 초과했는지 및 소스 IP 집중도 확인",
            "Response_Guide": "1. 상위 방화벽/ISP 레벨 Rate Limiting 적용\n2. 공격 유입 소스 IP 대역 블랙홀링 검토"
        },
        {
            "Standard_Feature_ID": "TOTAL_PACKET_VOLUME",
            "Standard_Feature_Name": "Total Packet Volume",
            "Risk_Group": "Volume",
            "Risk_Summary": "세션 내 총 패킷 전송량 비정상 증가",
            "Associated_Attack": "DoS / DDoS, Infiltration",
            "Investigation_Point": "정상 트래픽 대비 세션별 총 패킷 수 임계치 초과 여부 확인",
            "Response_Guide": "1. 대용량 고정 패킷 세션 강제 종료\n2. 트래픽 대역폭 제한 정책 적용"
        },
        {
            "Standard_Feature_ID": "TRAFFIC_BANDWIDTH_BPS",
            "Standard_Feature_Name": "Traffic Bandwidth (Bytes/s)",
            "Risk_Group": "Volume",
            "Risk_Summary": "대역폭 사용량(BPS) 포화 상태 도달",
            "Associated_Attack": "DoS / DDoS",
            "Investigation_Point": "초당 바이트 전송량(대역폭)이 네트워크 허용치를 초과했는지 확인",
            "Response_Guide": "1. QoS(품질 서비스) 정책 적용하여 중요 트래픽 우선 처리\n2. 대역폭 과다 유발 소스 트래픽 쉐이핑(Shaping)"
        }
    ]
    
    df_guide = pd.DataFrame(response_guide_data)
    df_guide.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"[INFO] '{filename}' 파일 생성 완료! (총 {len(df_guide)}개 가이드 항목)")
    return df_guide

if __name__ == "__main__":
    df_result = generate_full_response_guide_csv()
    print("\n=== 생성된 23개 가이드 룩업 테이블 요약 ===")
    print(df_result[['Risk_Group', 'Standard_Feature_ID']].to_string())