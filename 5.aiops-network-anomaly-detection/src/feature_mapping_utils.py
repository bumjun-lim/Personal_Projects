import pandas as pd

# 1. 개선된 의미 기반 통합 표준 피처 매핑 딕셔너리
SEMANTIC_FEATURE_MAP = {
    # --- Volume ---
    "Flow Packets/s": {"id": "PACKET_RATE_PER_SEC", "name": "Packet Rate Per Sec"},
    "Fwd Packets/s": {"id": "PACKET_RATE_PER_SEC", "name": "Packet Rate Per Sec"},
    "Bwd Packets/s": {"id": "PACKET_RATE_PER_SEC", "name": "Packet Rate Per Sec"},
    "network_interval-packets": {"id": "PACKET_RATE_PER_SEC", "name": "Packet Rate Per Sec"},
    
    "Total Fwd Packets": {"id": "TOTAL_PACKET_VOLUME", "name": "Total Packet Volume"},
    "Total Backward Packets": {"id": "TOTAL_PACKET_VOLUME", "name": "Total Packet Volume"},
    "network_packets_all_count": {"id": "TOTAL_PACKET_VOLUME", "name": "Total Packet Volume"},
    "network_packets_dst_count": {"id": "TOTAL_PACKET_VOLUME", "name": "Total Packet Volume"},
    "network_packets_src_count": {"id": "TOTAL_PACKET_VOLUME", "name": "Total Packet Volume"},
    
    "Flow Bytes/s": {"id": "TRAFFIC_BANDWIDTH_BPS", "name": "Traffic Bandwidth (Bytes/s)"},
    "log_messages_count": {"id": "LOG_EVENT_VOLUME", "name": "Log Event Volume"},

    # --- Size & Payload (분리 적용 완료) ---
    "Min Packet Length": {"id": "MIN_PACKET_SIZE", "name": "Min Packet Size"},
    "network_packet-size_min": {"id": "MIN_PACKET_SIZE", "name": "Min Packet Size"},
    
    "Max Packet Length": {"id": "MAX_PACKET_SIZE", "name": "Max Packet Size"},
    "Fwd Packet Length Max": {"id": "MAX_PACKET_SIZE", "name": "Max Packet Size"},
    "Bwd Packet Length Max": {"id": "MAX_PACKET_SIZE", "name": "Max Packet Size"},
    "network_packet-size_max": {"id": "MAX_PACKET_SIZE", "name": "Max Packet Size"},
    
    "Packet Length Mean": {"id": "AVG_PACKET_SIZE", "name": "Avg Packet Size"},
    "Average Packet Size": {"id": "AVG_PACKET_SIZE", "name": "Avg Packet Size"},
    "network_packet-size_avg": {"id": "AVG_PACKET_SIZE", "name": "Avg Packet Size"},
    
    # 짚어준 피드백대로 분리된 페이로드 및 IP 길이
    "network_payload-length_avg": {"id": "AVG_PAYLOAD_LENGTH", "name": "Avg Payload Length"},
    "network_ip-length_avg": {"id": "AVG_IP_LENGTH", "name": "Avg IP Length"},
    
    "Packet Length Std": {"id": "PACKET_SIZE_VOLATILITY", "name": "Packet Size Volatility"},
    "Packet Length Variance": {"id": "PACKET_SIZE_VOLATILITY", "name": "Packet Size Volatility"},
    "network_packet-size_std_deviation": {"id": "PACKET_SIZE_VOLATILITY", "name": "Packet Size Volatility"},
    
    "Total Length of Fwd Packets": {"id": "TOTAL_PAYLOAD_LENGTH", "name": "Total Payload Length"},
    "Total Length of Bwd Packets": {"id": "TOTAL_PAYLOAD_LENGTH", "name": "Total Payload Length"},

    # --- Topology ---
    "Destination Port": {"id": "PORT_ACTIVITY_COUNT", "name": "Port Activity Count"},
    "network_ports_all_count": {"id": "PORT_ACTIVITY_COUNT", "name": "Port Activity Count"},
    "network_ports_dst_count": {"id": "PORT_ACTIVITY_COUNT", "name": "Port Activity Count"},
    "network_ports_src_count": {"id": "PORT_ACTIVITY_COUNT", "name": "Port Activity Count"},
    
    "network_ips_all_count": {"id": "UNIQUE_IP_COUNT", "name": "Unique IP Count"},
    "network_ips_dst_count": {"id": "UNIQUE_IP_COUNT", "name": "Unique IP Count"},
    "network_ips_src_count": {"id": "UNIQUE_IP_COUNT", "name": "Unique IP Count"},
    
    "Down/Up Ratio": {"id": "TRAFFIC_FLOW_ASYMMETRY", "name": "Traffic Flow Asymmetry"},
    "Subflow Fwd Packets": {"id": "SUBFLOW_PACKET_COUNT", "name": "Subflow Packet Count"},
    "Subflow Bwd Packets": {"id": "SUBFLOW_PACKET_COUNT", "name": "Subflow Packet Count"},
    "Init_Win_bytes_forward": {"id": "INITIAL_WINDOW_SIZE", "name": "Initial Window Size"},
    "Init_Win_bytes_backward": {"id": "INITIAL_WINDOW_SIZE", "name": "Initial Window Size"},

    # --- Control State ---
    "SYN Flag Count": {"id": "TCP_SYN_FLAG_COUNT", "name": "TCP SYN Flag Count"},
    "network_tcp-flags-syn_count": {"id": "TCP_SYN_FLAG_COUNT", "name": "TCP SYN Flag Count"},
    
    "RST Flag Count": {"id": "TCP_RST_FLAG_COUNT", "name": "TCP RST Flag Count"},
    "network_tcp-flags-rst_count": {"id": "TCP_RST_FLAG_COUNT", "name": "TCP RST Flag Count"},
    
    "ACK Flag Count": {"id": "TCP_ACK_FLAG_COUNT", "name": "TCP ACK Flag Count"},
    "network_tcp-flags-ack_count": {"id": "TCP_ACK_FLAG_COUNT", "name": "TCP ACK Flag Count"},
    
    "FIN Flag Count": {"id": "TCP_FIN_FLAG_COUNT", "name": "TCP FIN Flag Count"},
    "network_tcp-flags-fin_count": {"id": "TCP_FIN_FLAG_COUNT", "name": "TCP FIN Flag Count"},
    
    "Flow IAT Mean": {"id": "CONNECTION_INTERVAL_TIME", "name": "Connection Interval Time"},
    "network_ttl_avg": {"id": "PACKET_TTL_AVERAGE", "name": "Packet TTL Average"},
    "network_window-size_avg": {"id": "TCP_WINDOW_SIZE_AVERAGE", "name": "TCP Window Size Average"}
}

# 원본 구조 데이터
META_FEATURE_MAPPING = {
    "2017": {
        "Volume": ["Total Fwd Packets", "Total Backward Packets", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s", "Flow Bytes/s"],
        "Size_Payload": ["Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "Average Packet Size", "Fwd Packet Length Max", "Bwd Packet Length Max", "Total Length of Fwd Packets", "Total Length of Bwd Packets"],
        "Topology": ["Destination Port", "Down/Up Ratio", "Subflow Fwd Packets", "Subflow Bwd Packets", "Init_Win_bytes_forward", "Init_Win_bytes_backward"],
        "Control_State": ["SYN Flag Count", "RST Flag Count", "ACK Flag Count", "FIN Flag Count", "Flow IAT Mean"],
    },
    "2025": {
        "Volume": ["network_packets_all_count", "network_packets_dst_count", "network_packets_src_count", "log_messages_count", "network_interval-packets"],
        "Size_Payload": ["network_packet-size_min", "network_packet-size_max", "network_packet-size_avg", "network_packet-size_std_deviation", "network_payload-length_avg", "network_ip-length_avg"],
        "Topology": ["network_ports_all_count", "network_ports_dst_count", "network_ports_src_count", "network_ips_all_count", "network_ips_dst_count", "network_ips_src_count"],
        "Control_State": ["network_tcp-flags-syn_count", "network_tcp-flags-rst_count", "network_tcp-flags-ack_count", "network_tcp-flags-fin_count", "network_ttl_avg", "network_window-size_avg"],
    },
}

def generate_mapping_csv(filename="feature_standard_mapping.csv"):
    rows = []
    
    for year, categories in META_FEATURE_MAPPING.items():
        for category, features in categories.items():
            for feat in features:
                if feat in SEMANTIC_FEATURE_MAP:
                    feat_info = SEMANTIC_FEATURE_MAP[feat]
                    std_id = feat_info["id"]
                    std_name = feat_info["name"]
                else:
                    std_id = f"UNASSIGNED_{feat.upper().replace(' ', '_')}"
                    std_name = feat
                
                rows.append({
                    "Year": int(year),
                    "Original_Feature": feat,
                    "Standard_Feature_ID": std_id,     # ID 추가 (유지보수용)
                    "Standard_Feature_Name": std_name, # 화면 표시용 이름
                    "Risk_Group": category
                })
                
    df_mapping = pd.DataFrame(rows)
    df_mapping.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"[INFO] '{filename}' 파일 생성 완료! (총 {len(df_mapping)}개 매핑 행)")
    return df_mapping

if __name__ == "__main__":
    df_result = generate_mapping_csv()
    print("\n=== 생성된 매핑 테이블 샘플 ===")
    print(df_result)