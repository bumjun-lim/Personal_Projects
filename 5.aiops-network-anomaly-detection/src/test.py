import glob
import os
import numpy as np
import pandas as pd

# 그룹별 피처 맵핑 정의 (출력 시 항목별 분류용)
META_FEATURE_MAPPING = {
    "2017": {
        "Volume": [
            "Total Fwd Packets",
            "Total Backward Packets",
            "Flow Packets/s",
            "Fwd Packets/s",
            "Bwd Packets/s",
            "Flow Bytes/s",
        ],
        "Size_Payload": [
            "Min Packet Length",
            "Max Packet Length",
            "Packet Length Mean",
            "Packet Length Std",
            "Packet Length Variance",
            "Average Packet Size",
            "Fwd Packet Length Max",
            "Bwd Packet Length Max",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
        ],
        "Topology": [
            "Destination Port",
            "Down/Up Ratio",
            "Subflow Fwd Packets",
            "Subflow Bwd Packets",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
        ],
        "Control_State": [
            "SYN Flag Count",
            "RST Flag Count",
            "ACK Flag Count",
            "FIN Flag Count",
            "Flow IAT Mean",
        ],
    },
    "2025": {
        "Volume": [
            "network_packets_all_count",
            "network_packets_dst_count",
            "network_packets_src_count",
            "log_messages_count",
            "network_interval-packets",
        ],
        "Size_Payload": [
            "network_packet-size_min",
            "network_packet-size_max",
            "network_packet-size_avg",
            "network_packet-size_std_deviation",
            "network_payload-length_avg",
            "network_ip-length_avg",
        ],
        "Topology": [
            "network_ports_all_count",
            "network_ports_dst_count",
            "network_ports_src_count",
            "network_ips_all_count",
            "network_ips_dst_count",
            "network_ips_src_count",
        ],
        "Control_State": [
            "network_tcp-flags-syn_count",
            "network_tcp-flags-rst_count",
            "network_tcp-flags-ack_count",
            "network_tcp-flags-fin_count",
            "network_ttl_avg",
            "network_window-size_avg",
        ],
    },
}

def main():
    path = "../data/processed_data/*.csv"
    files = glob.glob(path)

    if files:
        # CSV 파일들 통합 로드
        df_master = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        
        # 1. Normal 그룹만 필터링하고 연도별로 통계 확인
        # (만약 개별 점수 컬럼인 'Volume Score', 'Size Payload Score' 등이 이미 마스터에 있다면 바로 agg에 넣으시면 됩니다)
        
        # 예시: 지정해주신 5가지 핵심 점수 컬럼명이 데이터프레임에 존재한다는 가정 하에 그룹화
        target_scores = [
            "Risk_Score", 
            "Volume_Score", 
            "Size_Payload_Score", 
            "Topology_Score", 
            "Control_State_Score"
        ]
        
        # 데이터프레임에 해당 컬럼들이 있는지 확인 후 존재하는 것들만 추출
        available_scores = [col for col in target_scores if col in df_master.columns]
        
        if "Normal" in df_master["Attack_Group"].values:
            df_normal = df_master[df_master["Attack_Group"] == "Normal"]
        else:
            # 대소문자나 명칭이 다를 수 있어 안전장치 (예: 'normal')
            df_normal = df_master[df_master["Attack_Group"].str.lower() == "normal"]

        normal_stats = (
            df_normal
            .groupby("Year")[available_scores]
            .agg(["mean", "std", "min", "max", "median"])
        )
        
        print("=== [Normal 그룹 연도별 영역별 점수 통계 요약] ===")
        print(normal_stats)
        
    else:
        print(">> 지정된 경로에 CSV 파일이 없습니다.")

if __name__ == "__main__":
  main()