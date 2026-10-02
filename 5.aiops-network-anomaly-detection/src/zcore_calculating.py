import numpy as np
import pandas as pd

# 메타 피처 정의 (스코어 산출에 필요한 피처 그룹 맵핑)
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


def calculate_meta_scores(df, year_label, max_z_cap=5.0):
  """이미 연도와 공격 그룹 매핑이 완료된 데이터프레임을 인자로 받아,

  오직 메타 축별 Z-score 계산, Max Cap 5.0 적용, 그리고
  각 점수 컬럼(Volume_Score 등)과 Risk_Score를 장착하여 반환합니다.
  """
  df = df.copy()

  # 정상(Benign/Normal) 데이터 분리를 위한 레이블 컬럼 식별
  if year_label == "2017":
    label_candidates = [c for c in df.columns if c.lower() == "label"]
  else:
    label_candidates = [c for c in df.columns if c.lower() == "label2"]

  if not label_candidates:
    raise ValueError(f">> [{year_label}] 오류: 레이블 컬럼을 찾을 수 없습니다.")

  label_col = label_candidates[0]

  meta_groups = META_FEATURE_MAPPING[year_label]
  score_column_names = []

  # 2. 전체 데이터 대상 inf/nan 정제 (nanop.py 경고 원인 원천 차단)
  all_target_features = [
      feat for features in meta_groups.values() for feat in features if feat in df.columns
  ]
  df[all_target_features] = (
      df[all_target_features].replace([np.inf, -np.inf], np.nan).fillna(0)
  )

  # 정상 데이터 기준선 마스크
  series_label = df[label_col].astype(str).str.strip().str.upper()
  if year_label == "2017":
    benign_mask = series_label == "BENIGN"
  else:
    benign_mask = series_label.isin(["NORMAL", "BENIGN", "BENIGN/NORMAL"])

  df_benign = df[benign_mask]
  if len(df_benign) == 0:
    print(f">> [{year_label}] 경고: 정상 데이터가 없어 Z-score 산출 불가")
    return df

  # 3. 그룹별 점수 및 [개별 피처별 Z-Score 컬럼] 동시 생성
  for group_name, features in meta_groups.items():
    valid_features = [f for f in features if f in df.columns]
    score_col_name = f"{group_name}_Score"
    score_column_names.append(score_col_name)

    if not valid_features:
      df[score_col_name] = 0.0
      continue

    b_mean = df_benign[valid_features].mean()
    b_std = df_benign[valid_features].std().replace(0, 1.0)

    group_z_accumulator = 0.0
    for feat in valid_features:
      raw_z = np.abs((df[feat] - b_mean[feat]) / b_std[feat])
#========================scoring method===========================
      #capped_z = np.clip(raw_z, 0, max_z_cap)  # 상한선 5.0 적용 
      
      logged_z = np.log1p(raw_z)
      capped_z = np.clip(logged_z, 0, max_z_cap)
#========================scoring method===========================
      # 💡 [핵심] 개별 피처별 Z-Score를 세션별로 기록 (디버깅/조율용)
      df[f"Z__{feat}"] = capped_z
      group_z_accumulator += capped_z

    df[score_col_name] = group_z_accumulator / len(valid_features)

  # 최종 Risk_Score 합산 컬럼 장착
  df["Risk_Score"] = df[score_column_names].sum(axis=1)

  return df