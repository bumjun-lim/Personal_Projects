import glob
import os
import numpy as np
import pandas as pd


def comprehensive_cross_year_analysis():
  path_2017 = "../data/processed_data/cic-ids-2017/*.parquet"
  path_2025 = "../data/processed_data/cic-iiot-2025/*.parquet"

  files_2017 = glob.glob(path_2017)
  files_2025 = glob.glob(path_2025)

  print(f">> 2017년 파일 개수: {len(files_2017)}개 발견")
  print(f">> 2025년 파일 개수: {len(files_2025)}개 발견")

  if not files_2017 or not files_2025:
    print(">> 경로를 다시 확인해주세요.")
    return

  df_17 = pd.concat([pd.read_parquet(f) for f in files_2017], ignore_index=True)
  df_25 = pd.concat([pd.read_parquet(f) for f in files_2025], ignore_index=True)

  df_17.columns = df_17.columns.str.strip()
  df_25.columns = df_25.columns.str.strip()

  # [연도별 독립 메타 피처 매핑 구조 복원]
  meta_feature_mapping = {
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

  def analyze_dataset(df, meta_groups, year_label):
    if year_label == "2017":
      label_candidates = [c for c in df.columns if c.lower() == "label"]
    else:
      label_candidates = [c for c in df.columns if c.lower() == "label2"]

    if not label_candidates:
      print(f">> [{year_label}] 오류: 레이블용 컬럼을 찾을 수 없습니다.")
      return pd.DataFrame()

    label_col = label_candidates[0]
    print(f">> [{year_label}] 사용 레이블 컬럼: {label_col}")

    # 실제 존재하는 피처만 그룹별로 필터링
    active_meta_groups = {}
    all_mapped_features = []

    for g_name, g_cols in meta_groups.items():
      valid = [c for c in g_cols if c in df.columns]
      print(f">> [{year_label} - {g_name}] 매칭된 피처 수: {len(valid)}/{len(g_cols)}")
      if valid:
        active_meta_groups[g_name] = valid
        all_mapped_features.extend(valid)

    if not all_mapped_features:
      print(f">> [{year_label}] 오류: 매칭된 피처가 단 하나도 없습니다.")
      return pd.DataFrame()

    df_clean = (
        df[all_mapped_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # 정상/공격 판별 마스크
    series_label = df[label_col].astype(str).str.strip().str.upper()
    if year_label == "2017":
      benign_mask = series_label == "BENIGN"
    else:
      benign_mask = series_label.isin(["NORMAL", "BENIGN", "BENIGN/NORMAL"])

    df_benign = df_clean[benign_mask]
    df_attack = df_clean[~benign_mask]

    print(
        f">> [{year_label}] 정상 세션: {len(df_benign):,}건 | 공격 세션:"
        f" {len(df_attack):,}건"
    )

    if len(df_benign) == 0:
      print(f">> [{year_label}] 경고: 정상 데이터가 0건입니다.")
      return pd.DataFrame()

    stats_summary = []
    
    # 1단계: 각 그룹별 총 Z-Score 먼저 계산 (그룹 간 비교용)
    group_total_z = {}
    group_valid_features = {}
    
    for group_name, features in active_meta_groups.items():
      valid_features = [f for f in features if f in df_clean.columns]
      if not valid_features:
        continue
      group_valid_features[group_name] = valid_features
      
      # 개별 피처 Z-score 계산
      b_mean = df_benign[valid_features].mean()
      b_std = df_benign[valid_features].std().replace(0, 1.0)
      
      feat_z_scores = {}
      for feat in valid_features:
        if len(df_attack) > 0:
          z_vals = np.abs((df_attack[feat] - b_mean[feat]) / b_std[feat]).mean()
        else:
          z_vals = 0.0
        feat_z_scores[feat] = z_vals
        
      group_total_z[group_name] = sum(feat_z_scores.values())

    # 전체 그룹의 Z-score 총합 (전체 분모)
    total_system_z = sum(group_total_z.values()) if group_total_z else 1.0

    # 2단계: 그룹별 기여도 및 그룹 내 피처 기여도 산출
    for group_name, features in group_valid_features.items():
      g_total = group_total_z[group_name]
      # 전체 점수(예: 100점 만점 기준)에서 이 그룹이 차지하는 비중 (%)
      group_contribution_pct = (g_total / total_system_z) * 100 if total_system_z > 0 else 0.0

      b_mean = df_benign[features].mean()
      b_std = df_benign[features].std().replace(0, 1.0)
      b_median = df_benign[features].median()
      b_q75 = df_benign[features].quantile(0.75)
      b_q25 = df_benign[features].quantile(0.25)
      b_iqr = (b_q75 - b_q25).replace(0, 1.0)
      b_p99 = df_benign[features].quantile(0.99)

      for feat in features:
        if len(df_attack) > 0:
          z_vals = np.abs((df_attack[feat] - b_mean[feat]) / b_std[feat]).mean()
        else:
          z_vals = 0.0

        # 그룹 내에서 이 피처가 차지하는 비중 (%)
        intra_group_pct = (z_vals / g_total) * 100 if g_total > 0 else 0.0

        stats_summary.append({
            "Year": year_label,
            "Group": group_name,
            "Feature": feat,
            "Normal_Mean": b_mean[feat],
            "Normal_Median": b_median[feat],
            "Normal_Std": b_std[feat],
            "Normal_IQR": b_iqr[feat],
            "Normal_P99": b_p99[feat],
            "Attack_Z_Score": z_vals,
            "Group_Contribution_Pct": group_contribution_pct, # 전체 중 해당 그룹의 비중
            "Intra_Group_Pct": intra_group_pct,               # 그룹 내 해당 피처의 비중
        })

    return pd.DataFrame(stats_summary)

  df_res_17 = analyze_dataset(
      df_17, meta_feature_mapping["2017"], "2017"
  )
  df_res_25 = analyze_dataset(
      df_25, meta_feature_mapping["2025"], "2025"
  )

  df_total_report = pd.concat(
      [df for df in [df_res_17, df_res_25] if not df.empty], ignore_index=True
  )

  if not df_total_report.empty:
    print(
        "\n===================================================================================="
    )
    print(
        "                      [2017 vs 2025 메타 특성별 통계 및 이탈도 비교]"
    )
    print(
        "===================================================================================="
    )
    print(
        df_total_report.to_string(
            index=False,
            formatters={
                "Normal_Mean": "{:.2f}".format,
                "Normal_Median": "{:.2f}".format,
                "Normal_Std": "{:.2f}".format,
                "Normal_IQR": "{:.2f}".format,
                "Normal_P99": "{:.2f}".format,
                "Attack_Z_Score": "{:.4f}".format,
                "Contribution_Pct": "{:.2f}%".format,
                "Intra_Group_Pct": "{:.2f}%".format
            },
        )
    )
    print(
        "===================================================================================="
    )

    print("\n--- [그룹별 평균 공격 Z-Score 요약] ---")
    group_summary = (
        df_total_report.groupby(["Year", "Group"])["Attack_Z_Score"]
        .mean()
        .unstack(level=0)
    )
    print(group_summary)


if __name__ == "__main__":
  comprehensive_cross_year_analysis()