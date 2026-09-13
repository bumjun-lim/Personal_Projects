import pandas as pd
import numpy as np
import glob
import os

META_FEATURE_MAPPING = {
    'Volume': {
        '2025': [
            'network_packets_all_count', 'network_packets_dst_count', 
            'network_packets_src_count', 'log_messages_count', 'network_interval-packets'
        ]
    },
    'Size_Payload': {
        '2025': [
            'network_packet-size_min', 'network_packet-size_max', 
            'network_packet-size_avg', 'network_packet-size_std_deviation',
            'network_payload-length_avg', 'network_ip-length_avg'
        ]
    },
    'Topology': {
        '2025': [
            'network_ports_all_count', 'network_ports_dst_count', 'network_ports_src_count',
            'network_ips_all_count', 'network_ips_dst_count', 'network_ips_src_count'
        ]
    },
    'Control_State': {
        '2025': [
            'network_tcp-flags-syn_count', 'network_tcp-flags-rst_count', 
            'network_tcp-flags-ack_count', 'network_tcp-flags-fin_count',
            'network_ttl_avg', 'network_window-size_avg'
        ]
    }
}

def normalize_and_score_dataset(df):
    print(">> 데이터 스키마 정규화 및 메타 축 매핑 시작...")
    
    # 1. 컬럼명 공백 제거 안전장치
    df.columns = df.columns.str.strip()
    
    # 2. label2 컬럼 존재 확인 및 정상 데이터 마스킹 디버깅
    label_col = [c for c in df.columns if c.lower() == 'label2'][0]
    print(f">> 탐색된 레이블 컬럼: {label_col}")
    print(f">> 고유 레이블 값 종류: {df[label_col].unique()[:5]}...")
    
    benign_mask = df[label_col].astype(str).str.lower().str.strip() == 'benign'
    print(f">> 매칭된 정상(benign) 데이터 건수: {benign_mask.sum()}건 / 전체 {len(df)}건")
    
    if benign_mask.sum() == 0:
        print(">> [오류] 정상(benign) 데이터가 0건입니다. 레이블 값을 확인해주세요.")
        return df
        
    axis_scores = []
    for axis_name, mapping in META_FEATURE_MAPPING.items():
        # 실제 df에 존재하는 컬럼만 필터링
        target_cols = [c for c in mapping['2025'] if c in df.columns]
        if not target_cols:
            print(f">> [경고] '{axis_name}' 축에 매칭되는 컬럼이 없습니다.")
            continue
            
        df_axis = df[target_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        axis_benign = df_axis[benign_mask]
        mean = axis_benign.mean()
        std = axis_benign.std().replace(0, 1.0)
        
        z_scores = np.abs((df_axis - mean) / std)
        axis_intensity = np.clip(z_scores.mean(axis=1), 0, 20.0) / 20.0
        axis_scores.append(axis_intensity)
    
    if axis_scores:
        combined_score = np.mean(axis_scores, axis=0) * 100.0
        df['Risk_Score'] = np.clip(combined_score, 0, 100)
    else:
        df['Risk_Score'] = 0.0

    df['Risk_Grade'] = pd.cut(
        df['Risk_Score'],
        bins=[-1, 15.0, 50.0, 100.0],
        labels=['Normal', 'Warning', 'Danger']
    )
    
    return df

def run_pipeline():
    path_pattern = "../data/processed_data/cic-iiot-2025/*.parquet"
    files = glob.glob(path_pattern)
    
    if not files:
        print(f"경로를 확인해주세요: {path_pattern}")
        return

    df_list = [pd.read_parquet(f).sample(n=min(3000, pd.read_parquet(f).shape[0]), random_state=42) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    
    scored_df = normalize_and_score_dataset(df)
    
    print("\n--- [메타 축 정규화 기반 리스크 스코어링 결과] ---")
    print(scored_df.groupby('label2')['Risk_Score'].agg(['count', 'mean', 'max']))
    print("\n--- [위험 등급 분포] ---")
    print(scored_df['Risk_Grade'].value_counts())

if __name__ == "__main__":
    run_pipeline()