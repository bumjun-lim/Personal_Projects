import pandas as pd
import numpy as np
import glob
import os

# 2017년 및 2025년 통합 메타 축 매핑 딕셔너리
META_FEATURE_MAPPING = {
    'Volume': {
        '2017': ['Total Fwd Packets', 'Total Backward Packets', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s']
    },
    'Size_Payload': {
        '2017': ['Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'Average Packet Size', 'Fwd Packet Length Max', 'Bwd Packet Length Max']
    },
    'Topology': {
        '2017': ['Destination Port', 'Down/Up Ratio', 'Subflow Fwd Packets', 'Subflow Bwd Packets']
    },
    'Control_State': {
        '2017': ['SYN Flag Count', 'RST Flag Count', 'ACK Flag Count', 'FIN Flag Count', 'Flow IAT Mean', 'Init_Win_bytes_forward']
    }
}

def normalize_and_score_dataset_2017():
    # TODO: 2017년 데이터셋 경로에 맞게 수정 필요
    path_pattern = "../data/processed_data/cic-ids-2017/*.parquet"
    files = glob.glob(path_pattern)
    
    if not files:
        print(f"2017년 데이터 경로를 확인해주세요: {path_pattern}")
        # 테스트용으로 파일이 없을 경우 가이드 메시지 출력
        return

    print(">> [2017] 데이터 로드 및 정규화 시작...")
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()
    
    # 2017년 레이블 컬럼('Label') 탐색
    label_cols = [c for c in df.columns if c.lower() == 'label']
    if not label_cols:
        print(">> [오류] 2017년 데이터셋에 'Label' 컬럼이 존재하지 않습니다.")
        return
    label_col = label_cols[0]
    
    print(f">> 탐색된 레이블 컬럼: {label_col}")
    print(f">> 고유 레이블 값 종류: {df[label_col].unique()[:5]}...")
    
    # 정상(benign) 데이터 마스킹
    benign_mask = df[label_col].astype(str).str.lower().str.strip() == 'benign'
    print(f">> 매칭된 정상(benign) 데이터 건수: {benign_mask.sum()}건 / 전체 {len(df)}건")
    
    if benign_mask.sum() == 0:
        print(">> [오류] 정상(benign) 데이터가 0건입니다.")
        return
        
    axis_scores = []
    for axis_name, mapping in META_FEATURE_MAPPING.items():
        target_cols = [c for c in mapping['2017'] if c in df.columns]
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
    
    print("\n--- [2017년 메타 축 정규화 기반 리스크 스코어링 결과] ---")
    print(df.groupby(label_col)['Risk_Score'].agg(['count', 'mean', 'max']))
    print("\n--- [2017년 위험 등급 분포] ---")
    print(df['Risk_Grade'].value_counts())

if __name__ == "__main__":
    normalize_and_score_dataset_2017()