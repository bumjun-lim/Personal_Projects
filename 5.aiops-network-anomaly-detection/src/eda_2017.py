import pandas as pd
import glob
import os

def analyze_2017_data():
    # 2017년 Parquet 파일들이 저장된 경로 (사용자 환경에 맞게 폴더명 조정)
    path_pattern = "../data/processed_data/cic-ids-2017/*.parquet"
    files = glob.glob(path_pattern)
    
    if not files:
        print("변환된 2017년 파케트 파일이 없습니다. 경로를 확인해주세요.")
        return

    print(f">> 발견된 2017년 데이터 파일 수: {len(files)}개")
    
    # 예시로 첫 번째 파일(또는 전체 파일 중 일부) 로드 후 구조 파악
    # 메모리 안정을 위해 청크 혹은 샘플링 적용
    df_list = []
    for f in files:
        df_chunk = pd.read_parquet(f)
        # 전체 데이터가 너무 크면 파일당 10%만 샘플링하여 로드
        df_sample = df_chunk.sample(frac=0.1, random_state=42)
        df_list.append(df_sample)
        
    df_2017 = pd.concat(df_list, ignore_index=True)
    
    print("\n--- [1] 데이터 기본 정보 ---")
    print(f"샘플링된 2017 총 데이터 크기: {df_2017.shape}")
    
    # 레이블 컬럼명 확인 (보통 'Label' 또는 공백 포함 형태)
    label_col = [c for c in df_2017.columns if 'label' in c.lower()]
    print(f"타겟 레이블 컬럼명: {label_col}")
    
    if label_col:
        target = label_col[0]
        print("\n--- [2] 레이블 분포 (정상 vs 공격) ---")
        print(df_2017[target].value_counts(dropna=False))

    print("\n--- [3] 주요 통계 지표 요약 (일부 피처) ---")
    # 예시로 IAT나 Packet Length 관련 컬럼 요약
    numeric_cols = df_2017.select_dtypes(include=['float64', 'int64']).columns
    print(f"숫자형 피처 총 개수: {len(numeric_cols)}개")

if __name__ == "__main__":
    analyze_2017_data()