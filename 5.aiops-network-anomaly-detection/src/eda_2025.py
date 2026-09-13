import pandas as pd
import glob
import os

def analyze_2025_data():
    # 2025년 Parquet 파일
    path_pattern = "../data/processed_data/cic-iiot-2025/*.parquet"
    files = glob.glob(path_pattern)
    
    if not files:
        print("변환된 2025년 데이터 파일이 없습니다. 경로를 확인해주세요.")
        return

    print(f">> 발견된 2025년 데이터 파일 수: {len(files)}개")
    
    # 2017년 코드와 동일하게 청크별 샘플링 로드 적용
    df_list = []
    for f in files:
        df_chunk = pd.read_parquet(f)
        df_sample = df_chunk.sample(frac=0.1, random_state=42)
        df_list.append(df_sample)
        
    df_2025 = pd.concat(df_list, ignore_index=True)
    
    print("\n--- [1] 2025년 데이터 기본 정보 ---")
    print(f"샘플링된 2025 총 데이터 크기: {df_2025.shape}")
    
    # 레이블 컬럼명 확인 (2025 데이터셋은 컬럼명이 다를 수 있어 확장 탐색)
    label_col = [c for c in df_2025.columns if 'label' in c.lower() or 'class' in c.lower() or 'attack' in c.lower()]
    print(f"타겟 레이블 컬럼명 후보: {label_col}")
    
    if label_col:
        target = label_col[0]
        print(f"\n--- [2] 레이블 분포 (정상 vs 공격) [{target}] ---")
        print(df_2025[target].value_counts(dropna=False))

    print("\n--- [3] 주요 통계 지표 요약 (일부 피처) ---")
    numeric_cols = df_2025.select_dtypes(include=['float64', 'int64']).columns
    print(f"숫자형 피처 총 개수: {len(numeric_cols)}개")
    print(f"숫자형 피처 목록 일부: {list(numeric_cols[:10])}")

if __name__ == "__main__":
    analyze_2025_data()