import pandas as pd
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def extract_feature_importance_2025():
    # 2025년 데이터셋 경로 설정 (사용자 환경에 맞게 조정)
    path_pattern = "../data/processed_data/cic-iiot-2025/*.parquet"
    files = glob.glob(path_pattern)
    
    if not files:
        print("변환된 2025년 파케트 파일이 없습니다. 경로를 확인해주세요.")
        return

    df_list = []
    for f in files:
        df_chunk = pd.read_parquet(f)
        df_list.append(df_chunk.sample(frac=0.1, random_state=42))
        
    df = pd.concat(df_list, ignore_index=True)
    
    # 1. 레이블 이진화 (benign 포함 여부로 정상 0, 공격 1 통합)
    label_col = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower() or 'attack' in c.lower()][0]
    df['Binary_Label'] = df[label_col].apply(lambda x: 0 if 'benign' in str(x).lower() else 1)
    
    # 2. X(특성)와 y(타겟) 분리
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # 2025년 데이터의 레이블 관련 후보 컬럼들을 모두 제외 목록에 포함
    exclude_cols = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower() or 'attack' in c.lower()]
    exclude_cols.append('Binary_Label')
    
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    X = df[feature_cols]
    y = df['Binary_Label']
    
    # 무한대 및 결측치 클렌징
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = np.clip(X, -1e15, 1e15)
    
    print(">> 2025년 데이터 Random Forest 피처 중요도 추출 중...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # 3. 중요도 상위 피처 정렬 출력
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    
    print("\n--- [2025년 Top 10 핵심 위협 지표(Feature)] ---")
    print(top_features)

if __name__ == "__main__":
    extract_feature_importance_2025()