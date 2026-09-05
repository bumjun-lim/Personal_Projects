import pandas as pd
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def extract_feature_importance():
    path_pattern = "../data/processed_data/cic-ids-2017/*.parquet"
    files = glob.glob(path_pattern)
    
    df_list = []
    for f in files:
        df_chunk = pd.read_parquet(f)
        df_list.append(df_chunk.sample(frac=0.1, random_state=42))
        
    df = pd.concat(df_list, ignore_index=True)
    
    # 1. 레이블 이진화 (BENIGN은 0, 나머지 공격은 1로 통합)
    label_col = [c for c in df.columns if 'label' in c.lower()][0]
    df['Binary_Label'] = df[label_col].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
    
    # 2. X(특성)와 y(타겟) 분리
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # [수정] 숫자형 컬럼 목록에서 타겟 레이블과 파생된 이진 레이블은 강제로 제외
    exclude_cols = ['Binary_Label', 'Label'] # 원본 레이블이나 파생 레이블 컬럼명 모두 방어
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    X = df[feature_cols]
    y = df['Binary_Label']
    
    # 무한대(inf, -inf) 값을 NaN으로 변환 후 결측치와 함께 0으로 채우기
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 혹시라도 float32 범위를 넘어가는 극단적인 수치가 있다면 상하한선 클리핑 또는 타입 정리
    # (선택 사항이지만 안전을 위해 numpy 기반으로 클린징)
    X = np.clip(X, -1e15, 1e15)
    
    print(">> Random Forest 피처 중요도 추출 중...")
    # 가볍고 빠르게 중요도를 뽑기 위해 트리 개수(n_estimators)를 조절
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # 3. 중요도 상위 피처 정렬 출력
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    
    print("\n--- [Top 10 핵심 위협 지표(Feature)] ---")
    print(top_features)

if __name__ == "__main__":
    extract_feature_importance()