import numpy as np
import pandas as pd
import data_loader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler


# 1. 데이터로더를 통해 훈련 셋과 테스트 셋을 분리하여 로드
train_df = data_loader.load_2017_data(file_path='../data/processed_data/sample_2017.parquet', include_attack=True, split='train')
test_df = data_loader.load_2017_data(file_path='../data/processed_data/sample_2017.parquet', include_attack=True, split='test')

print(f"학습 셋 로드 완료: {train_df.shape}")
print(f"테스트 셋 로드 완료: {test_df.shape}")

# 2. 정답(Label) 추출 함수 정의 및 적용
def extract_labels(df):
    label_col = [c for c in df.columns if c.strip().lower() == 'label'][0]
    return df[label_col].astype(str).str.strip().str.upper().apply(lambda x: 0 if 'BENIGN' in x else 1)

y_train = extract_labels(train_df)
y_test = extract_labels(test_df)

# 3. 수치형 피처 추출 및 무한대/결측치 정제 함수 정의 및 적용
def extract_features(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    if 'anomaly_pred' in X.columns:
        X = X.drop(columns=['anomaly_pred'])
    return X

X_train_raw = extract_features(train_df)
X_test_raw = extract_features(test_df)

# X_train_raw에 어떤 컬럼들이 포함되어 있는지 확인
print("--- 모델에 학습된 피처 목록 ---")
print(X_train_raw.columns.tolist())

# 혹시 레이블과 관련된 단어(label, class, attack 등)가 포함되어 있는지 강제로 체크
suspicious_cols = [c for c in X_train_raw.columns if any(keyword in c.lower() for keyword in ['label', 'class', 'attack', 'flag'])]
print(f"의심스러운 컬럼 목록: {suspicious_cols}")

# 4. 스케일링 적용 (주의: 데이터 누수 방지를 위해 Train 기준으로 fit 후 Test는 transform만!)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 5. Random Forest (지도 학습) 모델 학습
print("\nRandom Forest 모델 학습 중...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# 6. 테스트 셋으로 예측 수행
y_pred = rf_model.predict(X_test_scaled)

# 7. 성능 평가 리포트 출력
print("\n=== Random Forest 모델 성능 평가 리포트 (Test Set 기준) ===")
print(classification_report(y_test, y_pred, target_names=['Normal (Benign)', 'Anomaly (Attack)'], zero_division=0))