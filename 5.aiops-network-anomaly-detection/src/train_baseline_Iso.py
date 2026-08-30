import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler

# 1. 2017년 샘플 데이터 로드 (함수 혹은 직접 읽기)
# 방금 만든 데이터로더의 로직을 반영해 공격/정상이 섞인 데이터를 안전하게 로드
df = pd.read_parquet('../data/processed_data/sample_2017.parquet')
print(f"데이터 로드 완료: {df.shape}")

# 2. 전처리: 라벨 컬럼 및 수치형 이외의 비정상 데이터(문자열 등) 원천 차단
# 텍스트 컬럼이나 Object 타입이 섞여 들어가면 스케일러에서 에러가 나므로 확실히 수치형만 골라내기.
numeric_cols = df.select_dtypes(include=[np.number]).columns
X = df[numeric_cols].fillna(0)

# 무한대(inf, -inf) 값을 NaN으로 변환한 뒤 0으로 채우기
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# 혹시라도 모델 학습에 방해가 되는 예측 결과 컬럼이 있다면 제외
if 'anomaly_pred' in X.columns:
    X = X.drop(columns=['anomaly_pred'])

print(f"학습에 사용할 수치형 피처 개수: {X.shape[1]}개")

# 3. 스케일링 적용 (RobustScaler)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 4. 베이스라인 이상 탐지 모델 학습 (Isolation Forest)
print("Isolation Forest 모델 학습 중...")
model = IsolationForest(contamination=0.5, random_state=42)
preds = model.fit_predict(X_scaled)

# 예측 결과 확인 (-1: 이상치/공격, 1: 정상)
df['anomaly_pred'] = preds
print("\n--- 모델 예측 결과 분포 (-1: 이상, 1: 정상) ---")
print(pd.Series(preds).value_counts())

# 5. 실제 정답(Label) 컬럼 동적 탐색 및 분포 확인
label_col = [c for c in df.columns if c.strip().lower() == 'label'][0]
print("\n--- 실제 정답(Label) 분포 ---")
print(df[label_col].value_counts())

# 6. 성능 평가 (F1-Score, Precision, Recall 출력)
# 실제 정답: 'BENIGN'이 포함되면 0(정상), 아니면 1(공격)
y_true = (
    df[label_col]
    .astype(str)
    .str.strip()
    .str.upper()
    .apply(lambda x: 0 if 'BENIGN' in x else 1)
)

# 모델 예측값 변환: -1(이상/공격) -> 1, 1(정상) -> 0
y_pred = np.where(preds == -1, 1, 0)

# 제로 디비전 경고 방지를 위해 zero_division=0 옵션 추가
print("\n=== 모델 성능 평가 리포트 ===")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=['Normal (Benign)', 'Anomaly (Attack)'],
        zero_division=0,
    )
)