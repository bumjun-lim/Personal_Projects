# ==============================================================================
# Day 2 신용카드 사기 탐지 모델링 개편(Fraud Detection System)
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score

def get_model_predictions(data_path='creditcard.csv'):
    # 1. 데이터 로드
    df = pd.read_csv(data_path)
    
    # 원본 Time, Amount 값을 시각화 및 필터링용으로 복사
    raw_time = df['Time'].copy()
    raw_amount = df['Amount'].copy()
    
    # 2. Time, Amount 전처리 (RobustScaler)
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_time'] = scaler.fit_transform(df[['Time']])
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    # 3. Feature / Target 분리
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 4. Train/Test Split (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. RandomForest 모델 학습
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # 6. 예측
    y_pred = rf_model.predict(X_test)

    # 7. 대시보드용 테스트 데이터프레임 복원 (원본 Time, Amount 복구)
    test_df = df.loc[X_test.index].copy()
    test_df['Time'] = raw_time.loc[X_test.index]
    test_df['Amount'] = raw_amount.loc[X_test.index]
    test_df['Pred_Class'] = y_pred

    # 정탐/오탐/미탐/정상 상태 컬럼 추가
    def get_status(row):
        if row['Class'] == 1 and row['Pred_Class'] == 1:
            return 'TP (True Positive)'
        elif row['Class'] == 0 and row['Pred_Class'] == 1:
            return 'FP (False Positive)'
        elif row['Class'] == 1 and row['Pred_Class'] == 0:
            return 'FN (False Negative)'
        else:
            return 'TN (True Negative)'

    test_df['Detection_Status'] = test_df.apply(get_status, axis=1)

    return test_df, y_test, y_pred

if __name__ == "__main__":
    test_df, y_test, y_pred = get_model_predictions()
    print("Day 2 모델 파이프라인 단독 실행 완료!")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")