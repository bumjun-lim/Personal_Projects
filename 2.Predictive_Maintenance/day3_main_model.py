"""
day3_ablation_study_model.py ➔ (소거 연구를 통해 검증 및 최종 메인 모델로 채택)

"""
import os
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np

def load_and_preprocess_raw_data(data_path='ai4i2020.csv'):
  """센서 데이터를 로드하고 anomaly_score 없이 순수 센서 및 롤링 특성만 준비.

  [입력]
      data_path: CSV 데이터셋 파일 경로
  [출력]
      df: 전처리가 완료된 데이터프레임
      feature_cols: anomaly_score가 제외된 센서 특성 리스트
  """
  if not os.path.exists(data_path):
    data_path = '../ai4i2020.csv'

  df = pd.read_csv(data_path)

  # 컬럼명 정제
  df.columns = [
      'UDI',
      'Product_ID',
      'Type',
      'Air_temperature',
      'Process_temperature',
      'Rotational_speed',
      'Torque',
      'Tool_wear',
      'Machine_failure',
      'TWF',
      'HDF',
      'PWF',
      'OSF',
      'RNF',
  ]

  sensor_cols = [
      'Air_temperature',
      'Process_temperature',
      'Rotational_speed',
      'Torque',
      'Tool_wear',
  ]

  # 롤링 특성 생성 (Rotational_speed 기반 윈도우 사이즈 5)
  df['Rotational_speed_roll_mean'] = (
      df['Rotational_speed'].rolling(window=5, min_periods=1).mean()
  )
  df['Rotational_speed_roll_std'] = (
      df['Rotational_speed']
      .rolling(window=5, min_periods=1)
      .std()
      .fillna(0)
  )

  # anomaly_score를 포함하지 않은 순수 특성 목록
  pure_feature_cols = sensor_cols + [
      'Rotational_speed_roll_mean',
      'Rotational_speed_roll_std',
  ]

  return df, pure_feature_cols


def train_models_without_anomaly_score(X_train, y_train):
  """anomaly_score 없이 클래스 가중치를 적용하여 XGBoost 및 LightGBM을 학습.

  [입력]
      X_train, y_train: anomaly_score가 제외된 학습용 데이터
  [출력]
      models: 학습 완료된 모델 딕셔너리
  """
  ratio = (len(y_train) - sum(y_train)) / sum(y_train)

  # XGBoost
  xgb_model = XGBClassifier(
      n_estimators=100,
      learning_rate=0.05,
      max_depth=5,
      scale_pos_weight=ratio,
      random_state=42,
  )
  xgb_model.fit(X_train, y_train)

  # LightGBM
  lgb_model = LGBMClassifier(
      n_estimators=100,
      learning_rate=0.05,
      max_depth=5,
      scale_pos_weight=ratio,
      random_state=42,
      verbose=-1,
  )
  lgb_model.fit(X_train, y_train)

  return {'XGBoost (No Anomaly Score)': xgb_model, 'LightGBM (No Anomaly Score)': lgb_model}


def evaluate_ablation_results(models, X_test, y_test):
  """anomaly_score 미포함 모델들의 성능 평가 및 결과를 출력.

  [입력]
      models: 학습된 모델 딕셔너리
      X_test, y_test: 평가용 검증 데이터셋
  """
  for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print('\n' + '=' * 60)
    print(f'[{name} 성능 평가 결과]')
    print('=' * 60)
    print(
        classification_report(
            y_test, y_pred, target_names=['Normal(0)', 'Failure(1)']
        )
    )
    print(f'{name} ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}')


# =========================================================================
# 메인 실행 파이프라인
# =========================================================================
if __name__ == '__main__':
  print('=== [Ablation Test] Anomaly Score 제외 모델 성능 검증 시작 ===')

  # 1. 데이터 로드 (anomaly_score 미포함)
  df_processed, pure_features = load_and_preprocess_raw_data()

  X = df_processed[pure_features]
  y = df_processed['Machine_failure']

  # 2. 동일한 random_state(42) 및 stratify 조건으로 Train/Test 분할
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  print(f'입력 피처 목록 ({len(pure_features)}개): {pure_features}')
  print(f'Train Data: {X_train.shape[0]}건, Test Data: {X_test.shape[0]}건 분할 완료')

  # 3. 모델 학습 및 성능 평가
  trained_models = train_models_without_anomaly_score(X_train, y_train)
  evaluate_ablation_results(trained_models, X_test, y_test)
  # --- [추가 2] 
  main_xgb = trained_models['XGBoost (No Anomaly Score)']
  importances = main_xgb.feature_importances_
  indices = np.argsort(importances)

  plt.figure(figsize=(10, 6))
  plt.title(
      'Feature Importance (Main XGBoost - Pure Sensors)', fontsize=14, pad=15
  )
  plt.barh(
      range(len(indices)),
      importances[indices],
      color='skyblue',
      align='center',
  )
  plt.yticks(range(len(indices)), [pure_features[i] for i in indices])
  plt.xlabel('Importance')
  plt.tight_layout()
  plt.savefig('day3_main_feature_importance.png', dpi=300)
  plt.close()
  print('\n[완료] 메인 모델 특성 중요도 차트가 저장되었습니다.')