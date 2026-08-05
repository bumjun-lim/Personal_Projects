import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def load_and_preprocess_data(data_path='ai4i2020.csv'):
  """센서 데이터를 로드하고 컬럼명 정제 및 롤링 특성을 생성.

  [입력]
      data_path: CSV 데이터셋 파일 경로
  [출력]
      df: 전처리가 완료된 데이터프레임
      feature_cols: 기본 + 파생 센서 특성 이름 리스트
  """
  if not os.path.exists(data_path):
    data_path = '../ai4i2020.csv'

  df = pd.read_csv(data_path)

  # 컬럼명 단순화 (공백 및 특수문자 제거)
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

  feature_cols = sensor_cols + [
      'Rotational_speed_roll_mean',
      'Rotational_speed_roll_std',
  ]

  return df, feature_cols


def extract_anomaly_score_feature(df, feature_cols):
  """Day 2의 Isolation Forest 모델을 적용하여 anomaly_score 특성을 산출.

  [입력]
      df: 전처리된 데이터프레임
      feature_cols: 1단계 센서 입력 특성 리스트
  [출력]
      df: anomaly_score 컬럼이 추가된 데이터프레임
      final_feature_cols: anomaly_score가 포함된 최종 지도학습 특성 리스트
  """
  X_raw = df[feature_cols]

  # Isolation Forest 비지도 학습 진행
  iso_forest = IsolationForest(
      n_estimators=100, contamination=0.035, random_state=42
  )
  iso_forest.fit(X_raw)

  # Anomaly Score 추출 (점수가 높을수록 이상치에 가깝도록 음수 반전)
  df['anomaly_score'] = -iso_forest.decision_function(X_raw)

  final_feature_cols = feature_cols + ['anomaly_score']

  return df, final_feature_cols


def train_supervised_models(X_train, y_train):
  """불균형 클래스 가중치(scale_pos_weight)를 적용하여 XGBoost 및 LightGBM을 학습.

  [입력]
      X_train, y_train: 학습용 특성 데이터 및 고장 라벨
  [출력]
      models: 학습 완료된 모델 객체 딕셔너리 {'XGBoost': model, 'LightGBM':
      model}
  """
  # 클래스 불균형 비율 계산 (정상 건수 / 고장 건수)
  ratio = (len(y_train) - sum(y_train)) / sum(y_train)

  # 1. XGBoost 모델 세팅 및 학습
  xgb_model = XGBClassifier(
      n_estimators=100,
      learning_rate=0.05,
      max_depth=5,
      scale_pos_weight=ratio,
      random_state=42,
  )
  xgb_model.fit(X_train, y_train)

  # 2. LightGBM 모델 세팅 및 학습
  lgb_model = LGBMClassifier(
      n_estimators=100,
      learning_rate=0.05,
      max_depth=5,
      scale_pos_weight=ratio,
      random_state=42,
      verbose=-1,
  )
  lgb_model.fit(X_train, y_train)

  return {'XGBoost': xgb_model, 'LightGBM': lgb_model}


def evaluate_and_visualize(models, X_test, y_test, feature_names):
  """모델별 성능(Precision, Recall, F1, ROC-AUC)을 평가하고 XGBoost 특성 중요도를 저장.

  [입력]
      models: 학습된 모델 딕셔너리
      X_test, y_test: 평가용 검증 데이터셋
      feature_names: 최종 특성 이름 리스트
  """
  for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print('\n' + '=' * 50)
    print(f'[{name} 성능 평가 결과]')
    print('=' * 50)
    print(
        classification_report(
            y_test, y_pred, target_names=['Normal(0)', 'Failure(1)']
        )
    )
    print(f'{name} ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}')

  # XGBoost 기반 특성 중요도 시각화
  xgb_model = models['XGBoost']
  plt.figure(figsize=(10, 6))
  importance = pd.Series(
      xgb_model.feature_importances_, index=feature_names
  ).sort_values(ascending=True)

  importance.plot(kind='barh', color='skyblue')
  plt.title('Feature Importance (XGBoost with Anomaly Score)')
  plt.xlabel('Importance')
  plt.tight_layout()

  # 그래프 파일 저장
  output_img = 'day3_feature_importance.png'
  plt.savefig(output_img)
  print(f"\n[완료] 특성 중요도 차트가 '{output_img}'로 저장되었습니다.")


# =========================================================================
# day3_model.py 직접 실행 시 실행되는 메인 파이프라인
# =========================================================================
if __name__ == '__main__':
  print('=== 데이터 로드 및 전처리 실행 ===')
  df_processed, base_features = load_and_preprocess_data()

  print('\n=== Isolation Forest Anomaly Score 특성 결합 ===')
  df_final, final_features = extract_anomaly_score_feature(
      df_processed, base_features
  )

  print('\n=== Train / Test 데이터셋 분할 ===')
  X = df_final[final_features]
  y = df_final['Machine_failure']

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )
  print(
      f'Train Data: {X_train.shape[0]}건, Test Data:'
      f' {X_test.shape[0]}건 분할 완료'
  )

  print('\n=== 지도학습 모델 학습 및 최종 성과 평가 ===')
  trained_models = train_supervised_models(X_train, y_train)
  evaluate_and_visualize(trained_models, X_test, y_test, final_features)