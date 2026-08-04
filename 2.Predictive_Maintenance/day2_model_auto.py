import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def train_isolation_forest_auto(df, sensor_cols):
  """[목적] contamination='auto' 옵션을 적용하여 Isolation Forest 모델을 비지도 학습시키고,
  정규화된 Anomaly Score(0~1)와 알고리즘 자체 기준의 이상 분류 플래그를 생성.

  [입력]
      df: 이동 평균/표준편차 파생 피처가 포함된 데이터프레임
      sensor_cols: 기본 수치형 센서 컬럼 리스트

  [출력]
      df: anomaly_score 및 is_anomaly_iforest 컬럼이 추가된 데이터프레임
      iso_forest: 학습된 모델 객체
      scaler: 피처 스케일러 객체
  """
  # 1. 학습 피처 조합 (원본 5개 + 이동평균 5개 + 이동표준편차 5개 = 총 15개)
  feature_cols = (
      sensor_cols
      + [f'{col}_roll_mean' for col in sensor_cols] # 이동평균
      + [f'{col}_roll_std' for col in sensor_cols]  # 이동표준편차
  )
  X = df[feature_cols]

  # 2. 피처 정규화 (StandardScaler)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # 3. Isolation Forest 모델 학습 (contamination='auto' 적용)
  # 비율을 인위적으로 고정하지 않고 알고리즘 자체 기준선으로 분할
  iso_forest = IsolationForest(
      n_estimators=100, contamination='auto', random_state=42, n_jobs=-1
  )
  iso_forest.fit(X_scaled)

  # 4. Anomaly Score (0~1 정규화) 및 auto 기본 예측 결과 산출
  raw_scores = iso_forest.decision_function(X_scaled)
  # df['anomaly_score'] 가 추후 설정할 threshold_candidates의 값과 비교했을때, 이상치 판별 기준이 됨
  df['anomaly_score'] = (raw_scores.max() - raw_scores) / (
      raw_scores.max() - raw_scores.min()
  )

  # auto 기준 예측 플래그 (-1 -> 1: 이상, 1 -> 0: 정상)
  df['is_anomaly_iforest'] = iso_forest.predict(X_scaled)
  df['is_anomaly_iforest'] = np.where(df['is_anomaly_iforest'] == -1, 1, 0)

  return df, iso_forest, scaler


def evaluate_thresholds(
    df,
    target_col='Machine failure',
    threshold_candidates=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
):
  """산출된 Anomaly Score에 대해 다양한 Threshold를 적용하여
  실제 고장 라벨 대비 오탐(FP), 미탐(FN), 정밀도(Precision), 재현율(Recall)을 산출.
  """
  if target_col not in df.columns:
    print(
        f"[경고] 데이터프레임에 정답 라벨 '{target_col}' 컬럼이 없어 임계값"
        ' 평가를 스킵합니다.'
    )
    return None

  y_true = df[target_col]
  results = []

  for th in threshold_candidates:
    # Threshold 이상일 때 1(이상치)로 판정
    pred_binary = (df['anomaly_score'] >= th).astype(int)

    # Confusion Matrix 산출 (적중 오탐 미탐 정상적중)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_binary).ravel()

    # 정밀도, 재현율, F1-Score 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    results.append({
        'Threshold': th,
        'TP(적중)': tp,
        'FP(오탐)': fp,
        'FN(미탐)': fn,
        'TN(정상적중)': tn,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
    })

  return pd.DataFrame(results)


# =========================================================================
# 모델 예측과 출력과정을 분리하여, 외부 호출 시 불필요한 출력 방지
# =========================================================================
if __name__ == '__main__':
  from day2_features import process_sensor_features

  print('=== 데이터 로드 및 피처 생성 실행 ===')
  df_processed, sensor_cols, _ = process_sensor_features()

  print('\n=== Isolation Forest 모델 학습 실행 (auto) ===')
  df_result, model, scaler = train_isolation_forest_auto(
      df_processed, sensor_cols
  )

  print(
      f"\n[contamination='auto' 결과] 탐지된 이상치:"
      f" {df_result['is_anomaly_iforest'].sum()}건"
  )

  print('\n=== Threshold(문턱값)별 오탐/미탐 성능 평가 ===')
  eval_df = evaluate_thresholds(df_result, target_col='Machine failure')

  if eval_df is not None:
    print(eval_df.to_string(index=False))