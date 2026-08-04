import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def train_isolation_forest(df, sensor_cols, contamination=0.03):
  """전처리된 센서 데이터와 파생 피처를 입력받아 Isolation Forest 비지도 학습 모델을 구축하고,
  각 시점별 Anomaly Score(이상 점수)와 이상 여부 플래그를 산출하여 데이터프레임에 추가.

  [입력]
      df: 파생 피처(roll_mean, roll_std)가 포함된 Pandas DataFrame
      sensor_cols: 기본 수치형 센서 컬럼 이름 리스트
      contamination: 전체 데이터 중 이상치(Anomaly) 비율 추정치 (기본값: 0.03 = 3%)

  [출력]
      df: anomaly_score 및 is_anomaly_iforest 컬럼이 추가된 데이터프레임
      iso_forest: 학습이 완료된 Isolation Forest 모델 객체
      scaler: 피처 정규화에 사용된 StandardScaler 객체
  """
  # =========================================================================
  # 1. 모델 학습용 피처(Feature) 세트 구성
  # =========================================================================

  # 원본 센서 5개 + 이동 평균 5개 + 이동 표준편차 5개 = 총 15개 피처를 다변량 분석 입력값으로 조합
  # 순간 센서 수치뿐만 아니라 최근 추세(Mean)와 불안정성(Std)을 함께 고려하여 다변량 이상 패턴 탐지
  feature_cols = (
      sensor_cols
      + [f'{col}_roll_mean' for col in sensor_cols]
      + [f'{col}_roll_std' for col in sensor_cols]
  )

  # 모델에 입력될 학습 데이터셋 추출
  X = df[feature_cols]

  # =========================================================================
  # 2. 피처 스케일링 (Feature Scaling / 정규화)
  # =========================================================================

  # 센서마다 단위(K, rpm, Nm, min 등)와 수치 범위 차이가 커서 특정 센서가 모델 분할을 독점하는 현상 방지
  # StandardScaler를 통해 모든 피처를 평균 0, 표준편차 1 단위의 표준정규분포로 스케일링
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # =========================================================================
  # 3. Isolation Forest 모델 객체 생성 및 비지도 학습(Fit)
  # =========================================================================

  # Isolation Forest 비지도 이상 탐지 모델 하이퍼파라미터 정의
  # - n_estimators=100: 100개의 무작위 의사결정 나무(Tree)를 구축하여 결과의 안정성 및 신뢰성 확보
  # - contamination=contamination: 전체 데이터 중 약 3%를 이상치로 가정 (현장 설비 고장율 도메인 가설 반영)
  # - random_state=42: 코드 재실행 시에도 항상 동일한 예측 결과가 산출되도록 난수 고정
  # - n_jobs=-1: 모든 CPU 코어를 병렬 활용하여 모델 학습 및 추론 속도 최적화
  iso_forest = IsolationForest(
      n_estimators=100,
      contamination=contamination,
      random_state=42,
      n_jobs=-1,
  )

  # 스케일링된 15개 센서 피처 데이터셋을 Isolation Forest 모델에 학습시킴
  iso_forest.fit(X_scaled)

  # =========================================================================
  # 4. Anomaly Score(이상 점수) 및 예측 결과 산출
  # =========================================================================

  # scikit-learn의 decision_function은 음수일수록 고립 깊이가 짧아 더 심각한 이상치임을 의미
  # 현장 운용자가 직관적으로 이해할 수 있도록 [0~1] 범위의 점수로 Min-Max 변환 (1에 가까울수록 위험)
  raw_scores = iso_forest.decision_function(X_scaled)
  df['anomaly_score'] = (raw_scores.max() - raw_scores) / (
      raw_scores.max() - raw_scores.min()
  )

  # predict()는 contamination(3%) 비율 기준으로 이상치를 -1, 정상을 1로 자동 분류
  # 데이터 분석 표준 직관에 맞게 -1(이상치) -> 1, 1(정상) -> 0으로 플래그값 변환
  df['is_anomaly_iforest'] = iso_forest.predict(X_scaled)
  df['is_anomaly_iforest'] = np.where(df['is_anomaly_iforest'] == -1, 1, 0)

  # 결과 데이터프레임과 학습된 모델/스케일러 객체를 리턴
  return df, iso_forest, scaler


# =========================================================================
# 모델 예측과 출력과정을 분리하여, 외부 호출 시 불필요한 출력 방지
# =========================================================================
if __name__ == '__main__':
  # 1. 이전 단계 모듈(day2_features)에서 전처리 데이터 로드
  from day2_features import process_sensor_features

  print('===  데이터 로드 및 피처 생성 모듈 실행 ===')
  df_processed, sensor_cols, _ = process_sensor_features()

  # 2. 모델 학습 함수 실행
  print('\n=== Isolation Forest 모델 학습 및 추론 실행 ===')
  df_result, model, scaler = train_isolation_forest(
      df_processed, sensor_cols, contamination=0.03
  )

  # =========================================================================
  # 5. 모델 결과 모니터링 콘솔 출력
  # =========================================================================

  print('\n=== Isolation Forest 이상 탐지 결과 요약 ===')
  print(f'전체 데이터 건수: {len(df_result)}건')
  print(
      '탐지된 이상치(Anomaly) 건수:'
      f" {df_result['is_anomaly_iforest'].sum()}건"
      f" ({df_result['is_anomaly_iforest'].mean()*100:.2f}%)"
  )

  print('\n=== Anomaly Score 상위 5개 시점 (가장 위험한 공정 상태) ===')
  print(
      df_result[[
          'Air temperature [K]',
          'Rotational speed [rpm]',
          'Torque [Nm]',
          'anomaly_score',
          'is_anomaly_iforest',
      ]]
      .sort_values(by='anomaly_score', ascending=False)
      .head()
  )