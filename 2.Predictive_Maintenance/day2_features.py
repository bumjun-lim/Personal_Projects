import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def process_sensor_features(
    csv_path='ai4i2020.csv', df=None, db_path='personal_projects.db'
):
  """센서 데이터를 로드하고, 상관계수 행렬 계산 및 이동 평균(Mean) / 변동성(Std) 파생 특성을 생성하여 리턴.

  csv_path: CSV 파일 경로, df: 전달받은 데이터프레임 (없을 경우 CSV/DB 로드)
  df: 파생 특성이 추가된 데이터프레임, sensor_cols: 센서 컬럼 목록, corr_matrix: 상관계수 행렬
  """
  # =========================================================================
  # 1. 데이터 로드 및 센서 컬럼 그룹화
  # =========================================================================

  # 외부에서 df를 전달받지 않은 경우, 파일/DB에서 데이터 로드
  if df is None:
    # SQLite DB에서 전처리된 sensor_logs 테이블을 불러와 Pandas DataFrame으로 변환
    # DB 커넥션을 맺고 SELECT 쿼리를 실행하여 분석 대상 데이터 프레임을 생성
    # (※ 실제 환경에서는 아래 2줄의 코드로 DB 데이터 로드.)
    # conn = sqlite3.connect(db_path)
    # df = pd.read_sql("SELECT * FROM sensor_logs", conn)

    # 데이터 로드 (CSV 파일 기본 로드)
    df = pd.read_csv(csv_path)

  # 다변량 이상 탐지 모델 학습에 사용할 물리 센서 데이터 컬럼을 명시적으로 그룹화
  # 도메인 관점에서 주요한 5개 센서 항목을 리스트 형태(sensor_cols)로 관리
  sensor_cols = [
      'Air temperature [K]',  # 대기 온도
      'Process temperature [K]',  # 공정 온도
      'Rotational speed [rpm]',  # 회전 속도
      'Torque [Nm]',  # 토크
      'Tool wear [min]',  # 공구 마모 시간
  ]

  # =========================================================================
  # 2. 센서 간 상관관계(Correlation) 계산
  # =========================================================================

  # 단변량 탐지에서 포착하기 어려운 센서 간 물리적 연관성(예: RPM 증가 시 토크 감소 등) 및 다공선성 파악
  # 지정한 센서 컬럼들에 대해 피어슨 상관계수(Pearson Correlation Coefficient) 행렬 산출
  corr_matrix = df[sensor_cols].corr()

  # =========================================================================
  # 3. 이동 평균(Rolling Window) 및 변동성 파생 특성 생성
  # =========================================================================

  # 순간적인 측정 오류(Spike Noise)로 인한 오탐(False Positive)을 줄이고, 최근 공정의 상태 추세를 반영할 윈도우 크기 정의
  # 최근 5개 수집 시점(Row)을 하나의 구간으로 묶는 타임 스텝 설정
  window_size = 5

  # 모든 핵심 센서 변수에 대해 시계열 평활화 및 변동성 특성을 일괄 생성
  # 센서 컬럼 리스트를 순회하며 이동 평균(Rolling Mean)과 이동 표준편차(Rolling Std) 파생 변수를 추가
  for col in sensor_cols:

    # 노이즈를 제거하여 연속적인 부하 증가 및 기계의 장기적인 추세(Trend)를 포착
    # rolling(window=5)을 적용해 이동 평균을 계산. min_periods=1을 통해 초기 1~4개 행에서도 NaN 없이 즉시 값 생성
    df[f'{col}_roll_mean'] = (
        df[col].rolling(window=window_size, min_periods=1).mean()
    )

    # 설비 작동 중 순간적인 진동, 충격, 불안정성(Volatility)을 정량화하여 기계의 미세 이상 징후 감지
    # rolling(window=5)으로 이동 표준편차를 구하고, 데이터 첫 번째 행에서 발생하는 NaN(표준편차 불가)은 fillna(0)으로 0 채움
    df[f'{col}_roll_std'] = (
        df[col].rolling(window=window_size, min_periods=1).std().fillna(0)
    )

  return df, sensor_cols, corr_matrix


# =========================================================================
# 파생 특성 생성과정과, 시각화 과정을 분리하여, 외부 호출 시 불필요한 연산 방지
# =========================================================================
if __name__ == '__main__':
  # 연산 함수 호출
  df_processed, sensor_cols, corr_matrix = process_sensor_features()

  # =========================================================================
  # 2-1. 상관관계 시각화 (직접 실행 시에만 차트 팝업 출력)
  # =========================================================================

  # 시각화 차트의 가독성을 확보하기 위해 전체 캔버스 크기를 가로 8, 세로 6 인치로 설정
  plt.figure(figsize=(8, 6))

  # 상관계수 행렬을 색상(Heatmap)과 수치로 직관적으로 표현하여 이상 패턴 추적의 기초 자료 확보
  # Seaborn 히트맵 생성 (annot=True: 수치 표시, fmt=".2f": 소수점 둘째 자리 표현, vmin/vmax: 색상 범위 -1~1 고정)
  sns.heatmap(
      corr_matrix,
      annot=True,
      fmt='.2f',
      cmap='coolwarm',
      vmin=-1,
      vmax=1,
  )

  # 차트의 제목 설정 및 여백 최적화로 시각화 보고서 전달력 향상
  # 타이틀 지정 및 레이아웃 정리 후 팝업 출력
  plt.title('Sensor Correlation Matrix', fontsize=14)
  plt.tight_layout()
  plt.show()

  # EDA 노트 기록 및 수치 검증을 위해 매트릭스 결과를 콘솔에 텍스트 형태로 직접 출력
  # 상관계수 데이터프레임 표준 출력
  print('=== 센서 간 상관계수 요약 ===')
  print(corr_matrix)

  # =========================================================================
  # 4. 파생 특성 생성 결과 관찰 및 검증 (직접 실행 시에만 콘솔 출력)
  # =========================================================================

  # 파생 변수가 원본 데이터 형태에 맞게 오차 없이 신규 컬럼으로 바인딩되었는지 최종 결과 확인
  # 'Rotational speed [rpm]' 센서의 원본, 이동 평균, 이동 표준편차 3개 컬럼의 상위 5개 행을 추출해 모니터링 출력
  print('\n=== 이동 평균 파생 특성 생성 완료 ===')
  print(
      df_processed[[
          'Rotational speed [rpm]',
          'Rotational speed [rpm]_roll_mean',
          'Rotational speed [rpm]_roll_std',
      ]].head()
  )
