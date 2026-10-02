import glob
import numpy as np
import pandas as pd


def calculate_risk_deviation(df_master: pd.DataFrame):
  """마스터 데이터프레임에 Normal 그룹 기준 연도별 Risk 통계 및

  위험 이탈도(Risk Deviation) 컬럼을 추가하여 반환합니다.
  """
  # 1. Normal 그룹 추출 및 연도별 Risk_Score 통계 산출
  df_normal = df_master[df_master["Attack_Group"].str.lower() == "normal"]

  normal_stats = (
      df_normal.groupby("Year")["Risk_Score"]
      .agg(Normal_Risk_Mean="mean", Normal_Risk_STD="std")
      .reset_index()
  )

  # 2. 마스터 데이터프레임에 통계치 병합 (Left Join)
  df_master = pd.merge(df_master, normal_stats, on="Year", how="left")

  # 3. 새로운 핵심 요소 추가: 위험 이탈도(Risk Deviation) 계산
  df_master["Risk_Deviation"] = (
      df_master["Risk_Score"] - df_master["Normal_Risk_Mean"]
  ) / df_master["Normal_Risk_STD"].replace(0, np.nan)

  # 메모리 및 용량 최적화 (float32 다운캐스팅)
  df_master["Normal_Risk_Mean"] = df_master["Normal_Risk_Mean"].astype(
      "float32"
  )
  df_master["Normal_Risk_STD"] = df_master["Normal_Risk_STD"].astype("float32")
  df_master["Risk_Deviation"] = df_master["Risk_Deviation"].astype("float32")

  return df_master, normal_stats


def print_risk_report(df_master: pd.DataFrame, normal_stats: pd.DataFrame):
  # 공격 그룹별 Risk Deviation 요약 데이터 생성
  deviation_summary = (
      df_master.groupby(["Year", "Attack_Group"])["Risk_Deviation"]
      .agg(
          mean="mean",
          median="median",
          p95=lambda x: np.nanpercentile(x, 95),
          p99=lambda x: np.nanpercentile(x, 99),
          max="max",
          count="count",
      )
      .reset_index()
  )

  print("\n=== [연도별 Normal Risk Score 통계] ===")
  print(normal_stats)

  print("\n=== [공격 그룹별 Risk Deviation 상세 통계 비교] ===")
  print(deviation_summary.to_string(index=False))

  return deviation_summary

if __name__ == "__main__":
  print(">> [Test] risk_utils.py 단독 실행 중...")

  path = "../data/processed_data/master_scored_dataset_Logged_z_cap5.parquet"
  files = glob.glob(path)

  if files:
    print(">> Parquet 파일 로딩 중...")
    df_master = pd.read_parquet(files[0])
    print(">> 로딩 완료!")

  # 함수 테스트 수행
  df_tested, stats_tested = calculate_risk_deviation(df_master)
  print_risk_report(df_tested, stats_tested)
