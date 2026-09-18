import glob
import os
import numpy as np
import pandas as pd


def compare_baseline_sample_sizes_with_benign():
  path_pattern = "../data/processed_data/cic-ids-2017/*.parquet"
  files = glob.glob(path_pattern)

  if not files:
    print(f">> 경로를 확인해주세요: {path_pattern}")
    return

  print(f">> 총 {len(files)}개의 Parquet 파일 로드 및 통합 중 (전체 데이터 적재)...")

  df_list = [pd.read_parquet(f) for f in files]
  df = pd.concat(df_list, ignore_index=True)
  df.columns = df.columns.str.strip()

  # 레이블 컬럼 동적 탐색
  label_col = [c for c in df.columns if "label" in c.lower()][0]

  # 숫자형 피처 분리
  exclude_cols = [label_col, "Binary_Label"]
  feature_cols = [
      c
      for c in df.select_dtypes(include=["float64", "int64"]).columns
      if c not in exclude_cols
  ]

  # 데이터 정제 (무한대/결측치 처리)
  df_clean = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

  # 정상(BENIGN) 데이터와 공격 데이터 분리
  benign_mask = df[label_col].str.strip().str.upper() == "BENIGN"
  
  df_benign_full = df[benign_mask].copy()
  df_benign_clean = df_clean[benign_mask]

  df_attack = df[~benign_mask].copy()
  df_attack_clean = df_clean[~benign_mask]

  total_benign_count = len(df_benign_full)
  print(f">> 데이터 로드 완료")
  print(f"   - 전체 정상(BENIGN) 세션: {total_benign_count:,}건")
  print(f"   - 전체 공격 세션: {len(df_attack):,}건")
  print(f"   - 활용되는 네트워크 통계 피처 수: {len(feature_cols)}개")

  # 비교할 정상 기준선(Baseline) 표본 크기 시나리오
  size_scenarios = {
      f"Full ({total_benign_count:,}건)": total_benign_count,
      "1,000,000건 (1M)": 1000000,
      "500,000건 (500k)": 500000,
      "300,000건 (300k)": 300000,
  }

  comparison_results = {}

  for config_name, sample_size in size_scenarios.items():
    print(
        f"\n>> [시나리오 실행] 정상 기준선 표본 크기 적용 중: {config_name}..."
    )

    # 1. 지정된 크기만큼 정상 데이터 샘플링 (기준선 수립용)
    if sample_size >= total_benign_count:
      df_benign_sample = df_benign_clean
    else:
      df_benign_sample = df_benign_clean.sample(n=sample_size, random_state=42)

    # 2. 해당 정상 표본을 기준으로 평균과 표준편차 추출
    baseline_mean = df_benign_sample.mean()
    baseline_std = df_benign_sample.std().replace(0, 1.0)

    # --- [A] 정상(BENIGN) 데이터 전체에 대한 리스크 점수 계산 ---
    z_scores_benign = np.abs((df_benign_clean - baseline_mean) / baseline_std)
    ext_dev_benign = (z_scores_benign > 3.0).sum(axis=1)
    ratio_benign = ext_dev_benign / len(feature_cols)
    intensity_benign = np.clip(z_scores_benign.mean(axis=1), 0, 20.0)
    raw_scores_benign = (ratio_benign * 70.0) + ((intensity_benign / 20.0) * 30.0)
    df_benign_full["Risk_Score"] = np.clip(raw_scores_benign, 0, 100)

    # --- [B] 공격 데이터 전체에 대한 리스크 점수 계산 ---
    z_scores_attack = np.abs((df_attack_clean - baseline_mean) / baseline_std)
    ext_dev_attack = (z_scores_attack > 3.0).sum(axis=1)
    ratio_attack = ext_dev_attack / len(feature_cols)
    intensity_attack = np.clip(z_scores_attack.mean(axis=1), 0, 20.0)
    raw_scores_attack = (ratio_attack * 70.0) + ((intensity_attack / 20.0) * 30.0)
    df_attack["Risk_Score"] = np.clip(raw_scores_attack, 0, 100)

    # 3. 결과 집계 (정상 평균/맥스 + 공격 유형별 평균)
    scenario_summary = {}
    scenario_summary["[BENIGN] 평균 점수"] = df_benign_full["Risk_Score"].mean()
    scenario_summary["[BENIGN] 최대 점수"] = df_benign_full["Risk_Score"].max()

    attack_means = df_attack.groupby(label_col)["Risk_Score"].mean()
    for atk_type, score in attack_means.items():
      scenario_summary[f"Attack: {atk_type.strip()}"] = score

    comparison_results[config_name] = scenario_summary

  # 최종 비교 결과 표 생성 및 출력
  df_comparison = pd.DataFrame(comparison_results)
  print(
      "\n===================================================================="
  )
  print(
      "     [정상 기준선 표본 크기별 정상 및 공격 유형별 평균 리스크 점수 비교]"
  )
  print(
      "===================================================================="
  )
  print(df_comparison.to_string())
  print(
      "===================================================================="
  )


if __name__ == "__main__":
  compare_baseline_sample_sizes_with_benign()