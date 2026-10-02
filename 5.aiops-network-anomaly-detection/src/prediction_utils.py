import glob

import pandas as pd
import numpy as np

def calculate_prediction_labels(df):
    """
    Risk_Deviation 분위수(P95, P99) 및 4대 메타 축 상위 10%(P90) 임계값을 기반으로
    계층형 이상 탐지 라벨(Normal / Caution / Danger)을 산출합니다.
    
    Parameters:
        df (pd.DataFrame): 마스터 데이터프레임 (Year, Risk_Deviation, 4대 메타 축 스코어 포함)
        
    Returns:
        pd.DataFrame: Pred_Label 컬럼이 추가된 데이터프레임
    """
    print(f"[INFO] 예측 라벨(Prediction Label) 산출 시작 (총 {len(df):,}행)...")
    result_df = df.copy()
    
    # 1. 연도별 P95, P99 동적 계산 및 병합
    print("[INFO] 연도별 Risk_Deviation P95, P99 임계값 산출 중...")
    global_p = result_df.groupby("Year")["Risk_Deviation"].agg(
        P95=lambda x: np.nanpercentile(x, 95),
        P99=lambda x: np.nanpercentile(x, 99)
    ).reset_index()
    
    result_df = result_df.merge(global_p, on="Year", how="left")
    
    # 2. 4대 메타 축별 상위 10%(P90) 임계값 산출
    print("[INFO] 4대 메타 축별 P90 임계값 산출 중...")
    sub_score_cols = ["Volume_Score", "Size_Payload_Score", "Topology_Score", "Control_State_Score"]
    sub_thresholds = {}
    
    for col in sub_score_cols:
        if col in result_df.columns:
            sub_thresholds[col] = result_df[col].quantile(0.90)
        else:
            sub_thresholds[col] = float('inf')
            
    # 3. 은닉형 공격 구제 조건(4개 메타 축 중 하나라도 P90 이상인지 여부) 벡터화 계산
    # 4대 축 중 단 하나라도 해당 임계값을 넘는 행을 마스크(True/False)로 추출
    hidden_attack_mask = np.zeros(len(result_df), dtype=bool)
    for col in sub_score_cols:
        if col in result_df.columns:
            cond = result_df[col].values >= sub_thresholds[col]
            hidden_attack_mask = np.logical_or(hidden_attack_mask, cond)
            
    # 4. np.select를 활용한 초고속 계층형 조건 판정 (iterrows 루프 제거로 350만 행 최적화)
    deviation = result_df["Risk_Deviation"].values
    p95 = result_df["P95"].values
    p99 = result_df["P99"].values
    
    conditions = [
        (deviation > p99),                               # 1순위: Danger (폭발형)
        (deviation > p95),                               # 2순위: Caution (1차 경계)
        (deviation <= p95) & (hidden_attack_mask)        # 3순위: Caution (은닉형 구제 룰 - 1개 영역만 튀어도 구제)
    ]
    
    choices = ["Danger", "Caution", "Caution"]
    
    # 조건에 해당하지 않는 나머지는 모두 "Normal"로 처리
    result_df["Pred_Label"] = np.select(conditions, choices, default="Normal")
    
    print("[INFO] 예측 라벨 분류 완료!")
    print(result_df["Pred_Label"].value_counts())
    print(p95)
    print(p99)
    return result_df


# ==========================================
# 단독 테스트를 위한 메인 블록
# ==========================================
if __name__ == "__main__":
   print(">> [Test] risk_utils.py 단독 실행 중...")
       
   path = "../data/processed_data/master_scored_dataset.parquet"
   files = glob.glob(path)
       
   if files:
     print(">> Parquet 파일 로딩 중...")
     df_master = pd.read_parquet(files[0])
     print(">> 로딩 완료!")
       
       # 함수 실행
   output_df = calculate_prediction_labels(df_master)
   print("\n[상위 5개 결과 미리보기]")
   print(output_df[['Year', 'Risk_Deviation', 'Pred_Label']].head())