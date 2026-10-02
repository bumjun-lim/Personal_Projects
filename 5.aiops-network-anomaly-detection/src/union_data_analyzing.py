import glob
import os
import numpy as np
import pandas as pd
from attack_mapping import apply_attack_group_and_year
from zcore_calculating import calculate_meta_scores
from risk_utils import calculate_risk_deviation
from rca_utils import calculate_rca_metrics
from prediction_utils import calculate_prediction_labels

def extract_essential_features(df):
    """
    원본 Raw 피처는 제거하고, 4대 대분류 스코어, 최종 Risk_Score, 세부 Z-Score, 메타 축만 남기는 함수
    """
    # 1. 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 2. 레이블 컬럼 표준화
    if "Label" not in df.columns:
        for alt_col in ["label", "label2", "attack_cat"]:
            if alt_col in df.columns:
                df = df.rename(columns={alt_col: "Label"})
                break

    # 3. 필요한 모든 컬럼들을 명시적으로 정의 (중복 원천 차단)
    base_cols = [
        "Year", 
        "Attack_Group", 
        "Label", 
        "Risk_Score",
        "Volume_Score",
        "Size_Payload_Score",
        "Topology_Score",
        "Control_State_Score"
    ]
    
    # Z__로 시작하는 개별 피처별 Z-Score 컬럼들
    z_score_cols = [col for col in df.columns if col.startswith("Z__")]
    
    # 최종 합치기 (중복 방지를 위해 dict.fromkeys 활용)
    target_cols = base_cols + z_score_cols
    existing_cols = [col for col in target_cols if col in df.columns]
    existing_cols = list(dict.fromkeys(existing_cols))

    df_slim = df[existing_cols].copy()

    # 4. 숫자형 컬럼 강제 캐스팅 (문자열 "Null", 공백, 이상치 완벽 차단)
    numeric_cols = [col for col in df_slim.columns if col not in ["Year", "Attack_Group", "Label"]]
    
    for col in numeric_cols:
        df_slim[col] = df_slim[col].replace(to_replace=[r"^[Nn]ull$", r"^\s*$", ""], value=np.nan, regex=True)
        df_slim[col] = pd.to_numeric(df_slim[col], errors="coerce").astype(float)

    return df_slim
    
def main():
    path_2017 = "../data/processed_data/cic-ids-2017/*.parquet"
    path_2025 = "../data/processed_data/cic-iiot-2025/*.parquet"

    files_2017 = glob.glob(path_2017)
    files_2025 = glob.glob(path_2025)
    
    print(f">> 2017년 파일 개수: {len(files_2017)}개 발견")
    print(f">> 2025년 파일 개수: {len(files_2025)}개 발견")
    
    if not files_2017 or not files_2025:
        print(">> 경로를 다시 확인해주세요.")
        return
    
    # 데이터 로드 및 병합
    df_17 = pd.concat([pd.read_parquet(f) for f in files_2017], ignore_index=True)
    df_25 = pd.concat([pd.read_parquet(f) for f in files_2025], ignore_index=True)
    
    print(f">> 로드 직후 Shape - 2017: {np.shape(df_17)} | 2025: {np.shape(df_25)}")  
    
    # 컬럼명 공백 제거
    df_17.columns = df_17.columns.str.strip()
    df_25.columns = df_25.columns.str.strip()

    # ==========================================
    #  1단계: 연도 및 상위 공격 그룹 맵핑 적용
    # ==========================================
    print(">> [1/3] 연도 및 상위 공격 그룹 맵핑 진행 중...")
    df_17_mapped = apply_attack_group_and_year(df_17, "2017", "Label")
    df_25_mapped = apply_attack_group_and_year(df_25, "2025", "label2")

    # ==========================================
    #  2단계: 메타 축별 점수 및 Risk_Score 산출 (Z-score 캡핑 적용)
    # ==========================================
    print(">> [2/3] 메타 축별 Z-score 스코어링 및 캡핑(Max=5.0) 적용 중...")
    df_17_scored = calculate_meta_scores(df_17_mapped, "2017", max_z_cap=5.0)
    df_25_scored = calculate_meta_scores(df_25_mapped, "2025", max_z_cap=5.0)

    # ------------------------------------------
    #  2.5 필요한 핵심 정보만 남기고 Raw 피처 드랍
    # ------------------------------------------
    print(">> [추가] 분석용 핵심 스코어 지표 및 메타 축만 추출 중...")
    df_17_slim = extract_essential_features(df_17_scored)
    df_25_slim = extract_essential_features(df_25_scored)


    # ==========================================
    #  3단계: 마스터 통합 및 최종 결과물 저장
    # ==========================================
    print(">> [3/3] 최종 마스터 데이터셋 통합 중...")
    df_master = pd.concat([df_17_slim, df_25_slim], ignore_index=True)


    
    print("\n>> 위험 이탈도, RCA Scores, Pred_Label 산출 중...")
    # ------------------------------------------
    # 3-1. 위험 이탈도(Risk Deviation) 계산 및 검증 리포트 출력
    # ------------------------------------------
    df_master, normal_stats = calculate_risk_deviation(df_master)

    # ------------------------------------------
    # 3-2. RCA Scores 계산
    # ------------------------------------------
    df_master = calculate_rca_metrics(df_master)

    # ------------------------------------------
    # 3-3. 예측라벨 (Pred_Label) 계산 
    # ------------------------------------------
    df_master = calculate_prediction_labels(df_master)

    # 최종 결과 저장
    print(f">> 최종 마스터 데이터셋 Shape: {df_master.shape}")
    
    output_dir = "../data/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "master_scored_dataset.parquet")
    
    df_master.to_parquet(output_path, index=False)
    print(f"\n>> ✨ 최종 계층형 통합 리포트 저장 완료: {output_path}")

if __name__ == "__main__":
    main()