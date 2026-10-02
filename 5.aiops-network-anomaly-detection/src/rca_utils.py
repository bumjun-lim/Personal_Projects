import glob
import pandas as pd
import numpy as np

def calculate_rca_metrics(df):
    """
    350만 건 이상 대용량 데이터셋에서 메모리 폭발 없이 
    RCA 8개 메타 컬럼을 초고속으로 산출합니다.
    """
    print(f"[INFO] RCA 메타 컬럼 산출 시작 (총 {len(df):,}행 대용량 최적화 처리 중...)")
    
    result_df = df.copy()
    
    # 1. 4대 메타 축 정의 및 처리
    meta_score_cols = [
        'Volume_Score', 
        'Size_Payload_Score', 
        'Topology_Score', 
        'Control_State_Score'
    ]
    meta_name_mapping = {
        'Volume_Score': 'Volume',
        'Size_Payload_Score': 'Size & Payload',
        'Topology_Score': 'Topology',
        'Control_State_Score': 'Control State'
    }
    
    meta_df = result_df[meta_score_cols]
    max_meta_col_idx = meta_df.values.argmax(axis=1)
    max_meta_vals = meta_df.values.max(axis=1)
    
    col_names = np.array(meta_score_cols)
    top_group_raw = col_names[max_meta_col_idx]
    top_group_groups = [meta_name_mapping.get(col, col) for col in top_group_raw]
    top_group_scores = np.round(max_meta_vals, 2)
    
    # 2. 세부 피처(Z-score) Top 3 초고속 추출
    z_feature_cols = [col for col in result_df.columns if col.startswith('Z__')]
    
    if z_feature_cols:
        z_array = result_df[z_feature_cols].abs().values  
        z_col_names = np.array([col.replace('Z__', '').replace('_', ' ') for col in z_feature_cols])
        
        n_rows, n_cols = z_array.shape
        top_k = min(3, n_cols)
        
        # argpartition을 통한 상위 k개 인덱스 고속 추출
        part_indices = np.argpartition(-z_array, top_k, axis=1)[:, :top_k]
        
        # 각 행별 정렬
        rows = np.arange(n_rows)
        sorted_offsets = np.argsort(-z_array[rows[:, None], part_indices], axis=1)
        sorted_top_indices = part_indices[rows[:, None], sorted_offsets]
        
        # [수정 포인트] 1차원 인덱싱을 사용하여 메모리 폭발(90 TiB 에러) 방지
        t1_idx = sorted_top_indices[:, 0]
        t1_names = z_col_names[t1_idx]
        t1_scores = np.round(z_array[rows, t1_idx], 2)
        
        if top_k > 1:
            t2_idx = sorted_top_indices[:, 1]
            t2_names = z_col_names[t2_idx]
            t2_scores = np.round(z_array[rows, t2_idx], 2)
        else:
            t2_names, t2_scores = [None] * n_rows, [0] * n_rows
            
        if top_k > 2:
            t3_idx = sorted_top_indices[:, 2]
            t3_names = z_col_names[t3_idx]
            t3_scores = np.round(z_array[rows, t3_idx], 2)
        else:
            t3_names, t3_scores = [None] * n_rows, [0] * n_rows
    else:
        t1_names = t2_names = t3_names = [None] * len(result_df)
        t1_scores = t2_scores = t3_scores = [0] * len(result_df)

    # 3. 8개 RCA 메타 컬럼 주입
    result_df['Top_Root_Cause_Group'] = top_group_groups
    result_df['Top_Group_Score'] = top_group_scores
    result_df['Top1_Root_Cause_Feature'] = t1_names
    result_df['Top1_Feature_Score'] = t1_scores
    result_df['Top2_Root_Cause_Feature'] = t2_names
    result_df['Top2_Feature_Score'] = t2_scores
    result_df['Top3_Root_Cause_Feature'] = t3_names
    result_df['Top3_Feature_Score'] = t3_scores

    print("[INFO] RCA 350만 행 벡터화 처리 및 메타 컬럼 8개 생성 완료.")
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
    output_df = calculate_rca_metrics(df_master)
    
    # 결과 확인
    rca_cols = [
        'Top_Root_Cause_Group', 'Top_Group_Score',
        'Top1_Root_Cause_Feature', 'Top1_Feature_Score',
        'Top2_Root_Cause_Feature', 'Top2_Feature_Score',
        'Top3_Root_Cause_Feature', 'Top3_Feature_Score'
    ]
    print("\n[테스트 결과]")
    print(output_df[rca_cols].head(5))