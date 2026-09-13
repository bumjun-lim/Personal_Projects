import pandas as pd
import numpy as np
import glob
import os

def build_transparent_risk_engine_2025():
    path_pattern = "../data/processed_data/cic-iiot-2025/*.parquet"
    files = glob.glob(path_pattern)
    
    if not files:
        print(f"경로를 확인해주세요: {path_pattern}")
        return

    print(f">> 총 {len(files)}개의 2025년 Parquet 파일 로드 및 통합 시작...")
    
    # 1. 전체 데이터 로드 및 결합 (메모리 최적화 및 균등 분포 확보를 위한 샘플링)
    df_list = []
    for f in files:
        df_chunk = pd.read_parquet(f)
        df_list.append(df_chunk.sample(n=min(len(df_chunk), 3000), random_state=42))
        
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()
    
    # [변경] 라벨 컬럼을 'label2'로 명시적 고정 (없을 경우를 대비해 소문자 처리 안전장치 추가)
    label2_cols = [c for c in df.columns if c.lower() == 'label2']
    if not label2_cols:
        print("데이터셋에 'label2' 컬럼이 존재하지 않습니다. 컬럼명을 확인해주세요.")
        return
    label_col = label2_cols[0]
    
    # 2. 숫자형 피처 분리 (모든 레이블 및 관련 파생 컬럼 철저한 제외)
    exclude_cols = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower() or 'attack' in c.lower()]
    feature_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c not in exclude_cols]
    
    # 데이터 정제 (무한대/결측치 처리)
    df_clean = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 3. [Baseline 수립] 오직 정상(benign) 트래픽만을 기준으로 평균과 표준편차 추출
    benign_mask = df[label_col].astype(str).str.lower().str.strip() == 'benign'
    df_benign = df_clean[benign_mask]
    
    baseline_mean = df_benign.mean()
    baseline_std = df_benign.std().replace(0, 1.0) # 표준편차 0인 경우 분모 0 방지
    
    print(f">> 2025년 정상 기준선 수립 완료 (전체 세션: {len(df)}건 중 참조 정상 세션: {len(df_benign)}건)")
    print(f">> 활용되는 네트워크 통계 피처 수: {len(feature_cols)}개")
    
    # 4. [이탈도 산출] 전체 세션에 대한 Z-Score 계산 (정상 평균으로부터의 거리 측정)
    z_scores = np.abs((df_clean - baseline_mean) / baseline_std)
    
    # 5. [환산 공식 적용]
    # - 1축: 전체 피처 중 3-Sigma를 초과한 피처의 비율
    extreme_deviations = (z_scores > 3.0).sum(axis=1)
    abnormal_feature_ratio = extreme_deviations / len(feature_cols)
    
    # - 2축: 전체 피처의 평균 이탈 강도
    mean_deviation_intensity = np.clip(z_scores.mean(axis=1), 0, 20.0)
    
    # 최종 리스크 스코어 수식 (비율 중심 70% + 강도 중심 30% 조합의 100점 만점 환산)
    raw_scores = (abnormal_feature_ratio * 70.0) + ((mean_deviation_intensity / 20.0) * 30.0)
    df['Risk_Score'] = np.clip(raw_scores, 0, 100)
    
    # 6. 위험 등급 분류
    df['Risk_Grade'] = pd.cut(
        df['Risk_Score'], 
        bins=[-1, 15.0, 50.0, 100.0], 
        labels=['Normal', 'Warning', 'Danger']
    )
    
    # 7. 결과 리포트 출력 (label2 대분류 기준 집계)
    print("\n--- [2025년 대분류(Label2) 공격 유형별 평균 리스크 점수 및 탐지 통계] ---")
    summary_report = df.groupby(label_col)['Risk_Score'].agg(['count', 'mean', 'max']).sort_values(by='mean', ascending=False)
    print(summary_report)
    
    print("\n--- [2025년 전체 위험 등급 분포] ---")
    print(df['Risk_Grade'].value_counts())
    
    # 대시보드 연동용 결과 저장
    output_dir = "../data/dashboard_ready"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "scored_network_flows_2025_label2.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\n>> 2025년 대시보드 연동용 데이터 저장 완료: {output_path}")

if __name__ == "__main__":
    build_transparent_risk_engine_2025()