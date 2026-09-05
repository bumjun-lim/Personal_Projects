import pandas as pd
import numpy as np
import glob
import os

def build_transparent_risk_engine():
    path_pattern = "../data/processed_data/cic-ids-2017/*.parquet"
    files = glob.glob(path_pattern)
    
    if not files:
        print(f"경로를 확인해주세요: {path_pattern}")
        return

    print(f">> 총 {len(files)}개의 Parquet 파일 로드 및 통합 시작...")
    
    # 1. 전체 데이터 로드 및 결합 (파일별 샘플링을 통해 메모리 최적화 및 균등 분포 확보)
    df_list = []
    for f in files:
        df_chunk = pd.read_parquet(f)
        # 파일당 적절한 샘플링 (필요시 전체 데이터 연산으로 변경 가능)
        df_list.append(df_chunk.sample(n=min(len(df_chunk), 3000), random_state=42))
        
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()
    
    # 레이블 컬럼 동적 탐색
    label_col = [c for c in df.columns if 'label' in c.lower()][0]
    
    # 2. 숫자형 피처 분리 (타겟 및 이진 레이블 컬럼 철저한 제외)
    exclude_cols = [label_col, 'Binary_Label']
    feature_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c not in exclude_cols]
    
    # 데이터 정제 (무한대/결측치 처리)
    df_clean = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 3. [Baseline 수립] 오직 BENIGN(정상) 트래픽만을 기준으로 평균과 표준편차 추출
    benign_mask = df[label_col].str.strip().str.upper() == 'BENIGN'
    df_benign = df_clean[benign_mask]
    
    baseline_mean = df_benign.mean()
    baseline_std = df_benign.std().replace(0, 1.0) # 표준편차 0인 경우 분모 0 방지
    
    print(f">> 정상 기준선 수립 완료 (전체 세션: {len(df)}건 중 참조 정상 세션: {len(df_benign)}건)")
    print(f">> 활용되는 네트워크 통계 피처 수: {len(feature_cols)}개")
    
    # 4. [이탈도 산출] 전체 세션에 대한 Z-Score 계산 (정상 평균으로부터의 거리 측정)
    z_scores = np.abs((df_clean - baseline_mean) / baseline_std)
    
    # 5. [투명한 환산 공식 적용 (매직 넘버 제거)]
    # - 1축: 78개 전체 피처 중 3-Sigma(정상 범위를 아득히 벗어난 지점)를 초과한 피처의 '비율(Percentage)'
    extreme_deviations = (z_scores > 3.0).sum(axis=1)
    abnormal_feature_ratio = extreme_deviations / len(feature_cols) # 0.0 ~ 1.0 사이 값
    
    # - 2축: 전체 피처의 평균 이탈 강도 (Log 스케일링 혹은 클리핑으로 이상치 폭등 방어)
    mean_deviation_intensity = np.clip(z_scores.mean(axis=1), 0, 20.0)
    
    # 최종 리스크 스코어 수식 (비율 중심 70% + 강도 중심 30% 조합의 100점 만점 환산)
    # 현업 설명력: "전체 피처 중 비정상 이탈 비율이 높고, 평균 편차가 클수록 점수가 상승합니다."
    raw_scores = (abnormal_feature_ratio * 70.0) + ((mean_deviation_intensity / 20.0) * 30.0)
    df['Risk_Score'] = np.clip(raw_scores, 0, 100)
    
    # 6. 위험 등급 분류 (BI 대시보드 연동용)
    df['Risk_Grade'] = pd.cut(
        df['Risk_Score'], 
        bins=[-1, 15.0, 50.0, 100.0], 
        labels=['Normal', 'Warning', 'Danger']
    )
    
    # 7. 결과 리포트 출력
    print("\n--- [공격 유형별 평균 리스크 점수 및 탐지 통계] ---")
    summary_report = df.groupby(label_col)['Risk_Score'].agg(['count', 'mean', 'max']).sort_values(by='mean', ascending=False)
    print(summary_report)
    
    print("\n--- [전체 위험 등급 분포] ---")
    print(df['Risk_Grade'].value_counts())
    
    # (선택) 대시보드 시각화용 결과 저장
    output_dir = "../data/dashboard_ready"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "scored_network_flows.parquet")
    df.to_parquet(output_path, index=False)
    print(f"\n>> 대시보드 연동용 데이터 저장 완료: {output_path}")

if __name__ == "__main__":
    build_transparent_risk_engine()