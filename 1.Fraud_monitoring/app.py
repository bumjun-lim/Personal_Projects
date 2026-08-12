import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

# Day 2 모델 파이프라인 모듈에서 함수 불러오기
from prediction_model import get_model_predictions

st.set_page_config(page_title="FDS 머신러닝 이상 거래 감시 모니터링", page_icon="💳", layout="wide")

st.title("💳 FDS 머신러닝 이상 거래 탐지 대시보드")
st.markdown("Day 2(`day2_model.py`) **RandomForest 모델 예측 결과** 및 Day 1 **금액 구간 설정** 연동")
st.markdown("---")

# 1. Day 2 모델 결과 로딩 (Streamlit 캐싱 적용으로 속도 최적화)
@st.cache_data
def load_data_and_model():
    return get_model_predictions('creditcard.csv')

with st.spinner('`day2_model.py` 파이프라인을 실행하여 예측 및 FDS 성과를 분석 중입니다...'):
    test_df, y_test, y_pred = load_data_and_model()

# 2. Day 1 기준: 거래 금액 구간 분류 (4단계로 수정)
def categorize_amount(amt):
    if amt < 10: return '1. 소액 (<$10)'
    elif amt < 100: return '2. 중액 ($10~$100)'
    elif amt < 1000: return '3. 고액 ($100~$1000)'
    else: return '4. 초고액 (≥$1000)'

test_df['Amount_Category'] = test_df['Amount'].apply(categorize_amount)

# 3. 사이드바 (필터 옵션 - 4단계 금액 구간 반영)
st.sidebar.header("🔍 FDS 필터 옵션")

amount_options = [
    '1. 소액 (<$10)', 
    '2. 중액 ($10~$100)', 
    '3. 고액 ($100~$1000)', 
    '4. 초고액 (≥$1000)'
]

selected_cats = st.sidebar.multiselect(
    "금액 구간 선택",
    options=amount_options,
    default=amount_options
)

show_only_fraud = st.sidebar.checkbox(
    "🚨 이상/사기 관련 거래만 집중 모니터링 (정탐/오탐/미탐)", 
    value=True,
    help="정상 거래(TN)를 숨기고 모델이 탐지하거나 놓친 사기 관련 거래만 집중하여 보여줍니다."
)

# 데이터 필터링 적용
filtered_df = test_df[test_df['Amount_Category'].isin(selected_cats)]
if show_only_fraud:
    filtered_df = filtered_df[filtered_df['Detection_Status'] != '정상 정탐 (TN)']

# 4. 상단 핵심 지표 (Day 2 ML 성과 실시간 표출)
f1 = f1_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Day 2 모델 F1-Score", f"{f1:.4f}")
col2.metric("사기 적발 (정탐 TP)", f"{tp} 건", delta="적발 완료")
col3.metric("사기 미탐 (FN - 금융 손실)", f"{fn} 건", delta=f"-{fn}건 놓침", delta_color="inverse")
col4.metric("고객 불편 (오탐 FP)", f"{fp} 건", delta_color="inverse")

st.markdown("---")

# 5. 차트 시각화 (그래프 한글 깨짐 방지: 영문/기호 라벨링 적용)
c1, c2 = st.columns(2)

# 그래프 내부 범례용 영문 레이블 매핑
status_en_map = {
    '정탐 (TP)': 'TP (True Positive)',
    '오탐 (FP)': 'FP (False Positive)',
    '미탐 (FN)': 'FN (False Negative)',
    '정상 정탐 (TN)': 'TN (True Negative)'
}

# 금액 구간 영문 레이블 매핑 (4단계 수정)
category_en_map = {
    '1. 소액 (<$10)': '1. Small (<$10)',
    '2. 중액 ($10~$100)': '2. Medium ($10-$100)',
    '3. 고액 ($100~$1000)': '3. Large ($100-$1k)',
    '4. 초고액 (≥$1000)': '4. Very Large (≥$1k)'
}

# 시각화용 임시 컬럼 생성
filtered_df_plot = filtered_df.copy()
filtered_df_plot['Status_EN'] = filtered_df_plot['Detection_Status'].map(status_en_map)
filtered_df_plot['Category_EN'] = filtered_df_plot['Amount_Category'].map(category_en_map)

# 영문 키값 팔레트
palette_colors = {
    'TP (True Positive)': '#2ecc71', 
    'FP (False Positive)': '#f1c40f', 
    'FN (False Negative)': '#e74c3c', 
    'TN (True Negative)': '#95a5a6'
}

if len(filtered_df_plot) > 0:
    with c1:
        st.subheader("📊 금액 구간별 FDS 탐지 현황")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            data=filtered_df_plot, 
            x='Category_EN', 
            hue='Status_EN', 
            ax=ax, 
            palette=palette_colors
        )
        ax.set_yscale('log')
        ax.set_xlabel("Amount Category")
        ax.set_ylabel("Count (Log Scale)")
        plt.xticks(rotation=15)
        st.pyplot(fig)

    with c2:
        st.subheader("⏰ 시간대별 이상 거래 분포 (Amount vs Time)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=filtered_df_plot, 
            x='Time', 
            y='Amount', 
            hue='Status_EN',
            style='Status_EN',
            s=60,
            ax=ax, 
            palette=palette_colors
        )
        ax.set_yscale('log')
        ax.set_ylabel("Amount ($) [Log Scale]")
        ax.set_xlabel("Time (Seconds)")
        st.pyplot(fig)
else:
    st.warning("⚠️ 선택한 필터 조건에 해당하는 데이터가 없습니다. 필터를 변경해 주세요.")

# 5.2'구간별 요약 메트릭/표' 
st.markdown("##### 📌 금액 구간별 사기 거래 분포 요약 (선택 필터 기준)")
if len(filtered_df_plot) > 0:
    # 1) 선택된 금액 구간들에 대해 원본 Test Set(test_df) 기준 전체/사기 거래수 집계
    selected_categories = filtered_df_plot['Amount_Category'].unique()
    base_df = test_df[test_df['Amount_Category'].isin(selected_categories)]
    
    # 구간별 총 거래수, 실제 사기 건수(Class=1) 계산
    group_summary = base_df.groupby('Amount_Category').agg(
        전체거래수=('Class', 'count'),
        사기건수=('Class', 'sum')
    )
    
    # 해당 금액대 거래 중 사기 비율 (%) 계산 (1일차 fraud_rate_percentage)
    group_summary['구간 내 사기 비율 (%)'] = (
        group_summary['사기건수'] / group_summary['전체거래수'] * 100
    ).round(3).astype(str) + '%'

    # 2) 현재 대시보드 모니터링 현황(TP, FN, FP) 집계
    fds_status = filtered_df_plot.groupby(['Amount_Category', 'Detection_Status']).size().unstack(fill_value=0)
    for col in ['정탐 (TP)', '오탐 (FP)', '미탐 (FN)']:
        if col not in fds_status.columns:
            fds_status[col] = 0

    # 3) 두 집계 데이터 병합
    final_summary = group_summary.join(fds_status[['정탐 (TP)', '미탐 (FN)', '오탐 (FP)']])
    
    # 컬럼 이름 및 순서 정돈 (1일차 SQL 결과창과 동일한 흐름)
    final_summary.columns = ['구간 전체 거래(건)', '실제 사기(건)', '구간 내 사기 비율(%)', '정탐/TP(건)', '미탐/FN(건)', '오탐/FP(건)']
    
    st.dataframe(final_summary, width='stretch')
# 5-3. 필터링 조건별 실시간 모델 성과(F1-Score, Precision, Recall) 시각화
st.markdown("---")
st.subheader("🎯 선택 구간별 모델 성과 지표 (F1-Score / Precision / Recall)")

# 선택된 필터링 데이터 기준 성과 지표 계산
if len(filtered_df) > 0 and filtered_df['Class'].sum() > 0 and filtered_df['Pred_Class'].sum() > 0:
    sub_f1 = f1_score(filtered_df['Class'], filtered_df['Pred_Class'])
    sub_prec = precision_score(filtered_df['Class'], filtered_df['Pred_Class'])
    sub_rec = recall_score(filtered_df['Class'], filtered_df['Pred_Class'])
    
    # 1) 지표 비교 바 차트 시각화
    metrics_df = pd.DataFrame({
        'Metric': ['F1-Score', 'Precision', 'Recall'],
        'Score': [sub_f1, sub_prec, sub_rec]
    })
    
    m_col1, m_col2 = st.columns([2, 1])
    
    with m_col1:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        bars = sns.barplot(
            data=metrics_df, 
            x='Score', 
            y='Metric', 
            palette=['#3498db', '#2ecc71', '#e74c3c'], 
            ax=ax
        )
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Score (0.0 ~ 1.0)")
        ax.set_ylabel("")
        
        # 막대 끝에 점수 수치(%) 표시
        for bar in bars.patches:
            width = bar.get_width()
            ax.text(
                width + 0.02, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.4f} ({width*100:.1f}%)', 
                va='center', 
                fontsize=10, 
                fontweight='bold'
            )
        st.pyplot(fig)
        
    with m_col2:
        st.markdown("**💡 성과 해석 요약**")
        st.write(f"- 현재 선택 구간 **F1-Score**: `{sub_f1:.4f}`")
        st.write(f"- 실제 사기를 맞춘 비율(**Recall**): `{sub_rec*100:.1f}%`")
        st.write(f"- 탐지 알람 중 실제 사기 비율(**Precision**): `{sub_prec*100:.1f}%`")

elif len(filtered_df) > 0:
    st.info("ℹ️ 현재 선택된 필터 구간에는 실제 사기 거래(Class=1)가 없거나 탐지 건수가 없어 성과 지표(F1-Score)를 계산할 수 없습니다.")

# 6. FDS 탐지 결과 데이터 테이블
st.markdown("---")
st.subheader("📋 실시간 탐지 내역 상세 데이터 (Test Set 56,962건 기반)")
st.caption("💡 정상 거래를 포함한 전체 데이터(56,962건)를 조회하려면 왼쪽 사이드바의 **'🚨 이상/사기 관련 거래만 집중 모니터링'** 체크를 해제해 주세요.")
st.dataframe(
    filtered_df[['Time', 'Amount', 'Amount_Category', 'Class', 'Pred_Class', 'Detection_Status']], 
    use_container_width='stretch'
)



# 6-2. FDS 오탐 및 미탐(Detection Errors) 결과 데이터 테이블
st.markdown("---")
st.subheader("📋 실시간 모델 탐지 오류 내역 상세 데이터 (FP + FN)")
st.caption("※ 실제 정답과 모델 예측이 불일치한 데이터만 추출하여 FDS 모델 개선 포인트로 활용합니다.")
st.caption("💡 구간 별로 오탐지 내역을 볼 수 있습니다 왼쪽 **금액 구간 선택**으로 조정해 주세요")
model_failure = filtered_df[filtered_df['Class'] != filtered_df['Pred_Class']]

st.dataframe( 
    model_failure[['Time', 'Amount', 'Amount_Category', 'Class', 'Pred_Class', 'Detection_Status']], 
    use_container_width='stretch'
)