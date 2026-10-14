import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 깨짐 방지 설정 (시스템 환경에 따라 선택)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 1. 페이지 기본 설정 (와이드 모드)
st.set_page_config(
    page_title="AIOps Network Anomaly Detection - Risk Overview",
    page_icon="🛡️",
    layout="wide"
)

# 2. session_state에서 데이터 가져오기 (app.py에서 로드된 데이터 활용)
df_master = st.session_state.get('df_master', None)
df_mapping = st.session_state.get('df_mapping', None)
df_feature_guide = st.session_state.get('df_feature_guide', None)
df_attack_knowledge = st.session_state.get('df_attack_knowledge', None)

if df_master is None or df_master.empty:
    st.error("⚠️ 마스터 데이터(`df_master`)가 존재하지 않습니다. 메인 페이지(`app.py`)를 먼저 실행하여 데이터를 로드해 주세요.")
    st.stop()

# 3.메인 타이틀 
st.title("🛡️ Screen ①: Risk Overview (전체 위험도 현황)")
st.markdown("전체 네트워크 트래픽 세션의 위험도 현황과 연도별·공격 그룹별 위험 이탈 분포를 관리자 관점에서 확인합니다.")
st.markdown("---")

# 4. Dataset Overview (데이터 개괄)
st.subheader("📊 Dataset Overview (데이터 개괄)")

total_rows = len(df_master)
total_features = len(df_master.columns)
years = sorted(df_master['Year'].unique()) if 'Year' in df_master.columns else [2017, 2025]
attack_types = df_master['Attack_Group'].nunique() if 'Attack_Group' in df_master.columns else 5

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("총 분석 세션(로그) 건수", f"{total_rows:,}")
with col2:
    st.metric("활용 피처(Feature) 수", f"{total_features}개")
with col3:
    st.metric("분류된 공격 유형 수", f"{attack_types}개 유형")
with col4:
    st.metric("분석 대상 연도", f"{len(years)}개년 ({min(years)}, {max(years)})")

st.markdown("---")

# 5. 연도별 공격 데이터 구성 현황
st.subheader("🔍 연도별 공격 데이터 구성 현황")
st.markdown("연도별 데이터 환경에서 탐지된 공격 그룹과 세션 분포를 비교하여 두 데이터셋의 구성 차이를 확인합니다.")

if 'Year' in df_master.columns and 'Attack_Group' in df_master.columns:
    available_years = sorted(df_master['Year'].unique())
    year_cols = st.columns(len(available_years))
    
    for idx, yr in enumerate(available_years):
        with year_cols[idx]:
            st.markdown(f"#### 📅 연도: {yr}")
            yr_data = df_master[df_master['Year'] == yr]['Attack_Group'].value_counts().reset_index()
            yr_data.columns = ['Attack Group', 'Session Count']
            st.dataframe(yr_data, use_container_width=True, hide_index=True)

st.markdown("---")

# 6. 분석 방법론 및 데이터 특성 정의
st.subheader("💡 분석 방법론 및 데이터 특성 정의")

st.markdown("""
> **"동일한 Feature라도 데이터 환경에 따라 분포와 이상치 특성이 다르게 나타나므로, 연도별 Normal 기준선을 적용하여 위험도를 산출하였습니다."**
""")

st.markdown("""
본 프로젝트에서는 공격 상황이 포함된 공개 네트워크 Flow 데이터를 기반으로 개별 세션의 위험도를 산출하였습니다. 
그러나 실제 보안 환경에서는 동일한 세션의 위험도뿐만 아니라 시간에 따른 위험도 변화와 반복적인 이상 패턴을 함께 고려할 필요가 있습니다.
""")

st.markdown("---")

# 7. Risk Summary (위험도 현황)
st.subheader("📊 Risk Summary")
st.markdown("전체 분석 세션 중 시스템의 위험 등급 판단 결과(`Pred_Label`)별 세션 분포 및 비율 현황입니다.")

if 'Pred_Label' in df_master.columns:
    pred_counts = df_master['Pred_Label'].value_counts()
    normal_cnt = pred_counts.get('Normal', 0)
    caution_cnt = pred_counts.get('Caution', 0)
    danger_cnt = pred_counts.get('Danger', 0)
else:
    normal_cnt, caution_cnt, danger_cnt = 0, 0, 0

normal_ratio = (normal_cnt / total_rows * 100) if total_rows > 0 else 0
caution_ratio = (caution_cnt / total_rows * 100) if total_rows > 0 else 0
danger_ratio = (danger_cnt / total_rows * 100) if total_rows > 0 else 0

r_col1, r_col2, r_col3 = st.columns(3)

with r_col1:
    st.metric(label="Normal (정상)", value=f"{normal_cnt:,}개", delta=f"전체의 {normal_ratio:.1f}%", delta_color="normal")
with r_col2:
    st.metric(label="Caution (주의)", value=f"{caution_cnt:,}개", delta=f"전체의 {caution_ratio:.1f}%", delta_color="off")
with r_col3:
    st.metric(label="Danger (위험)", value=f"{danger_cnt:,}개", delta=f"전체의 {danger_ratio:.1f}%", delta_color="inverse")

st.markdown("---")

# 8. Risk Deviation Boxplot (기존 세이본 박스플롯 유지 - 샘플링 없음)
st.subheader("📈 연도별 & 공격 그룹별 Risk Deviation 분포 비교")
st.markdown("💡 연도별 Normal 기준선에 따른 공격 그룹의 Risk Deviation 분포를 비교하여 데이터 환경별 위험도 차이를 확인합니다.")

if all(col in df_master.columns for col in ['Attack_Group', 'Risk_Deviation', 'Year']):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    sns.boxplot(
        data=df_master, 
        x="Attack_Group", 
        y="Risk_Deviation", 
        hue="Year",
        palette="Set2",
        ax=ax,
        showfliers=False
    )
    
    ax.set_title("Comparison of Risk Deviation Distribution by Year & Attack Group", fontsize=14, fontweight='bold')
    ax.set_xlabel("Attack Group", fontsize=12)
    ax.set_ylabel("Risk Deviation (vs Normal)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.legend(title="Year")
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close(fig)

st.markdown(
    """
    > **💡 Risk Deviation 해석 안내**:
    > Risk Deviation은 각 연도의 Normal 환경을 기준으로 현재 세션의 Risk Score가 얼마나 이탈했는지를 나타내는 상대적 위험도 지표입니다.
    > 따라서 서로 다른 데이터 환경을 가진 2017년과 2025년의 위험도를 비교할 때 절대 Risk Score보다 기준선 대비 Risk Deviation을 함께 확인할 수 있습니다.
    """
)

st.markdown("---")

# 9. Attack Group Risk Profile (그룹별 평균 Risk Deviation 막대 그래프)
st.subheader("🔍 Attack Group Risk Profile")
st.markdown("공격 그룹별 평균 Risk Deviation을 비교하여 상대적으로 높은 위험 이탈을 보이는 공격 그룹을 확인합니다.")

if all(col in df_master.columns for col in ['Year', 'Attack_Group', 'Risk_Deviation']):
    group_profile = (
        df_master
        .groupby(["Year", "Attack_Group"])["Risk_Deviation"]
        .mean()
        .reset_index()
    )
    
    fig_prof, ax_prof = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=group_profile,
        x="Attack_Group",
        y="Risk_Deviation",
        hue="Year",
        palette="muted",
        ax=ax_prof
    )
    
    ax_prof.set_title("Average Risk Deviation by Attack Group & Year", fontsize=13, fontweight='bold')
    ax_prof.set_xlabel("Attack Group", fontsize=11)
    ax_prof.set_ylabel("Mean Risk Deviation", fontsize=11)
    plt.xticks(rotation=35, ha='right')
    ax_prof.legend(title="Year", loc="upper right")
    plt.tight_layout()
    
    st.pyplot(fig_prof)
    plt.close(fig_prof)

st.markdown("---")

# 10. Yearly Risk Comparison (연도별 전체 평균 Risk Deviation 비교)
st.subheader("📅 Yearly Risk Comparison")
st.markdown("2017년과 2025년 전체 데이터의 연도별 평균 Risk Deviation을 비교하여 거시적인 상대 위험도 수준을 확인합니다.")

if 'Year' in df_master.columns and 'Risk_Deviation' in df_master.columns:
    yearly_risk = (
        df_master
        .groupby("Year")["Risk_Deviation"]
        .mean()
        .reset_index()
    )
    
    c_left, c_right = st.columns([1, 1])
    
    with c_left:
        fig_yr, ax_yr = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=yearly_risk,
            x="Year",
            y="Risk_Deviation",
            palette="Set1",
            ax=ax_yr
        )
        ax_yr.set_title("Overall Mean Risk Deviation by Year", fontsize=11, fontweight='bold')
        ax_yr.set_xlabel("Year", fontsize=10)
        ax_yr.set_ylabel("Mean Risk Deviation", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_yr)
        plt.close(fig_yr)
        
    with c_right:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background-color: #1e2530; padding: 20px; border-radius: 8px; border-left: 5px solid #3498db; height: 100%;">
                <h4 style="color: #3498db; margin-top: 0;">📌 연도별 위험도 비교 요약</h4>
                <p style="line-height: 1.6; margin-bottom: 0;">
                    2017년과 2025년 네트워크 트래픽은 수집 환경 및 프로토콜 구성에 차이가 존재합니다. 
                    연도별 전체 평균 Risk Deviation 비교를 통해 각 연도별 데이터셋이 지닌 거시적인 위험도 편차 수준을 가시적으로 확인할 수 있습니다.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br><br>", unsafe_allow_html=True)