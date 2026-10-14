import streamlit as st
import pandas as pd
import os

# 1. 페이지 기본 설정 (와이드 모드)
st.set_page_config(
    page_title="AIOps Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

# 2. 공통 데이터 일괄 로드 함수 (캐싱 적용)
@st.cache_data
def load_all_data():
    data_dir = "data/processed_data"
    
    master_path = os.path.join(data_dir, "master_scored_dataset.parquet")
    mapping_path = os.path.join(data_dir, "feature_standard_mapping.csv")
    feature_guide_path = os.path.join(data_dir, "feature_response_guide.csv")
    attack_guide_path = os.path.join(data_dir, "attack_response_guide.csv")
    
    df_master = pd.read_parquet(master_path) if os.path.exists(master_path) else None
    df_mapping = pd.read_csv(mapping_path) if os.path.exists(mapping_path) else None
    df_feature_guide = pd.read_csv(feature_guide_path) if os.path.exists(feature_guide_path) else None
    df_attack_knowledge = pd.read_csv(attack_guide_path) if os.path.exists(attack_guide_path) else None
    
    return df_master, df_mapping, df_feature_guide, df_attack_knowledge

# 데이터 로드 수행
df_master, df_mapping, df_feature_guide, df_attack_knowledge = load_all_data()

# session_state에 공통 데이터 저장 (다른 페이지에서 공용으로 활용)
st.session_state['df_master'] = df_master
st.session_state['df_mapping'] = df_mapping
st.session_state['df_feature_guide'] = df_feature_guide
st.session_state['df_attack_knowledge'] = df_attack_knowledge

# 3. 프로젝트 진입 화면 (Landing Page)
st.title("🛡️ 네트워크 이상 탐지 및 실무 대응 분석 플랫폼")
st.markdown("AIOps 기반 트래픽 분석 결과와 현업 보안 대응 가이드를 연동하여 직관적인 원인 분석(RCA)을 지원합니다.")
st.markdown("---")

# 4. 프로젝트 한 줄 소개
st.markdown("### 💡 플랫폼 개요")
st.markdown("""
네트워크 Flow 데이터를 기반으로 개별 세션의 위험도를 산출하고, 위험도가 높은 세션의 원인을 분석하여 실무 대응 가이드까지 연결하는 네트워크 이상 탐지 및 의사결정 지원 플랫폼입니다.
""")

st.markdown("<br>", unsafe_allow_html=True)

# 5. 프로젝트 분석 흐름 소개
st.markdown("### 🔄 분석 프로세스 흐름")
col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)

with col_f1:
    st.info("**1. Network Flow**\n\n원천 트래픽 및 세션 데이터 수집")
with col_f2:
    st.info("**2. Risk Scoring**\n\n위험도 및 이탈도(Deviation) 산출")
with col_f3:
    st.info("**3. Risk Overview**\n\n전체 위험도 및 그룹별 현황 파악")
with col_f4:
    st.info("**4. Session (RCA)**\n\n개별 세션 원인 및 Root Cause 분석")
with col_f5:
    st.info("**5. Response Guide**\n\n실무 보안 대응 조치 가이드 연동")

st.markdown("<br>", unsafe_allow_html=True)

# 6. 데이터 로드 상태 확인
st.markdown("### 📂 Project Data Status")
status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.metric("Master Dataset", "✅ 정상 로드" if df_master is not None else "❌ 로드 실패")
with status_col2:
    st.metric("Feature Mapping", "✅ 정상 로드" if df_mapping is not None else "❌ 로드 실패")
with status_col3:
    st.metric("Feature Guide", "✅ 정상 로드" if df_feature_guide is not None else "❌ 로드 실패")
with status_col4:
    st.metric("Attack Knowledge", "✅ 정상 로드" if df_attack_knowledge is not None else "❌ 로드 실패")

if df_master is None:
    st.error("⚠️ 핵심 데이터셋(`master_scored_dataset.parquet`)을 불러오지 못했습니다. `data/processed_data/` 경로를 확인해 주세요.")

st.markdown("---")

# 7. 상세 분석 페이지 안내
st.markdown("### 🚀 상세 분석 시작하기")
st.success("👇 상단의 사이드바(페이지 메뉴)에서 **Risk Overview**부터 상세 분석을 시작할 수 있습니다.")

st.markdown("""
- **1_Risk_Overview**: 전체 위험도 현황 및 데이터 분석 배경, 연도별 구성 비교
- **2_Risk_Analysis**: Risk Score와 Risk Deviation 산출 방법 및 분석 과정
- **3_Session_Investigation**: 개별 세션의 위험도 및 Root Cause 분석 (RCA)
- **4_Response_Guide**: RCA 결과를 기반으로 한 실무 대응 가이드 확인
- **5_Attack_Knowledge**: 공격 유형별 특징과 대응 방법을 확인하는 참고 레퍼런스
""")

# 사이드바 안내
st.sidebar.success("👈 상단의 페이지 메뉴를 선택하여 이동하세요.")
st.sidebar.markdown("---")
st.sidebar.info("📌 **System Status**: 공통 데이터 로드 완료")