import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Response & Mitigation Guide", 
    page_icon="🛠️", 
    layout="wide"
)

st.title("🛠️ Screen ④: Response & Mitigation Guide")
st.markdown("선택된 고위험 세션의 원인 분석 결과를 바탕으로, 표준 피처 매핑과 실무 대응 가이드를 제공합니다.")
st.markdown("---")

# 1. 세션 상태 확인
if 'selected_session_row' not in st.session_state or st.session_state['selected_session_row'] is None:
    st.warning("⚠️ 선택된 세션 정보가 없습니다. 먼저 **3_Session_Investigation** 페이지에서 분석할 세션을 선택해 주세요.")
    if st.button("👈 3번 페이지로 이동하기"):
        try:
            st.switch_page("pages/3_Session_Investigation.py")
        except Exception:
            st.info("좌측 메뉴에서 `3_Session_Investigation` 페이지를 선택해 주세요.")
    st.stop()

row = st.session_state['selected_session_row']

# 2. 데이터프레임 로드 및 확인
df_mapping = st.session_state.get('df_mapping', None)
df_feature_guide = st.session_state.get('df_feature_guide', None)

if df_mapping is None and os.path.exists("data/processed_data/feature_standard_mapping.csv"):
    df_mapping = pd.read_csv("data/processed_data/feature_standard_mapping.csv")
        
if df_feature_guide is None and os.path.exists("data/processed_data/feature_response_guide.csv"):
    df_feature_guide = pd.read_csv("data/processed_data/feature_response_guide.csv")

# 세션 컨텍스트 정보 추출
session_year = str(row.get('Year', '')).strip()
attack_group = row.get('Attack_Group', 'N/A')

st.subheader(f"📌 현재 분석 중인 세션 컨텍스트 (Attack Group: {attack_group} | Year: {session_year})")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Year", session_year if session_year else 'N/A')
with c2:
    st.metric("Attack Group", attack_group)
with c3:
    st.metric("Risk Score", f"{row.get('Risk_Score', 0):.2f}")
with c4:
    st.metric("Risk Deviation", f"{row.get('Risk_Deviation', 0):.2f}")
with c5:
    st.metric("Pred Label", row.get('Pred_Label', 'N/A'))
with c6:
    st.metric("Top Root Group", row.get('Top_Root_Cause_Group', 'N/A'))

st.markdown("---")
st.subheader("🔍 Root Cause Feature별 표준 매핑 및 실무 대응 가이드")

# 분석 대상 Top 3 피처 정의
top_features = [
    (row.get('Top1_Root_Cause_Feature', 'N/A'), row.get('Top1_Feature_Score', 0), "Top 1 (핵심 주원인)"),
    (row.get('Top2_Root_Cause_Feature', 'N/A'), row.get('Top2_Feature_Score', 0), "Top 2 (보조 원인)"),
    (row.get('Top3_Root_Cause_Feature', 'N/A'), row.get('Top3_Feature_Score', 0), "Top 3 (보조 원인)")
]

for idx, (feat_name, feat_score, rank_label) in enumerate(top_features, 1):
    if feat_name == 'N/A' or not feat_name:
        continue
        
    # =========================================================================
    # [비교 알고리즘 블록] 
    # 1. 세션의 피처명과 연도를 매핑 데이터프레임과 비교하기 위해 정제
    # 2. 1순위: (연도 + 피처명) 동시 일치 검색
    # 3. 2순위: 피처명 단독 일치 검색 (띄어쓰기/언더바 및 대소문자 유연 대응)
    # =========================================================================
    standard_id = "N/A"
    standard_name = "N/A"
    risk_group = "N/A"
    
    if df_mapping is not None and not df_mapping.empty:
        # 검색어 정제 (공백 제거, 소문자 변환, 언더바/띄어쓰기 표준화)
        clean_feat = str(feat_name).strip().lower().replace("_", " ")
        clean_year = session_year
        
        # 매핑용 임시 컬럼 생성 (원본 손상 방지)
        df_map_temp = df_mapping.copy()
        df_map_temp['_clean_orig'] = df_map_temp['Original_Feature'].astype(str).str.strip().str.lower().str.replace("_", " ")
        df_map_temp['_clean_year'] = df_map_temp['Year'].astype(str).str.strip() if 'Year' in df_map_temp.columns else ""
        
        # [비교 1] 연도와 피처명이 모두 일치하는 행 필터링
        matched_df = df_map_temp[
            (df_map_temp['_clean_orig'] == clean_feat) & 
            (df_map_temp['_clean_year'] == clean_year)
        ]
        
        # [비교 2] 만약 연도 매칭 결과가 없으면 피처명만으로 재검색
        if matched_df.empty:
            matched_df = df_map_temp[df_map_temp['_clean_orig'] == clean_feat]
            
        # 매칭 성공 시 정보 추출
        if not matched_df.empty:
            matched_row = matched_df.iloc[0]
            standard_id = str(matched_row.get('Standard_Feature_ID', 'N/A')).strip()
            standard_name = str(matched_row.get('Standard_Feature_Name', 'N/A')).strip()
            risk_group = str(matched_row.get('Risk_Group', 'N/A')).strip()

    # =========================================================================
    # [가이드 연동 비교 블록]
    # 1. 위에서 획득한 standard_id를 가이드 데이터프레임과 비교
    # =========================================================================
    guide_row = None
    if df_feature_guide is not None and not df_feature_guide.empty and standard_id != 'N/A':
        df_guide_temp = df_feature_guide.copy()
        df_guide_temp['_clean_guide_id'] = df_guide_temp['Standard_Feature_ID'].astype(str).str.strip()
        
        matched_guide = df_guide_temp[df_guide_temp['_clean_guide_id'] == standard_id]
        if not matched_guide.empty:
            guide_row = matched_guide.iloc[0]

    # =========================================================================
    # [출력 렌더링 블록] 비교가 모두 끝난 최종 결과값들을 화면에 출력
    # =========================================================================
    with st.container():
        if idx == 1:
            st.markdown(f"### 🔥 **{rank_label}: {feat_name}**")
        else:
            st.markdown(f"### 🔸 **{rank_label}: {feat_name}**")
            
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"- **Feature Score**: `{feat_score:.2f}`")
        with col_m2:
            st.markdown(f"- **Standard ID**: `{standard_id}`")
        with col_m3:
            st.markdown(f"- **Risk Group**: `{risk_group}`")
            
        st.markdown(f"- **Standard Name**: `{standard_name}`")

        # 가이드 내용이 존재할 경우 카드 형태로 출력
        if guide_row is not None:
            risk_summary = guide_row.get('Risk_Summary', '등록된 위험 의미 설명이 없습니다.')
            assoc_attack = guide_row.get('Associated_Attack', '관련 공격 정보 없음')
            investigation = guide_row.get('Investigation_Point', '조사 포인트 정보 없음')
            response = guide_row.get('Response_Guide', '실무 대응 가이드 정보 없음')
            
            st.markdown(f"> **[위험 의미 (Risk Summary)]**\n> {risk_summary}")
            
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.info(f"**🔍 확인할 사항 (Investigation Point)**\n\n{investigation}")
            with col_g2:
                st.success(f"**🛡️ 실무 대응 방법 (Response Guide)**\n\n{response}")
                
            # 연관 공격 그룹 및 주의 안내 문구 추가
            st.markdown(f"📌 **연관 공격 그룹 (Associated Attack)**: `{assoc_attack}`")
            st.caption("ℹ️ *연관 공격 그룹은 해당 Feature 이상 시 함께 검토할 공격 유형이며, 공격 유형을 직접 판정한 결과가 아닙니다.*")
        else:
            st.warning(f"⚠️ Standard ID (`{standard_id}`)에 해당하는 상세 가이드가 `feature_response_guide.csv`에 등록되어 있지 않습니다.")
            
        st.markdown("---")

if st.button("🔄 다른 세션 조사하러 가기", use_container_width=True):
    try:
        st.switch_page("pages/3_Session_Investigation.py")
    except Exception:
        st.success("좌측 메뉴에서 `3_Session_Investigation` 페이지를 선택해 주세요.")