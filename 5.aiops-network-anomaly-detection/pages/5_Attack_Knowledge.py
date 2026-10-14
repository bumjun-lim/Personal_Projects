import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Attack Knowledge Reference", 
    page_icon="📚", 
    layout="wide"
)

st.title("📚 Screen ⑤: Attack Knowledge (공격 유형별 보안 지식 레퍼런스)")
st.markdown(
    "주요 네트워크 공격 유형의 특징과 위험 요인, 관련 Feature 및 일반적인 대응 방향을 "
    "참고할 수 있는 보안 지식 레퍼런스입니다."
)

with st.expander("ℹ️ 페이지 이용 안내 (다른 메뉴와의 차이점)", expanded=False):
    st.markdown(
        """
        - **Screen 3 (Session Investigation)**: 특정 세션의 이상 원인을 분석합니다.
        - **Screen 4 (Response Guide)**: 분석된 원인을 바탕으로 실무 확인 및 대응 방법을 제시합니다.
        - **Screen 5 (Attack Knowledge)**: **[현재 페이지]** 특정 세션 분석 결과와 무관하게, 공격 유형 자체에 대한 일반적인 보안 지식을 찾아보는 독립적인 레퍼런스 사전입니다.
        """
    )

st.markdown("---")

df_knowledge = None

if 'df_attack_knowledge' in st.session_state and st.session_state['df_attack_knowledge'] is not None:
    df_knowledge = st.session_state['df_attack_knowledge']
elif 'df_knowledge' in st.session_state and st.session_state['df_knowledge'] is not None:
    df_knowledge = st.session_state['df_knowledge']
else:
    knowledge_path = "data/processed_data/Attack_Knowledge_Guide.csv"
    if os.path.exists(knowledge_path):
        try:
            df_knowledge = pd.read_csv(knowledge_path)
        except Exception as e:
            st.error(f"❌ 데이터 로드 중 오류가 발생했습니다: {e}")

if df_knowledge is None or df_knowledge.empty:
    st.error("⚠️ `Attack_Knowledge_Guide.csv` 데이터를 찾을 수 없거나 비어 있습니다. 앱 초기화 상태를 확인해 주세요.")
    st.stop()

df_knowledge.columns = [c.strip() for c in df_knowledge.columns]

required_cols = ['Attack_Type', 'Elevated_Features', 'Risk_Factor', 'Risk_Summary', 'Practical_Response', 'Additional_Check']
missing_cols = [col for col in required_cols if col not in df_knowledge.columns]
if missing_cols:
    st.error(f"❌ 데이터프레임에 필수 컬럼이 누락되었습니다: {missing_cols}")
    st.stop()

st.subheader("🔍 공격 유형 선택 (Dictionary Index)")
st.markdown("탐색하고자 하는 공격 유형을 선택하세요. (현재 분석 중인 세션과 무관하게 항상 동일한 지식 사전을 제공합니다)")

attack_types = df_knowledge['Attack_Type'].dropna().unique().tolist()

selected_attack = st.selectbox(
    "조회할 공격 유형을 선택하세요:",
    options=attack_types,
    index=0,
    label_visibility="collapsed"
)

st.markdown("##### 📌 주요 공격 유형 바로가기")

if 'selected_knowledge_attack' not in st.session_state:
    st.session_state['selected_knowledge_attack'] = attack_types[0] if attack_types else ""

btn_cols = st.columns(min(len(attack_types), 4))
for i, atk in enumerate(attack_types):
    col_idx = i % len(btn_cols)
    with btn_cols[col_idx]:
        if st.button(f"🛡️ {atk}", use_container_width=True):
            st.session_state['selected_knowledge_attack'] = atk
            selected_attack = atk
            st.rerun()

if selected_attack != st.session_state.get('selected_knowledge_attack'):
    selected_attack = st.session_state.get('selected_knowledge_attack', attack_types[0])

st.markdown("---")

row_data = df_knowledge[df_knowledge['Attack_Type'] == selected_attack]

if row_data.empty:
    st.warning(f"⚠️ '{selected_attack}'에 해당하는 지식 정보가 존재하지 않습니다.")
    st.stop()

info = row_data.iloc[0]

attack_type = info['Attack_Type']
risk_factor = info['Risk_Factor']
risk_summary = info['Risk_Summary']
elevated_features_raw = str(info['Elevated_Features'])
practical_response = info['Practical_Response']
additional_check = info['Additional_Check']

st.markdown(f"## 🔥 {attack_type}")
st.markdown("---")

col_l, col_r = st.columns([1.2, 0.8])

with col_l:
    st.markdown("### 📌 [위험 요인 (Risk Factor)]")
    st.info(f"{risk_factor}")
    
    st.markdown("### 📝 [위험 특성 요약 (Risk Summary)]")
    st.markdown(f"> {risk_summary}")

with col_r:
    st.markdown("### 🧬 [주요 관련 Feature]")
    st.caption("해당 공격 유형과 관련하여 관찰할 수 있는 대표적인 네트워크 특성입니다.")
    
    if ',' in elevated_features_raw:
        features_list = [f.strip() for f in elevated_features_raw.split(',') if f.strip()]
    else:
        features_list = [f.strip() for f in elevated_features_raw.splitlines() if f.strip()]
        
    if features_list:
        feature_tags_html = "".join([f"<span style='background-color: #262730; color: #ffffff; padding: 4px 8px; border-radius: 4px; margin: 3px; display: inline-block; font-family: monospace; font-size: 0.9em; border: 1px solid #4f4f4f;'>{feat}</span>" for feat in features_list])
        st.markdown(feature_tags_html, unsafe_allow_html=True)
    else:
        st.code(elevated_features_raw)
        
    st.caption("ℹ️ *해당 Feature는 공격 유형과 관련하여 관찰할 수 있는 대표적인 네트워크 특성이며, Feature만으로 공격 유형을 직접 판정하는 것은 아닙니다.*")

st.markdown("---")

st.markdown("### 🛡️ 실무 대응 및 검토 지침")

col_resp, col_check = st.columns(2)

with col_resp:
    st.markdown(
        """
        <div style="background-color: #1e2530; padding: 20px; border-radius: 8px; border-left: 5px solid #2e7d32; height: 100%;">
            <h4 style="color: #4caf50; margin-top: 0;">🛡️ Practical Response (일반적인 대응)</h4>
            <hr style="border-color: #333;">
            <p style="white-space: pre-line; line-height: 1.6;">{}</p>
        </div>
        """.format(practical_response),
        unsafe_allow_html=True
    )

with col_check:
    st.markdown(
        """
        <div style="background-color: #1e2530; padding: 20px; border-radius: 8px; border-left: 5px solid #0288d1; height: 100%;">
            <h4 style="color: #29b6f6; margin-top: 0;">🔎 Additional Check (추가 확인 사항)</h4>
            <hr style="border-color: #333;">
            <p style="white-space: pre-line; line-height: 1.6;">{}</p>
        </div>
        """.format(additional_check),
        unsafe_allow_html=True
    )

st.markdown("<br><br>", unsafe_allow_html=True)