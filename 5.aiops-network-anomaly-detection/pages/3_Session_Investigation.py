import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Session Investigation & RCA", 
    page_icon="🎯", 
    layout="wide"
)

st.title("🎯 Screen ③: Session Investigation & RCA")
st.markdown("개별 네트워크 세션의 이상 징후를 추적하고, 원인 영역(Risk Group)과 핵심 피처를 분석합니다.")
st.markdown("---")

# app.py 등 메인 진입점에서 공유된 master 데이터 가져오기 (없을 경우에만 백업 로드)
if 'df_master' in st.session_state and st.session_state['df_master'] is not None:
    df_master = st.session_state['df_master']
else:
    # 혹시 단독 실행될 경우를 대비한 안전 장치
    data_dir = "data/processed_data"
    master_path = os.path.join(data_dir, "master_scored_dataset.parquet")
    df_master = pd.read_parquet(master_path) if os.path.exists(master_path) else None

if df_master is None:
    st.error("⚠️ `master_scored_dataset.parquet` 파일을 찾을 수 없습니다. 메인 페이지(app.py)를 먼저 실행하거나 `data/processed_data/` 경로를 확인해주세요.")
else:
    # 1. 상단 사이드바 필터 (공간 최소화)
    st.sidebar.header("🔍 세션 필터 설정")
    
    years = sorted(df_master['Year'].unique()) if 'Year' in df_master.columns else []
    selected_years = st.sidebar.multiselect("연도 (Year)", years, default=years)
    
    attack_groups = sorted(df_master['Attack_Group'].unique()) if 'Attack_Group' in df_master.columns else []
    selected_attacks = st.sidebar.multiselect("공격 그룹 (Attack Group)", attack_groups, default=attack_groups)
    
    pred_labels = sorted(df_master['Pred_Label'].unique()) if 'Pred_Label' in df_master.columns else []
    selected_preds = st.sidebar.multiselect("예측 레이블 (Pred Label)", pred_labels, default=pred_labels)
    
    sort_column = st.sidebar.selectbox("정렬 기준", options=['Risk_Deviation', 'Risk_Score'], index=0)
    
    min_val = float(df_master[sort_column].min()) if sort_column in df_master.columns else 0.0
    max_val = float(df_master[sort_column].max()) if sort_column in df_master.columns else 100.0
    threshold = st.sidebar.slider(f"최소 {sort_column}", min_value=min_val, max_value=max_val, value=min_val)

    # 필터 적용
    filtered_df = df_master.copy()
    if selected_years and 'Year' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
    if selected_attacks and 'Attack_Group' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Attack_Group'].isin(selected_attacks)]
    if selected_preds and 'Pred_Label' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Pred_Label'].isin(selected_preds)]
    if sort_column in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[sort_column] >= threshold]
        filtered_df = filtered_df.sort_values(by=sort_column, ascending=False)

    total_matched = len(filtered_df)

    if total_matched > 0:
        # 2. 행 선택형 테이블 (드롭다운 대신 표에서 직접 클릭하여 선택)
        st.subheader("📋 고위험 세션 목록 (Target Session Selection)")
        st.caption(f"조건에 일치하는 총 {total_matched:,}개 세션 중 상위 고위험 세션 목록입니다. **표에서 조사할 행을 클릭**하세요.")

        display_df = filtered_df.head(100).copy()
        display_df_reset = display_df.reset_index(names="Session_Index")

        list_cols = ['Session_Index', 'Year', 'Attack_Group', 'Risk_Score', 'Risk_Deviation', 'Pred_Label', 'Top_Root_Cause_Group']
        existing_list_cols = [col for col in list_cols if col in display_df_reset.columns]
        
        # 스트림릿 네이티브 행 선택 기능 적용
        event = st.dataframe(
            display_df_reset[existing_list_cols],
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="session_selection_table"
        )

        st.markdown("---")

        # 사용자가 테이블에서 행을 클릭했는지 확인
        selected_rows = event.selection.rows if hasattr(event, "selection") else []

        if selected_rows:
            clicked_row_position = selected_rows[0]
            selected_idx = display_df_reset.loc[clicked_row_position, "Session_Index"]
            row = filtered_df.loc[selected_idx]
            
            # 3. 선택 세션 핵심 요약 카드 (4분할)
            st.subheader(f"📌 Selected Session Summary (Index: {selected_idx})")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Risk Score", f"{row.get('Risk_Score', 0):.2f}")
            with c2:
                st.metric("Risk Deviation", f"{row.get('Risk_Deviation', 0):.2f}")
            with c3:
                st.metric("Pred Label / Grade", f"{row.get('Pred_Label', 'N/A')}")
            with c4:
                st.metric("Top Root Cause Group", f"{row.get('Top_Root_Cause_Group', 'N/A')}")

            st.markdown("---")

            # 4. RCA 핵심 영역: 4개 Risk Group 비교 (시각적 바 형태 표현)
            st.subheader("🔍 Root Cause Analysis — 왜 이 세션이 위험한가?")
            st.markdown("4대 리스크 그룹별 스코어 비교를 통해 위험이 발생한 주원인 영역을 파악합니다.")

            group_scores = {
                "Topology": float(row.get('Topology_Score', 0)),
                "Control & State": float(row.get('Control_State_Score', 0)),
                "Volume": float(row.get('Volume_Score', 0)),
                "Size & Payload": float(row.get('Size_Payload_Score', 0))
            }
            
            sorted_groups = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
            max_score_val = max(group_scores.values()) if max(group_scores.values()) > 0 else 1.0

            for grp_name, sc_val in sorted_groups:
                bar_length = int((sc_val / max_score_val) * 60) if max_score_val > 0 else 0
                bar_str = "█" * bar_length
                
                if grp_name == row.get('Top_Root_Cause_Group', ''):
                    st.markdown(f"**`{grp_name:<16}`** `{bar_str}` **{sc_val:.2f}** 👈 *(Primary Root Cause)*")
                else:
                    st.markdown(f"`{grp_name:<16}` `{bar_str}` {sc_val:.2f}")

            st.markdown("")

            # 5. Top 3 Root Cause Feature 순위 형태 출력
            st.subheader("🎯 Top 3 Root Cause Features")
            
            f1 = row.get('Top1_Root_Cause_Feature', 'N/A')
            s1 = row.get('Top1_Feature_Score', 0)
            f2 = row.get('Top2_Root_Cause_Feature', 'N/A')
            s2 = row.get('Top2_Feature_Score', 0)
            f3 = row.get('Top3_Root_Cause_Feature', 'N/A')
            s3 = row.get('Top3_Feature_Score', 0)

            st.markdown(f"**① {f1}** : `Score {s1:.2f}`")
            st.markdown(f"**② {f2}** : `Score {s2:.2f}`")
            st.markdown(f"**③ {f3}** : `Score {s3:.2f}`")

            st.markdown("---")

            # 7. 전체 데이터프레임 형식으로 원본 속성 확인 (수정 반영됨)
            with st.expander("📂 전체 데이터프레임 및 원본 속성 보기"):
                st.caption("선택된 세션의 전체 피처 및 스코어 데이터 전체 목록입니다.")
                cols = [c for c in row.index]
                if cols:
                    df_row = pd.DataFrame(row[cols]).T
                    st.dataframe(df_row, use_container_width=True)
                else:
                    st.write(row)

            # 6 & 실무 연동 버튼: 다음 페이지(Response Guide)로 연결
            st.markdown("---")
            col_b1, col_b2 = st.columns([3, 1])
            with col_b1:
                st.info("💡 분석된 원인 피처와 공격 그룹을 바탕으로 즉각적인 실무 조치 방안을 확인하실 수 있습니다.")
            with col_b2:
                if st.button("🛠️ 대응 가이드 확인하기", type="primary", use_container_width=True):
                    # 4번 페이지가 읽어갈 수 있도록 전체 row 데이터를 session_state에 저장합니다!
                    st.session_state['selected_session_row'] = row
                    try:
                        st.switch_page("pages/4_Response_Guide.py")
                    except Exception:
                        st.success("✅ 대응 가이드로 연동되었습니다! 좌측 메뉴에서 '4_Response_Guide'를 선택해 주세요.")
        else:
            st.info("👆 위의 고위험 세션 목록 테이블에서 **조사할 행을 클릭**해주세요.")
    else:
        st.warning("⚠️ 선택한 조건에 부합하는 세션 데이터가 없습니다. 사이드바의 필터 조건을 조정해 주세요.")