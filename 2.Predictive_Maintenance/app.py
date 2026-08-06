import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="스마트팩토리 설비 고장 실시간 모니터링 & 분석",
    page_icon="⚙️",
    layout="wide"
)

# 2. day3_main_model 모듈 로드 및 데이터/모델 준비 (캐싱 적용)
@st.cache_resource
def init_main_pipeline():
    try:
        import day3_main_model as main_module
        
        # 1) 원본 데이터 로드 및 파생 특성 생성
        df, pure_feature_cols = main_module.load_and_preprocess_raw_data('ai4i2020.csv')
        
        # 2) 학습에 사용할 X, y 분리
        X = df[pure_feature_cols]
        y = df['Machine_failure']
        
        # 3) day3_main_model의 학습 함수를 호출하여 메인 XGBoost 모델 획득
        models = main_module.train_models_without_anomaly_score(X, y)
        xgb_model = models['XGBoost (No Anomaly Score)']
        
        # 전체 데이터셋에 대한 고장 예측 확률 미리 추론 (0~1 스케일)
        df['Failure_Prob'] = xgb_model.predict_proba(X)[:, 1]
        
        return df, pure_feature_cols, xgb_model
    except Exception as e:
        st.error(f"day3_main_model 연동 중 오류 발생: {e}")
        return None, None, None

df, pure_feature_cols, model = init_main_pipeline()

# 데이터 로드 실패 시 예외 처리
if df is None:
    st.stop()

# UDI 범례 및 최소/최대값 확인
has_udi = 'UDI' in df.columns
min_udi = int(df['UDI'].min()) if has_udi else 1
max_udi = int(df['UDI'].max()) if has_udi else len(df)
total_len = len(df)

# 백분율 파생 컬럼 미리 생성
df['Failure_Prob_Pct'] = df['Failure_Prob'] * 100

# 세션 스테이트 초기화 (시작 UDI 관리)
if 'current_start_udi' not in st.session_state:
    st.session_state.current_start_udi = min_udi

# 3. 사이드바 제어
st.sidebar.header("🕹️ 모니터링 제어")

selected_device = "Machine_01 (CNC Milling, AI4I 2020 Dataset)"

st.sidebar.info(
    f"⚙️ **모니터링 설비:** {selected_device}\n\n"

    f"📌 **데이터 특성:** 단일 설비 연속 가공 이력 (총 {total_len:,}건)"
)

st.sidebar.caption("Source: AI4I 2020 Predictive Maintenance Dataset")

product_type_filter = st.sidebar.selectbox(
    "가공 제품 유형 (Product Type)",
    ["전체 (All Types)", "L (Low Quality)", "M (Medium Quality)", "H (High Quality)"]
)

st.sidebar.markdown("### 🔍 UDI 구간 탐색")

window_options = [100, 300, 500, 1000, 2000, total_len, "직접 입력"]
selected_window_option = st.sidebar.selectbox(
    "한 번에 볼 관측 포인트 수 (Window Size)",
    options=window_options,
    index=2
)

if selected_window_option == "직접 입력":
    raw_window_size = st.sidebar.number_input(
        "원하는 관측 포인트 수 직접 입력",
        min_value=1,
        max_value=total_len * 2,
        value=500,
        step=50
    )
    window_size = max(1, min(raw_window_size, total_len))
else:
    window_size = selected_window_option

is_full_view = (window_size >= total_len)

if is_full_view:
    start_udi = min_udi
    end_udi = max_udi
    st.sidebar.caption(f"📍 전체 데이터 조망 모드 (UDI {min_udi} ~ {max_udi})")
else:
    max_start_udi = max(min_udi, max_udi - window_size + 1)

    # 세션 스테이트 값이 범위를 벗어나지 않도록 보정
    if st.session_state.current_start_udi > max_start_udi:
        st.session_state.current_start_udi = max_start_udi
    if st.session_state.current_start_udi < min_udi:
        st.session_state.current_start_udi = min_udi

    # 📌 조회 시작 지점 직접 입력 및 슬라이더 연동
    input_start_udi = st.sidebar.number_input(
        "조회 시작 지점 직접 입력 (Start UDI)",
        min_value=min_udi,
        max_value=max_start_udi,
        value=int(st.session_state.current_start_udi),
        step=50
    )
    
    slider_start_udi = st.sidebar.slider(
        "조회 시작 지점 슬라이더",
        min_value=min_udi,
        max_value=max_start_udi,
        value=input_start_udi,
        step=50
    )

    st.session_state.current_start_udi = slider_start_udi
    start_udi = st.session_state.current_start_udi
    end_udi = start_udi + window_size - 1
    st.sidebar.caption(f"📍 현재 조망 구간: UDI {start_udi} ~ {end_udi}")

threshold_slider = st.sidebar.slider("고장 위험 경보 임계치 (%)", 30, 90, 50, 5)
threshold = threshold_slider / 100.0

st.sidebar.markdown("### 🎯 특정 UDI 즉시 이동 (끝단 맞춤)")
target_udi_input = st.sidebar.number_input(
    "보고 싶은 특정 UDI 번호 입력",
    min_value=min_udi,
    max_value=max_udi,
    value=674,
    step=1
)

if st.sidebar.button("🚀 해당 UDI로 즉시 이동", key="jump_to_udi_btn"):
    if is_full_view:
        st.sidebar.warning("⚠️ 현재 전체 조망 모드입니다. 윈도우 크기를 먼저 설정해주세요.")
    else:
        # 입력한 UDI가 화면의 맨 우측(끝단)에 오도록 계산하되, 범위를 넘지 않게 조정
        new_start = target_udi_input - window_size + 1
        if new_start < min_udi:
            new_start = min_udi
        if new_start > max_start_udi:
            new_start = max_start_udi
            
        st.session_state.current_start_udi = int(new_start)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.success("✅ **메인 모델:** Single-stage XGBoost 연동 완료")


#st.sidebar.markdown("### data debug console")
#debug_udi = st.sidebar.number_input("조회할 UDI 번호", min_value=min_udi, max_value=max_udi, value=673)
#target_row = df[df['UDI'] == debug_udi]
#if not target_row.empty:
#    r = target_row.iloc[0]
#    st.sidebar.json({
#        "UDI": int(r['UDI']),
#        "Type": r['Type'],
#        "Tool_wear (min)": float(r['Tool_wear']),
#        "Torque (Nm)": float(r['Torque']),
#        "Failure_Prob (%)": float(r['Failure_Prob_Pct']),
#        "Actual_Failure": int(r['Machine_failure'])
#    })


# 4. 메인 화면 헤더
st.title("⚙️ 스마트팩토리 설비 고장 실시간 모니터링 & 심층 분석")
st.caption(
    f"선택 설비: **{selected_device}** | "
    f"XGBoost 실시간 고장 확률 추론 및 예지보전 3대 인사이트 분석 "
    f"(According to 'AI4I 2020 Predictive Maintenance Dataset' description)"
)

# 5. 시뮬레이션용 관측 데이터 슬라이싱 (UDI 기준 필터링)
filtered_df = df.copy()
if product_type_filter != "전체 (All Types)":
    target_type = product_type_filter.split()[0]
    filtered_df = filtered_df[filtered_df['Type'] == target_type]

if is_full_view:
    sim_df = filtered_df.copy()
else:
    if has_udi:
        sim_df = filtered_df[(filtered_df['UDI'] >= start_udi) & (filtered_df['UDI'] <= end_udi)].copy()
    else:
        start_pos = max(0, start_udi - 1)
        end_pos = min(len(filtered_df), start_pos + window_size)
        sim_df = filtered_df.iloc[start_pos:end_pos].copy()

if sim_df.empty:
    st.warning("선택한 조건 및 UDI 범위에 해당하는 관측 데이터가 없습니다.")
    st.stop()

latest_data = sim_df.iloc[-1]
latest_prob = latest_data['Failure_Prob']

# 6. 상단 Dynamic KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    if latest_prob >= threshold:
        st.metric(label="설비 상태", value="🚨 위험 (Critical)", delta="점검 필요", delta_color="inverse")
    else:
        st.metric(label="설비 상태", value="✅ 정상 (Normal)", delta="안전", delta_color="normal")

with col2:
    prev_prob = sim_df.iloc[-2]['Failure_Prob'] if len(sim_df) > 1 else latest_prob
    st.metric(
        label="고장 예측 확률 (XGBoost)",
        value=f"{latest_prob * 100:.1f}%",
        delta=f"{(latest_prob - prev_prob) * 100:+.1f}%p",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="회전 속도 (Rotational Speed)",
        value=f"{latest_data['Rotational_speed']:.0f} rpm",
        delta=f"{latest_data['Rotational_speed'] - sim_df.iloc[-2]['Rotational_speed']:.0f} rpm" if len(sim_df) > 1 else "0 rpm"
    )

with col4:
    st.metric(
        label="공구 마모 (Tool Wear)",
        value=f"{latest_data['Tool_wear']:.0f} min",
        delta=f"{latest_data['Tool_wear'] - sim_df.iloc[-2]['Tool_wear']:.0f} min" if len(sim_df) > 1 else "0 min"
    )

st.markdown("---")

# 7. 구간 이동 버튼 배치
if not is_full_view:
    nav_col1, nav_col2, nav_space = st.columns([1, 1, 4])

    with nav_col1:
        if st.button("◀ 이전 구간 이동", use_container_width=True):
            new_val = max(min_udi, st.session_state.current_start_udi - window_size)
            st.session_state.current_start_udi = new_val
            st.rerun()

    with nav_col2:
        if st.button("다음 구간 이동 ▶", use_container_width=True):
            new_val = min(max_start_udi, st.session_state.current_start_udi + window_size)
            st.session_state.current_start_udi = new_val
            st.rerun()

# 8. 메인 실시간 차트 & 위험 단계 알람
left_col, right_col = st.columns([3.2, 1])

x_axis_col = 'UDI' if has_udi else sim_df.index

with left_col:
    st.subheader("📈 실시간 센서 추이 및 고장 예측 확률")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=sim_df[x_axis_col], y=sim_df['Tool_wear'], name="Tool Wear (min)", line=dict(color='#00CC96', width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=sim_df[x_axis_col], y=sim_df['Torque'], name="Torque (Nm)", line=dict(color='#AB63FA', width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=sim_df[x_axis_col], y=sim_df['Failure_Prob_Pct'], name="Failure Prob (%)", line=dict(color='#EF553B', width=2, dash='dot')),
        secondary_y=True,
    )
    
    fig.add_hline(y=threshold_slider, line_dash="dash", line_color="red", secondary_y=True, annotation_text=f"Danger Threshold ({threshold_slider}%)")

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        dragmode=False
    )
    
    if not sim_df.empty:
        fig.update_xaxes(range=[sim_df[x_axis_col].iloc[0], sim_df[x_axis_col].iloc[-1]])
    
    fig.update_yaxes(title_text="Sensor Values (min / Nm)", secondary_y=False)
    fig.update_yaxes(title_text="Failure Probability (%)", range=[0, 100], secondary_y=True)
    
    st.plotly_chart(
        fig, 
        use_container_width=True, 
        config={'displayModeBar': False, 'scrollZoom': False, 'doubleClick': False}
    )

with right_col:
    st.subheader("⚠️ 센서별 알람")
    
    tool_wear_val = latest_data['Tool_wear']
    torque_val = latest_data['Torque']
    
    st.markdown("**1. Tool Wear (마모도)**")
    st.progress(min(int(tool_wear_val / 240 * 100), 100))
    if tool_wear_val > 200:
        st.error(f"🚨 임박: {tool_wear_val:.0f} min (200 초과)")
    else:
        st.success(f"✅ 정상: {tool_wear_val:.0f} min")
        
    st.markdown("---")
    
    st.markdown("**2. Torque (토크)**")
    st.progress(min(int(torque_val / 80 * 100), 100))
    if torque_val > 60:
        st.warning(f"⚠️ 과부하: {torque_val:.1f} Nm (60 초과)")
    else:
        st.success(f"✅ 적정: {torque_val:.1f} Nm")

    st.markdown("---")
    if latest_prob >= threshold:
        st.error(f"🔥 **[경고]** 고장 예측 확률 {latest_prob*100:.1f}% 도달!")

# 9. 실무 예지보전 3대 심층 분석 섹션
st.markdown("---")
st.subheader("🔍 예지보전 핵심 인사이트 심층 분석")

tab1, tab2, tab3 = st.tabs([
    "🎯 1. 골든 타임 (변곡점) 자동 탐지", 
    "⚖️ 2. 제품 유형(Type)별 마모-토크 상관 비교", 
    "🚨 3. 위험 징후 및 오탐지(FP) 모아보기"
])

with tab1:
    st.markdown("#### ⏳ 공구 마모 200분 초과 구간(위험 마지노선)에서의 AI 예측 검증")
    st.caption("기존 분석 인사이트('200분 이상부터 고장률이 급증함')를 바탕으로, 실제 공구 마모도가 200분을 넘어설 때 XGBoost 모델이 고장 위험을 얼마나 잘 포착하는지 검증합니다.")
    
    wear_over_200 = df[df['Tool_wear'] >= 200]
    
    if not wear_over_200.empty:
        high_prob_in_over200 = wear_over_200[wear_over_200['Failure_Prob'] >= threshold]
        detection_rate = (len(high_prob_in_over200) / len(wear_over_200)) * 100
        
        col_inf1, col_inf2, col_inf3 = st.columns(3)
        with col_inf1:
            st.metric("마모 200분 초과 총 가공 건수", f"{len(wear_over_200):,} 건")
        with col_inf2:
            st.metric("모델 위험 경고 적중 건수", f"{len(high_prob_in_over200):,} 건")
        with col_inf3:
            st.metric("위험 구간 AI 탐지율", f"{detection_rate:.1f}%")
        
        st.info(f"💡 **분석 가이드:** 공구가 **200분 이상 마모된 상태**에서는 실제 고장 위험이 급증하며, 본 AI 모델은 이 구간의 데이터를 **{detection_rate:.1f}%**의 확률로 정확히 위험(Critical)으로 감지해내고 있습니다. 따라서 현장에서는 공구 마모가 200분에 도달하기 전(예: 180~190분 선)을 최종 교체 주기로 삼는 것이 가장 이상적입니다.")
    else:
        st.success("현재 데이터셋 내에 공구 마모가 200분을 초과한 구간이 존재하지 않습니다.")

with tab2:
    st.markdown("#### 📊 가공 제품 유형(L / M / H)별 공구 마모 vs 토크(Torque) 상관관계")
    st.caption("고품질(H) 제품과 저품질(L) 제품 가공 시 동일 마모도에서 발생하는 부하(Torque) 특성 차이를 비교합니다.")
    
    fig_scatter = px.scatter(
        df, 
        x="Tool_wear", 
        y="Torque", 
        color="Type", 
        opacity=0.6,
        template="plotly_dark",
        labels={"Tool_wear": "Tool Wear (min)", "Torque": "Torque (Nm)", "Type": "Product Type"}
    )
    fig_scatter.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), dragmode=False)
    st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': False, 'doubleClick': False})
    
    st.markdown("---")
    
    st.markdown("#### 🔍 토크(부하) 구간별 가공 제품 유형(Type) 점유율 분석")
    st.caption("설비에 걸리는 토크 부하를 구간별로 나누어, 각 구간에서 어떤 제품(L, M, H)이 주로 생산되며 비율을 차지하는지 분석합니다.")
    
    df_analyzed = df.copy()
    df_analyzed['Type'] = pd.Categorical(df_analyzed['Type'], categories=['L', 'M', 'H'], ordered=True)
    
    bins = [0, 40, 60, df['Torque'].max() + 1]
    labels = ['40 Nm 미만 (저부하)', '40~60 Nm (중부하)', '60 Nm 초과 (고부하)']
    df_analyzed['Torque_Bin'] = pd.cut(df_analyzed['Torque'], bins=bins, labels=labels, right=False)
    
    torque_type_ct = pd.crosstab(df_analyzed['Torque_Bin'], df_analyzed['Type'], normalize='columns') * 100
    torque_type_count = pd.crosstab(df_analyzed['Torque_Bin'], df_analyzed['Type'])
    
    display_table = torque_type_ct.round(1).astype(str) + " % (" + torque_type_count.astype(str) + "건)"
    display_table = display_table[['L', 'M', 'H']]
    
    st.dataframe(display_table, use_container_width=True)

with tab3:
    st.markdown("#### 🕵️ 모델 예측 위험(>임계치) vs 실제 정상(Machine_failure = 0) 데이터 조회")
    st.caption("모델은 고장 위험이 높다고 경고했으나 실제 고장은 발생하지 않은 **'아슬아슬한 위험 구간 (Near-miss / False Positive)'**의 특성을 분석합니다.")
    
    near_miss_df = df[(df['Failure_Prob'] >= threshold) & (df['Machine_failure'] == 0)]
    
    col_nm1, col_nm2 = st.columns([1, 3])
    with col_nm1:
        st.metric("아슬아슬한 위험 건수", f"{len(near_miss_df):,} 건")
        
    if not near_miss_df.empty:
        st.dataframe(
            near_miss_df[['UDI', 'Type', 'Rotational_speed', 'Torque', 'Tool_wear', 'Failure_Prob_Pct', 'Machine_failure']].style.format({
                'Failure_Prob_Pct': '{:.1f}%',
                'Torque': '{:.1f}'
            }),
            use_container_width=True,
            height=250
        )
        st.warning("⚠️ **현장 검증 포인트:** 실제 고장(1)은 나지 않았지만 모델이 경고를 보낸 위 데이터들은 **순간적인 과부하(Torque 튐)나 마모 임계치 임박**으로 인해 설비 피로도가 극에 달했던 순간들입니다. 정비 엔지니어의 교차 점검이 권장됩니다.")
    else:
        st.success("해당 임계치 조건에서 아슬아슬한 위험 징후(Near-miss) 데이터가 발견되지 않았습니다.")

# 10. 하단 데이터 상세 내역 및 구간 분석 요약 지표
st.markdown("---")
st.subheader("📋 현재 조망 구간 데이터 상세 내역")

total_count = len(sim_df)
exceeded_df = sim_df[sim_df['Failure_Prob'] >= threshold]
exceeded_count = len(exceeded_df)
exceeded_ratio = (exceeded_count / total_count * 100) if total_count > 0 else 0.0

sum_col1, sum_col2, sum_col3 = st.columns(3)
with sum_col1:
    st.metric(label="현재 조망 구간 전체 데이터 수", value=f"{total_count:,} 건")
with sum_col2:
    st.metric(label=f"임계치({threshold_slider}%) 초과 데이터 수", value=f"{exceeded_count:,} 건", delta=f"{exceeded_ratio:.1f}%", delta_color="inverse")
with sum_col3:
    st.metric(label="임계치 초과 비율", value=f"{exceeded_ratio:.2f}%")

st.markdown("")

display_cols = ['UDI', 'Type', 'Air_temperature[K]', 'Process_temperature[K]', 'Rotational_speed[rpm]', 'Torque[Nm]', 'Tool_wear[min]', 'Failure_Prob_Pct', 'Machine_failure']
available_cols = [col for col in display_cols if col in sim_df.columns]

st.dataframe(
    sim_df[available_cols].style.format({
        'Failure_Prob_Pct': '{:.1f}%',
        'Torque[Nm]': '{:.1f}'
    }),
    use_container_width=True,
    height=300
)