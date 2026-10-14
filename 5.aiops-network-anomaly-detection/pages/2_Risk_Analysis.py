import streamlit as st
import pandas as pd
import numpy as np

# 페이지 기본 설정
st.set_page_config(
    page_title="Risk Score 설계 및 검증",
    page_icon="📐",
    layout="wide"
)

st.title("📐 Screen ②: Risk Score 설계 및 검증 과정")
st.markdown("**부제:** 연도별 네트워크 환경 차이를 반영한 위험도 산출 과정과 검증 결과")
st.markdown("---")

# Session State 데이터 확인 안내
if "df_master" in st.session_state and st.session_state["df_master"] is not None:
    st.success("✅ 마스터 데이터셋 및 매핑 정보가 성공적으로 연동되어 있습니다.")
else:
    st.info("💡 안내: app.py를 통해 데이터 로드가 완료되면 마스터 데이터와 연동됩니다. 현재 페이지는 검증 실험 리포트 중심로 구성되어 있습니다.")

st.markdown("""
### 🧭 분석 및 검증 프로세스 흐름
연도별 Normal 환경 차이 -> Feature별 Z-score 이탈도 -> 4대 Risk Group 통합 -> 극단값 기여도 왜곡 확인 -> Cap 적용 -> Log1p + Cap 검증 -> Risk Score 분포 검증 -> Risk Deviation 도입 -> Risk Grade / Threshold 검증
""")
st.markdown("---")

# ==========================================
# SECTION 1: 연도별 Normal 기준선 검증
# ==========================================
st.subheader("1. 연도별 Normal 기준선 검증")
st.markdown("**핵심 결론:** 2017년과 2025년 데이터는 수집 환경 자체가 다르므로, 단일 공통 Normal baseline보다 연도별 Normal baseline을 분리 적용하는 것이 적절함.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("2017 Normal 표본", "2,273,097건", "표본 축소 테스트 완료")
with col2:
    st.metric("2025 Normal 표본", "400,672건", "독립 IIoT 환경 반영")
with col3:
    st.metric("Baseline 전략", "연도별 분리", "트렌드 유지 검증됨")

with st.expander("🔍 상세 분석 내용 보기 (표본 크기별 테스트 결과)"):
    st.markdown("""
    - **분석 배경:** 2017년 방대한 정상 데이터(220만+건)로 인해 표준편차가 부풀려지는 현상이 관측되어, 표본 수를 대폭 축소(1M, 500k, 300k)하여 테스트를 수행함.
    - **결과:** 표본 수를 줄여도 리스크 점수의 전반적인 트렌드와 경향성은 안정적으로 유지됨을 확인.
    - **시사점:** 이는 단순 표본 수의 문제가 아니라 수집된 네트워크 환경 자체의 차이에서 기인한 것이므로, 연도별 독립 Normal baseline을 유지하기로 확정함.
    """)

st.markdown("---")

# ==========================================
# SECTION 2: Feature 분포 및 4대 Risk Group
# ==========================================
st.subheader("2. Feature 분포 차이와 Risk Group 설계")
st.markdown("**핵심 결론:** 이질적인 두 시대(2017 vs 2025)의 컬럼 스키마 격차를 극복하기 위해 네트워크 트래픽을 관통하는 4대 메타 축(Meta Feature)으로 추상화함.")

r_col1, r_col2, r_col3, r_col4 = st.columns(4)
with r_col1:
    st.markdown("##### 📦 Volume")
    st.caption("단위 시간당 패킷 양과 발생 빈도 (DoS/DDoS 탐지 핵심)")
with r_col2:
    st.markdown("##### 📐 Size & Payload")
    st.caption("바이트 길이 통계 및 구조적 특성 (웹 취약점/악성코드 탐지)")
with r_col3:
    st.markdown("##### 🌐 Topology")
    st.caption("IP 대역, 포트 번호 다양성 및 연결 구조 (Recon/스캔 탐지)")
with r_col4:
    st.markdown("##### ⚙️ Control & State")
    st.caption("TCP 플래그, IAT, 윈도우 크기 (세션 제어 및 브루트포스 추적)")

st.markdown("---")

# ==========================================
# SECTION 3: Feature Group별 Risk Score 기여도
# ==========================================
st.subheader("3. Feature Group별 Risk Score 기여도 분석")
st.markdown("**핵심 결론:** 동일한 Z-score 방식이라도 네트워크 특성에 따라 위험 기여 영역이 완전히 달라짐을 실측함.")

df_contrib = pd.DataFrame({
    "Risk Group": ["Volume", "Size & Payload", "Topology", "Control & State"],
    "2017 기여도 (%)": [5.76, 73.72, 8.56, 11.96],
    "2025 기여도 (%)": [4.01, 0.004, 4.67, 91.31]
})

st.table(df_contrib)
st.markdown("""
- **2017년 특징:** Size & Payload 영역이 약 `73.72%`로 압도적인 기여도를 기록 (전통적 IT 환경의 패킷 크기 변동성 중심).
- **2025년 특징:** Control & State 영역이 약 `91.31%`로 지배적 기여도를 기록 (IIoT 스마트팩토리 환경의 규칙적 통신 중 발생하는 TCP 플래그/상태 이상 포착).
""")

st.markdown("---")

# ==========================================
# SECTION 4: 극단적인 Z-score 문제 분석
# ==========================================
st.subheader("4. 극단적인 Z-score가 Risk Score를 지배하는 문제")
st.markdown("**핵심 결론:** 일부 특성에서 발생한 극단적인 Z-score가 특정 Risk Group의 점수를 과도하게 왜곡하는 현상을 확인.")

st.markdown("""
- **문제점:** Normal baseline 대비 편차를 구할 때 일부 스파이크성 Feature 값이 수만 배 이상 튀면서, Random Forest 중요도와 비교했을 때도 기여도 편차가 지나치게 극심해지는 현상 발견.
- **대책 필요성:** 특성 본연의 이상 정도는 반영하되, 단일 극단값 하나가 전체 Risk Score를 지배하지 않도록 완화 장치가 필요함을 도출.
""")

st.markdown("---")

# ==========================================
# SECTION 5 & 6: Z-score Cap 적용 및 Log1p + Cap 비교
# ==========================================
st.subheader("5 & 6. Z-score Cap(5.0) 및 Log1p + Cap 검증")
st.markdown("**핵심 결론:** 극단값 통제를 위해 `Cap = 5.0` 및 `Scaled_Z = min(log1p(Z), 5.0)` 방식을 교차 검증함.")

c_col1, c_col2 = st.columns(2)
with c_col1:
    st.markdown("##### 🔹 Cap = 5.0 적용")
    st.info("""
    - **방식:** 원본 Z-score가 5.0을 초과할 경우 5로 고정.
    - **장점:** 단일 특성의 과도한 점수 지배력 완화.
    - **한계:** 극단 구간에서의 원본 데이터 의미 다소 희석.
    """)
with c_col2:
    st.markdown("##### 🔸 Log1p + Cap 추가 검증")
    st.info("""
    - **방식:** `min(log1p(Z), 5.0)` 적용.
    - **결과:** 전반적인 공격군 패턴은 유지되면서, 2025 DoS/DDoS 공격군에서 상대적 변별력이 세밀하게 조정됨을 확인.
    """)

st.markdown("---")

# ==========================================
# SECTION 7: 연도별 Risk Score 분포 검증
# ==========================================
st.subheader("7. Risk Score 분포 검증")
st.markdown("**핵심 결론:** Cap과 Log1p를 모두 적용한 결과, 원시 Z-score에 비해 점수 분산 차이가 상당 부분 완화되었으나 여전히 연도별 환경 차이에 의한 분포 간극은 잔존함.")

df_score_dist = pd.DataFrame({
    "Year": ["2017", "2025"],
    "Mean Score": [1.02, 1.23],
    "Median Score": [0.94, 1.18],
    "Std Dev": [0.32, 0.60],
    "Max Score": [6.42, 9.98]
})
st.table(df_score_dist)
st.markdown("👉 절대적인 단일 Threshold를 두 연도에 동시 적용하기에는 한계가 있으므로, **상대적 이탈도(Risk Deviation)** 도입의 근거가 됨.")

st.markdown("---")

# ==========================================
# SECTION 8 & 9: Risk Deviation 도입 및 공격군 분석
# ==========================================
st.subheader("8 & 9. Risk Deviation (상대적 위험도) 도입 및 공격 유형별 분석")
st.markdown("**핵심 결론:** 각 연도의 Normal 환경 대비 얼마나 이례적인지를 나타내는 `Risk Deviation = (Risk Score - Mean) / STD` 지표를 도입함.")

st.markdown("""
- **Risk Score vs Risk Deviation:**
  - **Risk Score:** 절대적인 위험 수치화
  - **Risk Deviation:** 해당 연도 Normal 환경 대비 상대적 이례성(Standardized Deviation) 해석
- **공격군별 분석 결과:**
  - **2017년:** DoS/DDoS 중심으로 높은 Deviation을 보였으나, Recon/Scan, Malware 등은 Normal과 분포가 다수 겹치는 양상
  - **2025년:** DoS/DDoS, Recon/Scan, Malware/Botnet, Web Attack 등 다수 공격군에서 Deviation이 높게 치솟아 Normal과 뚜렷한 분리 양상 확인
""")

st.markdown("---")

# ==========================================
# SECTION 10 & 11: Risk Grade 기준 및 Threshold Trade-off
# ==========================================
st.subheader("10 & 11. Risk Grade 기준 및 Threshold Trade-off 검증")
st.markdown("**핵심 결론:** P95/P99 기반 등급화와 함께 4대 메타 축 서브 스코어 조건을 조합하여 탐지 성능과 실무 트레이드오프를 검증함.")

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.markdown("##### 📌 조건 A: 상위 10% (P90) 초과 항목 1개 이상")
    st.metric("Binary F1-Score", "0.5933", "균형 잡힌 탐지 성능")
    st.caption("Precision: 0.58 / Recall: 0.61")
with t_col2:
    st.markdown("##### 📌 조건 B: 상위 10% (P90) 초과 항목 2개 이상 (강화)")
    st.metric("Binary F1-Score", "0.5119", "미탐(Recall) 급증 발생")
    st.caption("Precision: 0.78 / Recall: 0.38 (미탐 심화)")

st.markdown("""
- **구조적 한계 및 시사점:**
  - DoS/DDoS나 Malware처럼 볼륨/제어 플래그가 크게 요동치는 공격은 잘 탐지됨.
  - 반면 Recon/Scan, Brute Force 등 스텔스성 공격은 특정 축만 건드리므로 조건을 2개 이상으로 엄격히 걸 경우 미탐이 급증함.
  - 따라서 본 시스템의 Risk Score는 완벽한 자동 차단 분류기가 아니라, **BI 모니터링 및 실무 RCA 의사결정을 지원하는 정량적 지표**로 정의함.
""")

st.markdown("---")

# ==========================================
# 최종 Conclusion
# ==========================================
st.markdown("### 🎯 최종 요약 (Conclusion)")
st.info("""
본 페이지는 2017년 표준 벤치마크와 2025년 대규모 IIoT 트래픽 간의 환경적 차이를 통계적 EDA와 Cap/Log1p 정규화, 그리고 Risk Deviation 변환을 통해 만들어낸 **통합 리스크 스코어링 엔진의 설계 및 검증 일지**입니다. 
단순한 기계학습 블랙박스 추론에서 벗어나, 현업 관리자가 네트워크 인프라의 이상 징후를 다각도로 추적하고 후속 RCA 및 실무 대응으로 연계할 수 있도록 BI 의사결정 기준을 확립했습니다.
""")