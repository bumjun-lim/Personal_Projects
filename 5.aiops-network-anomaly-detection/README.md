# AIOps Network & IIoT Flow Data Risk Scoring & BI Decision System

표준 네트워크 데이터(2017)와 대규모 최신 IIoT 트래픽 데이터(2025)를 통계적 EDA와 머신러닝으로 분석하여, 인프라 위협 징후를 정량적 지표로 발굴하고 현업 관리자의 실시간 의사결정을 지원하는 BI 대시보드 구축 프로젝트.

## 📌 프로젝트 개요

- **목적:** 수백만 건에 달하는 이질적인 네트워크 플로우 데이터를 통계적으로 정규화하고, 딥러닝 블랙박스 구조에서 벗어나 투명한 '통합 리스크 스코어링 엔진'을 구축하여 인프라 이상 징후 감지 및 실무 대응(`RCA`) 체계 마련.
- **주요 해결 과제:**
  * 수 GB에 달하는 대규모 원본 데이터를 Chunk 분할 로딩 및 Parquet 포맷 변환을 통해 메모리 에러(OOM) 없이 처리하는 대용량 최적화 파이프라인 정립.
  * 2017년(전통 IT)과 2025년(IIoT) 간의 수집 환경 및 스키마 격차를 극복하기 위해 4대 메타 축(Volume, Size & Payload, Topology, Control & State) 기반의 의미적 매핑 설계.
  * 단일 Z-score 방식에서 발생하는 극단값의 기여도 왜곡 문제를 완화하기 위해 `Cap=5.0` 및 `Log1p` 정규화 적용.
  * 연도별 Normal 분포 차이를 보정하기 위한 상대적 이탈도(`Risk Deviation`) 지표 도입 및 현업 의사결정을 위한 3단계 위험 등급 산출.

## 🔗 상세 프로젝트 기록 (Notion)

- [노션 포트폴리오 바로가기](https://app.notion.com/p/05-IIoT-BI-3c9c6bb576d38018acaccd6578270011?source=copy_link)

## 🛠️ 사용 기술 및 도구 (Tech Stack)

- **Language:** Python
- **Data Processing & ML:** Pandas, NumPy, Scikit-Learn (Random Forest)
- **Data Optimization:** Chunk Processing, Apache Parquet
- **Dashboard & UI:** Streamlit (`st.session_state` 기반 멀티페이지 아키텍처)
- **Visualization:** Matplotlib, Seaborn, Plotly

## 💡 주요 수행 내용 & 성과 (Key Highlights & Insights)

1. **대규모 네트워크 데이터의 메모리 최적화 및 스키마 정규화 (Day 1 ~ Day 2)**
   * 수백만 건의 이질적인 2017년 벤치마크 및 2025년 IIoT 데이터를 Parquet 포맷으로 일괄 변환하여 고속 입출력 및 메모리 안정성 확보.
   * 서로 다른 컬럼 명칭과 Granularity를 가진 원본 특성을 트래픽 본질을 관통하는 4대 메타 축(Volume, Size & Payload, Topology, Control & State)으로 추상화하여 상호 호환 스키마 구축.
2. **통계적 특성 분석 및 리스크 스코어링 엔진 설계 (Day 3 ~ Day 7)**
   * 오직 정상(BENIGN) 트래픽만을 기준으로 하는 연도별 독립 Normal Baseline을 수립하고, Z-score 이탈도 기반의 0~100점 스코어링 파이프라인 정립.
   * 초기 스코어링 과정에서 일부 스파이크성 Feature가 전체 점수를 지배하는 왜곡 현상을 발견하고, `Cap=5.0`과 `Log1p` 변환을 교차 검증하여 점수 분산 안정화.
   * 절대적 Risk Score의 한계를 극복하고자 연도별 Normal 환경 대비 상대적 이례성을 나타내는 `Risk Deviation` 변환 도입 및 P95/P99 기반의 3단계 위험 등급(Normal, Caution, Danger) 체계 확립.
3. **Streamlit 기반의 인터랙티브 BI 대시보드 및 RCA 구현 (Day 7 ~ Day 8)**
   * 초기 검토했던 Tableau의 동적 인터랙션 한계를 극복하고자, Python 데이터 가공 레이블과 긴밀히 결합된 `Streamlit` 멀티페이지 아키텍처 도입.
   * 전체 위험 현황 모니터링(`Risk Overview`)부터 고위험 세션의 주원인 그룹 및 Top 3 피처를 역추적하는 원인 분석(`Session Investigation & RCA`), 그리고 현업 조치 가이드(`Response Guide`) 및 공격 백과사전(`Attack Knowledge`)으로 이어지는 유기적인 의사결정 파이프라인 완성.
   * *"완벽한 자동 차단 모델이란 존재하지 않으며, 인프라의 미세한 이탈 징후를 다각도로 추적하고 현업 관리자의 실무 조치를 조율하는 정량적 의사결정 지표가 진짜 분석가의 몫"*이라는 실무적 감각 체득.

## 🗂️ 프로젝트 구조 (Directory Structure)
```text
5.aiops-network-anomaly-detection/
│
├── data/                    # 원천 및 가공된 Parquet/CSV 데이터셋
├── notebooks/               # EDA 및 스코어링 검증 Jupyter Notebook
├── pages/                   # Streamlit 멀티페이지 구성
│   ├── 1_Risk_Overview.py   # 전체 위험 현황 대시보드
│   ├── 2_Risk_Analysis.py   # Risk Score 설계 및 검증 리포트
│   ├── 3_Session_Investigation.py # 개별 세션 RCA 및 원인 분석
│   ├── 4_Response_Guide.py  # 실무 대응 및 조치 가이드 연동
│   └── 5_Attack_Knowledge.py # 공격 유형별 보안 지식 레퍼런스
├── outputs/                 # 분석 결과물 및 로그
├── src/                     # 공통 유틸리티 및 스코어링 파이프라인 모듈
├── app.py                   # Streamlit 진입점 (Landing & Data Loader)
├── requirements.txt         # 패키지 의존성 목록
└── README.md
```

## 🚀 프로젝트 실행 방법

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. Streamlit 대시보드 실행
#### 프로젝트 루트 디렉토리에서 아래 명령어를 실행하여 웹 대시보드를 구동합니다.

```bash
streamlit run app.py
```