# Smart Factory Predictive Maintenance & Real-time Dashboard

제조 센서 데이터를 활용하여 라벨 없는 환경에서의 이상 징후 선제 감지, XGBoost 기반의 고장 예측 확률 산출, 그리고 현장 작업자가 직관적으로 확인할 수 있는 Streamlit 모니터링 대시보드를 구현한 프로젝트.

## 📌 프로젝트 개요

* **목적:** 설비 센서 데이터를 분석하여 돌발 고장을 사전에 예측하고, 현장 운영에 맞는 최적의 임계치(Threshold)와 오탐지 관리 체계를 수립할 수 있는 실시간 모니터링 환경 구축
* **주요 해결 과제:**
  * 제조 공정 데이터 전처리 및 SQL을 활용한 데이터 탐지·구조화
  * 단순 정확도가 아닌 현장 리스크 관리에 직결되는 고장 예측 확률(`Failure Prob`) 및 임계치 최적화
  * 공구 마모도(Tool Wear)와 토크(Torque) 등 복합 센서 간 상관관계 및 위험 징후(Near-miss) 시각화

## 🔗 상세 프로젝트 기록 (Notion)

* [노션 포트폴리오 바로가기](https://app.notion.com/p/02-SQL-Python-3b0c6bb576d380008896c6e598875582) 

## 🛠️ 사용 기술 및 도구 (Tech Stack)

* **Language:** Python
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn, XGBoost
* **Database & Query:** SQL (SQLite / BigQuery)
* **Visualization & Dashboard:** Streamlit, Matplotlib, Seaborn

## 💡 주요 수행 내용 & 성과

1. **데이터 전처리 및 탐색 (Preprocessing & SQL)**
   * SQL을 활용한 데이터 조회 및 결측치·이상치 검증을 통한 데이터 구조화
   * 제조 공정 로그 및 센서 데이터 특성 파악
2. **모델링 및 평가 (Modeling & Evaluation)**
   * 단일 스테이지 XGBoost 모델을 연동하여 실시간 고장 예측 확률 산출
   * 골든 타임(마모도 200분 초과 구간) 및 제품 품질 등급(L, M, H)별 부하 특성 분석
3. **대시보드 구축 (Streamlit Dashboard)**
   * 실시간 센서별 알람 패널 및 시계열 고장 예측 확률 추이 시각화
   * 임계치 설정 변경에 따른 위험 감지 건수 및 상세 데이터 조회 기능 구현

## 🚀 대시보드 실행 방법

### 1. 가상환경 구축 및 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. Streamlit 앱 실행
```bash
streamlit run app.py
```