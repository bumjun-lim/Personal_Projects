# Credit Card Fraud Detection & Real-time Dashboard

신용카드 거래 데이터를 활용하여 극심한 데이터 불균형 문제에 대해, F1-Score 기반의 사기 거래 탐지 모델 구축 및 Streamlit 모니터링 대시보드를 구현한 프로젝트.

## 📌 프로젝트 개요

* **목적:** 이상 거래(Fraud) 데이터를 선제적으로 탐지하여 금융 손실을 최소화하고, 오탐(FP)과 미탐(FN)을 추적할 수 있는 실시간 모니터링 환경 구축
* **주요 해결 과제:**
  * 극심한 클래스 불균형(Fraud 비율 약 0.17%) 해소
  * 단순 정확도(Accuracy)가 아닌 비즈니스 손실과 직결되는 **F1-Score / Recall** 최적화
  * 이상 거래 패턴 및 금액대별 시각화 모니터링 구축

## 🔗 상세 프로젝트 기록 (Notion)

👉 **[노션 포트폴리오 바로가기](https://app.notion.com/p/SQL-239c6bb576d3802fab81ecd7cb5cb766?source=copy_link)**

## 🛠️ 사용 기술 및 도구 (Tech Stack)

* **Language:** Python
* **Data Processing & ML:** Pandas, NumPy, Scikit-Learn (`RobustScaler`, `RandomForestClassifier`)
* **Visualization & Dashboard:** Streamlit, Matplotlib, Seaborn

## 💡 주요 수행 내용 & 성과

1. **데이터 전처리 (Preprocessing)**
   * `Time`, `Amount` 변수의 이상치 영향 최소화를 위한 `RobustScaler` 적용
2. **모델링 및 평가 (Modeling & Evaluation)**
   * `RandomForestClassifier` 기반의 분류 모델 구축
   * 데이터 불균형 환경에 맞춰 **F1-Score**를 메인 평가 지표로 채택
3. **대시보드 구축 (Streamlit Dashboard)**
   * 금액대별 사기 거래 발생 패턴 시각화
   * 오탐(False Positive) 및 미탐(False Negative) 건수를 실시간 추적할 수 있는 인터페이스 구현

## 🚀 대시보드 실행 방법


# 1. 가상환경 구축 및 패키지 설치
pip install -r requirements.txt

# 2. Streamlit 앱 실행
streamlit run app.py