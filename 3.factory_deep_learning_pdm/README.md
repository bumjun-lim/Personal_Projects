# Smart Factory Predictive Maintenance & LSTM Model Limitation Study

제조 센서 데이터를 활용해 기존 머신러닝(XGBoost) 단계를 넘어 PyTorch 기반의 시계열 딥러닝(LSTM) 모델을 도입하고, 데이터 특성과 모델 적합성을 검증한 프로젝트.

## 📌 프로젝트 개요

* **목적:** 기존 머신러닝 모델을 고도화하기 위해 PyTorch LSTM을 도입하고, 설비 센서 데이터의 시계열 패턴을 학습하여 돌발 고장 징후를 선제적으로 예측하고자 함
* **주요 해결 과제:**
  * 텐서(Tensor) 구조 설계 및 슬라이딩 윈도우(Sliding Window) 기반 시퀀스 데이터 구조화
  * PyTorch 기반의 시계열 고장 예측 모델 학습 및 역전파(Backpropagation) 최적화
  * 기존 XGBoost 모델과 LSTM 모델의 성능 비교 및 딥러닝 도입의 실효성 검증

## 🔗 상세 프로젝트 기록 (Notion)

* [노션 포트폴리오 바로가기](https://app.notion.com/p/03-PyTorch-LSTM-3b6c6bb576d3803387dfc9076b0a05ab?source=copy_link) 

## 🛠️ 사용 기술 및 도구 (Tech Stack)

* **Language:** Python
* **Deep Learning:** PyTorch (LSTM, DataLoader, Tensor)
* **Machine Learning & Processing:** Pandas, NumPy, Scikit-Learn, XGBoost
* **Database & Query:** SQL
* **Visualization:** Matplotlib, Seaborn

## 💡 주요 수행 내용 & 회고 (Project Conclusion)

1. **데이터 파이프라인 정비 및 텐서 변환**
   * 기존 스마트팩토리 센서 데이터를 SQL로 재검증 및 시계열 정렬
   * 슬라이딩 윈도우 기법을 적용하여 PyTorch 학습용 시퀀스 데이터셋 및 DataLoader 구축
2. **PyTorch LSTM 모델링 및 성능 비교**
   * Hidden Size, Layer 등을 조정하며 LSTM 신경망 아키텍처 설계 및 학습 루프 구현
   * 기존 XGBoost 모델과의 성능 비교(Recall, F1-Score)를 통한 어블레이션(Ablation) 연구 수행
3. **프로젝트 중단 및 기술적 회고 (Critical Learning)**
   * **한계점 도출:** 제조업 예지보전 데이터 특성상 기계가 서서히 변하는 흐름보다 단발성 토크 변화가 주된 고장 요인임을 파악. LSTM은 긴 시계열 흐름을 잡는 데 최적화되어 있어 해당 데이터의 성격과 맞지 않음을 확인.
   * **결론:** 무조건 복잡한 딥러닝을 적용하는 것보다 도메인 데이터의 특성을 파악하고 적절한 알고리즘을 선택하는 것이 중요함을 깨달음. 본 프로젝트를 통해, 데이터 도메인에 대한 이해를 바탕으로 모델을 설계하는것이 좋은 설계법이라는것을 배움.

## 🚀 실행 방법 (Reference)

```bash
# 패키지 설치
pip install -r requirements.txt
```