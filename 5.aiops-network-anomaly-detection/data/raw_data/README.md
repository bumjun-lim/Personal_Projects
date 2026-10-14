# 📂 Raw Data Directory Guide

이 디렉토리는 네트워크 이상 탐지 모델 학습 및 분석을 위한 원천 데이터(`.csv`)를 보관하는 공간입니다. 
대용량 파일의 특성상 깃허브 용량 제한으로 인해 원본 CSV 파일들은 직접 업로드되어 있지 않습니다. 

프로젝트를 정상적으로 실행하려면 아래의 디렉토리 구조에 맞추어 데이터를 수동으로 배치해야 합니다.

---

## 📁 Required Directory Structure

```
data/
└── raw_data/
    ├── .gitkeep
    ├── cic-ids-2017/
    │   └── (CIC-IDS-2017 원본 CSV 파일들 위치,2017년도는 여기서 압축 해제 해주세요.)
    └── cic-iiot-2025/ (2025년도는 압축해제한 파일의 두 종류 폴더를 잘 보고 이동시키면 됩니다.)
        ├── attack_data/
        │   └── (2025년 공격 데이터 CSV 파일들 위치)
        └── benign_data/
            └── (2025년 정상 데이터 CSV 파일들 위치)
```
---

## 📥 Data Download & Setup

1. **데이터 다운로드 링크**: [데이터셋1](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset) , [데이터셋2](https://www.kaggle.com/datasets/baderbotaiban/cic-iiot-2025-datasense-csv-mirror)
2. 위 링크에서 데이터를 다운로드합니다.
3. 압축을 해제한 뒤, 위 디렉토리 구조에 맞추어 CSV 파일들을 위치시켜 줍니다.
4. 데이터 배치가 완료되면 `src/` 디렉토리로 이동하여 전처리 스크립트를 실행해 파케(Parquet) 파일로 변환합니다.
   ```bash
   cd src
   python data_process.py
   ```
5. 마지막으로 스트림릿 구동에 필요한 4가지 파일을 만들기 위한 스크립트들을 실행해주시고, 프로젝트 폴더(상위폴더) 로 이동해주세요
   ```bash
   python union_data_analyzing.py
   python attack_response_guide.py
   python feature_mapping_utils.py
   python feature_response_guide.py
   ```
6. 이후 스트림릿 구동을 진행 할 수 있습니다.
   ```bash
   streamlit run app.py
   ```