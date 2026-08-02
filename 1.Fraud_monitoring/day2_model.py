# ==============================================================================
# [Day 2] 신용카드 사기 탐지 모델링 (Fraud Detection System)
# 
# 📌 프로젝트 핵심 배경:
# 1. V1 ~ V28 변수: 이미 금융 보안을 위해 PCA(주성분 분석)로 암호화 및 스케일링이 완료된 상태
# 2. Time, Amount 변수: 수치 단위가 너무 크고 이상치(Outlier)가 많아 별도의 스케일링 전처리가 필수
# 3. Class (거래가 사기인가 아닌가): 0(정상)이 99.83%, 1(사기)이 0.17%인 '극심한 데이터 불균형' 상태
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score ,accuracy_score


# To-do list
# 데이터 셋 전처리
# 모델학습
# 모델 평가 F1-Score vs Accuracy 성과 비교 및 혼동 행렬


# ------------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리 (Preprocessing)
# ------------------------------------------------------------------------------

#같은 폴더에있는 데이터 가져오기
print("1. 데이터 로드중")
df = pd.read_csv('creditcard.csv')

#--------------------데이터 스케일링--------------------

print("2. Time(시간), Amount(거래금액) 변수 전처리 (RobustScaler)")
scaler = RobustScaler()
# [왜 RobustScaler를 쓰는가?]
# Standard/MinMax Scaler는 '1,000만 원 결제' 같은 극단적 이상치에 평균이 오염됨.
# RobustScaler는 '중앙값(Median)'을 기준으로 스케일링하므로 이상치에 훨씬 안정적임.

# V1~V28과 수치 단위를 맞춰주기 위해 Time과 Amount만 스케일링을 진행.
df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time'] = scaler.fit_transform(df[['Time']])

# 스케일링되기 전의 원본 컬럼(Time, Amount)은 더 이상 필요 없으므로 삭제.
df.drop(['Time', 'Amount'], axis=1, inplace=True)

#--------------------테스트 셋 만들기--------------------

# Feature(문제지: X)와 Target(정답지: y)을 분리.
X = df.drop('Class', axis=1) # class 를 제되한 30개 변수
y = df['Class']              # 사기거래인지 아닌지에 대한 정답지

# 테스트 셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# [왜 Stratified Split을 쓰는가?]
# 사기 데이터(1)가 0.17%밖에 안 되기 때문에, 그냥 무작위로 쪼개면 테스트 세트에 사기 건수가 들어가지 않는 경우가 발생.
# 'stratify=y' 옵션을 주면 학습용/테스트용 데이터셋 모두에 사기 비율(0.17%)을 똑같이 유지하며 분할할 수 있으므로, 균등한 테스트데이터셋을 만들 수 있음.

print(f" - 학습 데이터 크기: {X_train.shape}")
print(f" - 테스트 데이터 크기: {X_test.shape}")
print(f" - 테스트 데이터 내 실제 사기 건수: {y_test.sum()}건\n")


# ------------------------------------------------------------------------------
# 2. 모델 학습 (Model Training)
# ------------------------------------------------------------------------------

print("3. RandomForest 모델 학습 시작")
# [왜 RandomForest 모델을 선택했는가?]
# 여러 개의 의사결정나무(Decision Tree)를 조합한 앙상블 모델.
# 복잡한 비선형 패턴을 잘 찾고, 데이터 불균형이 있거나 변수가 많을 때도 과적합 위험이 적고 안정적.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,) # n_estimators: 트리의 개수(100개), n_jobs=-1: 모든 CPU 코어 활용
#rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,class_weight='balanced') # 이상거래에 가중치를 부여하는 옵션 (성능 하락으로인해 폐기)

rf_model.fit(X_train, y_train)

# 테스트 데이터 예측
y_pred = rf_model.predict(X_test)
print(" - 모델 학습 완료 및 테스트 데이터 예측 끝\n")


# ------------------------------------------------------------------------------
# 3. 모델 평가 (Evaluation)
# ------------------------------------------------------------------------------
# [왜 Accuracy(정확도) 대신 F1-Score를 메인으로 봐야 하는가?]
# 테스트 데이터 56,962건 중 사기는 단 98건.
# 모델이 아무것도 안 하고 "전부 정상이다"라고만 예측해도 Accuracy는 99.83%.
# 따라서 단순 정확도에 속지 않고, 정밀도(Precision)와 재현율(Recall)을 조합한 'F1-Score'로 평가해야 함.

print("================ [ 모델 평가 결과 ] ================")
print(f"1. Accuracy (정확도) : {accuracy_score(y_test, y_pred)*100:.3f}%")      # 전체 거래중 모델이 맞게 예측한 비율 (TP + TN) / (TP + TN + FP + FN)
print(f"2. Precision (정밀도): {precision_score(y_test, y_pred):.4f}")          # 모델이 사기거래라고 예측한 것중 실제로 사기거래였던 비율 TP / (TP + FP)
print(f"3. Recall (재현율)   : {recall_score(y_test, y_pred):.4f}")             # 실제 사기거래중 모델이 사기거래라고 예측한 비율 TP / (TP + FN)
print(f"4. F1-Score (F1 점수): {f1_score(y_test, y_pred):.4f}")                 # 정밀도와 재현율의 조화평균 (Precision * Recall) / (Precision + Recall)
print("===================================================\n")

# 혼동 행렬(Confusion Matrix) 출력
cm = confusion_matrix(y_test, y_pred)
print("혼동 행렬 (Confusion Matrix):")
print(f"  [ 정상(TN): {cm[0][0]}건 | 오탐(FP): {cm[0][1]}건 ]")
print(f"  [ 미탐(FN): {cm[1][0]}건 | 정탐(TP): {cm[1][1]}건 ]\n")                 # 금융사에서는 FN(실제 사기를 놓침) 에 대한 처리비용? 혹은 리스크가 가장 큼 

print("상세 분류 리포트 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Fraud(1)']))