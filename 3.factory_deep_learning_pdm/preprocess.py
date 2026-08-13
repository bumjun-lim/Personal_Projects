import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from imblearn.over_sampling import SMOTE

# PyTorch Dataset 정의
class PdMDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) #행렬곱 형태의 연산 차원맞추기

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 슬라이딩 윈도우 함수 정의
def create_sequences(data, labels, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : (i + window_size)])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)


# 외부에서 호출할 메인 전처리 함수
def get_pdm_dataloader(
        file_path="ai4i2020.csv", window_size = 10, batch_size=32
):
    # 1. 데이터 로드
    ##db 로드 방식
    # 현재 파일 위치 기준 상위(..)로 갔다가 data 폴더 안의 DB 접근
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    #db_path = os.path.join(current_dir, "..", "data", "personal_projects.db")

    #conn = sqlite3.connect(db_path)

    ## SQL 쿼리로 sensor_logs 테이블 데이터 조회
    #query = """
    #SELECT * 
    #FROM sensor_logs
    #"""
    #df = pd.read_sql(query, conn)
    #conn.close()

    #print("--- 공용 DB에서 불러온 데이터 상위 5개 ---")
    #print(df.head())
    #'''

    # CSV 데이터 로드 방식
    df = pd.read_csv(file_path)

    # 2. 사용할 센서 특징 및 정답 컬럼 지정 
    feature_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    target_col = "Machine failure"
# 3. 전체 데이터 시간 순서대로 Train / Val / Test 쪼개기 (비율: 70% / 15% / 15%)
    total_len = len(df)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.85)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    # 4. 스케일러(Scaler)는 반드시 'Train 데이터' 기준으로만 fit 해야 합니다! (데이터 누수 방지)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(df_train[feature_cols])
    # Val과 Test는 Train 기준 스케일러로 transform만 수행
    scaled_val = scaler.transform(df_val[feature_cols])
    scaled_test = scaler.transform(df_test[feature_cols])

#    # 4.1. Train 데이터에만 SMOTE 오버샘플링 적용 
#    print(f"SMOTE 적용 전 Train 데이터 형태: {scaled_train.shape}, 고장 개수: {sum(df_train[target_col])}")
#    smote = SMOTE(random_state=42)

#    # train의 피처와 타겟을 합쳐서 리밸런싱 후 다시 분리
#    scaled_train_resampled, train_target_resampled = smote.fit_resample(
#        scaled_train, df_train[target_col].values
#    )
#    print(f"SMOTE 적용 후 Train 데이터 형태: {scaled_train_resampled.shape}, 고장 개수: {sum(train_target_resampled)}")

# SMOTE기법은, 기존 데이터의 시계열흐름을 깨기 때문에 폐기 처분하도록 결정함

    # 5. 각각 슬라이딩 윈도우 적용
#   X_train, y_train = create_sequences(scaled_train_resampled, train_target_resampled, window_size) #SMOTE기법 적용된 데이터를 학습데이터로 넘기기 
    
    X_train, y_train = create_sequences(scaled_train, df_train[target_col].values, window_size)
    X_val, y_val = create_sequences(scaled_val, df_val[target_col].values, window_size)
    X_test, y_test = create_sequences(scaled_test, df_test[target_col].values, window_size)

    # 6. PyTorch Dataset 생성
    train_dataset = PdMDataset(X_train, y_train)
    val_dataset = PdMDataset(X_val, y_val)
    test_dataset = PdMDataset(X_test, y_test)

    # 7. DataLoader 생성 (Train만 셔플을 켜고, Val과 Test는 순서대로 평가해야 하므로 shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def main():
    train_loader, val_loader, test_loader = get_pdm_dataloader(
        file_path="ai4i2020.csv", window_size=10, batch_size=32
    )
    # 테스트 확인
    for batch_X, batch_y in train_loader:
        print("로드된 배치 X 형태:", batch_X.shape)
        print("로드된 배치 y 형태:", batch_y.shape)
        break
if __name__ == '__main__':
    main()
   