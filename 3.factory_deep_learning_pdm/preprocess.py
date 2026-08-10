import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


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
    file_path="ai4i2020.csv", window_size=10, batch_size=32
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

    # 3. 데이터 스케일링
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])

    # 4. 슬라이딩 윈도우 시퀀스 구조화
    labels = df[target_col].values
    X_seq, y_seq = create_sequences(scaled_features, labels, window_size)

    # 5. PyTorch Dataset 및 DataLoader 생성
    dataset = PdMDataset(X_seq, y_seq)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def main():
    dataloader = get_pdm_dataloader(file_path="ai4i2020.csv", window_size=10, batch_size=32)
    # 테스트 확인
    for batch_X, batch_y in dataloader:
        print("로드된 배치 X 형태:", batch_X.shape)
        print("로드된 배치 y 형태:", batch_y.shape)
        break
if __name__ == '__main__':
    main()
   