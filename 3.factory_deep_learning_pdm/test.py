from preprocess import get_pdm_dataloader

# 필요한 거(창문 크기, 배치 크기)만 던져주면 DataLoader가 뚝딱 나옴!
dataloader = get_pdm_dataloader(
    file_path="ai4i2020.csv", window_size=10, batch_size=32
)

# 테스트 확인
for batch_X, batch_y in dataloader:
    print("로드된 배치 X 형태:", batch_X.shape)
    print("로드된 배치 y 형태:", batch_y.shape)
    break