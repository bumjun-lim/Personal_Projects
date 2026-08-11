import torch
from model import PdMLSTM
from preprocess import get_pdm_dataloader  # 본인의 전처리 함수 임포트

# 1. 똑같은 뼈대의 모델 객체 생성
model = PdMLSTM(input_dim=5, hidden_dim=64, num_layers=1)

# 2. 저장해 둔 .pth 파일의 가중치 불러오기
model.load_state_dict(torch.load("model_weights.pth", weights_only=True))

# 3. 모델을 평가 모드로 전환
model.eval()

# 4. [추가] 테스트용 데이터로더에서 데이터 한 묶음(배치) 꺼내오기
test_loader = get_pdm_dataloader(
    file_path="ai4i2020.csv",window_size = 10 ,batch_size=5
)  #직접 확인용으로 5개만 가져옴
X_sample, y_sample = next(iter(test_loader))

# 5. 모델 결과 출력
with torch.no_grad():
  outputs = model(X_sample)
  print(">>> 모델이 예측한 고장 확률값들:")
  print(outputs)
  print("\n>>> 실제 정답(Label):")
  print(y_sample)
