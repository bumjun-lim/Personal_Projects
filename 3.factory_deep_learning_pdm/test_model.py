import torch
from model import PdMLSTM
from preprocess import get_pdm_dataloader
from sklearn.metrics import f1_score, recall_score, precision_score

# 1. 모델 설정 (학습 때와 동일하게)
INPUT_DIM, HIDDEN_DIM, NUM_LAYERS = 5, 64, 1
model = PdMLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
model.load_state_dict(torch.load("best_model_weights.pth", weights_only=True))
model.eval()

# 2. 테스트 데이터 로드
_, _, test_loader = get_pdm_dataloader(file_path="ai4i2020.csv", window_size=10, batch_size=32)

# 3. 예측 수행
y_true = []
y_pred = []

with torch.no_grad():
    for X_test, y_test in test_loader:
        outputs = model(X_test)
        preds = (outputs > 0.1).float() # 0.5를 임계값으로 이진 분류
        y_true.extend(y_test.numpy())
        y_pred.extend(preds.numpy())

# 4. 성능 지표 산출
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"--- 테스트셋 평가 결과 ---")
print(f"Precision (정밀도): {precision:.4f}")
print(f"Recall (재현율): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")