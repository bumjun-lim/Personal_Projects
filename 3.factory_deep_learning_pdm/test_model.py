import torch
from model import PdMLSTM
from preprocess import get_pdm_dataloader
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve

# 1. 모델 설정 (학습 때와 동일하게)
INPUT_DIM, HIDDEN_DIM, NUM_LAYERS = 5, 64, 1
model = PdMLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
model.load_state_dict(torch.load("best_model_weights.pth", weights_only=True))
model.eval()

# 2. 테스트 데이터 로드
_, _, test_loader = get_pdm_dataloader(file_path="ai4i2020.csv", window_size=5, batch_size=32)

# 3. 예측 수행
y_true = []
y_pred = []
all_outputs = [] # 진단용 리스트

with torch.no_grad():
    for X_test, y_test in test_loader:
        outputs = model(X_test)

        # [핵심] Logits 출력을 시그모이드 함수에 통과시켜 0~1 사이 확률로 변환
        probs = torch.sigmoid(outputs)

        #all_outputs.extend(outputs.numpy()) 
        all_outputs.extend(probs.numpy()) # Focal Loss 용 
        preds = (probs > 0.5).float() # 0.5를 임계값으로 이진 분류
        y_true.extend(y_test.numpy())
        y_pred.extend(preds.numpy())

# 진단: 모델이 뱉어낸 확률의 최댓값과 평균 확인하기
import numpy as np
print(f">>> 모델 출력 확률 최댓값: {np.max(all_outputs):.4f}, 평균값: {np.mean(all_outputs):.4f}")

# 4. 성능 지표 산출
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"--- 테스트셋 평가 결과 ---")
print(f"Precision (정밀도): {precision:.4f}")
print(f"Recall (재현율): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


precisions, recalls, thresholds = precision_recall_curve(y_true, all_outputs)

# F1-Score가 가장 높아지는 임계값 찾기
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f">>> 최적의 임계값(Threshold): {best_threshold:.4f}")
print(f">>> 그때의 최고 F1-Score: {best_f1:.4f}")