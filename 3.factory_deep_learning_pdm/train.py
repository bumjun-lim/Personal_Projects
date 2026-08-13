# ==========================================
# [TODO] PyTorch LSTM 시계열 고장 예측 모델 학습
# ==========================================

# 1. 모델 및 데이터 연동 확인
# - model.py에서 PdMLSTM 클래스 임포트 확인
# - preprocess.py에서 3차원 슬라이딩 윈도우 텐서([Batch, Window, Feature]) 및 DataLoader 구성 확인

# 2. 손실 함수(Loss Function) 선택 및 기준 정의
# [선택 기준]
# - 현재 모델의 최종 출력단에 nn.Sigmoid()가 포함되어 있으므로, 
#   0과 1 사이의 확률값을 직접 비교하는 'nn.BCELoss()'를 사용.
# - (참고) 만약 모델에서 Sigmoid를 제거하고 날것의 점수(Logits)를 반환하도록 수정한다면,
#   수치 안정성(Numerical Stability)이 더 뛰어난 'nn.BCEWithLogitsLoss()'로 변경.

# 3. 옵티마이저(Optimizer) 선택 및 기준 정의
# [선택 기준]
# - 'optim.Adam'은 모멘텀(Momentum)과 적응적 학습률(RMSprop)의 장점을 결합한 알고리즘으로,
#   시계열 및 딥러닝 학습에서 가장 무난하고 빠르게 수렴 성능을 보여주므로 채택.
# - 초기 학습률(Learning Rate)은 보통 0.001 또는 0.0001로 설정하여 오차 수렴 양상을 모니터링.

# 4. 학습 루프(Training Loop) 구현
# - model.train()을 통해 모델을 학습 모드로 전환
# - 매 반복(Iteration)마다 optimizer.zero_grad()로 이전 기울기 초기화
# - 순전파(Forward) 후 criterion을 통한 Loss 계산
# - loss.backward()를 통한 역전파(Backpropagation) 수행 및 optimizer.step()으로 가중치 업데이트
# - 지정한 Epoch 마다 Loss 수렴 여부(Loss 감소 추이) 콘솔 출력 확인

# 5. 학습 완료된 모델 가중치 저장
# - torch.save()를 활용해 학습된 모델의 state_dict를 .pth 파일로 저장

# ==========================================
# [Training Log] 
# ==========================================
# 1.모델에 시그모이드 적용 후 criterion = nn.BCELoss() -> 0.00
# 2.Train 데이터에 SMOTE 적용 후 슬라이딩 윈도우 진행 ➔ 시계열 흐름이 깨져서 여전히 0.00 (이후 SMOTE는 폐기함)
# 3.SMOTE 폐기 + 모델 시그모이드 제거 + nn.BCEWithLogitsLoss(pos_weight=pos_weight) 적용 ➔ 드디어 수치 출력 시작 (Recall 42%, F1 0.26 달성) 그럼에도 부족
# 4.윈도우사이즈를 30으로 늘리고, 고장에 대한 패널티 가중치 25 -> recall 0.7857 그러나 정밀도가 바닥을 기고있음
# 5.윈도우사이즈를 5로 줄이고, 고장에 대한 패널티 가중치 25 -> recall 0.1071 
# 6.Focal Loss적용해보기  0.000
# 7.Focal Loss, 윈도우 줄여보기 0.000

import torch.nn as nn
import torch.optim as optim
# trial 5
import torch
import torch.nn.functional as F

from model import PdMLSTM  # 어제 만든 모델 클래스
from preprocess import get_pdm_dataloader  # 전처리 및 데이터로더 함수
# Focal Loss 클래스 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # 고장 데이터에 주는 가중치 역할
        self.gamma = gamma # 쉬운 데이터를 얼마나 무시할지 결정 (보통 2.0 사용)

    def forward(self, inputs, targets):
        # inputs는 시그모이드를 거치지 않은 Logits, targets는 실제 라벨
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
     
        return torch.mean(focal_loss)


# 1. 하이퍼파라미터 설정 (세부 조절 도구들)
INPUT_DIM = 5
HIDDEN_DIM = 64
NUM_LAYERS = 1
BATCH_SIZE = 32
WINDOW_SIZE = 5
LEARNING_RATE = 0.001
EPOCHS = 50  # 테스트용

print(">>> 데이터 로딩 및 모델 초기화 중...")
# 2. 데이터 및 모델 로드
train_loader, val_loader, test_loader = get_pdm_dataloader(
    file_path="ai4i2020.csv", window_size = WINDOW_SIZE, batch_size=BATCH_SIZE
)
model = PdMLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)

# 3. 손실 함수 및 옵티마이저 설정
# - BCELoss: 최종 출력에 Sigmoid가 있으므로 확률 오차를 계산하기 위해 사용
# 만약 sigmoid를 사용하지 않고 출력되었다면 BCEWithLogitsLoss() 를 사용해 시그모이드를 적용하여 손실 계산
#criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()

#모델 고도화 과정에서, 고장에 패널티 가중치를 부과하기로 결정

# 3.1. 고장(1) 데이터에 강력한 패널티 가중치 부여 (예: 25배)
pos_weight = torch.tensor([25.0])

# 3.2. BCEWithLogitsLoss 사용 (수치 안정성이 뛰어나고 pos_weight 적용 가능)
#criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#trial 5
criterion = FocalLoss(alpha=0.75, gamma=2.0)

# - Adam: 가중치를 잘게 쪼개어 최적화하는 메인 엔진
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(">>> 학습(Training) 루프 시작")
best_val_loss = float('inf')  # 최고 성능(가장 낮은 검증 Loss)을 기록하기 위한 변수


# 4. 학습 루프 (잘게 쪼갠 미니배치 단위로 반복 학습)
for epoch in range(EPOCHS):
    # 4.1 Training 
    model.train()  # 모델을 학습 모드로 전환 (이전엔 학습만을 했지만 이번엔 검증까지 해야하므로, 에포크시작할때마다 모드전환)
    total_loss = 0
    for X_batch, y_batch in train_loader:
        # 1) 이전 스텝에서 쌓인 기울기(Gradient) 초기화
        optimizer.zero_grad()
        
        # 2) 순전파(Forward): 모델에 데이터 넣고 예측값 산출
        outputs = model(X_batch)
        
        # 3) 오차(Loss) 계산
        loss = criterion(outputs, y_batch)
        
        # 4) 역전파(Backward): 미분을 통해 각 가중치의 기여도와 수정 방향 계산
        loss.backward()
        
        # 5) 최적화(Step): 계산된 방향대로 가중치를 살짝 수정
        optimizer.step()
        
        total_loss += loss.item()
    
    # Epoch마다 오차가 잘 줄어들고 있는지 중간 점검 출력
    avg_train_loss = total_loss / len(train_loader)

    # 4.2 validation
    model.eval()  # 모델을 평가 모드로 전환 (드롭아웃 등 비활성화)
    val_loss = 0
    # 검증 시에는 가중치를 업데이트할 필요가 없으므로 gradient 계산을 끄기 (메모리 절약 및 속도 향상)
    with torch.no_grad():
        for X_val, y_val in val_loader:
            val_outputs = model(X_val)
            v_loss = criterion(val_outputs, y_val)
            val_loss += v_loss.item()
            
    avg_val_loss = val_loss / len(val_loader)

    # 4.3 print result & checkpoint save
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 검증 Loss가 이전 기록보다 더 낮아졌다면, 지금 모델이 최고 성적이라는 뜻
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model_weights.pth")
        print(f"[Checkpoint] Validation Loss 개선 최고 가중치 저장 완료 (Val Loss: {avg_val_loss:.4f})")

# 5. 학습 완료된 가중치 저장
print(">>> 최종 학습 완료 'best_model_weights.pth'로 모델 가중치가 저장되었습니다.")