import torch
import torch.nn as nn


class PdMLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(PdMLSTM, self).__init__()

        # 1. LSTM 레이어 정의
        # input_dim: 센서 종류 개수 (5개)
        # hidden_dim: LSTM 내부의 은닉 상태(메모리) 차원 크기 (보통 32, 64 등 사용)
        # num_layers: LSTM 층을 몇 개로 쌓을 것인지
        # batch_first=True: 입력 데이터의 첫 번째 차원이 배치 크기임을 알려줌 ([배치, 시간, 특징])
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # 2. Fully Connected (선형) 레이어 정의
        # LSTM이 시간 흐름을 압축해서 뱉어낸 마지막 결과물을 받아 최종 확률값(0~1)으로 변환
        self.fc = nn.Linear(hidden_dim, output_dim)

        # 3. 시그모이드 활성화 함수 (0과 1 사이의 확률값으로 압축)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x 형태: [Batch Size, Window Size, Feature Size] (예: [32, 10, 5])

        # LSTM 통과
        # out: 각 시간대별 은닉 상태들, (hn, cn): 마지막 시점의 숨겨진 상태와 셀 상태
        out, (hn, cn) = self.lstm(x)

        # out 중에서 '가장 마지막 시간(10번째)'의 결과만 꺼냄
        # out 형태: [Batch Size, Window Size, hidden_dim] -> [Batch Size, hidden_dim]
        out = out[:, -1, :]

        # Fully Connected 레이어 통과 -> [Batch Size, 1]
        out = self.fc(out)

        # 시그모이드 통과하여 0~1 사이 확률값으로 변환
        out = self.sigmoid(out)

        return out