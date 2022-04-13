import torch
import torch.nn as nn

# 초기 설정
## input 설정
input = torch.Tensor(1, 1, 28, 28)

## conv layer 설정
conv1 = nn.Conv2d(1, 5, 5)

## pooling layer 설정
pool = nn.MaxPool2d(2)


# 연산 진행
out = conv1(input)
out2 = pool(out)


# 출력값 사이즈 확인
print(out.size())
print(out2.size())