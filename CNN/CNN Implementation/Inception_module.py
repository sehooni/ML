import torch
import torch.nn as nn

# 초기 설정
## input 설정
input = torch.Tensor(1, 256, 28, 28)

## 3*3 conv layer 설정
conv_3 = nn.Conv2d(256, 192, 3, stride=1, padding=1)

## 5*5 conv layer 설정
conv_5 = nn.Conv2d(256, 96, 5, stride=1, padding=2)

## 3*3 pooling layer 설정
pool_3 = nn.MaxPool2d(3, stride=1, padding=1)


# 연산 진행
out_0 = conv_3(input)
out_1 = conv_5(input)
out_2 = pool_3(input)


# 출력값 사이즈 확인
print(out_0.size())
print(out_1.size())
print(out_2.size())