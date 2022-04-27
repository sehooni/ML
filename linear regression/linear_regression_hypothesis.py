import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(2)    # seed 고정

# 1회차
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = x_train * W + b
# print(hypothesis)
cost = torch.mean((hypothesis - y_train) ** 2)
# print(cost)

optimizer = optim.SGD([W, b], lr=0.01)
optimizer.zero_grad()
cost.backward()
optimizer.step()
# print(W)
# print(b)
# print(hypothesis)
hypothesis = x_train * W + b
print(hypothesis)
cost = torch.mean((hypothesis - y_train)**2)
print(cost)

# 1번 더 돌려서 문제에서 요구한 2번째 회차를 실행.
optimizer = optim.SGD([W, b], lr=0.01)
optimizer.zero_grad()
cost.backward()
optimizer.step()
# print(W)
# print(b)
# print(hypothesis)
hypothesis = x_train * W + b
print(hypothesis)
cost = torch.mean((hypothesis - y_train)**2)
print(cost)
