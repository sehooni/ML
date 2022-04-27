import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# seed 고정
torch.manual_seed(0)

z = torch.FloatTensor([1, 2, 3])
# print(z)
hypothesis = F.softmax(z, dim=0)
print("F.softmax를 통해 구한 값:")
print(hypothesis)

print("----------------------------")    # 출력창을 깔끔하게 하기 위해 추가

# 실제 그런지 확인
from numpy import exp
# print(exp(1))
print("softmax를 (1, 2, 3)에 적용한 값:")
print(exp(1)/(exp(1) + exp(2) + exp(3)))
print(exp(2)/(exp(1) + exp(2) + exp(3)))
print(exp(3)/(exp(1) + exp(2) + exp(3)))

print("----------------------------")    # 출력창을 깔끔하게 하기 위해 추가

# 이어서 softmax 적용하여 그대로 진행
z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)

# one-hot code 적용
y = torch.randint(5, (3,)).long()
y_one_hot = torch.zeros_like(hypothesis)
print(y)
print(y_one_hot)
print(y_one_hot.scatter_(1, y.unsqueeze(1), 1))

# loss check
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

# low level로 구하기
z2 = torch.log(F.softmax(z, dim=1))
print(z2)

# 결과 확인
print((y_one_hot * -z2).sum(dim=1).mean())
print(F.nll_loss(F.log_softmax(z, dim=1), y))

# 이것도 복잡하니까 더 간단하게 만들 수 있다. → cross entropy 사용
print(F.cross_entropy(z, y))

# 이는 간단하게 확인한 거!