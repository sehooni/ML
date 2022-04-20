# perceptron.py 에서 linear classification을 사용함에 따라 XOR 문제를 분류하지 못하였다.
# 이를 MLP를 사용하여 XOR 문제를 해결할 수 있다.
# perceptron.py과의 차이점은 은닉층이 추가된다는 것이다.

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu' # GPU의 유무 확인

# XOR문제를 확인하기 위하여 X와 Y설정
X = torch.FloatTensor(([0, 0], [0, 1], [1, 0], [1, 1])).to(device)
Y = torch.FloatTensor(([0], [1], [1], [0])).to(device)

# Neural Net model 사용을 위한 설정
linear1 = torch.nn.Linear(2, 2, bias=True)
linear2 = torch.nn.Linear(2, 1, bias=True)
sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 실제로 약 10,000번정도 돌리기
for step in range(10001):
    optimizer.zero_grad()   # optimizer 0으로 초기화
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()
    # 출력을 100번마다 한번 씩 적용
    if step % 100 == 0:
        print(step, cost.item())

# 정확히 예측하였는지 확인
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis>0.5).float()
    accuracy = (predicted == Y).float().mean()  # accuracy가 4개밖에 없으니 다 맞으면 1이 출력된다.
    print('\nHypothesis: ', hypothesis.detach(), '\nCorrect: ', predicted.detach(), '\nAccuracy: ', accuracy.item())
