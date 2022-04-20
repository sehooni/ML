import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu' # GPU의 유무 확인

# XOR문제를 확인하기 위하여 X와 Y설정
X = torch.FloatTensor(([0, 0], [0, 1], [1, 0], [1, 1])).to(device)
Y = torch.FloatTensor(([0], [1], [1], [0])).to(device)

# Neural Net model 사용을 위한 설정
linear = torch.nn.Linear(2, 1, bias=True)
sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear, sigmoid).to(device) # 선언한 linear와 sigmoid 붙이기
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

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

# 결과를 보면 10000번을 돌렸는데 발전이 없음이 보인다..
# 그 말은 즉슨, 학습이 제대로 안됐다는 이야기
    # 추적을 안하도록 설정 후 체크
with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis>0.5).float()
    accuracy = (predicted == Y).float().mean()  # accuracy가 4개밖에 없으니 다 맞으면 1이 출력된다.
    print('\nHypothesis: ', hypothesis.detach(), '\nCorrect: ', predicted.detach(), '\nAccuracy: ', accuracy.item())

# 이유는 linear classification을 사용하기 때문에 학습이 안되는 것. (XOR 문제)
