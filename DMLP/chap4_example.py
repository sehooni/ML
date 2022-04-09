import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

# 5. hidden layer가 3개인 DMLP
# 5.2. 로지스틱 시그모이드의 출력을 구하시오.

# U Matrix 정의
u1 = np.array([
    [-0.3, 1.0, 1.2],
    [1.6, -1.0, -1.1]
])
u2 = np.array([
    [1.0, 1.0, -1.0],
    [0.7, 0.5, 1.0]
])
u3 = np.array([
    [0.5, -0.8, 1.0],
    [-0.1, 0.3, 0.4]
])
u4 = np.array([
    [1.0, 0.1, -0.2],
    [-0.2, 1.3, -0.4]
])

# Calculation the output when using sigmoid function
## Define the function of Sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))
    # return expit(z)

## progressing the calculation
x = np.array([1, 1, 0])
z1_sig = sigmoid(np.matmul(u1, x))
### 바이어스 행렬에 추가
z1_sig = np.insert(z1_sig, 0, 1)
# print(z1_sig)
z2_sig = sigmoid(np.matmul(u2, z1_sig))
z2_sig = np.insert(z2_sig, 0 ,1)
z3_sig = sigmoid(np.matmul(u3, z2_sig))
z3_sig = np.insert(z3_sig, 0 ,1)
out_sig = sigmoid(np.matmul(u4, z3_sig))
# print(out)

## 결과값 출력
print("활성함수로 로지스틱 시그모이드를 사용하였을 때의 출력값: o1={}, o2={}".format(out_sig[0],out_sig[1]))

# ## 그래프화
# plt.plot(out)

# 5.3. ReLU의 출력을 구하시오.
# Calculation the output when using ReLU function
## Define the function of ReLU
def relu(x):
    return np.maximum(0,x)

## progressing the calculation
x = np.array([1, 1, 0])
z1_relu = relu(np.matmul(u1, x))
z1_relu = np.insert(z1_relu, 0, 1)
z2_relu = relu(np.matmul(u2, z1_relu))
z2_relu = np.insert(z2_relu, 0 ,1)
z3_relu = relu(np.matmul(u3, z2_relu))
z3_relu = np.insert(z3_relu, 0 ,1)
out_relu = relu(np.matmul(u4, z3_relu))

## 결과값 출력
print("활성함수로 ReLU를 사용하였을 때의 출력값: o1={}, o2={}".format(out_relu[0],out_relu[1]))

# 5.4. 가중치를 줄일 경우, 오차에 어떠한 영향을 미치는지 설명하시오.
    # 기존 u3의 [1,3]은 1.0이었다.
    # 이를 0.9로 줄이고 결과 값을 확인하여 본다.
    # 이때 기대 출력은 [0, 1]'이다.
## 3rd hidden layer element 변경 후 새로운 변수 리스트에 저장
u3_new =np.array([
    [0.5, -0.8, 0.9],
    [-0.1, 0.3, 0.4]
])

## activation function으로 sigmoid 사용 시
## 변경후 결과 값들을 새로운 리스트들에 저장
z1_new_sig = sigmoid(np.matmul(u1, x))
z1_new_sig = np.insert(z1_new_sig, 0, 1)
z2_new_sig = sigmoid(np.matmul(u2, z1_new_sig))
z2_new_sig = np.insert(z2_new_sig, 0, 1)
z3_new_sig = sigmoid(np.matmul(u3_new, z2_new_sig))
z3_new_sig = np.insert(z3_new_sig, 0, 1)
out_new_sig = sigmoid(np.matmul(u4, z3_new_sig))

## 오차 확인
print('변경 전: 활성함수 시그모이드 사용하였을 때 오차: {}'.format(np.abs(0-out_sig[0]) + np.abs(1-out_sig[1])))
print('변경 후: 활성함수 시그모이드 사용하였을 때 오차: {}'.format(np.abs(0-out_new_sig[0]) + np.abs(1-out_new_sig[1])))

## activation function으로 ReLU 사용 시
## 변경후 결과 값들을 새로운 리스트들에 저장
z1_new_relu = relu(np.matmul(u1, x))
z1_new_relu = np.insert(z1_new_relu, 0, 1)
z2_new_relu = relu(np.matmul(u2, z1_new_relu))
z2_new_relu = np.insert(z2_new_relu, 0, 1)
z3_new_relu = relu(np.matmul(u3_new, z2_new_relu))
z3_new_relu = np.insert(z3_new_relu, 0, 1)
out_new_relu = relu(np.matmul(u4, z3_new_relu))

## 오차 확인
print('변경 전: 활성함수 시그모이드 사용하였을 때 오차: {}'.format(np.abs(0-out_relu[0]) + np.abs(1-out_relu[1])))
print('변경 후: 활성함수 시그모이드 사용하였을 때 오차: {}'.format(np.abs(0-out_new_relu[0]) + np.abs(1-out_new_relu[1])))

### 가중치를 줄이게 될 경우 오류(오차)가 더 증가함을 확인할 수 있다.