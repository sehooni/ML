import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt

# hidden layer가 3개인 DMLP
# 1. 로지스틱 시그모이드의 출력을 구하시오.


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
z1 = sigmoid(np.matmul(u1, x))
### 바이어스 행렬에 추가
z1 = np.insert(z1, 0, 1)
# print(z1)
z2 = sigmoid(np.matmul(u2, z1))
z2 = np.insert(z2, 0 ,1)
z3 = sigmoid(np.matmul(u3, z2))
z3 = np.insert(z3, 0 ,1)
out = sigmoid(np.matmul(u4, z3))
# print(out)

## 결과값 출력
print("활성함수로 로지스틱 시그모이드를 사용하였을 때의 출력값: o1={}, o2={}".format(out[0],out[1]))

## 그래프화
plt.plot(out)

# 2. ReLU의 출력을 구하시오.
# Calculation the output when using ReLU function
## Define the function of ReLU
def relu(x):
    return np.maximum(0,x)

## progressing the calculation
x = np.array([1, 1, 0])
z1 = relu(np.matmul(u1, x))
z1 = np.insert(z1, 0, 1)
z2 = relu(np.matmul(u2, z1))
z2 = np.insert(z2, 0 ,1)
z3 = relu(np.matmul(u3, z2))
z3 = np.insert(z3, 0 ,1)
out = relu(np.matmul(u4, z3))

## 결과값 출력
print("활성함수로 ReLU를 사용하였을 때의 출력값: o1={}, o2={}".format(out[0],out[1]))
