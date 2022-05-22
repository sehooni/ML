# 2. softmax를 적용한 후 출력이 (0.001, 0.9, 0.001, 0.0098)**T 이고 레이블 정보가 (0, 0, 0, 1)**T일 때, 세 가지 목적함수, 평균제곱 오차, 교차 엔트로피, 로그우도를 계산하시오.

# 세 가지 목적함수 정의
print("세 가지 목적함수로, 평균제곱 오차와 교차 엔트로피, 로그우도 함수가 있다.")

# 평균제곱 오차 계산
import numpy as np
from sklearn.metrics import mean_squared_error

y_pred = np.array([0.001, 0.9, 0.001, 0.098])
y_real = np.array([0, 0, 0, 1])

print("평균제곱 오차 계산")
print(mean_squared_error(y_pred, y_real))
print(((0.001 - 0)**2 + (0.9 - 0)**2 + (0.001 - 0)**2 + (0.0989 - 1)**2) / 4)


# 교차 엔트로피 계산
def cross_entropy_error(y, t):
        # 로그 함수는 x = 0에서 무한대로 발산하는 함수이기 때문에 x = 0이 들어가서는 안된다.
        # 따라서, 매우 작은 값을 넣어 - 무한대가 나오는 것을 방지한다.
    delta = 1e-7
    return -np.sum(np.log2(y + delta) * t + np.log2(1-y + delta) * (1-t))

print("교차 엔트로피 오차 계산")
print(cross_entropy_error(y_pred, y_real))
print(-(np.log2(0.001 + 1e-7)*0 + np.log2(1-0.001 + 1e-7) * (1-0) + np.log2(0.9 + 1e-7)*0 + np.log2(1-0.9 + 1e-7)*(1-0) + np.log2(0.001 + 1e-7)*0 + np.log2(1-0.001 + 1e-7) * (1-0) + np.log2(0.098+1e-7)*1 + np.log2(1-0.098 + 1e-7)*(1-1)))


# 로그우도 계산
print("로그우도 계산")
print(-np.log2(0.098))