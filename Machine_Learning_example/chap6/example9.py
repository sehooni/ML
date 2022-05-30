import numpy as np
import numpy.linalg as lin
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# (k-평균) / 표준편차
def standardization(s):
    return (s-s.mean()) / s.std(ddof=0) # 모집단이라고 가정


def new_coordinates(X, eigenvectors):
    for i in range(eigenvectors.shape[0]):
        if i == 0:
            new = [X.dot(eigenvectors.T[i])]
        else:
            new = np.concatenate((new, [X.dot(eigenvectors.T[i])]), axis=0)
    return new.T


def main():
    # df = pd.DataFrame(columns=['x', 'y'])
    x1 = [1, 2]
    x2 = [2, 2]
    x3 = [3, 2]
    x4 = [3, 3]
    x5 = [4, 3]
    x6 = [4, 4]
    x7 = [5, 4]
    x8 = [6, 4]

    X = np.stack((x1, x2, x3, x4, x5, x6, x7, x8), axis=0)
    X = pd.DataFrame(X.T, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'])
    print(X)


    # 이때 x의 평균은 3.5, y의 평균은 3이고, 분산은 1로 하여 데이터 스케일링을 한다.
    # scaler = StandardScaler()
    # X_std = scaler.fit_transform(X)
    # X_std = pd.DataFrame({
    #     'x1': standardization(X['x1']),
    #     'x2': standardization(X['x2']),
    #     'x3': standardization(X['x3']),
    #     'x4': standardization(X['x4']),
    #     'x5': standardization(X['x5']),
    #     'x6': standardization(X['x6']),
    #     'x7': standardization(X['x7']),
    #     'x8': standardization(X['x8'])
    # })

    x1_std = [-2.5, -3]
    x2_std = [-1.5, -1]
    x3_std = [-0.5, -1]
    x4_std = [-0.5, 0]
    x5_std = [0.5, 0]
    x6_std = [0.5, 1]
    x7_std = [1.5, 1]
    x8_std = [2.5, 1]
    X_std = np.stack((x1_std, x2_std, x3_std, x4_std, x5_std, x6_std, x7_std, x8_std), axis=0)
    # X_std = pd.DataFrame(X_std.T, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'])
    print(f"X_std는 {X_std}")

    features = X_std.T
    print(f"features는 {features}")

    cov_matrix = np.cov(features)
    print(f"cov_matrix는 {cov_matrix}")

    # 고유값 고유 벡터 구하기
    # 공분산 행렬의 eigenvalue, eigenvector
    eigenvalues = lin.eig(cov_matrix)[0]
    eigenvectors = lin.eig(cov_matrix)[1]
    print(f"eigenvalues는 {eigenvalues}")
    print(f"eigenvectors는 {eigenvectors}")
    #
    # # mat = np.zeros((2, 1))
    # # # print(mat)
    # # # symmetric matrix = P*D*P.T로 분해
    # # mat[0][0] = eigenvalues[1]
    # # mat[1][1] = eigenvalues[2]
    # # mat[2][2] = eigenvalues[3]
    # # print(mat)
    #
    print('explained variance ratio:', eigenvalues/eigenvalues.sum())

    mat = np.diag(eigenvalues)
    print(f"mat는 {mat}")
    value = np.dot(np.dot(eigenvectors, mat), eigenvectors.T)
    print(f"value는 {value}")

    print(cov_matrix)

    print(X_std)
    data = new_coordinates(X_std, eigenvectors)
    print(f"data는 새로운 축에 나타난 데이터 : {data}")
    #
    # pca = MYPCA(X, 8)
    # print(f"pca를 이용한 새로운 데이터 : {pca}")


if __name__ == "__main__":
    main()