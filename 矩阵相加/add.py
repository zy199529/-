import numpy as np


def read_data(filename):
    matrix = []

    file = open(filename)
    for line in file:
        if line:
            element = list(map(int, line.split(',')))
            matrix.append(element)
    matrix = np.mat(matrix)
    return matrix


if __name__ == '__main__':
    matrixA = read_data('./data/matrixA.txt')
    matrixB = read_data('./data/matrixB.txt')
    matrixC = matrixA * matrixB
    x = np.sort(matrixC)  # 按降序排列
    # print(x[0][0])
    np.savetxt("./data/ans_one.txt", x, fmt="%d", delimiter="\n")
