# -*- coding:UTF-8 -*-
import csv

from linear import *

if __name__ == '__main__':
    data = []
    for i in range(18):
        data.append([])

    # read data
    n_row = 0
    text = open('./data/train.csv', 'r', encoding='big5')
    row = csv.reader(text, delimiter=',')
    for r in row:
        if n_row != 0:
            for i in range(3, 27):
                if r[i] != "NR":
                    data[(n_row - 1) % 18].append(float(r[i]))
                else:
                    data[(n_row - 1) % 18].append(float(0))
        n_row = n_row + 1
    text.close
    # data 为18种特征，每个特征为24(小时)*20(天数)*12(月数)
    x = []
    y = []
    for i in range(12):  # 按PM2.5来看一个月少9个小时，因为是前9个小时预测的
        for j in range(471):
            x.append([])
            for t in range(18):
                for s in range(9):
                    x[471 * i + j].append(data[t][480 * i + j + s])
            y.append(data[9][480 * i + j + 9])
    trainX = np.array(x)  # 每一行有9*18个数 每9个代表9天的某一种污染物
    trainY = np.array(y)
    test_x = []
    n_row = 0
    text = open('./data/test.csv', 'r')
    row = csv.reader(text, delimiter=',')
    for r in row:
        if n_row % 18 == 0:
            test_x.append([])
            for i in range(2, 11):
                test_x[n_row // 18].append(float(r[i]))
        else:
            for i in range(2, 11):
                if r[i] != 'NR':
                    test_x[n_row // 18].append(float(r[i]))
                else:
                    test_x[n_row // 18].append(float(0))
        n_row = n_row + 1
    text.close()
    test_x = np.array(test_x)
    # parse anser
    ans_y = []
    n_row = 0
    text = open('data/ans.csv', "r")
    row = csv.reader(text, delimiter=",")

    for r in row:
        ans_y.append(r[1])

    ans_y = ans_y[1:]
    ans_y = np.array(list(map(int, ans_y)))
    test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis=1)
    trainX = np.concatenate((np.ones((trainX.shape[0], 1)), trainX), axis=1)
    classify = linear()
    print("start")
    w = np.zeros(len(trainX[0]))  # 162种
    w, loss_list = classify.SGD(trainX, trainY, w, eta=0.0001, iteration=20000, lambdaL2=0)

    print("end")
    y_gd = classify.predict(test_x, w)
    print(y_gd)

    ans = []
    for i in range(len(test_x)):
        ans.append(["id" + str(i)])
        a = np.dot(w, test_x[i])
        ans[i].append(a)
    filename = "./data/predict.csv"
    text = open(filename, "w+")
    s = csv.writer(text, delimiter=',', lineterminator='\n')
    for i in range(len(ans)):
        s.writerow(ans[i])
    text.close()
