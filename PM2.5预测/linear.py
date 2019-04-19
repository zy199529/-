import numpy as np


# 线性回归模型的类
class linear(object):
    def SGD(self, X, Y, w, eta, iteration, lambdaL2):
        list_cost = []
        for i in range(iteration):
            hypo = np.dot(X, w)
            loss = hypo - Y
            cost = np.sum(loss ** 2) / len(X)
            list_cost.append(cost)

            rand = np.random.randint(0, len(X))
            grad = X[rand] * loss[rand] / len(X) + lambdaL2 * w
            w = w - eta * grad
            print(loss)
        return w, list_cost

    def predict(self, X_test, w):
        y_pred = X_test.dot(w)
        return y_pred
