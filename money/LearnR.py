from math import floor

import numpy as np


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, (1 - (1e-8)))


def _shuffle(X, Y):  # X and Y are np.array
    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


def split_valid_set(X, Y, percentage):
    all_size = X.shape[0]
    valid_size = int(floor(all_size * percentage))
    X, Y = _shuffle(X, Y)
    X_valid, Y_valid = X[: valid_size], Y[: valid_size]
    X_train, Y_train = X[valid_size:], Y[valid_size:]

    return X_train, Y_train, X_valid, Y_valid


def valid(X, Y, w):
    a = np.dot(w, X.T)
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y) == y_)
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return y_


def train(X_train, Y_train):
    # valid_set_percentage = 0.1
    # X_train, Y_train, X_valid, Y_valid = split_valid_set(X, Y, valid_set_percentage)

    w = np.zeros(len(X_train[0]))

    l_rate = 0.001
    batch_size = 32
    train_dataz_size = len(X_train)
    step_num = int(floor(train_dataz_size / batch_size))
    epoch_num = 300
    list_cost = []
    total_loss = 0.0
    for epoch in range(1, epoch_num):
        total_loss = 0.0
        X_train, Y_train = _shuffle(X_train, Y_train)
        for idx in range(1, step_num):
            X = X_train[idx * batch_size:(idx + 1) * batch_size]
            Y = Y_train[idx * batch_size:(idx + 1) * batch_size]
            z = np.dot(X, w)
            y = sigmoid(z)
            cross_entropy = -1 * (
                    np.dot(np.squeeze(Y.T), np.log(y)) + np.dot((1 - np.squeeze(Y.T)), np.log(1 - y))) / len(Y)
            total_loss += cross_entropy

            grad = np.sum(-1 * X * (np.squeeze(Y) - y).reshape((batch_size, 1)), axis=0)
            w = w - l_rate * grad
        list_cost.append(total_loss)
    return w
