from math import floor

import numpy as np


class LR(object):
    def train(self):
        return

    def sigmoid(self, z):
        res = 1 / (1.0 + np.exp(-z))
        return np.clip(res, 1e-8, (1 - (1e-8)))

    def _shuffle(self, X, Y):
        rand = np.arange(X.shape[0])

        def train(self, X_train, Y_train):
            w = np.zeros(len(X_train[0]))
            l_rate = 0.001
            batch_size = 32
            data_size = len(X_train)
            step_num = int(floor(data_size / batch_size))
            epochs_num = 300
            list_cost = []
            for epoch in range(1, epochs_num):
                total_loss = 0.0
                X_train, Y_train = _shuffle(X_train, Y_train)
            return
