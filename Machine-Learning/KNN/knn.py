import numpy as np


class KNN:
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)
        if self.train_y.dtype == bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test, i_train] = sum(abs(X[i_test]-self.train_X[i_train]))
        return dists

    def compute_distances_one_loop(self, X):
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        self.train_X = np.array(self.train_X)
        X = np.array(X)
        for i_test in range(num_test):
            dists[i_test] = abs(self.train_X - X[i_test]).sum(axis=1)
        return dists

    def compute_distances_no_loops(self, X):
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = ((X[:, np.newaxis] - self.train_X).reshape(-1, X.shape[1]))
        dists = np.sum((abs(dists)), axis=1)
        dists = dists.reshape(num_test, num_train)
        return dists

    def predict_labels_binary(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, bool)
        for i in range(num_test):
            neigh_id = np.argsort(dists[i])[:self.k]
            if self.k == 1:
                pred[i] = self.train_y[neigh_id]
                continue
            pred[i] = np.sum(self.train_y[neigh_id]) >= (self.k // 2)
        return pred

    def predict_labels_multiclass(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, int)

        for i in range(num_test):
            neighbours = {}
            neigh_id = np.argsort(dists[i])[:self.k]
            for j in neigh_id:
                if self.train_y[j] in neighbours:
                    neighbours[self.train_y[j]] += 1
                else:
                    neighbours[self.train_y[j]] = 1
            pred[i] = max(neighbours, key=neighbours.get)
        return pred
