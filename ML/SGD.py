import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
import pandas as pd


class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg = None, l1_coef = None, l2_coef = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.metrics = {
            'mse': lambda y, _y: np.mean(np.square(y - _y)),
            'mae': lambda y, _y: np.mean(np.abs(y - _y)),
            'rmse': lambda y, _y: np.sqrt(np.mean(np.square(y - _y))),
            'r2': lambda y, _y: 1 - (np.sum(np.square(y - _y)) / (np.sum(np.square(y - np.mean(y))))),
            'mape': lambda y, _y: 100 * np.mean(np.abs((y - _y) / y))
        }
        self.last_metric = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.regs = {
            'l1': lambda x: self.l1_coef * np.sum(np.abs(x)),
            'l2': lambda x: self.l2_coef * np.sum(np.square(x)),
            'elasticnet': lambda x: self.l1_coef * np.sum(np.abs(x)) + self.l2_coef * np.sum(np.square(x))
        }
        self.dregs = {
            'l1': lambda x: self.l1_coef * self.sgn(x),
            'l2': lambda x: self.l2_coef * 2 * x,
            'elasticnet': lambda x: self.l1_coef * self.sgn(x) + self.l2_coef * 2 * x
        }

    def __repr__(self):
        class_name = self.__class__.__name__
        params = self.dict
        params_str = ", ".join(f"{key}={value}" for key, value in params.items())
        return f"{class_name} class: {params_str}"

    def Li(self,w, x, y):
        M = np.dot(w, x) * y
        return 2 / (1 + np.exp(M))

    # производная сигмоидной функции потерь по вектору w
    def gradLi(self,w, x, y):
        M = np.dot(w, x) * y
        return -2 * (1 + np.exp(M)) ** (-2) * np.exp(M) * x * y

    def sgn(self, w):
        transformed_matrix = np.where(w > 0, 1, np.where(w < 0, -1, 0))
        return transformed_matrix

    def suffle(self, x, y):
        f = np.random.permutation(len(y))
        return (x[f[1]], y[f[1]])
    def fit(self, x, y, lmb, verbose=False):
        X = np.c_[np.ones(x.shape[0]), x]
        self.weights = np.zeros(X.shape[1])


        Q = np.mean([self.Li(self.weights, a, b) for a, b in zip(X, y)])
        Q_plot = [Q]
        for i in range(self.n_iter):
            xk, yk = self.suffle(X, y)
            ek = self.Li(self.weights, xk, yk)
            self.weights = self.weights - self.learning_rate * self.gradLi(self.weights, xk, yk)
            Q = lmb * ek + (1 - lmb) * Q
            Q_plot.append(Q)
            if i % 10 == 0:
                print(f"{i} | Q: {Q}")
        return Q_plot

    def get_coef(self):
        # Проверка, что веса уже были вычислены
        if self.weights is not None:
            # Возвращение весов, начиная со второго значения
            return self.weights[1:]
        else:
            raise ValueError(
                "The model has not been trained yet. Please call the fit method before getting coefficients.")

    def predict(self, x):
        X = np.c_[np.ones(x.shape[0]), x]
        return np.dot(X, self.weights)

    def get_best_score(self):
        if self.last_metric is not None:
            return self.last_metric


cl = MyLineReg(n_iter=1000, learning_rate=0.001, metric='r2', reg='l1', l1_coef=0.1, l2_coef=0.1)

x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
X = np.array(x_train)
y = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])


N = 100

Q = cl.fit(X, y, 2/(N+1), 10)
x_plt = np.linspace(0, 100, len(Q))
plt.plot(x_plt, Q)
plt.show()


