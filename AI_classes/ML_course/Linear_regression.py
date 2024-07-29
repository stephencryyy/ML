import random
import numpy as np
from sklearn.datasets import make_regression
import pandas as pd

class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=None, l2_coef=None,
                 sgd_sample=None, random_state=42):
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
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self):
        class_name = self.__class__.__name__
        params = self.__dict__
        params_str = ", ".join(f"{key}={value}" for key, value in params.items())
        return f"{class_name} class: {params_str}"

    def sgn(self, w):
        transformed_matrix = np.where(w > 0, 1, np.where(w < 0, -1, 0))
        return transformed_matrix

    def fit(self, x, y, verbose=False):
        random.seed(self.random_state)


        X = np.c_[np.ones(x.shape[0]), x]
        self.weights = np.ones(X.shape[1])

        _y = np.dot(X, self.weights)
        mse = np.mean((_y - y) ** 2) + self.regs[self.reg](self.weights) if self.reg else np.mean((_y - y) ** 2)
        metric_value = None
        if self.metric:
            metric_value = self.metrics[self.metric](y, _y)
        if verbose:
            print(f"start | loss: {mse:.2f} | {self.metric}: {metric_value}")
        for i in range(self.n_iter):
            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, float):
                    sample_size = int(len(X) * self.sgd_sample)
                else:
                    sample_size = self.sgd_sample
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_batch = X[indices]
                y_batch = y.iloc[indices]
            else:
                X_batch = X
                y_batch = y

            _y_batch = np.dot(X_batch, self.weights)
            _y = np.dot(X, self.weights)
            errors = _y_batch - y_batch if self.sgd_sample else _y-y
            if self.sgd_sample:
                gradients = (2 * X_batch.T.dot(errors) / len(y_batch)) + self.dregs[self.reg](
                    self.weights) if self.reg else (
                        2 * X_batch.T.dot(errors) / len(y_batch))
            else:
                gradients = (2 * X.T.dot(errors) / len(y)) + self.dregs[self.reg](self.weights) if self.reg else (
                        2 * X.T.dot(errors) / len(y))
            self.weights -= self.learning_rate * gradients
            mse = np.mean(np.square(_y - y)) + self.regs[self.reg](self.weights) if self.reg else np.mean((_y - y) ** 2)
            if self.metric:
                metric_value = self.metrics[self.metric](y, _y)
            if verbose and i % verbose == 0:
                print(f"{i} | loss: {mse:.2f}|{self.metric}: {metric_value}")
        self.last_metric = metric_value

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


cl = MyLineReg(n_iter=100, learning_rate=0.1, metric='r2', reg='l1', l1_coef=0.1, l2_coef=0.1, sgd_sample=0.6)

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

cl.fit(X, y, 10)
print(np.mean(cl.get_coef()))
print(cl.get_best_score())
