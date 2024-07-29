import random

import numpy as np
from sklearn.datasets import make_classification
import pandas as pd


class MyLogReg():
    def __init__(self, n_iter=10, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=None, l2_coef=None,
                 sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.metrics = {
            'accuracy': lambda x: np.trace(x) / np.sum(x),
            'precision': lambda x: x[1, 1] / np.sum(x[:, 1]) if np.sum(x[:, 1]) != 0 else 0,
            'recall': lambda x: x[1, 1] / np.sum(x[1, :]) if np.sum(x[1, :]) != 0 else 0,
            'f1': lambda x: 2 * self.metrics['precision'](x) * self.metrics['recall'](x) / (
                    self.metrics['precision'](x) + self.metrics['recall'](x)) if (
                                                                                         self.metrics['precision'](x) +
                                                                                         self.metrics['recall'](
                                                                                             x)) != 0 else 0,
            'roc_auc': lambda x, y: self.roc_auc(x, y)
        }
        self.best_metric = None
        self.reg = reg if reg is not None else 'None'
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.regs = {
            'l1': lambda x: self.l1_coef * np.sum(np.abs(x)),
            'l2': lambda x: self.l2_coef * np.sum(np.square(x)),
            'elasticnet': lambda x: self.l1_coef * np.sum(np.abs(x)) + self.l2_coef * np.sum(np.square(x)),
            'None': lambda x: 0
        }
        self.dregs = {'l1': lambda x: self.l1_coef * self.sgn(x),
                      'l2': lambda x: self.l2_coef * 2 * x,
                      'elasticnet': lambda x: self.l1_coef * self.sgn(x) + self.l2_coef * 2 * x,
                      'None': lambda x: 0
                      }
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self):
        class_name = self.__class__.__name__
        params = self.__dict__
        params_str = ", ".join(f"{key}={value}" for key, value in params.items())
        return f"{class_name} class: {params_str}"

    def roc_auc(self, y_true, y_pred):
        sorted_indices = np.argsort(y_pred)[::-1]  # Сортировка по убыванию
        y_true_sorted = y_true[sorted_indices]

        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)

        tpr = tps / np.sum(y_true_sorted)
        fpr = fps / np.sum(1 - y_true_sorted)

        return np.trapz(tpr, fpr)

    def sgn(self, w):
        transformed_matrix = np.where(w > 0, 1, np.where(w < 0, -1, 0))
        return transformed_matrix

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.dot(x, self.weights)))

    def confusion_matrix(self, y_true, y_pred, threshold=0.5):
        # Бинаризация предсказаний на основе заданного порога
        _y_class = (y_pred >= threshold).astype(int)

        # Инициализация матрицы ошибок
        cm = np.zeros((2, 2), dtype=int)

        # Заполнение матрицы ошибок
        for true_label, pred_label in zip(y_true, _y_class):
            cm[true_label, pred_label] += 1

        return cm

    def fit(self, x, y, verbose=False):
        random.seed(self.random_state)

        eps = 10 ** (-15)
        X = np.c_[np.ones(x.shape[0]), x]
        self.weights = np.ones(X.shape[1])
        _y = self.sigmoid(X)
        for i in range(1, self.n_iter+1):
            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, float):
                    indices = random.sample(range(X.shape[0]), int(len(X) * self.sgd_sample))
                else:
                    indices = random.sample(range(X.shape[0]), self.sgd_sample)

                X_batch = X[indices]
                y_batch = y.iloc[indices]
            else:
                X_batch = X
                y_batch = y

            _y_batch = self.sigmoid(X_batch)
            _y = self.sigmoid(X)
            error_matrix = self.confusion_matrix(y, _y)

            if self.metric:
                self.best_metric = self.metrics[self.metric](
                    error_matrix) if self.metric != 'roc_auc' else self.roc_auc(y, _y)

            LogLoss = -np.mean((np.log(_y + eps) * y) + ((1 - y) * np.log(1 - _y + eps))) + self.regs[self.reg](
                self.weights) if self.reg else -np.mean((np.log(_y + eps) * y) + ((1 - y) * np.log(1 - _y + eps)))

            err = _y - y if self.sgd_sample is None else _y_batch - y_batch

            grad = X_batch.T.dot(err) / len(y_batch) + self.dregs[self.reg](self.weights)

            self.weights -= self.learning_rate(i) * grad if (not isinstance(self.learning_rate, int)
            and not isinstance(self.learning_rate, float)) else self.learning_rate * grad

            if verbose and i % 10 == 0:
                print(f"{i} | loss: {LogLoss:.2f}| metric: {self.best_metric:.4f} ")

        if self.metric == 'roc_auc':
            _y = self.sigmoid(X)
            err = _y - y
            grad = X.T.dot(err) / len(y) + self.dregs[self.reg](self.weights)
            self.weights -= self.learning_rate(i) * grad if (not isinstance(self.learning_rate, int)
                                                             and not isinstance(self.learning_rate,
                                                                                float)) else self.learning_rate * grad
            self.best_metric = self.roc_auc(y, _y)

    def get_coef(self):
        return self.weights[1:]

    def predict(self, x):
        X = np.c_[np.ones(x.shape[0]), x]
        _y = self.sigmoid(X)
        return np.where(_y > 0.5, 1, 0)

    def predict_proba(self, x):
        X = np.c_[np.ones(x.shape[0]), x]
        return self.sigmoid(X)

    def get_best_score(self):
        return self.best_metric


# Генерация данных для классификации
X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, n_classes=2, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

cl = MyLogReg(n_iter=11, learning_rate= lambda iter: 0.5 * (0.85 ** iter), weights=None, metric='roc_auc')
cl.fit(X, y, verbose=10)
