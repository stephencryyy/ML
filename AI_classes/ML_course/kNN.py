import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from scipy.stats import mode

class MyKNNClf():
    def __init__(self, k=3, metric=None, wights='uniform'):
        self.k = k
        self.train_size = None
        self.X = None
        self.y = None
        self.metric = metric
        self.metrics = {
            'euclidean': lambda x: np.sqrt(((self.X - x) ** 2).sum(axis=1)),
            'manhattan': lambda x: np.sum(np.abs((self.X - x)),axis=1),
            'chebyshev': lambda x, _x: np.max(np.abs((x - _x))),
            'cosine': lambda x: self.cosine_distance(x)

        }
    def __repr__(self):
        class_name = self.__class__.__name__
        params = self.__dict__
        params_str = ", ".join(f"{key}={value}" for key, value in params.items())
        return f'{class_name} class: {params_str}'

    def cosine_distance(self, x):
        dot_product = np.sum(self.X * x, axis=1)
        norm_train = np.sqrt(np.sum(self.X ** 2, axis=1))
        norm_test = np.sqrt(np.sum(x ** 2))
        return 1 - (dot_product / (norm_train * norm_test))

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.X = X.values
        else:
            self.X = X

        if isinstance(y, pd.Series):
            self.y = y.values
        else:
            self.y = y

        self.train_size = (np.size(X, axis=0), np.size(X, axis=1))




    def euclidean_distance_matrix(self, X):
        # Вычисление евклидовых расстояний между всеми парами точек в обучающей выборке
        dists = np.sqrt(((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        return dists

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = []
        for test_point in X:
            # Вычисление евклидовых расстояний от тестовой точки до всех обучающих точек
            distances = np.array([self.metrics[self.metric](test_point, train_point) for train_point in self.X]) if self.metric == 'chebyshev' else self.metrics[self.metric](test_point)
            # Нахождение k ближайших соседей
            nearest_neighbor_indices = np.argsort(distances)[:self.k]
            nearest_neighbor_classes = self.y[nearest_neighbor_indices]
            # Нахождение моды (наиболее частого класса среди ближайших соседей)
            summ = np.sum(nearest_neighbor_classes)/len(nearest_neighbor_classes)
            predictions.append(1) if summ >= 0.5 else predictions.append(0)

        return np.array(predictions)




    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = []
        for test_point in X:
            # Вычисление евклидовых расстояний от тестовой точки до всех обучающих точек
            distances = np.array([self.metrics[self.metric](test_point, train_point) for train_point in
                                  self.X]) if self.metric == 'chebyshev' else self.metrics[self.metric](test_point)

            # Нахождение k ближайших соседей
            nearest_neighbor_indices = np.argsort(distances)[:self.k]
            nearest_neighbor_classes = self.y[nearest_neighbor_indices]
            # Нахождение моды (наиболее частого класса среди ближайших соседей)
            possibility = np.sum([nearest_neighbor_classes==1])/len(nearest_neighbor_classes)
            predictions.append(possibility)
        return np.array(predictions)




X, y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]
cl = MyKNNClf(k=3, metric='cosine')
cl.fit(X,y)