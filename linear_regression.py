import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = 0

    def initialize_params(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def predict(self, x):
        return np.dot(x, self.w) + self.b

    def compute_cost(self, x, y):
        m = x.shape[0]
        predictions = self.predict(x)
        cost = np.sum((predictions - y) ** 2) / (2 * m)
        return cost

    def compute_gradient(self, x, y):
        m, n = x.shape
        dj_dw = np.zeros(n)
        dj_db = 0

        predictions = self.predict(x)
        errors = predictions - y

        for i in range(m):
            for j in range(n):
                dj_dw[j] += errors[i] * x[i, j]
            dj_db += errors[i]

        dj_dw /= m
        dj_db /= m

        return dj_dw, dj_db

    def fit(self, x, y, alpha=0.01, num_iters=1000, verbose=False):
        m, n = x.shape
        self.initialize_params(n)
        cost_history = []

        for i in range(num_iters):
            dj_dw, dj_db = self.compute_gradient(x, y)
            self.w -= alpha * dj_dw
            self.b -= alpha * dj_db

            cost = self.compute_cost(x, y)
            cost_history.append(cost)

            if verbose and i % max(1, num_iters // 10) == 0:
                print(f"Epoch {i:4d}: Cost = {cost:.4f}")

        return cost_history

