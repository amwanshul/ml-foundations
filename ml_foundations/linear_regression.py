import numpy as np


class LinearRegression:
    """Linear regression fitted with batch gradient descent."""

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_ = None
        self.bias_ = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and match y in length")

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.epochs):
            predictions = X @ self.weights_ + self.bias_
            error = predictions - y
            loss = np.mean(error ** 2)
            self.loss_history_.append(float(loss))

            grad_w = (2.0 / n_samples) * (X.T @ error)
            grad_b = 2.0 * np.mean(error)

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

        return self

    def predict(self, X):
        if self.weights_ is None:
            raise RuntimeError("Call fit before predict")
        X = np.asarray(X, dtype=float)
        return X @ self.weights_ + self.bias_
