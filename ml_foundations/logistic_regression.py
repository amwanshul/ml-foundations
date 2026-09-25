import numpy as np


class LogisticRegression:
    """Binary logistic regression fitted with batch gradient descent."""

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_ = None
        self.bias_ = None
        self.loss_history_ = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
            raise ValueError("X must be 2D, non-empty, and match y in length")

        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0.0
        self.loss_history_ = []

        for _ in range(self.epochs):
            logits = X @ self.weights_ + self.bias_
            probabilities = self._sigmoid(logits)

            eps = 1e-12
            loss = -np.mean(
                y * np.log(probabilities + eps)
                + (1 - y) * np.log(1 - probabilities + eps)
            )
            self.loss_history_.append(float(loss))

            error = probabilities - y
            grad_w = (X.T @ error) / n_samples
            grad_b = np.mean(error)

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

        return self

    def predict_proba(self, X):
        if self.weights_ is None:
            raise RuntimeError("Call fit before predict")
        X = np.asarray(X, dtype=float)
        return self._sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
