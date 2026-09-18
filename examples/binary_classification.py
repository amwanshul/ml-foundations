import numpy as np

from ml_foundations.logistic_regression import LogisticRegression
from ml_foundations.metrics import accuracy, precision, recall, f1_score
from ml_foundations.preprocessing import StandardScaler


def main():
    # A tiny synthetic dataset: two features, two linearly separable classes.
    X = np.array([
        [1.0, 1.2], [1.4, 1.0], [1.2, 1.8], [1.8, 1.5],
        [4.0, 4.2], [4.5, 3.8], [5.0, 4.7], [4.2, 5.1],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(learning_rate=0.2, epochs=1500)
    model.fit(X_scaled, y)
    predictions = model.predict(X_scaled)

    print(f"accuracy : {accuracy(y, predictions):.3f}")
    print(f"precision: {precision(y, predictions):.3f}")
    print(f"recall   : {recall(y, predictions):.3f}")
    print(f"f1       : {f1_score(y, predictions):.3f}")
    print(f"final loss: {model.loss_history_[-1]:.6f}")


if __name__ == "__main__":
    main()
