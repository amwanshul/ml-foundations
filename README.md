# ML Foundations

A small, transparent machine-learning lab focused on understanding the mechanics behind common models instead of hiding everything behind a library call.

## What is here

- Linear regression trained with gradient descent
- Binary logistic regression trained with gradient descent
- Standardization implemented explicitly
- Classification metrics: accuracy, precision, recall and F1
- Small deterministic experiments that make the math inspectable
- Tests for the core numerical utilities

## Why this project exists

This is a learning-oriented implementation. The goal is to make the relationship between the equations, the optimization loop and the resulting predictions easy to inspect.

It is intentionally **not** a replacement for scikit-learn. Production ML should use mature, tested libraries unless there is a specific reason to implement an algorithm yourself.

## Structure

```text
ml-foundations/
├── ml_foundations/
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── preprocessing.py
│   └── metrics.py
├── examples/
│   └── binary_classification.py
├── tests/
│   └── test_core.py
├── requirements.txt
└── README.md
```

## Run

```bash
python -m pip install -r requirements.txt
python examples/binary_classification.py
python -m unittest discover -s tests -v
```

## Design notes

The implementations use NumPy for vectorized numerical operations, but the learning logic is written directly in Python:

- Linear regression minimizes mean squared error.
- Logistic regression uses the sigmoid function and binary cross-entropy.
- Gradient descent updates parameters using the gradient of the objective.
- Standardization uses statistics calculated from the training data.

## Scope

This repository is deliberately small. Future experiments can add regularization, multiclass classification, learning-rate experiments and comparisons against scikit-learn.

## License

MIT
