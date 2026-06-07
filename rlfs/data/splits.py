# rlfs/data/splits.py
from __future__ import annotations

import numpy as np
from typing import Tuple


def train_test_split_fixed(
    X: np.ndarray,
    y: np.ndarray,
    train_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if train_size >= len(X):
        raise ValueError("train_size must be smaller than dataset size")

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test
