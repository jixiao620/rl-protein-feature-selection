# rlfs/rewards/accuracy.py
from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class AccuracyReward:
    """
    Accuracy-based terminal reward with caching.

    Usage:
        reward = AccuracyReward(X, y, model="svc")
        env = FeatureSelectionEnv(..., reward_fn=reward)

    The instance itself is callable.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: int = 15_000,
        model: str = "svc",          # "svc" | "logreg"
        svc_kernel: str = "rbf",
        svc_C: float = 1.0,
        logreg_C: float = 1.0,
        random_state: int = 0,
    ):
        self.X = X
        self.y = y
        self.train_size = train_size
        self.model = model

        self.svc_kernel = svc_kernel
        self.svc_C = svc_C
        self.logreg_C = logreg_C
        self.random_state = random_state

        self._cache: Dict[Tuple[int, ...], float] = {}

    def __call__(self, selected_features: List[int]) -> float:
        if len(selected_features) == 0:
            return -1.0

        key = tuple(sorted(selected_features))
        if key in self._cache:
            return self._cache[key]

        X_train = self.X[: self.train_size, selected_features]
        y_train = self.y[: self.train_size]
        X_test = self.X[self.train_size :, selected_features]
        y_test = self.y[self.train_size :]

        if self.model == "svc":
            clf = SVC(
                kernel=self.svc_kernel,
                C=self.svc_C,
                probability=False,
            )
        elif self.model == "logreg":
            clf = LogisticRegression(
                C=self.logreg_C,
                max_iter=1000,
                n_jobs=1,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model}")

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        self._cache[key] = float(acc)
        return float(acc)

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)
