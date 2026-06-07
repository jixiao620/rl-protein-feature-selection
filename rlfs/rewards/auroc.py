# rlfs/rewards/auroc.py
from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class AUROCReward:
    """
    AUROC-based terminal reward.

    Replaces AccuracyReward for imbalanced disease datasets where accuracy
    is uninformative (predicting all-negative gives >99% accuracy).

    reward = 10 * auroc - lam * |selected|

    Uses LogisticRegression with predict_proba for AUROC computation.
    If only one class present in test set, falls back to 0.5.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_size: int = 15_000,
        logreg_C: float = 1.0,
        random_state: int = 0,
    ):
        self.X = X
        self.y = y
        self.train_size = min(train_size, int(len(y) * 0.75))
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

        # Skip if test set has only one class
        if len(np.unique(y_test)) < 2:
            return 0.5

        clf = LogisticRegression(
            C=self.logreg_C,
            max_iter=1000,
            n_jobs=1,
            random_state=self.random_state,
        )
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]
        auroc = float(roc_auc_score(y_test, y_score))

        self._cache[key] = auroc
        return auroc

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)
