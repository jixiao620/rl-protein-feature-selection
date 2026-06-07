# rlfs/data/pvalue.py
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List, Tuple


def compute_logit_pvalues(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
):
    results: List[Tuple[int, str, float]] = []

    for i in range(X.shape[1]):
        xi = sm.add_constant(X[:, i])
        model = sm.Logit(y, xi)
        res = model.fit(disp=0)
        pval = res.pvalues[1]
        results.append((i, feature_names[i], float(pval)))

    return results


def select_topk_features(
    pvalue_results: List[Tuple[int, str, float]],
    k: int,
) -> np.ndarray:
    sorted_res = sorted(pvalue_results, key=lambda x: x[2])
    return np.array([idx for idx, _, _ in sorted_res[:k]], dtype=int)


def export_topk_npy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_idx: np.ndarray,
    prefix: str,
):
    np.save(f"{prefix}_Training_X.npy", X_train[:, selected_idx])
    np.save(f"{prefix}_Test_X.npy", X_test[:, selected_idx])
    np.save(f"{prefix}_Training_y.npy", y_train)
    np.save(f"{prefix}_Test_y.npy", y_test)
