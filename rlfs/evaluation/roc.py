# rlfs/evaluation/roc.py
from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score

from rlfs.models.q_network import QNetwork
from rlfs.env.feature_selection import FeatureSelectionEnv


def load_qnetwork_from_checkpoint(
    checkpoint_path: str,
    obs_dim: int,
    n_actions: int,
    hidden: int = 512,
    map_location: str = "cpu",
) -> QNetwork:
    ckpt = torch.load(checkpoint_path, map_location=torch.device(map_location), weights_only=False)
    q = QNetwork(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden)
    q.load_state_dict(ckpt["model_state_dict"])
    q.eval()
    return q


def rollout_selected_features(
    q_net: QNetwork,
    env: FeatureSelectionEnv,
    device: str = "cpu",
) -> List[int]:
    s = env.reset()
    done = False

    dev = torch.device(device)

    while not done:
        with torch.no_grad():
            qs = q_net(torch.as_tensor(s, device=dev).unsqueeze(0))[0]
            legal_mask = env.legal_actions_mask()
            q_values = qs.clone()
            illegal = torch.as_tensor(~legal_mask, device=dev)
            q_values[illegal] = -1e9
            a = int(torch.argmax(q_values).item())

        s, _, done, _ = env.step(a)

    return list(env.selected)


def compute_roc_auc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    svc_kernel: str = "rbf",
    svc_C: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Returns:
        fpr, tpr, roc_auc, accuracy
    """
    clf = SVC(kernel=svc_kernel, C=svc_C, probability=True)
    clf.fit(X_train, y_train)

    y_score = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, float(roc_auc), float(acc)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
) -> None:
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
