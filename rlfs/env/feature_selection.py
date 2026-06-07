# rlfs/env/feature_selection.py
from __future__ import annotations

from typing import List, Tuple, Callable, Dict, Optional
import numpy as np


class FeatureSelectionEnv:
    """
    Combinatorial RL environment for sequential feature selection.

    State:
        - Binary mask of selected features: shape (n_feat,)
        - Normalized step position: shape (1,)
        - Disease hyperbolic embedding: shape (emb_dim,)  [optional]

    Actions:
        - 0 .. n_feat-1 : select feature i (if not selected)
        (STOP action removed - episode ends at max_steps)

    Reward:
        - 0 for intermediate steps
        - Terminal reward: 10 * acc - lam * |selected_features|
    """

    def __init__(
        self,
        n_feat: int,
        max_steps: int,
        reward_fn: Callable[[List[int]], float],
        lam: float = 0.0,
        disease_embedding: Optional[np.ndarray] = None,
    ):
        self.n_feat = n_feat
        self.n_actions = n_feat  # no STOP action
        self.max_steps = max_steps
        self.reward_fn = reward_fn
        self.lam = lam

        if disease_embedding is not None:
            self.disease_embedding = disease_embedding.astype(np.float32)
            self.emb_dim = len(disease_embedding)
        else:
            self.disease_embedding = None
            self.emb_dim = 0

        self.obs_dim = n_feat + 1 + self.emb_dim
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.selected: List[int] = []
        self.done = False
        return self._state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("step() called on terminated episode; call reset()")

        info: Dict = {}
        reward = 0.0
        self.t += 1

        if 0 <= action < self.n_feat and action not in self.selected:
            self.selected.append(action)
        # illegal action (already selected): just skip, don't terminate

        if self.t >= self.max_steps:
            self.done = True

        if len(self.selected) == self.n_feat:
            self.done = True

        if self.done:
            acc = self.reward_fn(self.selected)
            reward = 10.0 * acc - self.lam * len(self.selected)
            info["acc_val"] = acc
            info["num_features"] = len(self.selected)

        return self._state(), float(reward), bool(self.done), info

    def legal_actions_mask(self) -> np.ndarray:
        mask = np.ones(self.n_actions, dtype=bool)
        if self.selected:
            mask[self.selected] = False
        return mask

    def _state(self) -> np.ndarray:
        mask = np.zeros(self.n_feat, dtype=np.float32)
        if self.selected:
            mask[np.array(self.selected, dtype=int)] = 1.0

        pos = np.array([self.t / max(1, self.max_steps)], dtype=np.float32)

        parts = [mask, pos]
        if self.disease_embedding is not None:
            parts.append(self.disease_embedding)

        return np.concatenate(parts, axis=0)
