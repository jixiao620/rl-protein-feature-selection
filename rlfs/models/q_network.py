# rlfs/models/q_network.py
from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Simple MLP Q-network for feature selection.

    Input:
        obs_dim = n_feat + 1

    Output:
        Q-values for each action, shape (n_actions,)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
