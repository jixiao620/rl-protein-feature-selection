# rlfs/models/factorized_q_network.py
from __future__ import annotations

import torch
import torch.nn as nn


class FactorizedQNetwork(nn.Module):
    """
    Action-factorized Q-network for large protein feature spaces.

    Instead of Q(state) -> R^{n_proteins} (one output per protein),
    we factorize:

        state_emb = state_encoder(state)          # [batch, emb_dim]
        Q(s, i)   = state_emb · protein_emb[i]   # dot product

    This allows:
    1. Scaling to 2941 proteins without a massive output layer
    2. Parameter sharing across proteins (better generalization)
    3. Efficient computation via a single matrix multiply

    The protein embeddings are learned from scratch (random init),
    so the model discovers protein structure purely from RL signal.

    State layout: [binary_mask(n_proteins) | timestep(1) | disease_emb(10)]
    """

    def __init__(
        self,
        state_dim: int,
        n_proteins: int,
        protein_emb_dim: int = 64,
        hidden: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_proteins = n_proteins
        self.protein_emb_dim = protein_emb_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, protein_emb_dim),
        )

        # Learned embedding for each protein, randomly initialized
        self.protein_emb = nn.Embedding(n_proteins, protein_emb_dim)
        nn.init.xavier_uniform_(self.protein_emb.weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch, state_dim]
        Returns:
            q_values: [batch, n_proteins]
        """
        state_emb = self.state_encoder(state)           # [batch, emb_dim]
        protein_embs = self.protein_emb.weight           # [n_proteins, emb_dim]
        return state_emb @ protein_embs.T                # [batch, n_proteins]
