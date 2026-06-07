from __future__ import annotations

import torch
import torch.nn as nn


class FiLMFactorizedQNetwork(nn.Module):
    """
    Factorized Q-network with FiLM conditioning.

    Q(s, i) = state_encoder(s_base) · (protein_emb[i] * gamma(d) + beta(d))

    where d = disease embedding extracted from the last `disease_emb_dim` dims
    of the state, and (gamma, beta) are produced by a small MLP.

    This ensures the disease embedding directly modulates protein representations
    at inference time, so different diseases select different proteins.
    """

    def __init__(
        self,
        state_dim: int,
        n_proteins: int,
        protein_emb_dim: int = 64,
        hidden: int = 256,
        disease_emb_dim: int = 10,
    ):
        super().__init__()
        self.protein_emb_dim = protein_emb_dim
        self.disease_emb_dim = disease_emb_dim
        self.n_proteins = n_proteins

        # State encoder ignores disease embedding (last disease_emb_dim dims)
        base_dim = state_dim - disease_emb_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(base_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, protein_emb_dim),
        )

        # Protein embeddings (shared across diseases; FiLM adapts them)
        self.protein_emb = nn.Embedding(n_proteins, protein_emb_dim)
        nn.init.xavier_uniform_(self.protein_emb.weight)

        # FiLM network: disease_emb -> (gamma, beta), each protein_emb_dim
        self.film_net = nn.Sequential(
            nn.Linear(disease_emb_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, protein_emb_dim * 2),
        )
        # Init FiLM to identity (gamma=1, beta=0) so training starts stable
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.constant_(self.film_net[-1].bias[:protein_emb_dim], 1.0)   # gamma -> 1
        nn.init.constant_(self.film_net[-1].bias[protein_emb_dim:], 0.0)   # beta  -> 0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch, state_dim]  last disease_emb_dim dims = disease embedding
        Returns:
            q_values: [batch, n_proteins]
        """
        d = state[:, -self.disease_emb_dim:]          # [batch, 10]
        s = state[:, :-self.disease_emb_dim]           # [batch, state_dim - 10]

        state_repr = self.state_encoder(s)             # [batch, emb_dim]

        film_out = self.film_net(d)                    # [batch, emb_dim*2]
        gamma = film_out[:, :self.protein_emb_dim]     # [batch, emb_dim]
        beta  = film_out[:, self.protein_emb_dim:]     # [batch, emb_dim]

        protein_embs = self.protein_emb.weight  # [n_proteins, emb_dim]

        # Q(s,i) = state_repr · (protein_emb[i]*gamma + beta)
        #        = (state_repr*gamma) · protein_emb[i]  +  (state_repr·beta)
        # Avoids the [batch, n_proteins, emb_dim] intermediate (~1.5 GB at batch=2048).
        modulated = state_repr * gamma                          # [batch, emb_dim]
        bias      = (state_repr * beta).sum(dim=1, keepdim=True)  # [batch, 1]
        q_vals    = modulated @ protein_embs.T + bias           # [batch, n_proteins]

        return q_vals
