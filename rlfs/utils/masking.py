# rlfs/utils/masking.py
from __future__ import annotations

import torch


def mask_illegal_actions(q_values: torch.Tensor, legal_mask: torch.Tensor, neg_value: float = -1e9) -> torch.Tensor:
    """
    Args:
        q_values: shape (..., n_actions)
        legal_mask: bool tensor with same last dim
    Returns:
        masked q_values where illegal entries are set to neg_value
    """
    neg = torch.full_like(q_values, neg_value)
    return torch.where(legal_mask, q_values, neg)
