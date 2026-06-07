# rlfs/config/schema.py
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class FSConfig:
    n_feat: int
    max_steps: int

    lam: float = 0.0          # feature-count penalty in terminal reward
    gamma: float = 1.0        # episodic tasks typically use gamma=1.0

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 50_000   # steps for epsilon linear decay

    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 50_000

    target_tau: float = 0.005          # hard update when 1.0; else soft update (Polyak)
    target_update_every: int = 1000    # steps between hard updates if tau==1

    train_start: int = 5_000           # fill buffer before training
    train_freq: int = 1
    hidden: int = 512

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_save_freq: int = 5000   # episodes between saving model checkpoints
    evaluate_freq: int = 50            # episodes between evaluations
