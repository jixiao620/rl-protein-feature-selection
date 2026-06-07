# rlfs/replay/buffer.py
from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        capacity: int,
        device: torch.device,
    ):
        self.capacity = int(capacity)
        self.storage_device = torch.device("cpu")
        self.device = device

        self.ptr = 0
        self.full = False

        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.storage_device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=self.storage_device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.storage_device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.storage_device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=self.storage_device)
        self.next_legal_masks = torch.zeros((capacity, n_actions), dtype=torch.bool, device=self.storage_device)

    def add(self, s, a, r, s2, done, next_legal_mask) -> None:
        i = self.ptr

        self.obs[i] = torch.as_tensor(s, dtype=torch.float32, device=self.storage_device)
        self.actions[i] = int(a)
        self.rewards[i] = float(r)
        self.next_obs[i] = torch.as_tensor(s2, dtype=torch.float32, device=self.storage_device)
        self.dones[i] = float(done)
        self.next_legal_masks[i] = torch.as_tensor(next_legal_mask, dtype=torch.bool, device=self.storage_device)

        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or (self.ptr == 0)

    def sample(self, batch_size: int):
        n = self.capacity if self.full else self.ptr
        if n == 0:
            raise RuntimeError("ReplayBuffer is empty; cannot sample.")

        idx = torch.randint(0, n, (batch_size,), device=self.storage_device)

        return (
            self.obs[idx].to(self.device),
            self.actions[idx].to(self.device),
            self.rewards[idx].to(self.device),
            self.next_obs[idx].to(self.device),
            self.dones[idx].to(self.device),
            self.next_legal_masks[idx].to(self.device),
        )
