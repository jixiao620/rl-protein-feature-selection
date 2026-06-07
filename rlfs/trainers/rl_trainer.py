# rlfs/trainers/dqn_trainer.py
from __future__ import annotations

import os
import time
import json
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from rlfs.config.schema import FSConfig
from rlfs.env.feature_selection import FeatureSelectionEnv
from rlfs.models.q_network import QNetwork
from rlfs.replay.buffer import ReplayBuffer


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate_policy(
    env: FeatureSelectionEnv,
    q_net: QNetwork,
    n_episodes: int = 10,
) -> float:
    returns = []

    device = next(q_net.parameters()).device

    for _ in range(n_episodes):
        s = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            with torch.no_grad():
                qs = q_net(torch.as_tensor(s, device=device).unsqueeze(0))[0]
                legal_mask = env.legal_actions_mask()
                q_values = qs.clone()
                illegal = torch.as_tensor(~legal_mask, device=device)
                q_values[illegal] = -1e9
                a = int(torch.argmax(q_values).item())

            s, r, done, _ = env.step(a)
            ep_return += r

        returns.append(ep_return)

    return float(np.mean(returns))


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------

def save_checkpoint(
    model: Dict[str, QNetwork],
    filepath: str,
    info: Optional[Dict] = None,
) -> None:
    payload = {
        "model_state_dict": model["q"].state_dict(),
        "model_target_state_dict": model["q_target"].state_dict(),
    }
    if info is not None:
        payload.update(info)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(payload, filepath)


# ----------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------

def dqn_train(
    env: FeatureSelectionEnv,
    cfg: FSConfig,
    max_total_steps: int = 1_000_000,
    seed: int = 0,
    run_name: Optional[str] = None,
):
    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}_{run_name}" if run_name else timestamp
    checkpoint_dir = os.path.join("runs", run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(checkpoint_dir, "fs_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # ------------------------------------------------------------------
    # Init networks & buffer
    # ------------------------------------------------------------------
    obs_dim = env.obs_dim
    n_actions = env.n_actions
    device = torch.device(cfg.device)

    q = QNetwork(obs_dim, n_actions, hidden=cfg.hidden).to(device)
    q_target = QNetwork(obs_dim, n_actions, hidden=cfg.hidden).to(device)
    q_target.load_state_dict(q.state_dict())

    optimizer = Adam(q.parameters(), lr=cfg.lr)

    buffer = ReplayBuffer(
        obs_dim=obs_dim,
        n_actions=n_actions,
        capacity=cfg.buffer_size,
        device=device,
    )

    # ------------------------------------------------------------------
    # Epsilon schedule (linear decay)
    # ------------------------------------------------------------------
    def epsilon_by_step(step: int) -> float:
        frac = max(0.0, 1.0 - step / max(1, cfg.eps_decay))
        return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * frac

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    best_eval_return = -float("inf")
    evaluated_returns = []

    episode = 0
    print("Starting DQN training...")

    while True:
        episode += 1
        s = env.reset()
        done = False

        while not done:
            eps = epsilon_by_step(global_step)
            legal_mask = env.legal_actions_mask()

            if random.random() < eps:
                legal_idx = np.flatnonzero(legal_mask)
                a = int(np.random.choice(legal_idx))
            else:
                with torch.no_grad():
                    qs = q(torch.as_tensor(s, device=device).unsqueeze(0))[0]
                    q_values = qs.clone()
                    illegal = torch.as_tensor(~legal_mask, device=device)
                    q_values[illegal] = -1e9
                    a = int(torch.argmax(q_values).item())

            s2, r, done, _ = env.step(a)
            next_legal = env.legal_actions_mask()

            buffer.add(s, a, r, s2, done, next_legal)
            s = s2
            global_step += 1

            # ----------------------------------------------------------
            # Learning step
            # ----------------------------------------------------------
            if global_step > cfg.train_start and global_step % cfg.train_freq == 0:
                (
                    b_s,
                    b_a,
                    b_r,
                    b_s2,
                    b_done,
                    b_next_legal,
                ) = buffer.sample(cfg.batch_size)

                q_sa = q(b_s).gather(1, b_a)

                with torch.no_grad():
                    q_next_all = q_target(b_s2)
                    neg_inf = torch.full_like(q_next_all, -1e9)
                    q_next_masked = torch.where(
                        b_next_legal, q_next_all, neg_inf
                    )
                    q_next_max = q_next_masked.max(dim=1, keepdim=True).values
                    target = b_r + (1.0 - b_done) * cfg.gamma * q_next_max

                loss = F.smooth_l1_loss(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Target update
                if cfg.target_tau >= 1.0:
                    if global_step % cfg.target_update_every == 0:
                        q_target.load_state_dict(q.state_dict())
                else:
                    with torch.no_grad():
                        for p, p_tgt in zip(q.parameters(), q_target.parameters()):
                            p_tgt.mul_(1.0 - cfg.target_tau).add_(
                                cfg.target_tau * p
                            )

        # ------------------------------------------------------------------
        # Evaluation & checkpoint
        # ------------------------------------------------------------------
        if episode % cfg.evaluate_freq == 0:
            eval_ret = evaluate_policy(env, q, n_episodes=10)
            evaluated_returns.append((global_step, eval_ret))

            if eval_ret > best_eval_return:
                best_eval_return = eval_ret
                save_checkpoint(
                    {"q": q, "q_target": q_target},
                    filepath=os.path.join(
                        checkpoint_dir,
                        f"best_model_ep{episode}_step{global_step}.pth",
                    ),
                    info={
                        "global_step": global_step,
                        "episode": episode,
                        "eval_return": eval_ret,
                    },
                )

            np.save(
                os.path.join(checkpoint_dir, "evaluated_returns.npy"),
                np.array(evaluated_returns),
            )

            print(
                f"Episode {episode} | eval_return={eval_ret:.4f} | step={global_step}"
            )

        if (
            episode % cfg.checkpoint_save_freq == 0
            or global_step >= max_total_steps
        ):
            save_checkpoint(
                {"q": q, "q_target": q_target},
                filepath=os.path.join(
                    checkpoint_dir,
                    f"checkpoint_ep{episode}_step{global_step}.pth",
                ),
                info={
                    "global_step": global_step,
                    "episode": episode,
                },
            )

        if max_total_steps and global_step >= max_total_steps:
            break

    return {
        "q": q,
        "q_target": q_target,
        "run_dir": checkpoint_dir,
    }
