# scripts/train_dqn.py
from __future__ import annotations

import argparse
import os
import numpy as np

from rlfs.config.schema import FSConfig
from rlfs.env.feature_selection import FeatureSelectionEnv
from rlfs.rewards.accuracy import AccuracyReward
from rlfs.trainers.dqn_trainer import dqn_train


def parse_args():
    p = argparse.ArgumentParser("Train DQN for RL Feature Selection")
    p.add_argument("--train-x", type=str, required=True)
    p.add_argument("--train-y", type=str, required=True)
    p.add_argument("--max-total-steps", type=int, default=1_000_000)
    p.add_argument("--episode-max-steps", type=int, default=50)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    # reward options
    p.add_argument("--reward-model", type=str, default="svc", choices=["svc", "logreg"])
    p.add_argument("--train-size", type=int, default=15000)

    # epsilon
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--eps-decay", type=int, default=10000)

    # optimization
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--target-tau", type=float, default=0.005)

    return p.parse_args()


def main():
    args = parse_args()

    X = np.load(args.train_x)
    y = np.load(args.train_y)
    n_feat = X.shape[1]

    reward = AccuracyReward(
        X=X,
        y=y,
        train_size=args.train_size,
        model=args.reward_model,
    )

    env = FeatureSelectionEnv(
        n_feat=n_feat,
        max_steps=args.episode_max_steps,
        reward_fn=reward,
        lam=0.0,
    )

    cfg = FSConfig(
        n_feat=n_feat,
        max_steps=args.max_total_steps,
    )
    cfg.lr = args.lr
    cfg.eps_end = args.eps_end
    cfg.eps_decay = args.eps_decay
    cfg.batch_size = args.batch_size
    cfg.buffer_size = args.buffer_size
    cfg.target_tau = args.target_tau

    result = dqn_train(
        env=env,
        cfg=cfg,
        max_total_steps=args.max_total_steps,
        seed=args.seed,
        run_name=args.run_name,
    )

    print("Training finished.")
    print("Run directory:", result["run_dir"])


if __name__ == "__main__":
    main()
