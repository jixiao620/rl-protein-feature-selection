#!/usr/bin/env python3
"""
Multi-disease training: randomly sample one disease per episode.
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import numpy as np
import torch
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_processing import build_training_data
from rlfs.data.pvalue import compute_logit_pvalues, select_topk_features
from rlfs.data.splits import train_test_split_fixed
from rlfs.config.schema import FSConfig
from rlfs.env.feature_selection import FeatureSelectionEnv
from rlfs.rewards.accuracy import AccuracyReward
from rlfs.models.q_network import QNetwork
from rlfs.replay.buffer import ReplayBuffer
import torch.nn.functional as F
from torch.optim import Adam


def load_disease_embedding(tsv_path: str, coding: str) -> np.ndarray:
    import csv
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["coding"].strip() == coding:
                dims = [float(row[f"dim_{i}"]) for i in range(1, 11)]
                return np.array(dims, dtype=np.float32)
    raise ValueError(f"Disease coding '{coding}' not found in {tsv_path}")


def parse_args():
    p = argparse.ArgumentParser("Multi-disease RL training")
    p.add_argument("--train-disease-codes", type=str, nargs="+", required=True)
    p.add_argument("--disease-path", type=str, required=True)
    p.add_argument("--protein-paths", type=str, nargs="+", required=True)
    p.add_argument("--embedding-path", type=str, required=True)
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--train-size", type=int, default=15000)
    p.add_argument("--sample-n", type=int, default=20000)
    p.add_argument("--episode-max-steps", type=int, default=50)
    p.add_argument("--max-total-steps", type=int, default=500000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reward-model", type=str, default="logreg")
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--buffer-size", type=int, default=200000)
    p.add_argument("--target-tau", type=float, default=0.005)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--run-name", type=str, default="multidisease")
    return p.parse_args()


def epsilon_by_step(step, eps_end=0.1, eps_decay=10000):
    frac = max(0.0, 1.0 - step / max(1, eps_decay))
    return eps_end + (1.0 - eps_end) * frac


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print(f"Training diseases: {args.train_disease_codes}")
    print("=" * 60)

    # Load top-k indices using I10
    print("\nLoading I10 data for p-value filtering...")
    X_i10, y_i10, feat_cols = build_training_data(
        disease_path=args.disease_path,
        protein_paths=args.protein_paths,
        target_code="I10",
        sample_n=args.sample_n,
    )
    X_train_i10, _, y_train_i10, _ = train_test_split_fixed(
        X_i10, y_i10, train_size=args.train_size
    )
    pvals = compute_logit_pvalues(X_train_i10, y_train_i10, feat_cols)
    top_idx = select_topk_features(pvals, args.topk)
    print(f"  Top-{args.topk} protein indices from I10 p-value filter.")

    # Load each training disease
    disease_data = {}
    for code in args.train_disease_codes:
        X, y, _ = build_training_data(
            disease_path=args.disease_path,
            protein_paths=args.protein_paths,
            target_code=code,
            sample_n=args.sample_n,
        )
        X_k = X[:, top_idx]
        disease_data[code] = {
            "X": X_k,
            "y": y,
            "embedding": load_disease_embedding(args.embedding_path, code),
            "reward_fn": AccuracyReward(
                X=X_k,
                y=y,
                train_size=min(args.train_size, int(len(y) * 0.75)),
                model=args.reward_model,
            ),
        }
        print(f"  {code}: {X.shape[0]} patients, positive rate: {y.mean():.3f}")

    # Setup DQN
    emb_dim = 10
    obs_dim = args.topk + 1 + emb_dim
    n_actions = args.topk
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}, obs_dim: {obs_dim}, n_actions: {n_actions}")

    q = QNetwork(obs_dim, n_actions, hidden=args.hidden).to(device)
    q_target = QNetwork(obs_dim, n_actions, hidden=args.hidden).to(device)
    q_target.load_state_dict(q.state_dict())
    optimizer = Adam(q.parameters(), lr=args.lr)

    buffer = ReplayBuffer(
        obs_dim=obs_dim,
        n_actions=n_actions,
        capacity=args.buffer_size,
        device=device,
    )

    cfg = FSConfig(
        n_feat=args.topk,
        max_steps=args.episode_max_steps,
        lam=args.lam,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_tau=args.target_tau,
        hidden=args.hidden,
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", f"{timestamp}_{args.run_name}")
    os.makedirs(run_dir, exist_ok=True)

    global_step = 0
    episode = 0
    best_eval_return = -float("inf")
    evaluated_returns = []
    codes = args.train_disease_codes

    print("\nStarting multi-disease DQN training...")

    while global_step < args.max_total_steps:
        episode += 1
        code = random.choice(codes)
        d = disease_data[code]

        env = FeatureSelectionEnv(
            n_feat=args.topk,
            max_steps=args.episode_max_steps,
            reward_fn=d["reward_fn"],
            lam=args.lam,
            disease_embedding=d["embedding"],
        )

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
                    q_vals = qs.clone()
                    q_vals[torch.as_tensor(~legal_mask, device=device)] = -1e9
                    a = int(torch.argmax(q_vals).item())

            s2, r, done, _ = env.step(a)
            next_legal = env.legal_actions_mask()
            buffer.add(s, a, r, s2, done, next_legal)
            s = s2
            global_step += 1

            if global_step > cfg.train_start and global_step % cfg.train_freq == 0:
                b_s, b_a, b_r, b_s2, b_done, b_next_legal = buffer.sample(cfg.batch_size)
                q_sa = q(b_s).gather(1, b_a)
                with torch.no_grad():
                    q_next_all = q_target(b_s2)
                    neg_inf = torch.full_like(q_next_all, -1e9)
                    q_next_masked = torch.where(b_next_legal, q_next_all, neg_inf)
                    q_next_max = q_next_masked.max(dim=1, keepdim=True).values
                    target = b_r + (1.0 - b_done) * cfg.gamma * q_next_max
                loss = F.smooth_l1_loss(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for p, p_tgt in zip(q.parameters(), q_target.parameters()):
                        p_tgt.mul_(1.0 - cfg.target_tau).add_(cfg.target_tau * p)

        if episode % 50 == 0:
            d_eval = disease_data[codes[0]]
            env_eval = FeatureSelectionEnv(
                n_feat=args.topk,
                max_steps=args.episode_max_steps,
                reward_fn=d_eval["reward_fn"],
                lam=args.lam,
                disease_embedding=d_eval["embedding"],
            )
            returns = []
            for _ in range(10):
                s_e = env_eval.reset()
                done_e = False
                ep_r = 0.0
                while not done_e:
                    with torch.no_grad():
                        qs = q(torch.as_tensor(s_e, device=device).unsqueeze(0))[0]
                        lm = env_eval.legal_actions_mask()
                        qv = qs.clone()
                        qv[torch.as_tensor(~lm, device=device)] = -1e9
                        a_e = int(torch.argmax(qv).item())
                    s_e, r_e, done_e, _ = env_eval.step(a_e)
                    ep_r += r_e
                returns.append(ep_r)
            eval_ret = float(np.mean(returns))
            evaluated_returns.append((global_step, eval_ret))
            print(f"Episode {episode} | eval_return={eval_ret:.4f} | step={global_step} | disease={code}")

            if eval_ret > best_eval_return:
                best_eval_return = eval_ret
                torch.save({
                    "model_state_dict": q.state_dict(),
                    "model_target_state_dict": q_target.state_dict(),
                    "global_step": global_step,
                    "episode": episode,
                    "eval_return": eval_ret,
                }, os.path.join(run_dir, f"best_model_ep{episode}_step{global_step}.pth"))

            np.save(os.path.join(run_dir, "evaluated_returns.npy"), np.array(evaluated_returns))

    torch.save({
        "model_state_dict": q.state_dict(),
        "model_target_state_dict": q_target.state_dict(),
        "global_step": global_step,
        "episode": episode,
    }, os.path.join(run_dir, f"checkpoint_ep{episode}_step{global_step}.pth"))

    print(f"\nTraining complete. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
