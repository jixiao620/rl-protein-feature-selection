#!/usr/bin/env python3
"""
Multi-disease training with Action-Factorized DQN on full protein feature space.

Key differences from train_multidisease.py:
  - No p-value pre-filtering: all 2941 proteins are candidates
  - FactorizedQNetwork: Q(s, protein_i) = state_emb · protein_emb[i]
  - AUROCReward instead of AccuracyReward (handles class imbalance)
  - Disease hyperbolic embedding still in state (main generalization mechanism)
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import time
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_processing import build_training_data
from rlfs.data.splits import train_test_split_fixed
from rlfs.env.feature_selection import FeatureSelectionEnv
from rlfs.rewards.auroc import AUROCReward
from rlfs.models.factorized_q_network import FactorizedQNetwork
from rlfs.replay.buffer import ReplayBuffer


def load_disease_embedding(tsv_path: str, coding: str) -> np.ndarray:
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["coding"].strip() == coding:
                return np.array([float(row[f"dim_{i}"]) for i in range(1, 11)], dtype=np.float32)
    raise ValueError(f"Disease coding '{coding}' not found in {tsv_path}")


def parse_args():
    p = argparse.ArgumentParser("Factorized DQN: full-feature multi-disease training")
    p.add_argument("--train-disease-codes", type=str, nargs="+", required=True)
    p.add_argument("--disease-path", type=str, required=True)
    p.add_argument("--protein-paths", type=str, nargs="+", required=True)
    p.add_argument("--embedding-path", type=str, required=True)
    p.add_argument("--train-size", type=int, default=15000)
    p.add_argument("--sample-n", type=int, default=20000)
    p.add_argument("--episode-max-steps", type=int, default=50)
    p.add_argument("--max-total-steps", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--buffer-size", type=int, default=200000)
    p.add_argument("--target-tau", type=float, default=0.005)
    p.add_argument("--protein-emb-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=int, default=50000)
    p.add_argument("--no-embedding", action="store_true",
                   help="Ablation: disable disease embedding (state = mask+timestep only)")
    p.add_argument("--run-name", type=str, default="factorized_multidisease")
    return p.parse_args()


def epsilon_by_step(step, eps_end=0.05, eps_decay=50000):
    frac = max(0.0, 1.0 - step / max(1, eps_decay))
    return eps_end + (1.0 - eps_end) * frac


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Factorized DQN — Full Feature Space (no p-value pre-filtering)")
    print(f"Training diseases: {args.train_disease_codes}")
    print("=" * 60)

    # Load all training diseases (full feature space, no filtering)
    disease_data = {}
    n_proteins = None
    for code in args.train_disease_codes:
        print(f"\nLoading {code}...")
        X, y, feat_cols = build_training_data(
            disease_path=args.disease_path,
            protein_paths=args.protein_paths,
            target_code=code,
            sample_n=args.sample_n,
        )
        if n_proteins is None:
            n_proteins = X.shape[1]
        assert X.shape[1] == n_proteins, f"Protein count mismatch: {X.shape[1]} vs {n_proteins}"

        disease_data[code] = {
            "X": X,
            "y": y,
            "embedding": None if args.no_embedding else load_disease_embedding(args.embedding_path, code),
            "reward_fn": AUROCReward(
                X=X,
                y=y,
                train_size=args.train_size,
            ),
        }
        print(f"  {code}: {X.shape[0]} patients, {n_proteins} proteins, positive rate: {y.mean():.4f}")

    print(f"\nFull protein space: {n_proteins} proteins")

    # Network setup
    emb_dim = 0 if args.no_embedding else 10
    state_dim = n_proteins + 1 + emb_dim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"State dim: {state_dim}, Device: {device}")

    q = FactorizedQNetwork(
        state_dim=state_dim,
        n_proteins=n_proteins,
        protein_emb_dim=args.protein_emb_dim,
        hidden=args.hidden,
    ).to(device)
    q_target = FactorizedQNetwork(
        state_dim=state_dim,
        n_proteins=n_proteins,
        protein_emb_dim=args.protein_emb_dim,
        hidden=args.hidden,
    ).to(device)
    q_target.load_state_dict(q.state_dict())
    optimizer = Adam(q.parameters(), lr=args.lr, weight_decay=1e-4)

    buffer = ReplayBuffer(
        obs_dim=state_dim,
        n_actions=n_proteins,
        capacity=args.buffer_size,
        device=device,
    )

    train_start = 10_000
    train_freq = 1
    gamma = 1.0
    evaluate_freq = 50
    checkpoint_save_freq = 5000

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", f"{timestamp}_{args.run_name}")
    os.makedirs(run_dir, exist_ok=True)

    config_dict = vars(args)
    config_dict["n_proteins"] = n_proteins
    config_dict["state_dim"] = state_dim
    with open(os.path.join(run_dir, "fs_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    codes = args.train_disease_codes
    global_step = 0
    episode = 0
    best_eval_return = -float("inf")
    evaluated_returns = []

    print("\nStarting training...")

    while global_step < args.max_total_steps:
        episode += 1
        code = random.choice(codes)
        d = disease_data[code]

        env = FeatureSelectionEnv(
            n_feat=n_proteins,
            max_steps=args.episode_max_steps,
            reward_fn=d["reward_fn"],
            lam=0.0,
            disease_embedding=d["embedding"],
        )

        s = env.reset()
        done = False

        while not done:
            eps = epsilon_by_step(global_step, eps_end=args.eps_end, eps_decay=args.eps_decay)
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

            if global_step > train_start and global_step % train_freq == 0:
                b_s, b_a, b_r, b_s2, b_done, b_next_legal = buffer.sample(args.batch_size)
                q_sa = q(b_s).gather(1, b_a)
                with torch.no_grad():
                    q_next_all = q_target(b_s2)
                    neg_inf = torch.full_like(q_next_all, -1e9)
                    q_next_masked = torch.where(b_next_legal, q_next_all, neg_inf)
                    q_next_max = q_next_masked.max(dim=1, keepdim=True).values
                    target = b_r + (1.0 - b_done) * gamma * q_next_max
                loss = F.smooth_l1_loss(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), max_norm=10.0)
                optimizer.step()
                with torch.no_grad():
                    for p, p_tgt in zip(q.parameters(), q_target.parameters()):
                        p_tgt.mul_(1.0 - args.target_tau).add_(args.target_tau * p)

        if episode % evaluate_freq == 0:
            # Evaluate on first training disease
            d_eval = disease_data[codes[0]]
            env_eval = FeatureSelectionEnv(
                n_feat=n_proteins,
                max_steps=args.episode_max_steps,
                reward_fn=d_eval["reward_fn"],
                lam=0.0,
                disease_embedding=d_eval["embedding"],
            )
            returns = []
            for _ in range(20):
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
            with torch.no_grad():
                q_sample = q(torch.zeros(1, state_dim, device=device))
                q_range = (q_sample.max() - q_sample.min()).item()
            print(f"Ep {episode} | step={global_step} | eval_return={eval_ret:.4f} | q_range={q_range:.2f} | last_disease={code}")

            if eval_ret > best_eval_return:
                best_eval_return = eval_ret
                torch.save({
                    "model_state_dict": q.state_dict(),
                    "model_target_state_dict": q_target.state_dict(),
                    "global_step": global_step,
                    "episode": episode,
                    "eval_return": eval_ret,
                    "n_proteins": n_proteins,
                    "state_dim": state_dim,
                    "protein_emb_dim": args.protein_emb_dim,
                    "hidden": args.hidden,
                }, os.path.join(run_dir, f"best_model_ep{episode}_step{global_step}.pth"))

            np.save(os.path.join(run_dir, "evaluated_returns.npy"), np.array(evaluated_returns))

        if episode % checkpoint_save_freq == 0 or global_step >= args.max_total_steps:
            torch.save({
                "model_state_dict": q.state_dict(),
                "model_target_state_dict": q_target.state_dict(),
                "global_step": global_step,
                "episode": episode,
                "n_proteins": n_proteins,
                "state_dim": state_dim,
                "protein_emb_dim": args.protein_emb_dim,
                "hidden": args.hidden,
            }, os.path.join(run_dir, f"checkpoint_ep{episode}_step{global_step}.pth"))

    print(f"\nTraining complete. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
