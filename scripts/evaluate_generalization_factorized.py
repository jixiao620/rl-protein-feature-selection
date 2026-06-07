#!/usr/bin/env python3
"""
Zero-shot generalization evaluation for FactorizedQNetwork.

Loads a trained factorized checkpoint, swaps the disease embedding,
runs greedy rollout to select proteins, then evaluates AUROC on test disease.
No p-value pre-filtering — uses all proteins.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_processing import build_training_data
from rlfs.data.splits import train_test_split_fixed
from rlfs.env.feature_selection import FeatureSelectionEnv
from rlfs.models.factorized_q_network import FactorizedQNetwork
from rlfs.models.film_factorized_q_network import FiLMFactorizedQNetwork
from rlfs.evaluation.roc import compute_roc_auc, plot_roc_curve


def load_disease_embedding(tsv_path: str, coding: str) -> np.ndarray:
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["coding"].strip() == coding:
                return np.array([float(row[f"dim_{i}"]) for i in range(1, 11)], dtype=np.float32)
    raise ValueError(f"Disease coding '{coding}' not found in {tsv_path}")


def parse_args():
    p = argparse.ArgumentParser("Zero-shot generalization eval — factorized model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--disease-codes", type=str, nargs="+", required=True)
    p.add_argument("--disease-path", type=str, required=True)
    p.add_argument("--protein-paths", type=str, nargs="+", required=True)
    p.add_argument("--embedding-path", type=str, required=True)
    p.add_argument("--train-size", type=int, default=15000)
    p.add_argument("--sample-n", type=int, default=20000)
    p.add_argument("--episode-max-steps", type=int, default=50)
    p.add_argument("--no-embedding", action="store_true",
                   help="Ablation: no disease embedding (state = mask+timestep only)")
    p.add_argument("--out-dir", type=str, default="generalization_results_factorized")
    return p.parse_args()


def dummy_reward_fn(selected):
    return 0.0


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    n_proteins = ckpt["n_proteins"]
    state_dim = ckpt["state_dim"]
    protein_emb_dim = ckpt.get("protein_emb_dim", 64)
    hidden = ckpt.get("hidden", 256)
    print(f"  n_proteins={n_proteins}, state_dim={state_dim}, emb_dim={protein_emb_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = ckpt.get("model_type", "standard")
    if model_type == "film":
        q_net = FiLMFactorizedQNetwork(
            state_dim=state_dim,
            n_proteins=n_proteins,
            protein_emb_dim=protein_emb_dim,
            hidden=hidden,
        )
        print(f"  Model type: FiLM-conditioned")
    else:
        q_net = FactorizedQNetwork(
            state_dim=state_dim,
            n_proteins=n_proteins,
            protein_emb_dim=protein_emb_dim,
            hidden=hidden,
        )
        print(f"  Model type: standard factorized")
    q_net.load_state_dict(ckpt["model_state_dict"])
    q_net = q_net.to(device)
    q_net.eval()
    print(f"  Running on {device}")

    results = {}

    for code in args.disease_codes:
        print(f"\n{'='*50}")
        print(f"Evaluating: {code}")

        try:
            X_dis, y_dis, _ = build_training_data(
                disease_path=args.disease_path,
                protein_paths=args.protein_paths,
                target_code=code,
                sample_n=args.sample_n,
            )
        except ValueError as e:
            print(f"  Skipping {code}: {e}")
            continue

        print(f"  Patients: {X_dis.shape[0]}, Positive rate: {y_dis.mean():.4f}")
        assert X_dis.shape[1] == n_proteins, f"Protein count mismatch: {X_dis.shape[1]} vs {n_proteins}"

        train_size = min(args.train_size, int(len(y_dis) * 0.75))
        X_train, X_test, y_train, y_test = train_test_split_fixed(X_dis, y_dis, train_size=train_size)

        if args.no_embedding:
            disease_emb = None
        else:
            disease_emb = load_disease_embedding(args.embedding_path, code)
            print(f"  Embedding norm: {np.linalg.norm(disease_emb):.4f}")

        env = FeatureSelectionEnv(
            n_feat=n_proteins,
            max_steps=args.episode_max_steps,
            reward_fn=dummy_reward_fn,
            lam=0.0,
            disease_embedding=disease_emb,
        )

        s = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                qs = q_net(torch.as_tensor(s, device=device).unsqueeze(0))[0]
                legal_mask = env.legal_actions_mask()
                q_vals = qs.clone()
                q_vals[torch.as_tensor(~legal_mask, device=device)] = -1e9
                a = int(torch.argmax(q_vals).item())
            s, _, done, _ = env.step(a)

        selected = env.selected
        print(f"  Selected {len(selected)} proteins")

        if len(selected) == 0:
            print(f"  No features selected, skipping.")
            continue

        fpr, tpr, auroc, acc = compute_roc_auc(
            X_train=X_train[:, selected],
            y_train=y_train,
            X_test=X_test[:, selected],
            y_test=y_test,
        )

        roc_path = os.path.join(args.out_dir, f"roc_{code}.png")
        plot_roc_curve(fpr, tpr, auroc, title=f"{code} ROC (zero-shot, factorized)", save_path=roc_path)

        print(f"  AUROC: {auroc:.4f}  Accuracy: {acc:.4f}")
        np.save(os.path.join(args.out_dir, f"selected_{code}.npy"), np.array(selected))
        results[code] = {"auroc": auroc, "accuracy": acc, "n_selected": len(selected)}

    print(f"\n{'='*50}")
    print("Summary:")
    for code, res in results.items():
        print(f"  {code}: AUROC={res['auroc']:.4f}  n_feat={res['n_selected']}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
