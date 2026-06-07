#!/usr/bin/env python3
"""
Zero-shot generalization evaluation.

Load a trained DQN checkpoint from I10, swap the disease embedding
to a new disease, run greedy rollout to select features, then evaluate
AUROC on that disease's patient data.
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_processing import build_training_data
from rlfs.data.pvalue import compute_logit_pvalues, select_topk_features
from rlfs.data.splits import train_test_split_fixed
from rlfs.env.feature_selection import FeatureSelectionEnv
from rlfs.evaluation.roc import (
    load_qnetwork_from_checkpoint,
    compute_roc_auc,
    plot_roc_curve,
)


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
    p = argparse.ArgumentParser("Zero-shot generalization evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--disease-codes", type=str, nargs="+", required=True)
    p.add_argument("--disease-path", type=str, required=True)
    p.add_argument("--protein-paths", type=str, nargs="+", required=True)
    p.add_argument("--embedding-path", type=str, required=True)
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--train-size", type=int, default=15000)
    p.add_argument("--sample-n", type=int, default=20000)
    p.add_argument("--episode-max-steps", type=int, default=200)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--out-dir", type=str, default="generalization_results")
    return p.parse_args()


def dummy_reward_fn(selected):
    return 0.0


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading I10 data to reproduce p-value feature filtering...")
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
    print(f"  Top-{args.topk} protein indices reproduced.")

    emb_dim = 10
    obs_dim = args.topk + 1 + emb_dim
    n_actions = args.topk

    print(f"\nLoading checkpoint: {args.checkpoint}")
    q_net = load_qnetwork_from_checkpoint(
        checkpoint_path=args.checkpoint,
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden=args.hidden,
        map_location="cpu",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_net = q_net.to(device)
    q_net.eval()
    print(f"  Checkpoint loaded, running on {device}")

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

        print(f"  Patients: {X_dis.shape[0]}, Positive rate: {y_dis.mean():.3f}")

        X_dis_k = X_dis[:, top_idx]
        train_size = min(args.train_size, int(len(X_dis) * 0.75))
        X_train_k, X_test_k, y_train, y_test = train_test_split_fixed(
            X_dis_k, y_dis, train_size=train_size
        )

        disease_emb = load_disease_embedding(args.embedding_path, code)
        print(f"  Embedding norm: {np.linalg.norm(disease_emb):.4f}")

        env = FeatureSelectionEnv(
            n_feat=args.topk,
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
            X_train=X_train_k[:, selected],
            y_train=y_train,
            X_test=X_test_k[:, selected],
            y_test=y_test,
        )

        roc_path = os.path.join(args.out_dir, f"roc_{code}.png")
        plot_roc_curve(fpr, tpr, auroc, title=f"{code} ROC (zero-shot from I10)", save_path=roc_path)

        print(f"  AUROC: {auroc:.4f}  Accuracy: {acc:.4f}")
        np.save(os.path.join(args.out_dir, f"selected_{code}.npy"), np.array(selected))
        results[code] = {"auroc": auroc, "accuracy": acc, "n_selected": len(selected)}

    print(f"\n{'='*50}")
    print("Summary:")
    for code, res in results.items():
        print(f"  {code}: AUROC={res['auroc']:.4f}  Acc={res['accuracy']:.4f}  n_feat={res['n_selected']}")

    import json
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
