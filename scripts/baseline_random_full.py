#!/usr/bin/env python3
"""
Random baseline: select N proteins uniformly at random from all 2941 proteins.

This is the fair comparison for the factorized DQN — both methods operate
on the full feature space without p-value pre-filtering.
Runs multiple seeds and reports mean ± std AUROC.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_processing import build_training_data
from rlfs.data.splits import train_test_split_fixed
from rlfs.evaluation.roc import compute_roc_auc


def parse_args():
    p = argparse.ArgumentParser("Random baseline — full feature space")
    p.add_argument("--disease-codes", type=str, nargs="+", required=True)
    p.add_argument("--disease-path", type=str, required=True)
    p.add_argument("--protein-paths", type=str, nargs="+", required=True)
    p.add_argument("--n-select", type=int, default=50,
                   help="Number of proteins to select randomly (matches RL episode_max_steps)")
    p.add_argument("--n-trials", type=int, default=20,
                   help="Number of random seeds to average over")
    p.add_argument("--train-size", type=int, default=15000)
    p.add_argument("--sample-n", type=int, default=20000)
    p.add_argument("--out-path", type=str, default="baseline_results_full/random_baseline.json")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    results = {}

    for code in args.disease_codes:
        print(f"\n{'='*50}")
        print(f"Disease: {code}")

        try:
            X, y, _ = build_training_data(
                disease_path=args.disease_path,
                protein_paths=args.protein_paths,
                target_code=code,
                sample_n=args.sample_n,
            )
        except ValueError as e:
            print(f"  Skipping {code}: {e}")
            continue

        n_proteins = X.shape[1]
        train_size = min(args.train_size, int(len(y) * 0.75))
        X_train, X_test, y_train, y_test = train_test_split_fixed(X, y, train_size=train_size)

        print(f"  Patients: {X.shape[0]}, Proteins: {n_proteins}, Pos rate: {y.mean():.4f}")
        print(f"  Randomly selecting {args.n_select} proteins (avg over {args.n_trials} seeds)...")

        aurocs = []
        for trial in range(args.n_trials):
            selected = random.sample(range(n_proteins), args.n_select)
            _, _, auroc, _ = compute_roc_auc(
                X_train=X_train[:, selected],
                y_train=y_train,
                X_test=X_test[:, selected],
                y_test=y_test,
            )
            aurocs.append(auroc)

        mean_auroc = float(np.mean(aurocs))
        std_auroc = float(np.std(aurocs))
        print(f"  AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")

        results[code] = {
            "mean_auroc": mean_auroc,
            "std_auroc": std_auroc,
            "n_select": args.n_select,
            "n_proteins": n_proteins,
            "n_trials": args.n_trials,
            "all_aurocs": aurocs,
        }

    print(f"\n{'='*50}")
    print("Summary:")
    for code, r in results.items():
        print(f"  {code}: AUROC={r['mean_auroc']:.4f} ± {r['std_auroc']:.4f}")

    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out_path}")


if __name__ == "__main__":
    main()
