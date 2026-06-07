#!/usr/bin/env python3
"""
Baseline 1: Random feature selection.
Randomly select N features, train classifier, compute AUROC.
Repeat multiple times and average.
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_processing import build_training_data
from rlfs.data.pvalue import compute_logit_pvalues, select_topk_features
from rlfs.data.splits import train_test_split_fixed
from rlfs.evaluation.roc import compute_roc_auc


def parse_args():
    p = argparse.ArgumentParser("Random feature selection baseline")
    p.add_argument("--disease-codes", type=str, nargs="+", required=True)
    p.add_argument("--disease-path", type=str, required=True)
    p.add_argument("--protein-paths", type=str, nargs="+", required=True)
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--n-select", type=int, default=200)
    p.add_argument("--n-repeat", type=int, default=20)
    p.add_argument("--train-size", type=int, default=15000)
    p.add_argument("--sample-n", type=int, default=20000)
    p.add_argument("--out-dir", type=str, default="baseline_results")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Reproduce same top-k filter using I10
    print("Loading I10 data for p-value filtering...")
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

    results = {}

    for code in args.disease_codes:
        print(f"\n{'='*50}")
        print(f"Evaluating: {code} (random baseline)")

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

        aurocs = []
        for i in range(args.n_repeat):
            selected = np.random.choice(args.topk, size=args.n_select, replace=False).tolist()
            _, _, auroc, _ = compute_roc_auc(
                X_train=X_train_k[:, selected],
                y_train=y_train,
                X_test=X_test_k[:, selected],
                y_test=y_test,
            )
            aurocs.append(auroc)

        mean_auroc = float(np.mean(aurocs))
        std_auroc = float(np.std(aurocs))
        print(f"  AUROC: {mean_auroc:.4f} ± {std_auroc:.4f} (over {args.n_repeat} runs)")
        results[code] = {"auroc_mean": mean_auroc, "auroc_std": std_auroc, "n_select": args.n_select}

    print(f"\n{'='*50}")
    print("Summary (Random Baseline):")
    for code, res in results.items():
        print(f"  {code}: AUROC={res['auroc_mean']:.4f} ± {res['auroc_std']:.4f}")

    with open(os.path.join(args.out_dir, "random_baseline.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out_dir}/random_baseline.json")


if __name__ == "__main__":
    main()
