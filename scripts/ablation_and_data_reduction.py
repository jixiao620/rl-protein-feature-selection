"""
Runs three analyses in one pass (loads UKB data once, reuses):

1. Supervised upper bound: L2 LR on all 2941 proteins per held-out disease
2. Zero-shot AUROC of each seed's 50-protein panel with LR
   (comparable, apples-to-apples with ceiling)
3. Data-reduction robustness sweep: for each seed's 50-protein panel,
   fit LR at classifier train_size ∈ {500, 1500, 5000, 15000} — test set fixed

Same protocol as evaluate_generalization_factorized.py:
  - sample_n = 20000
  - train_size ordering: first N rows → train, rest → test (fixed, deterministic)
  - class weight = 'balanced' handles class imbalance
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_processing import load_binary_csv, load_protein_data, build_single_disease_table

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


HELDOUT = ["I11", "I119", "I129", "I80"]
SEED_COLS = ["seed42", "seed1", "seed2"]
DATA_ROOT = "/work/jl1401/icl_protein_disease/data/data/Original_extracted/Original/data"
DISEASE_PATH = f"{DATA_ROOT}/binary_csv.gz"
PROTEIN_PATHS = [f"{DATA_ROOT}/xa{c}.gz" for c in "abcdefghij"] + [f"{DATA_ROOT}/xaa_2.gz"]

TRAIN_SIZES = [500, 1500, 5000, 15000]
SAMPLE_N = 20000
NP_SEED = 20260921  # for the random baseline panel selection


def fit_lr_auroc(X_train, y_train, X_test, y_test):
    """L2 LR with column standardisation. Returns AUROC on test."""
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced",
        max_iter=2000, solver="liblinear",
    )
    clf.fit(scaler.transform(X_train), y_train)
    y_score = clf.predict_proba(scaler.transform(X_test))[:, 1]
    return float(roc_auc_score(y_test, y_score))


def gene_from_col(col: str) -> str:
    return col.split(":")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="analysis_new")
    ap.add_argument("--panels-csv", default="selected_proteins_3seeds.csv")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load raw data once
    print("[1/4] Loading raw UKB data ...", flush=True)
    df_patient = load_binary_csv(DISEASE_PATH)
    df_protein = load_protein_data(PROTEIN_PATHS)
    feat_cols = [c for c in df_protein.columns if c != "userID"]
    gene_to_idx = {gene_from_col(c): i for i, c in enumerate(feat_cols)}
    print(f"  feat cols: {len(feat_cols)}, unique genes in header: {len(set(gene_to_idx))}", flush=True)

    # 2. Selected panels per seed
    panels_df = pd.read_csv(args.panels_csv)
    seed_panels: dict[str, list[int]] = {}
    for scol in SEED_COLS:
        gene_col = f"{scol}_gene"
        genes = panels_df[gene_col].dropna().tolist()
        idx = []
        missing = []
        for g in genes:
            # panel may have joined-name like "CGB3_CGB5_CGB8" — try each fragment
            hit = None
            for token in [g] + g.split("_"):
                if token in gene_to_idx:
                    hit = gene_to_idx[token]; break
            if hit is None:
                missing.append(g)
            else:
                idx.append(hit)
        seed_panels[scol] = idx
        print(f"  panel {scol}: {len(idx)} matched, missing: {missing}", flush=True)

    # 3. Random baseline panel (fixed rng) — same n as seed panel (50)
    rng = np.random.default_rng(NP_SEED)
    random_panels = {f"rand{i}": rng.choice(len(feat_cols), 50, replace=False).tolist() for i in range(3)}

    all_results: dict[str, dict] = {}

    for code in HELDOUT:
        print(f"\n[2/4] Building data for {code} ...", flush=True)
        try:
            X, y, _ = build_single_disease_table(df_patient, df_protein, code, sample_n=SAMPLE_N)
        except ValueError as e:
            print(f"  skip {code}: {e}", flush=True); continue

        # Same fixed split as train_test_split_fixed: first 15000 → train, rest → test
        MAX_TRAIN = 15000
        assert len(y) > MAX_TRAIN
        X_train_full, y_train_full = X[:MAX_TRAIN], y[:MAX_TRAIN]
        X_test, y_test = X[MAX_TRAIN:], y[MAX_TRAIN:]
        print(f"  {code}: {X.shape[0]} pts, prev_train={y_train_full.mean():.4f}, prev_test={y_test.mean():.4f}", flush=True)

        code_out: dict[str, dict] = {}

        # ---- 3a: Ceiling (LR on all 2941, train=15000) ----
        print(f"  ceiling LR on all {len(feat_cols)} proteins ...", flush=True)
        auroc_ceiling = fit_lr_auroc(X_train_full, y_train_full, X_test, y_test)
        code_out["ceiling_all2941"] = auroc_ceiling
        print(f"    AUROC = {auroc_ceiling:.4f}", flush=True)

        # ---- 3b: Seed panels (LR on 50 proteins, train=15000) ----
        for scol in SEED_COLS:
            cols = seed_panels[scol]
            a = fit_lr_auroc(X_train_full[:, cols], y_train_full, X_test[:, cols], y_test)
            code_out[f"panel_{scol}"] = a
        code_out["panel_mean"] = float(np.mean([code_out[f"panel_{s}"] for s in SEED_COLS]))
        code_out["panel_std"] = float(np.std([code_out[f"panel_{s}"] for s in SEED_COLS], ddof=1))
        print(f"    panel mean AUROC = {code_out['panel_mean']:.4f} ± {code_out['panel_std']:.4f}", flush=True)

        # ---- 3c: Random panels (LR on 50 random proteins) ----
        rand_aurocs = []
        for rname, cols in random_panels.items():
            a = fit_lr_auroc(X_train_full[:, cols], y_train_full, X_test[:, cols], y_test)
            rand_aurocs.append(a)
        code_out["random_aurocs"] = rand_aurocs
        code_out["random_mean"] = float(np.mean(rand_aurocs))
        code_out["random_std"] = float(np.std(rand_aurocs, ddof=1))
        print(f"    random mean AUROC = {code_out['random_mean']:.4f} ± {code_out['random_std']:.4f}", flush=True)

        # ---- 3d: Data-reduction sweep (fixed test, varying train size) ----
        print(f"  data-reduction sweep ...", flush=True)
        dr = {}
        for tsize in TRAIN_SIZES:
            per_seed = []
            for scol in SEED_COLS:
                cols = seed_panels[scol]
                Xtr, ytr = X_train_full[:tsize, cols], y_train_full[:tsize]
                a = fit_lr_auroc(Xtr, ytr, X_test[:, cols], y_test)
                per_seed.append(a)
            dr[tsize] = {
                "per_seed": per_seed,
                "mean": float(np.mean(per_seed)),
                "std": float(np.std(per_seed, ddof=1)),
            }
            print(f"    train_size={tsize:>5}: mean={dr[tsize]['mean']:.4f} ± {dr[tsize]['std']:.4f}", flush=True)
        code_out["data_reduction"] = dr

        all_results[code] = code_out

    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[4/4] Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
