#!/usr/bin/env python3
"""
Correlation analysis and minimal set identification for RLFS selected proteins.

Steps:
1. Load protein expression for the 150 proteins selected by seeds 42/1/2
2. Compute pairwise Spearman correlation matrix
3. Hierarchical clustering → find correlated groups
4. Pick representative per cluster → minimal non-redundant set
5. Evaluate minimal set AUROC on the 4 test diseases
6. Save results + annotated protein list
"""

import sys, os
sys.path.insert(0, "/work/jl1401/rl_feature_selection")

import numpy as np
import pandas as pd
import json
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.data_processing import build_training_data
from rlfs.data.splits import train_test_split_fixed
from rlfs.evaluation.roc import compute_roc_auc

ORIG = "/work/jl1401/icl_protein_disease/data/data/Original_extracted/Original"
R    = "/work/jl1401/rl_feature_selection"
OUT  = os.path.join(R, "correlation_analysis")
os.makedirs(OUT, exist_ok=True)

protein_paths = [
    f"{ORIG}/data/xaa.gz",   f"{ORIG}/data/xaa_2.gz",
    f"{ORIG}/data/xab.gz",   f"{ORIG}/data/xac.gz",
    f"{ORIG}/data/xad.gz",   f"{ORIG}/data/xae.gz",
    f"{ORIG}/data/xaf.gz",   f"{ORIG}/data/xag.gz",
    f"{ORIG}/data/xah.gz",   f"{ORIG}/data/xai.gz",
    f"{ORIG}/data/xaj.gz",
]
disease_path = f"{ORIG}/data/binary_csv.gz"

# --- 1. Load selected protein indices ---
dirs = {
    "seed42": f"{R}/generalization_results_factorized_final",
    "seed1":  f"{R}/generalization_results_seed1",
    "seed2":  f"{R}/generalization_results_seed2",
}
# Each seed selects same proteins for all diseases; use I11
seed_selected = {}
for s, d in dirs.items():
    seed_selected[s] = np.load(f"{d}/selected_I11.npy").tolist()

all_idx = sorted(set(seed_selected["seed42"] + seed_selected["seed1"] + seed_selected["seed2"]))
print(f"Total unique proteins across 3 seeds: {len(all_idx)}")

# Load feature names
feat_cols = np.load(f"{R}/feat_cols.npy")
selected_names = [feat_cols[i] for i in all_idx]
selected_genes = [n.split(":")[0] for n in selected_names]
selected_uniprot = [n.split(":")[1] if ":" in n else "" for n in selected_names]

# --- 2. Load protein expression for these proteins only ---
print("Loading protein expression data (I10 as reference disease, large sample)...")
X_all, y_all, feat_cols_loaded = build_training_data(
    disease_path=disease_path,
    protein_paths=protein_paths,
    target_code="I10",
    sample_n=40000,  # use as many patients as possible for stable correlations
)
print(f"  Loaded: {X_all.shape[0]} patients x {X_all.shape[1]} proteins")

# Extract just the selected columns
X_sel = X_all[:, all_idx]
print(f"  Selected subset: {X_sel.shape}")

# --- 3. Compute Spearman correlation matrix ---
print("Computing Spearman correlation matrix...")
corr_matrix, _ = spearmanr(X_sel)
if np.isscalar(corr_matrix):
    corr_matrix = np.array([[1.0]])
print(f"  Correlation matrix: {corr_matrix.shape}")

# Save raw correlation
np.save(f"{OUT}/corr_matrix.npy", corr_matrix)

# --- 4. Plot heatmap (full) ---
print("Plotting correlation heatmap...")
fig, ax = plt.subplots(figsize=(20, 18))
# Sort by seed membership for visual clarity
seed_label = []
for i in all_idx:
    if i in seed_selected["seed42"]: seed_label.append(0)
    elif i in seed_selected["seed1"]: seed_label.append(1)
    else: seed_label.append(2)
sort_order = np.argsort(seed_label)
corr_sorted = corr_matrix[np.ix_(sort_order, sort_order)]
genes_sorted = [selected_genes[i] for i in sort_order]
labels_sorted = [seed_label[i] for i in sort_order]

im = ax.imshow(corr_sorted, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
plt.colorbar(im, ax=ax, label="Spearman r")
ax.set_xticks([])
ax.set_yticks([])
# Color bars for seed membership
colors = ["#2196F3", "#FF9800", "#4CAF50"]
for j, lbl in enumerate(labels_sorted):
    ax.add_patch(plt.Rectangle((j, -2), 1, 2, color=colors[lbl], clip_on=False))
ax.set_title("Spearman correlation: 150 proteins selected across 3 seeds\n(Blue=seed42, Orange=seed1, Green=seed2)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap_full.png", dpi=150)
plt.close()
print(f"  Saved: correlation_heatmap_full.png")

# --- 5. Cross-seed correlation: are seed42 proteins correlated with seed1/2 proteins? ---
print("\nCross-seed correlation analysis...")
s42 = [i for i, idx in enumerate(all_idx) if idx in seed_selected["seed42"]]
s1  = [i for i, idx in enumerate(all_idx) if idx in seed_selected["seed1"]]
s2  = [i for i, idx in enumerate(all_idx) if idx in seed_selected["seed2"]]

corr_42_1  = corr_matrix[np.ix_(s42, s1)]
corr_42_2  = corr_matrix[np.ix_(s42, s2)]
corr_1_2   = corr_matrix[np.ix_(s1, s2)]
corr_42_42 = corr_matrix[np.ix_(s42, s42)]

def off_diag(m):
    n = m.shape[0]
    mask = ~np.eye(n, dtype=bool) if m.shape[0] == m.shape[1] else np.ones(m.shape, dtype=bool)
    return np.abs(m[mask])

print(f"  Within-seed42  |r|: mean={off_diag(corr_42_42).mean():.3f}, max={off_diag(corr_42_42).max():.3f}")
print(f"  Seed42 vs Seed1 |r|: mean={np.abs(corr_42_1).mean():.3f}, max={np.abs(corr_42_1).max():.3f}")
print(f"  Seed42 vs Seed2 |r|: mean={np.abs(corr_42_2).mean():.3f}, max={np.abs(corr_42_2).max():.3f}")
print(f"  Seed1  vs Seed2 |r|: mean={np.abs(corr_1_2).mean():.3f},  max={np.abs(corr_1_2).max():.3f}")

# Find highly correlated cross-seed pairs (r > 0.6)
cross_pairs = []
for ia, ga in zip(s42, [selected_genes[i] for i in s42]):
    for ib, gb in zip(s1, [selected_genes[i] for i in s1]):
        r = corr_matrix[ia, ib]
        if abs(r) > 0.6:
            cross_pairs.append((abs(r), r, ga, gb, "42_vs_1"))
for ia, ga in zip(s42, [selected_genes[i] for i in s42]):
    for ib, gb in zip(s2, [selected_genes[i] for i in s2]):
        r = corr_matrix[ia, ib]
        if abs(r) > 0.6:
            cross_pairs.append((abs(r), r, ga, gb, "42_vs_2"))
for ia, ga in zip(s1, [selected_genes[i] for i in s1]):
    for ib, gb in zip(s2, [selected_genes[i] for i in s2]):
        r = corr_matrix[ia, ib]
        if abs(r) > 0.6:
            cross_pairs.append((abs(r), r, ga, gb, "1_vs_2"))

cross_pairs.sort(reverse=True)
print(f"\n  Highly correlated cross-seed pairs (|r|>0.6): {len(cross_pairs)}")
for abs_r, r, ga, gb, pair in cross_pairs[:20]:
    print(f"    {ga} — {gb}  r={r:.3f}  ({pair})")

# --- 6. Hierarchical clustering → minimal set ---
print("\nHierarchical clustering for minimal set...")
# Convert correlation to distance: d = 1 - |r|
dist_matrix = 1 - np.abs(corr_matrix)
np.fill_diagonal(dist_matrix, 0)
condensed = squareform(dist_matrix, checks=False)
Z = linkage(condensed, method="average")

# Try different thresholds
results_by_thresh = {}
for thresh in [0.2, 0.3, 0.4]:  # d < thresh → |r| > 1-thresh
    labels = fcluster(Z, t=thresh, criterion="distance")
    n_clusters = labels.max()
    results_by_thresh[thresh] = (n_clusters, labels)
    print(f"  Threshold d<{thresh} (|r|>{1-thresh:.1f}): {n_clusters} clusters")

# Use d<0.3 (|r|>0.7) as the main minimal set
thresh_main = 0.3
_, cluster_labels = results_by_thresh[thresh_main]
n_clusters = cluster_labels.max()

# For each cluster, pick the protein with highest mean |correlation| to all others in cluster
minimal_set_idx = []  # indices into all_idx
cluster_info = []

for c in range(1, n_clusters + 1):
    members = [i for i, lbl in enumerate(cluster_labels) if lbl == c]
    if len(members) == 1:
        rep = members[0]
    else:
        # Pick protein with highest mean |r| to others in cluster
        sub = np.abs(corr_matrix[np.ix_(members, members)])
        np.fill_diagonal(sub, 0)
        mean_r = sub.mean(axis=1)
        rep = members[np.argmax(mean_r)]
    minimal_set_idx.append(rep)

    # Determine which seeds contributed to this cluster
    seeds_in = set()
    for i in members:
        orig_idx = all_idx[i]
        if orig_idx in seed_selected["seed42"]: seeds_in.add("42")
        if orig_idx in seed_selected["seed1"]:  seeds_in.add("1")
        if orig_idx in seed_selected["seed2"]:  seeds_in.add("2")

    cluster_info.append({
        "cluster": c,
        "size": len(members),
        "representative_gene": selected_genes[rep],
        "representative_uniprot": selected_uniprot[rep],
        "representative_full": selected_names[rep],
        "representative_feat_idx": all_idx[rep],
        "members_genes": [selected_genes[m] for m in members],
        "seeds_contributing": sorted(seeds_in),
    })

minimal_feat_idx = [all_idx[i] for i in minimal_set_idx]
print(f"\nMinimal set size: {len(minimal_feat_idx)} proteins (from 150, threshold |r|>0.7)")

# Save cluster info
with open(f"{OUT}/cluster_info.json", "w") as f:
    json.dump(cluster_info, f, indent=2)

# --- 7. Evaluate minimal set AUROC on test diseases ---
print("\nEvaluating minimal set on test diseases...")
test_diseases = ["I11", "I119", "I129", "I80"]
minimal_aurocs = {}
full_aurocs = {}  # all 150 proteins

for disease in test_diseases:
    print(f"  {disease}...")
    X_dis, y_dis, _ = build_training_data(
        disease_path=disease_path,
        protein_paths=protein_paths,
        target_code=disease,
        sample_n=20000,
    )
    train_size = min(15000, int(len(y_dis) * 0.75))
    X_tr, X_te, y_tr, y_te = train_test_split_fixed(X_dis, y_dis, train_size=train_size)

    # Minimal set
    _, _, auroc_min, _ = compute_roc_auc(
        X_train=X_tr[:, minimal_feat_idx],
        y_train=y_tr,
        X_test=X_te[:, minimal_feat_idx],
        y_test=y_te,
    )
    # All 150
    _, _, auroc_all, _ = compute_roc_auc(
        X_train=X_tr[:, all_idx],
        y_train=y_tr,
        X_test=X_te[:, all_idx],
        y_test=y_te,
    )
    minimal_aurocs[disease] = auroc_min
    full_aurocs[disease] = auroc_all
    print(f"    minimal ({len(minimal_feat_idx)}): {auroc_min:.4f}  all_150: {auroc_all:.4f}")

# --- 8. Save annotated protein table ---
print("\nSaving annotated protein table...")
rows = []
for info in cluster_info:
    rows.append({
        "cluster": info["cluster"],
        "cluster_size": info["size"],
        "gene": info["representative_gene"],
        "uniprot": info["representative_uniprot"],
        "full_name": info["representative_full"],
        "panel": info["representative_full"].split(":")[4] if info["representative_full"].count(":") >= 4 else "",
        "seeds": "+".join(info["seeds_contributing"]),
        "cluster_members": ", ".join(info["members_genes"]),
        "in_seed42": info["representative_feat_idx"] in seed_selected["seed42"],
        "in_seed1": info["representative_feat_idx"] in seed_selected["seed1"],
        "in_seed2": info["representative_feat_idx"] in seed_selected["seed2"],
    })

df_out = pd.DataFrame(rows)
df_out = df_out.sort_values(["cluster_size", "gene"], ascending=[False, True])
df_out.to_csv(f"{OUT}/minimal_set_proteins.csv", index=False)

# Save summary JSON
summary = {
    "n_total_selected": len(all_idx),
    "n_minimal_set": len(minimal_feat_idx),
    "clustering_threshold_r": 0.7,
    "cross_seed_corr_42_1_mean": float(np.abs(corr_42_1).mean()),
    "cross_seed_corr_42_2_mean": float(np.abs(corr_42_2).mean()),
    "cross_seed_corr_1_2_mean":  float(np.abs(corr_1_2).mean()),
    "n_high_corr_cross_seed_pairs": len(cross_pairs),
    "minimal_set_auroc": minimal_aurocs,
    "all_150_auroc": full_aurocs,
    "minimal_set_mean_auroc": float(np.mean(list(minimal_aurocs.values()))),
    "all_150_mean_auroc": float(np.mean(list(full_aurocs.values()))),
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Total proteins (union of 3 seeds): {len(all_idx)}")
print(f"Minimal set (|r|>0.7 clustering):  {len(minimal_feat_idx)}")
print(f"\nCross-seed correlation:")
print(f"  seed42 vs seed1: mean |r| = {np.abs(corr_42_1).mean():.3f}")
print(f"  seed42 vs seed2: mean |r| = {np.abs(corr_42_2).mean():.3f}")
print(f"  seed1  vs seed2: mean |r| = {np.abs(corr_1_2).mean():.3f}")
print(f"  High-corr cross-seed pairs (|r|>0.6): {len(cross_pairs)}")
print(f"\nAUROC comparison:")
print(f"  {'Disease':<8} {'Minimal':>10} {'All 150':>10}")
for d in test_diseases:
    print(f"  {d:<8} {minimal_aurocs[d]:>10.4f} {full_aurocs[d]:>10.4f}")
print(f"  {'Mean':<8} {np.mean(list(minimal_aurocs.values())):>10.4f} {np.mean(list(full_aurocs.values())):>10.4f}")
print(f"\nResults saved to: {OUT}/")
print(f"  - minimal_set_proteins.csv")
print(f"  - cluster_info.json")
print(f"  - correlation_heatmap_full.png")
print(f"  - summary.json")
