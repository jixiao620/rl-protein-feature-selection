#!/usr/bin/env python3
"""
Consensus panel analysis across N seeds.
Counts how many seeds selected each protein, identifies stable proteins.
"""

import sys, os
sys.path.insert(0, "/work/jl1401/rl_feature_selection")

import numpy as np
import pandas as pd
import json

R = "/work/jl1401/rl_feature_selection"
OUT = os.path.join(R, "consensus_analysis")
os.makedirs(OUT, exist_ok=True)

feat_cols = np.load(f"{R}/feat_cols.npy")

# Collect selected proteins from all available seeds
seed_dirs = {
    42: f"{R}/generalization_results_factorized_final",
    1:  f"{R}/generalization_results_seed1",
    2:  f"{R}/generalization_results_seed2",
}
for s in range(3, 10):
    d = f"{R}/generalization_results_seed{s}"
    if os.path.isdir(d):
        seed_dirs[s] = d

print(f"Seeds found: {sorted(seed_dirs.keys())}")

seed_selections = {}
for seed, d in seed_dirs.items():
    f = os.path.join(d, "selected_I11.npy")
    if os.path.exists(f):
        seed_selections[seed] = np.load(f).tolist()
        print(f"  seed{seed}: {len(seed_selections[seed])} proteins")

n_seeds = len(seed_selections)
print(f"\nTotal seeds with results: {n_seeds}")

# Count frequency of each protein
counts = np.zeros(len(feat_cols), dtype=int)
for sel in seed_selections.values():
    for idx in sel:
        counts[idx] += 1

# Build frequency table
rows = []
for i, c in enumerate(counts):
    if c > 0:
        name = feat_cols[i]
        parts = name.split(":")
        rows.append({
            "feat_idx": i,
            "gene": parts[0],
            "uniprot": parts[1] if len(parts) > 1 else "",
            "panel": parts[4] if len(parts) > 4 else "",
            "n_seeds": c,
            "freq": c / n_seeds,
            "full_name": name,
        })

df = pd.DataFrame(rows).sort_values("n_seeds", ascending=False)
df.to_csv(f"{OUT}/protein_frequency.csv", index=False)

print(f"\n{'='*60}")
print(f"CONSENSUS PANEL RESULTS ({n_seeds} seeds)")
print(f"{'='*60}")

for thresh in range(n_seeds, 0, -1):
    sub = df[df["n_seeds"] >= thresh]
    if len(sub) > 0:
        print(f"\n  ≥{thresh}/{n_seeds} seeds: {len(sub)} proteins")
        if len(sub) <= 30:
            for _, row in sub.iterrows():
                print(f"    [{row['n_seeds']:2d}x] {row['gene']:<12} {row['uniprot']:<10} {row['panel']}")

# Consensus panel = proteins selected by ≥ half of seeds
half = n_seeds // 2 + (1 if n_seeds % 2 else 0)
consensus = df[df["n_seeds"] >= half]
print(f"\n{'='*60}")
print(f"CONSENSUS PANEL (≥{half}/{n_seeds} seeds): {len(consensus)} proteins")
print(f"{'='*60}")
for _, row in consensus.iterrows():
    print(f"  [{row['n_seeds']:2d}x] {row['gene']:<12} {row['uniprot']:<10} {row['panel']}")

# Panel distribution
print(f"\nPanel distribution:")
print(consensus["panel"].value_counts().to_string())

# Pairwise overlaps
seeds = sorted(seed_selections.keys())
print(f"\nPairwise overlaps (n proteins in common / 50):")
for i, s1 in enumerate(seeds):
    for s2 in seeds[i+1:]:
        overlap = len(set(seed_selections[s1]) & set(seed_selections[s2]))
        print(f"  seed{s1} ∩ seed{s2}: {overlap}/50")
all_seeds = set.intersection(*[set(v) for v in seed_selections.values()])
print(f"  All {n_seeds} seeds: {len(all_seeds)}/50")

summary = {
    "n_seeds": n_seeds,
    "seeds": seeds,
    "consensus_threshold": int(half),
    "n_consensus_proteins": int(len(consensus)),
    "consensus_proteins": consensus[["gene","uniprot","panel","n_seeds","feat_idx"]].to_dict("records"),
    "n_seen_by_any_seed": int(len(df)),
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUT}/")
